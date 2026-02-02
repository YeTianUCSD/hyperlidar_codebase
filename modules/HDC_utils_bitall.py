from torchhd import functional
from torchhd import embeddings

import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

#self.hd_nbits = int(self.ARCH.get("train", {}).get("hd_quant_bits", 6))



def quantize_signed_nbit(x: torch.Tensor, n_bits: int = 6, eps: float = 1e-8):
    """
    Quantize a float tensor to signed n-bit with 2's complement style range.

    For example:
        n_bits = 6 -> q in [-32, 31]
        n_bits = 4 -> q in [ -8,  7]

    Args:
        x: float tensor to quantize.
        n_bits: number of bits (e.g., 4, 6, 8).
        eps: small constant to avoid division by zero.

    Returns:
        q: int8 tensor with values in [qmin, qmax].
        scale: scalar float, so that x_hat ≈ q * scale.
    """
    # Signed range for n-bit 2's complement
    qmax = (1 << (n_bits - 1)) - 1   # e.g.,  31 for 6-bit, 7 for 4-bit
    qmin = - (1 << (n_bits - 1))     # e.g., -32 for 6-bit, -8 for 4-bit

    # Use symmetric range based on max absolute value
    alpha = x.abs().max()
    alpha = torch.clamp(alpha, min=eps)

    # Map [-alpha, alpha] approximately to [qmin, qmax]
    scale = alpha / float(qmax)

    # Real quantization: round to integer grid, then clamp to n-bit range
    q = torch.round(x / scale)
    q = torch.clamp(q, qmin, qmax).to(torch.int8)

    return q, scale


class Model(nn.Module):
    def __init__(self, ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
        super(Model, self).__init__()

        self.device = device

        # Record the current number of class hypervectors
        self.num_classes = num_classes      # Used in supervised HD
        self.hd_dim = 10000
        self.temperature = 0.01

        self.flatten = torch.nn.Flatten()

        # Set the input dimension of CNN features
        self.input_dim = 128
        self.ARCH = ARCH

        # Global bit-width for HD quantization (shared with BasicHD)
        # You can set ARCH["train"]["hd_quant_bits"] in YAML, e.g., 8 / 6 / 4 / 2
        self.hd_nbits = int(self.ARCH.get("train", {}).get("hd_quant_bits", 4))
        print(f"[HD-Model] Using {self.hd_nbits}-bit quantization inside Model.")

        # ------------------------------------------------------------------
        # Load CNN backbone
        # ------------------------------------------------------------------
        with torch.no_grad():
            torch.nn.Module.dump_patches = True
            if self.ARCH["train"]["pipeline"] == "hardnet":
                from modules.network.HarDNet import HarDNet
                # self.num_classes comes from semantic-*.yaml (e.g., 17/20)
                self.net = HarDNet(self.num_classes, self.ARCH["train"]["aux_loss"])

            if self.ARCH["train"]["pipeline"] == "res":
                from modules.network.ResNet import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

            if self.ARCH["train"]["pipeline"] == "fid":
                from modules.network.Fid import ResNet_34
                self.net = ResNet_34(self.num_classes, self.ARCH["train"]["aux_loss"])

                def convert_relu_to_softplus(model, act):
                    for child_name, child in model.named_children():
                        if isinstance(child, nn.LeakyReLU):
                            setattr(model, child_name, act)
                        else:
                            convert_relu_to_softplus(child, act)

                if self.ARCH["train"]["act"] == "Hardswish":
                    convert_relu_to_softplus(self.net, nn.Hardswish())
                elif self.ARCH["train"]["act"] == "SiLU":
                    convert_relu_to_softplus(self.net, nn.SiLU())

        w_dict = torch.load(modeldir + "/SENet_valid_best",
                            map_location=lambda storage, loc: storage)

        # Debug before loading state_dict
        try:
            print("[DEBUG] head out_channels (before load):",
                  getattr(self.net, "semantic_output").weight.shape[0])
        except Exception as e:
            print("[DEBUG] cannot read head channels:", e)

        self.net.load_state_dict(w_dict['state_dict'], strict=True)
        self.net.eval()
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.net.cuda()
        else:
            self.gpu = False

        # ------------------------------------------------------------------
        # HD encoder setup
        # ------------------------------------------------------------------
        self.hd_encoder = hd_encoder
        if self.hd_encoder == 'rp':  # Random projection encoding
            # Generate a random projection matrix
            self.projection = embeddings.Projection(self.input_dim, self.hd_dim)

        elif self.hd_encoder == 'idlevel':  # ID-level encoding
            # Generate id-level value hv for each floating value
            self.value = embeddings.Level(num_levels, self.hd_dim,
                                          randomness=randomness)
            print("self.value", self.value.weight.shape)   # [num_levels, hd_dim]
            # Create a random hv for each position, for binding with the value hv
            self.position = embeddings.Random(self.input_dim, self.hd_dim)
            print("self.position", self.position.weight.shape)  # [input_dim, hd_dim]

        elif self.hd_encoder == 'nonlinear':  # Nonlinear encoding
            self.nonlinear_projection = embeddings.Sinusoid(self.input_dim, self.hd_dim)

        else:  # No encoder, use raw samples
            self.hd_dim = self.input_dim

        # ------------------------------------------------------------------
        # Classification head in HD space
        # ------------------------------------------------------------------
        self.classify = nn.Linear(self.hd_dim, self.num_classes, bias=False)
        self.classify_sample_cnt = torch.zeros((self.num_classes, 1)).to(self.device)

        # Initialize classifier weights to zero
        self.classify.weight.data.fill_(0.0)

        # classify_weights is the sum of all hypervectors per class
        self.classify_weights = copy.deepcopy(self.classify.weight)
        # shape: [num_classes, hd_dim]

    # ----------------------------------------------------------------------
    # Encode CNN features into HD hypervectors
    # ----------------------------------------------------------------------
    def encode(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)
        # x shape example before backbone: [B, C_in, 64, 512]

        with torch.cuda.amp.autocast(enabled=True):
            # CNN forward: x -> feature map [B, 128, H, W]
            x = self.net(x, True)

        # Rearrange to [B*H*W, C=128] as point-wise features
        x = x.permute(0, 2, 3, 1)   # [B, H, W, 128]
        x = x.reshape(-1, 128)      # [B*H*W, 128]

        if PERCENTAGE is not None:
            # Select a subset of positions based on loss information (existing logic)
            num_samples = int(x.shape[0] * PERCENTAGE)
            num_wrongdata = num_samples // 2

            # is_wrong here is a "loss-like" value; sort in descending order
            sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            top_indices = sorted_indices[:num_wrongdata]

            all_indices = torch.arange(is_wrong.shape[0], device=x.device)
            temp = torch.ones_like(is_wrong, dtype=torch.bool)
            temp[top_indices] = False
            remaining_indices = all_indices[temp]

            remaining = num_samples - num_wrongdata
            if remaining_indices.numel() >= remaining:
                random_fill_indices = remaining_indices[
                    torch.randperm(remaining_indices.shape[0], device=x.device)[:remaining]
                ]
            else:
                # If not enough remaining, take all of them
                random_fill_indices = remaining_indices

            selected_indices = torch.cat([top_indices, random_fill_indices], dim=0)
            is_wrong[selected_indices] = 0  # Mark selected indices as used

            # Optionally re-sort by updated is_wrong (kept for compatibility)
            sorted_loss, sorted_indices = torch.sort(is_wrong, descending=True)
            selected_indices = sorted_indices[:num_samples]
            is_wrong[selected_indices] = 0.0

            # Filter input features
            x = x[selected_indices]
        else:
            # Use all positions
            selected_indices = torch.arange(x.shape[0], device=x.device)

        # Allocate tensor for hypervectors
        sample_hv = torch.zeros((x.shape[0], self.hd_dim),
                                device=self.device, dtype=x.dtype)

        # ------------------------------------------------------------------
        # HD encoding
        # ------------------------------------------------------------------
        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)
            sample_hv[:, mask] = self.projection(x)[:, mask]

        elif self.hd_encoder == 'idlevel':
            tmp_hv = functional.bind(
                self.position.weight[:, mask],
                self.value(x)[:, :, mask]
            )  # [N, num_features, hd_dim]
            sample_hv[:, mask] = functional.multiset(tmp_hv)  # [N, hd_dim]

        elif self.hd_encoder == 'nonlinear':
            sample_hv[:, mask] = self.nonlinear_projection(x)[:, mask]

        else:  # No encoder, just return raw CNN features
            return x, selected_indices, is_wrong

        # Binarize to bipolar hypervectors in {-1, +1}
        sample_hv[:, mask] = functional.hard_quantize(sample_hv[:, mask])
        # sample_hv shape example: [B*H*W, hd_dim]

        # ------------------------------------------------------------------
        # Compute and quantize frame-averaged hypervector to signed n-bit
        # NOTE:
        #   - We only compute a meaningful frame-level HV when using all samples
        #     (PERCENTAGE is None). In the retraining case with sampling, the
        #     average would not represent the full frame, so we skip it.
        # ------------------------------------------------------------------
        if PERCENTAGE is None:
            # Compute frame-averaged HV over all positions in this batch
            # If batch size is 1 (typical), this gives [1, hd_dim].
            frame_avg_hv = sample_hv.mean(dim=0, keepdim=True)  # [1, hd_dim]

            # n-bit quantization (e.g., 4/6/8-bit) using global hd_nbits
            frame_avg_hv_q, frame_avg_hv_scale = quantize_signed_nbit(
                frame_avg_hv, n_bits=self.hd_nbits
            )

            # De-quantized float representation
            frame_avg_hv_dequant = frame_avg_hv_q.to(sample_hv.dtype) * frame_avg_hv_scale

            # Store for later export / inspection
            self.last_frame_avg_hv = frame_avg_hv_dequant        # [1, hd_dim], float
            self.last_frame_avg_hv_q = frame_avg_hv_q            # [1, hd_dim], int8
            self.last_frame_avg_hv_scale = frame_avg_hv_scale    # scalar float

        # Return per-position hypervectors for segmentation training/inference
        return sample_hv, selected_indices, is_wrong

    # ----------------------------------------------------------------------
    # Forward: compute logits from input images
    # ----------------------------------------------------------------------
    def forward(self, x, mask=None, PERCENTAGE=None, is_wrong=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        # Encode to HD space
        enc, indices, is_wrong_left = self.encode(x, mask, PERCENTAGE, is_wrong)

        # Compute class scores (cosine-like) using float weights
        if enc.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc.dtype)
        logits = self.classify(F.normalize(enc))

        return logits, F.normalize(enc), indices, is_wrong_left

    # ----------------------------------------------------------------------
    # Get predictions (supports float or quantized n-bit path)
    # ----------------------------------------------------------------------
    def get_predictions(self, enc, use_quantized: bool = False):
        """
        Compute class scores for encoded hypervectors.

        Args:
            enc: tensor of shape [N, hd_dim], float hypervectors.
            use_quantized: if True and quantized class weights are available,
                           use int8 dot-product with n-bit quantization.
                           Otherwise fall back to the original float path.

        Returns:
            logits: tensor of shape [N, num_classes], float scores.
        """
        # 1) Fallback: original float path (used during training or if not quantized)
        if (not use_quantized) or (not hasattr(self, "classify_weights_q")):
            # Compute cosine-distance-like scores: W * normalize(enc)^T
            if enc.dtype != self.classify.weight.dtype:
                self.classify = self.classify.to(enc.dtype)
            logits = self.classify(F.normalize(enc))
            return logits

        # 2) Quantized inference path: real n-bit (int8) dot-product
        #    We assume class hypervectors have already been quantized by
        #    BasicHD.quantize_class_hv_nbit(), which sets:
        #       - self.classify_weights_q: [num_classes, hd_dim], int8
        #       - self.classify_weights_scale: scalar float
        w_q = self.classify_weights_q          # int8, [C, D]
        w_scale = self.classify_weights_scale  # scalar float

        # Normalize the input hypervectors to match the training behavior
        enc_norm = F.normalize(enc)  # [N, D], float in roughly [-1, 1]

        # Quantize enc_norm to signed n-bit as well
        # enc_q: int8 in [qmin, qmax], enc_scale: scalar float
        enc_q, enc_scale = quantize_signed_nbit(enc_norm, n_bits=self.hd_nbits)

        # Compute integer logits: shape [N, C]
        # Use int32 to avoid overflow when summing products of int8 values.
        logits_int = torch.matmul(
            enc_q.to(torch.int32),           # [N, D]
            w_q.t().to(torch.int32)         # [D, C]
        )

        # Total real scale factor for dot-product scores
        full_scale = enc_scale * w_scale  # scalar

        # Convert back to float for the rest of the pipeline
        logits = logits_int.to(enc.dtype) * full_scale

        return logits

    # ----------------------------------------------------------------------
    # Extractors (unchanged)
    # ----------------------------------------------------------------------
    def extract_class_hv(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        else:  # self.method == 'BasicHD'
            # class_hv = self.classify_weights / self.classify_sample_cnt
            class_hv = self.classify.weight[:, mask]
        return class_hv.detach().cpu().numpy()

    def extract_pair_simil(self, mask=None):
        if mask is None:
            mask = torch.ones(self.hd_dim, device=self.device).type(torch.bool)

        if self.method == 'LifeHD' or self.method == 'LifeHDsemi':
            class_hv = self.classify.weight[:self.cur_classes, mask]
        elif self.method == 'BasicHD':
            class_hv = self.classify.weight[:, mask]
        else:
            raise ValueError('method not supported: {}'.format(self.method))
        pair_simil = class_hv @ class_hv.T

        if self.method == 'LifeHDsemi':
            pair_simil[:self.num_classes, :self.num_classes] = torch.eye(self.num_classes)
        return pair_simil.detach().cpu().numpy(), class_hv.detach().cpu().numpy()


def set_model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device):
    return Model(ARCH, modeldir, hd_encoder, num_levels, randomness, num_classes, device)
