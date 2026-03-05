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
    No-zero symmetric signed n-bit quantization.

    For n_bits=4 -> q in {-8..-1, +1..+8}  (no 0)
    For n_bits=6 -> q in {-32..-1, +1..+32} (no 0)

    NOTE: This is not two's-complement 4-bit representable set (since +8 exists),
          but it's fine since we store in int8 and enforce the range logically.
    """
    assert 1 <= n_bits <= 8, f"n_bits must be in [1,8], got {n_bits}"

    levels = 1 << (n_bits - 1)  # 4->8, 6->32
    alpha = x.abs().max().clamp_min(eps)

    # scale maps integer levels back to real values
    scale = alpha / float(levels)

    q = torch.round(x / scale)
    q = torch.clamp(q, -levels, levels)

    # remove zeros: push 0 to +/-1 according to sign of x (0 treated as +)
    q = torch.where(
        q == 0,
        torch.where(x >= 0, torch.ones_like(q), -torch.ones_like(q)),
        q
    )

    return q.to(torch.int8), scale



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
        # Memory controls for large-frame HD encoding.
        self.encode_chunk_size = int(self.ARCH.get("train", {}).get("hd_encode_chunk_size", 2048))
        self.encode_store_int8 = bool(self.ARCH.get("train", {}).get("hd_encode_store_int8", True))
        print(
            f"[HD-Model] encode_chunk_size={self.encode_chunk_size}, "
            f"encode_store_int8={self.encode_store_int8}"
        )

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

        # ------------------------------------------------------------------
        # HD encoding (chunked for memory safety on large N)
        # ------------------------------------------------------------------
        n_points = x.shape[0]
        out_dtype = torch.int8 if self.encode_store_int8 else x.dtype
        sample_hv = torch.empty((n_points, self.hd_dim), device=self.device, dtype=out_dtype)
        need_frame_avg = (PERCENTAGE is None)
        frame_sum_hv = None
        if need_frame_avg:
            frame_sum_hv = torch.zeros((1, self.hd_dim), device=self.device, dtype=torch.float32)

        if self.hd_encoder == 'rp':
            if x.dtype != self.projection.weight.dtype:
                self.projection = self.projection.to(x.dtype).to(self.device)

            chunk = max(1, int(self.encode_chunk_size))
            for s in range(0, n_points, chunk):
                e = min(s + chunk, n_points)
                hv_chunk = self.projection(x[s:e])
                hv_chunk = functional.hard_quantize(hv_chunk[:, mask])
                if need_frame_avg:
                    frame_sum_hv += hv_chunk.to(torch.float32).sum(dim=0, keepdim=True)
                if self.encode_store_int8:
                    sample_hv[s:e].fill_(1)
                    sample_hv[s:e, mask] = hv_chunk.to(torch.int8)
                else:
                    sample_hv[s:e].zero_()
                    sample_hv[s:e, mask] = hv_chunk

        elif self.hd_encoder == 'idlevel':
            # Keep original path: idlevel bind/multiset is already expensive and
            # less common in this repo's lidar setup.
            sample_hv_fp = torch.zeros((n_points, self.hd_dim), device=self.device, dtype=x.dtype)
            tmp_hv = functional.bind(
                self.position.weight[:, mask],
                self.value(x)[:, :, mask]
            )  # [N, num_features, hd_dim]
            sample_hv_fp[:, mask] = functional.multiset(tmp_hv)  # [N, hd_dim]
            sample_hv_fp[:, mask] = functional.hard_quantize(sample_hv_fp[:, mask])
            if need_frame_avg:
                frame_sum_hv += sample_hv_fp.to(torch.float32).sum(dim=0, keepdim=True)
            sample_hv = sample_hv_fp.to(torch.int8) if self.encode_store_int8 else sample_hv_fp

        elif self.hd_encoder == 'nonlinear':
            chunk = max(1, int(self.encode_chunk_size))
            for s in range(0, n_points, chunk):
                e = min(s + chunk, n_points)
                hv_chunk = self.nonlinear_projection(x[s:e])
                hv_chunk = functional.hard_quantize(hv_chunk[:, mask])
                if need_frame_avg:
                    frame_sum_hv += hv_chunk.to(torch.float32).sum(dim=0, keepdim=True)
                if self.encode_store_int8:
                    sample_hv[s:e].fill_(1)
                    sample_hv[s:e, mask] = hv_chunk.to(torch.int8)
                else:
                    sample_hv[s:e].zero_()
                    sample_hv[s:e, mask] = hv_chunk

        else:  # No encoder, just return raw CNN features
            return x, selected_indices, is_wrong
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
            frame_avg_hv = frame_sum_hv / float(max(1, n_points))  # [1, hd_dim]

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
        enc_for_cls = enc
        if not torch.is_floating_point(enc_for_cls):
            enc_for_cls = enc_for_cls.to(self.classify.weight.dtype)

        # Compute class scores (cosine-like) using float weights
        if enc_for_cls.dtype != self.classify.weight.dtype:
            self.classify = self.classify.to(enc_for_cls.dtype)
        logits = self.classify(F.normalize(enc_for_cls))

        return logits, F.normalize(enc_for_cls), indices, is_wrong_left

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
            # Fast path for float enc
            if torch.is_floating_point(enc):
                if enc.dtype != self.classify.weight.dtype:
                    self.classify = self.classify.to(enc.dtype)
                logits = self.classify(F.normalize(enc))
                return logits

            # Memory-safe path for int8 enc: compute logits chunk-by-chunk
            w = self.classify.weight
            if w.dtype != torch.float32:
                w = w.to(torch.float32)
            w_n = F.normalize(w, dim=1)  # [C, D]

            n = enc.shape[0]
            c = w_n.shape[0]
            logits = torch.empty((n, c), device=enc.device, dtype=torch.float32)
            chunk = max(1, int(self.encode_chunk_size))
            for s in range(0, n, chunk):
                e = min(s + chunk, n)
                enc_chunk = enc[s:e].to(torch.float32)
                enc_chunk = F.normalize(enc_chunk, dim=1)
                logits[s:e] = torch.matmul(enc_chunk, w_n.t())
            return logits

        # 2) Quantized inference path: real n-bit (int8) dot-product
        #    We assume class hypervectors have already been quantized by
        #    BasicHD.quantize_class_hv_nbit(), which sets:
        #       - self.classify_weights_q: [num_classes, hd_dim], int8
        #       - self.classify_weights_scale: scalar float
        w_q = self.classify_weights_q          # int8, [C, D]
        w_scale = self.classify_weights_scale  # scalar float

        # Chunked quantized path to avoid OOM on large validation batches.
        # 1) normalize in chunks and estimate a global quantization scale
        # 2) quantize+matmul in chunks with that shared scale
        n = enc.shape[0]
        c = w_q.shape[0]
        chunk = max(1, int(self.encode_chunk_size))
        w_q_t = w_q.t().to(torch.float32)  # [D, C]
        logits = torch.empty((n, c), device=enc.device, dtype=torch.float32)

        levels = 1 << (self.hd_nbits - 1)
        eps = 1e-8
        alpha = torch.tensor(0.0, device=enc.device, dtype=torch.float32)

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            enc_chunk = enc[s:e]
            if not torch.is_floating_point(enc_chunk):
                enc_chunk = enc_chunk.to(torch.float32)
            enc_chunk = F.normalize(enc_chunk, dim=1)
            alpha = torch.maximum(alpha, enc_chunk.abs().max())

        alpha = alpha.clamp_min(eps)
        enc_scale = alpha / float(levels)
        full_scale = enc_scale * w_scale

        for s in range(0, n, chunk):
            e = min(s + chunk, n)
            enc_chunk = enc[s:e]
            if not torch.is_floating_point(enc_chunk):
                enc_chunk = enc_chunk.to(torch.float32)
            enc_chunk = F.normalize(enc_chunk, dim=1)

            enc_q = torch.round(enc_chunk / enc_scale)
            enc_q = torch.clamp(enc_q, -levels, levels)
            enc_q = torch.where(
                enc_q == 0,
                torch.where(enc_chunk >= 0, torch.ones_like(enc_q), -torch.ones_like(enc_q)),
                enc_q
            )

            logits_fp = torch.matmul(enc_q.to(torch.float32), w_q_t)
            logits[s:e] = logits_fp * full_scale

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
