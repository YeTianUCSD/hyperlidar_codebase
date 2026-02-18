from __future__ import print_function

import os
import copy
import numpy as np
import sys
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from modules.HDC_utils import set_model
from modules.ioueval import *
import torch.backends.cudnn as cudnn
from postproc.KNN import KNN
from common.avgmeter import *

from torchhd import functional
from torchhd import embeddings


#self.hd_nbits = int(self.ARCH.get("train", {}).get("hd_quant_bits", 4))

VAL_CNT = 10


def quantize_signed_nbit(x: torch.Tensor, n_bits: int = 6, eps: float = 1e-8):
    """
    Quantize a float tensor to signed n-bit with 2's complement style range.

    For example:
        n_bits = 6 -> q in [-32, 31]
        n_bits = 4 -> q in [ -8,  7]

    Args:
        x: float tensor to quantize.
        n_bits: number of bits (e.g., 4 or 6).
        eps: small constant to avoid division by zero.

    Returns:
        q: int8 tensor with values in [qmin, qmax].
        scale: scalar float so that x_hat ≈ q * scale.
    """
    # Signed range for n-bit 2's complement
    qmax = (1 << (n_bits - 1)) - 1          # e.g.,  31 for 6-bit, 7 for 4-bit
    qmin = - (1 << (n_bits - 1))            # e.g., -32 for 6-bit, -8 for 4-bit

    # Use symmetric range based on max absolute value
    alpha = x.abs().max()
    alpha = torch.clamp(alpha, min=eps)

    # Map [-alpha, alpha] approximately to [qmin, qmax]
    scale = alpha / float(qmax)

    # Real quantization: round to integer grid, then clamp
    q = torch.round(x / scale)
    q = torch.clamp(q, qmin, qmax).to(torch.int8)

    return q, scale


class BasicHD():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, logger):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir

        # Number of retraining epochs for HD
        self.epochs = int(self.ARCH.get("train", {}).get("hd_retrain_epochs", 20))

        # Global n-bit setting for HD quantization (class HV + input HV for quantized inference)
        self.hd_nbits = int(self.ARCH.get("train", {}).get("hd_quant_bits", 4))
        print(f"[HD] Using {self.hd_nbits}-bit quantization for HD vectors.")
        # HD quant mode: "float" or "nbit"
        self.hd_mode = str(self.ARCH.get("train", {}).get("hd_quant_mode", "nbit")).lower()
        if self.hd_mode not in ("float", "nbit"):
            raise ValueError(f"Invalid hd_quant_mode={self.hd_mode}, must be 'float' or 'nbit'")

        # int8-backed quantization only supports <= 8 bits
        if self.hd_mode == "nbit" and not (2 <= self.hd_nbits <= 8):
            raise ValueError(f"hd_quant_bits must be in [2,8] for int8 quant, got {self.hd_nbits}")

        print(f"[HD] hd_quant_mode={self.hd_mode}, hd_quant_bits={self.hd_nbits}")

        # ------------------------------------------------------------------
        # Dataset parser and number of classes
        # ------------------------------------------------------------------
        from dataset.kitti.parser import Parser
        labels_for_training = self.DATA.get("labels_coarse", self.DATA["labels"])
        self.parser = Parser(
            root=self.datadir,
            train_sequences=self.DATA["split"]["train"],
            valid_sequences=self.DATA["split"]["valid"],
            test_sequences=None,
            labels=labels_for_training,
            color_map=self.DATA["color_map"],
            learning_map=self.DATA["learning_map"],
            learning_map_inv=self.DATA["learning_map_inv"],
            sensor=self.ARCH["dataset"]["sensor"],
            max_points=self.ARCH["dataset"]["max_points"],
            batch_size=self.ARCH["train"]["batch_size"],
            workers=self.ARCH["train"]["workers"],
            gt=True,
            shuffle_train=False
        )

        # Infer num_classes from learning_map (ignore negative mappings)
        lmap = self.DATA.get("learning_map", {}) or {}
        tgt = [v for v in lmap.values() if isinstance(v, int) and v >= 0]
        if not tgt:
            raise RuntimeError("Cannot infer num_classes from data cfg")
        self.num_classes = max(tgt) + 1
        print("[DEBUG] num_classes (from learning_map) =", self.num_classes)
        print("[DEBUG] parser.get_n_classes()          =", self.parser.get_n_classes())

        # ------------------------------------------------------------------
        # Class-wise loss weights from content frequencies
        # ------------------------------------------------------------------
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.num_classes, dtype=torch.float)

        for cl, freq in self.DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq

        self.loss_w = 1.0 / (content + epsilon_w)
        for x_cl, w in enumerate(self.loss_w):
            if self.DATA["learning_ignore"][x_cl]:
                # Do not weigh ignored classes
                self.loss_w[x_cl] = 0.0
        print("Loss weights from content: ", self.loss_w.data)

        # ------------------------------------------------------------------
        # Build HD model (CNN backbone + HD encoder + classifier)
        # ------------------------------------------------------------------
        self.model = set_model(self.ARCH, self.modeldir, 'rp', 0, 0, self.num_classes, self.device)
        print("[HD] num_classes =", self.num_classes)

        # Optional KNN post-processing
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.num_classes)

        # GPU setup
        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        # Mask over HD dimensions (if None → use all dims)
        self.mask = None
        # Will be initialized during train()
        self.is_wrong_list = None

    # ----------------------------------------------------------------------
    # Quantize class hypervectors to n-bit after training
    # ----------------------------------------------------------------------
    def quantize_class_hv_nbit(self):
        """
        Quantize class hypervectors to signed n-bit and update the model in-place.

        After this call:
            - self.model.classify_weights_q: int8 tensor in [qmin, qmax]
            - self.model.classify_weights_scale: scalar float scale
            - self.model.classify_weights: de-quantized float32 (q * scale)
            - self.model.classify.weight: same de-quantized values, used by the classifier
        """
        if not hasattr(self.model, "classify_weights"):
            raise RuntimeError("Model has no attribute 'classify_weights'")

        self.model.eval()
        with torch.no_grad():
            # 1) Current float32 class hypervectors: [num_classes, hd_dim]
            w_fp32 = self.model.classify_weights

            # 2) Normalize before quantization (typical for cosine/dot products)
            w_norm = F.normalize(w_fp32, dim=1)

            # 3) Quantize to signed n-bit
            q, scale = quantize_signed_nbit(w_norm, n_bits=self.hd_nbits)

            # 4) Store integer weights and scale for export / hardware
            self.model.classify_weights_q = q          # [num_classes, hd_dim], int8
            self.model.classify_weights_scale = scale  # scalar float

            # 5) Overwrite float32 weights with the de-quantized version
            w_dequant = q.to(torch.float32) * scale

            # Update the internal prototype tensor used for training/inference
            self.model.classify_weights.copy_(w_dequant)

            # Also update the classifier layer used during forward passes
            if hasattr(self.model, "classify") and hasattr(self.model.classify, "weight"):
                self.model.classify.weight.data.copy_(w_dequant)

            print(f"[HD] Class hypervectors quantized to {self.hd_nbits}-bit.")
            print("[HD] Quantization scale (scalar):", float(scale))

    # ----------------------------------------------------------------------
    # Training entry point
    # ----------------------------------------------------------------------
    def start(self):
        print("Starting training with the HDC online learning:")
        self.model.eval()

        # Classes to ignore in IoU evaluation
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")

        self.evaluator = iouEval(self.num_classes, self.device, self.ignore_class)

        # ------------------------------------------------------------------
        # First pass: accumulate class hypervectors (one-pass online training)
        # ------------------------------------------------------------------
        time1 = time.time()
        self.train(self.parser.get_train_set(), self.model, self.logger)
        time2 = time.time()
        print('Initial HD train pass, total time {:.2f}s'.format(time2 - time1))

        # ------------------------------------------------------------------
        # Retraining epochs (float32 HD)
        # ------------------------------------------------------------------
        for epoch in range(1, self.epochs + 1):
            time1 = time.time()
            self.retrain(self.parser.get_train_set(), self.model, epoch, self.logger)
            time2 = time.time()
            print('retrain epoch {}, total time {:.2f}s'.format(epoch, time2 - time1))

            # Validation with float HD
            acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator, use_quantized=False)
            print('Stream final acc (float HD): {}'.format(acc))

    
        # ------------------------------------------------------------------
        # Quantized eval (optional): controlled by hd_quant_mode
        # ------------------------------------------------------------------
        if self.hd_mode == "nbit":
            print(f"[HD] Finished HD training. Quantizing class hypervectors to {self.hd_nbits}-bit...")
            self.quantize_class_hv_nbit()

            acc_q = self.validate(self.parser.get_valid_set(), self.model, self.evaluator, use_quantized=True)
            print('Stream final acc ({}-bit quantized): {}'.format(self.hd_nbits, acc_q))
        else:
            print("[HD] hd_quant_mode=float -> skip quantization & quantized eval.")


    # ----------------------------------------------------------------------
    # First-pass training: accumulate class prototypes
    # ----------------------------------------------------------------------
    def train(self, train_loader, model, logger):
        """Training on single-pass of data (prototype accumulation)."""
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            self.mask = None
            self.is_wrong_list = [None] * len(train_loader)  # store the loss / wrongness per pixel
            train_time = []

            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq,
                    path_name, p_x, p_y, proj_range, unproj_range,
                    _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Training")):

                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()
                # encode() returns per-pixel hypervectors: [B*H*W, hd_dim]
                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                # Flatten labels: shape [B, H, W] -> [B*H*W]
                proj_labels = proj_labels.view(-1).to(self.device)

                # Update class hypervectors (sum of all samples in the class)
                model.classify_weights.index_add_(0, proj_labels, samples_hv)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                train_time.append(res)
                start = time.time()

                # Compute predictions with float HD (for "wrongness" statistics)
                predictions = self.model.get_predictions(samples_hv, use_quantized=False)
                argmax = predictions.argmax(dim=1)  # [N]

                is_wrong = proj_labels != argmax

                # Keep only wrong predictions
                proj_labels_wrong = proj_labels[is_wrong]
                argmax_wrong = argmax[is_wrong]
                samples_hv_wrong = samples_hv[is_wrong]

                # Loss = score_wrong - score_true (margin-like signal)
                true_scores = predictions[is_wrong, proj_labels_wrong]  # [N_wrong]
                wrong_scores = predictions[is_wrong, argmax_wrong]      # [N_wrong]
                losses = wrong_scores - true_scores                     # [N_wrong]

                # Initialize or resize is_wrong_list[i]
                if self.is_wrong_list[i] is None or self.is_wrong_list[i].shape != is_wrong.shape:
                    self.is_wrong_list[i] = torch.zeros_like(is_wrong, dtype=losses.dtype)

                # Store "loss" at wrong positions (others keep 0)
                self.is_wrong_list[i][is_wrong] = losses

            # Normalize class weights for classifier layer
            model.classify.weight[:] = F.normalize(model.classify_weights)
            print("sum of is_wrong_list: ",
                  sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))

    # ----------------------------------------------------------------------
    # Retraining: refine prototypes based on misclassifications
    # ----------------------------------------------------------------------
    def retrain(self, train_loader, model, epoch, logger):
        """Training of one epoch on single-pass of data (refinement)."""
        buffer_percent = 0.05
        print("Retraining with buffer_percent =", buffer_percent)

        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            total_miss = 0
            retrain_time = []

            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq,
                    path_name, p_x, p_y, proj_range, unproj_range,
                    _, _, _, _, npoints) in enumerate(tqdm(train_loader, desc="Retraining")):

                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()
                model.classify.weight[:] = F.normalize(model.classify_weights)

                # Forward with buffer sampling based on self.is_wrong_list[i]
                predictions, samples_hv, indices, self.is_wrong_list[i] = model(
                    proj_in, self.mask, buffer_percent, self.is_wrong_list[i]
                )

                argmax = predictions.argmax(dim=1)  # [N_selected]

                proj_labels_flat = proj_labels.view(-1).to(self.device)
                proj_labels_sel = proj_labels_flat[indices]  # map to selected hypervectors

                is_wrong = proj_labels_sel != argmax

                if is_wrong.sum().item() == 0:
                    # Nothing to update in this batch
                    continue

                total_miss += is_wrong.sum().item()

                # Keep only wrong predictions
                proj_labels_wrong = proj_labels_sel[is_wrong]
                argmax_wrong = argmax[is_wrong]
                samples_hv_wrong = samples_hv[is_wrong]
                samples_hv_wrong = samples_hv_wrong.to(model.classify_weights.dtype)

                # Compute margin-like loss for wrong samples
                true_scores = predictions[is_wrong, proj_labels_wrong]
                wrong_scores = predictions[is_wrong, argmax_wrong]
                losses = wrong_scores - true_scores

                if losses.sum().item() < 0:
                    print("Warning: negative losses detected, this is not expected")

                # Map indices back to the full frame index space
                wrong_indices_within_selected = is_wrong.nonzero(as_tuple=False).squeeze()
                actual_wrong_indices = indices[wrong_indices_within_selected]

                # Update stored "wrongness" info
                self.is_wrong_list[i][actual_wrong_indices] = losses.to(self.is_wrong_list[i].dtype)

                # Update class prototypes:
                #   - Add samples to true class
                #   - Subtract from wrong predicted class
                model.classify_weights.index_add_(0, proj_labels_wrong, samples_hv_wrong)
                model.classify_weights.index_add_(0, proj_labels_wrong, samples_hv_wrong)
                model.classify_weights.index_add_(0, argmax_wrong, -samples_hv_wrong)
                model.classify_weights.index_add_(0, argmax_wrong, -samples_hv_wrong)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                retrain_time.append(res)

            print("total_miss: ", total_miss)
            print("sum of is_wrong_list: ",
                  sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC retraining time:{}\t std:{}".format(np.mean(retrain_time), np.std(retrain_time)))

    # ----------------------------------------------------------------------
    # Validation: can use float HD or quantized HD
    # ----------------------------------------------------------------------
    def validate(self, val_loader, model, evaluator, use_quantized: bool = False):
        """Validation: evaluate segmentation IoU / accuracy."""
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        rand_imgs = []
        evaluator.reset()
        validation_time = []

        with torch.no_grad():
            for i, (proj_in, proj_mask, proj_labels, unproj_labels, path_seq,
                    path_name, p_x, p_y, proj_range, unproj_range,
                    _, _, _, _, npoints) in enumerate(tqdm(val_loader, desc="Validation")):

                path_seq = path_seq[0]
                path_name = path_name[0]
                B, C, H, W = proj_in.shape

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                start = time.time()

                if not use_quantized:
                    # ---- Original float path ----
                    predictions, _, _, _ = model(proj_in, self.mask)
                    # predictions: [B*H*W, num_classes]
                    predictions = predictions.view(B, H, W, self.num_classes)
                    predictions = predictions.permute(0, 3, 1, 2)  # [B, C, H, W]
                else:
                    # ---- Quantized n-bit path ----
                    # 1) Encode to hypervectors (float, bipolar)
                    samples_hv, _, _ = model.encode(proj_in, self.mask)
                    # samples_hv: [B*H*W, hd_dim]

                    # 2) Get logits using int8 dot-product with n-bit quantization
                    predictions = model.get_predictions(samples_hv, use_quantized=True)
                    # predictions: [B*H*W, num_classes]
                    predictions = predictions.view(B, H, W, self.num_classes)
                    predictions = predictions.permute(0, 3, 1, 2)  # [B, C, H, W]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)

                # Argmax over classes
                argmax = predictions.argmax(dim=1)   # [B, H, W]
                argmax = argmax.squeeze(0)           # [H, W] for B=1

                proj_labels = proj_labels.to(self.device)
                evaluator.addBatch(argmax, proj_labels)

        print("Mean HDC validation time:{}\t std:{}".format(np.mean(validation_time), np.std(validation_time)))
        accuracy = evaluator.getacc()
        jaccard, class_jaccard = evaluator.getIoU()
        acc.update(accuracy.item(), proj_in.size(0))
        iou.update(jaccard.item(), proj_in.size(0))

        time_avg = float(np.mean(validation_time)) if len(validation_time) > 0 else float("nan")
        time_std = float(np.std(validation_time)) if len(validation_time) > 0 else float("nan")

        print('Validation set:\n'
              'Time avg per batch {:.3f} (std {:.3f})\n'
              'Loss avg {loss.avg:.4f}\n'
              'Jaccard avg {jac.avg:.4f}\n'
              'WCE avg {wces.avg:.4f}\n'
              'Acc avg {acc.avg:.3f}\n'
              'IoU avg {iou.avg:.3f}'.format(
                  time_avg, time_std, loss=losses, jac=jaccs, wces=wces, acc=acc, iou=iou))

        print('Class Jaccard: ', class_jaccard)
        return iou.avg
