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


VAL_CNT = 10


def quantize_6bit_signed(x: torch.Tensor, eps: float = 1e-8):
    """
    Quantize a float tensor to signed 6-bit in the range [-32, 31].

    Args:
        x: float tensor.
        eps: small constant to avoid division by zero.

    Returns:
        q: int8 tensor with values in [-32, 31].
        scale: scalar float, so that x_hat ≈ q * scale.
    """
    qmin = -32
    qmax = 31

    # Use symmetric range based on max absolute value
    alpha = x.abs().max()
    alpha = torch.clamp(alpha, min=eps)

    # Map [-alpha, alpha] approximately to [-31, 31]
    scale = alpha / float(qmax)

    # Real quantization: round to integer grid, then clamp to 6-bit range
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
        self.epochs = int(self.ARCH.get("train", {}).get("hd_retrain_epochs", 20))

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

        # Infer number of classes from learning_map (ignore -1)
        lmap = self.DATA.get("learning_map", {}) or {}
        tgt = [v for v in lmap.values() if isinstance(v, int) and v >= 0]
        if not tgt:
            raise RuntimeError("Cannot infer num_classes from data cfg")
        self.num_classes = max(tgt) + 1  # e.g., 17

        print("[DEBUG] num_classes (from learning_map) =", self.num_classes)
        print("[DEBUG] parser.get_n_classes()          =", self.parser.get_n_classes())

        # Class-frequency-based loss weights
        epsilon_w = self.ARCH["train"]["epsilon_w"]
        content = torch.zeros(self.num_classes, dtype=torch.float)

        for cl, freq in DATA["content"].items():
            x_cl = self.parser.to_xentropy(cl)  # map actual class to xentropy class
            content[x_cl] += freq
        self.loss_w = 1 / (content + epsilon_w)  # get weights

        for x_cl, w in enumerate(self.loss_w):  # ignore the ones necessary to ignore
            if DATA["learning_ignore"][x_cl]:
                # don't weigh
                self.loss_w[x_cl] = 0
        print("Loss weights from content: ", self.loss_w.data)

        # Build HD model (encoder + HD head)
        self.model = set_model(ARCH, modeldir, 'rp', 0, 0, self.num_classes, self.device)
        print(self.num_classes)

        # Optional post-processing
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"], self.num_classes)
        print(self.num_classes)

        # GPU settings
        self.gpu = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

        # HD mask (if you want to disable some dimensions)
        self.mask = None

    # -------------------- 6-bit quantization for class HV -------------------- #

    def quantize_class_hv_6bit(self):
        """
        Quantize class hypervectors to signed 6-bit and update the model in-place.

        After this call:
            - self.model.classify_weights_q: int8 tensor in [-32, 31]
            - self.model.classify_weights_scale: scalar float
            - self.model.classify_weights: de-quantized float32 (q * scale)
            - self.model.classify.weight: same de-quantized values, used by the classifier
        """
        if not hasattr(self.model, "classify_weights"):
            raise RuntimeError("Model has no attribute 'classify_weights'")

        self.model.eval()
        with torch.no_grad():
            # 1) Current float32 class hypervectors: [num_classes, hd_dim]
            w_fp32 = self.model.classify_weights

            # 2) Normalize before quantization (typical for cosine / dot products)
            w_norm = F.normalize(w_fp32, dim=1)

            # 3) Quantize to signed 6-bit [-32, 31]
            q, scale = quantize_6bit_signed(w_norm)

            # 4) Store integer weights and scale for export / hardware
            self.model.classify_weights_q = q          # [num_classes, hd_dim], int8
            self.model.classify_weights_scale = scale  # scalar float

            # 5) Overwrite float32 weights with de-quantized version
            w_dequant = q.to(torch.float32) * scale

            # Update the internal prototype tensor
            self.model.classify_weights.copy_(w_dequant)

            # Also update the classifier layer used during forward passes
            if hasattr(self.model, "classify") and hasattr(self.model.classify, "weight"):
                self.model.classify.weight.data.copy_(w_dequant)

            print("[HD] Class hypervectors quantized to 6-bit in [-32, 31].")
            print("[HD] Quantization scale (scalar):", float(scale))

    # ------------------------------ main pipeline ---------------------------- #

    def start(self):
        print("Starting training with the HDC online learning:")
        self.model.eval()

        # Build IoU evaluator and ignore list
        self.ignore_class = []
        for i, w in enumerate(self.loss_w):
            if w < 1e-10:
                self.ignore_class.append(i)
                print("Ignoring class ", i, " in IoU evaluation")
        self.evaluator = iouEval(self.num_classes, self.device, self.ignore_class)

        # Initial online HD training pass
        for e in range(1, 2):
            time1 = time.time()
            self.train(self.parser.get_train_set(), self.model, self.logger)
            time2 = time.time()
            print('train epoch {}, total time {:.2f}'.format(e, time2 - time1))

        # Retraining epochs + validation (float model)
        for epoch in range(1, self.epochs + 1):
            time1 = time.time()
            self.retrain(self.parser.get_train_set(), self.model, epoch, self.logger)
            time2 = time.time()
            print('retrain epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            acc = self.validate(self.parser.get_valid_set(), self.model, self.evaluator)
            print('Stream final acc (float): {}'.format(acc))

        # Quantize class hypervectors to 6-bit
        print("[HD] Finished HD training. Quantizing class hypervectors to 6-bit...")
        self.quantize_class_hv_6bit()

        # Evaluate quantized 6-bit model on validation set
        acc_q = self.validate(self.parser.get_valid_set(), self.model, self.evaluator, use_quantized=True)
        print('Stream final acc (6-bit quantized): {}'.format(acc_q))

    # ------------------------------ training loop ---------------------------- #

    def train(self, train_loader, model, logger):
        """Training on single-pass of data (online HD learning)."""
        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            idx = 0  # batch index
            cur_class = -1
            self.mask = None
            train_time = []
            # store the wrong classification (or loss) for each batch
            self.is_wrong_list = [None] * len(train_loader)

            for i, (proj_in, proj_mask, proj_labels, unproj_labels,
                    path_seq, path_name, p_x, p_y,
                    proj_range, unproj_range, _, _, _, _, npoints) in enumerate(
                        tqdm(train_loader, desc="Training")
                    ):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()
                # Encode each pixel / position into a hypervector
                samples_hv, _, _ = self.model.encode(proj_in, self.mask)
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                # Flatten labels: [1, 64, 512] -> [N]
                proj_labels = proj_labels.view(-1).to(self.device)

                # Update class prototypes (classify_weights)
                model.classify_weights.index_add_(0, proj_labels, samples_hv)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                train_time.append(res)

                # Compute predictions with current class prototypes
                start = time.time()
                predictions = self.model.get_predictions(samples_hv)  # float path
                argmax = predictions.argmax(dim=1)  # [N]

                # Select misclassified samples
                is_wrong = proj_labels != argmax
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                # Compute losses = score_wrong - score_true for misclassified samples
                true_scores = predictions[is_wrong, proj_labels]
                wrong_scores = predictions[is_wrong, argmax]
                losses = wrong_scores - true_scores

                # Initialize per-batch loss buffer if needed
                if self.is_wrong_list[i] is None or self.is_wrong_list[i].shape != is_wrong.shape:
                    self.is_wrong_list[i] = torch.zeros_like(is_wrong, dtype=losses.dtype)
                self.is_wrong_list[i][is_wrong] = losses

            # Normalize class prototypes
            model.classify.weight[:] = F.normalize(model.classify_weights)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC training time:{}\t std:{}".format(np.mean(train_time), np.std(train_time)))

    def retrain(self, train_loader, model, epoch, logger):
        """Training of one epoch on single-pass of data (sample-based retraining)."""
        buffer_percent = 0.05
        print("Training in ", buffer_percent)

        batchs_per_class = np.floor(len(train_loader) / self.num_classes).astype('int')
        if self.gpu:
            torch.cuda.empty_cache()

        with torch.no_grad():
            idx = 0  # batch index
            cur_class = -1
            total_miss = 0
            retrain_time = []

            for i, (proj_in, proj_mask, proj_labels, unproj_labels,
                    path_seq, path_name, p_x, p_y,
                    proj_range, unproj_range, _, _, _, _, npoints) in enumerate(
                        tqdm(train_loader, desc="Retraining")
                    ):
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()

                start = time.time()
                model.classify.weight[:] = F.normalize(model.classify_weights)

                # Forward: sample a subset of hypervectors based on stored losses
                predictions, samples_hv, indices, self.is_wrong_list[i] = model(
                    proj_in, self.mask, buffer_percent, self.is_wrong_list[i]
                )
                argmax = predictions.argmax(dim=1)  # [N_sampled]

                proj_labels = proj_labels.view(-1).to(self.device)
                proj_labels = proj_labels[indices]  # map to sampled hypervectors

                is_wrong = proj_labels != argmax
                if is_wrong.sum().item() == 0:
                    continue

                total_miss += is_wrong.sum().item()
                proj_labels = proj_labels[is_wrong]
                argmax = argmax[is_wrong]
                samples_hv = samples_hv[is_wrong]
                samples_hv = samples_hv.to(model.classify_weights.dtype)

                # Compute score differences for sampled misclassified points
                true_scores = predictions[is_wrong, proj_labels]
                wrong_scores = predictions[is_wrong, argmax]
                losses = wrong_scores - true_scores
                if losses.sum().item() < 0:
                    print("Warning: negative losses detected, this is not expected")

                wrong_indices_within_selected = is_wrong.nonzero(as_tuple=False).squeeze()
                actual_wrong_indices = indices[wrong_indices_within_selected]
                self.is_wrong_list[i][actual_wrong_indices] = losses.to(self.is_wrong_list[i].dtype)

                # Update class prototypes
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, proj_labels, samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)
                model.classify_weights.index_add_(0, argmax, -samples_hv)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                retrain_time.append(res)

            print("total_miss: ", total_miss)
            print("sum of is_wrong_list: ", sum([x.sum().item() for x in self.is_wrong_list if x is not None]))
            print("Mean HDC retraining time:{}\t std:{}".format(np.mean(retrain_time), np.std(retrain_time)))

    # ------------------------------ validation ------------------------------- #

    def validate(self, val_loader, model, evaluator, use_quantized: bool = False):
        """Validation, evaluate linear classification accuracy and kNN accuracy."""
        losses = AverageMeter()
        jaccs = AverageMeter()
        wces = AverageMeter()
        acc = AverageMeter()
        iou = AverageMeter()
        validation_time = []

        evaluator.reset()

        with torch.no_grad():
            for i, (proj_in, proj_mask, proj_labels, unproj_labels,
                    path_seq, path_name, p_x, p_y,
                    proj_range, unproj_range, _, _, _, _, npoints) in enumerate(
                        tqdm(val_loader, desc="Validation")
                    ):
                path_seq = path_seq[0]
                path_name = path_name[0]

                B, C, H, W = proj_in.shape[0], proj_in.shape[1], proj_in.shape[2], proj_in.shape[3]

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
                    # ---- Quantized 6-bit path ----
                    # 1) Encode to hypervectors
                    samples_hv, _, _ = model.encode(proj_in, self.mask)
                    # 2) Get logits using int8 dot-product with 6-bit quantization
                    predictions = model.get_predictions(samples_hv, use_quantized=True)
                    predictions = predictions.view(B, H, W, self.num_classes)
                    predictions = predictions.permute(0, 3, 1, 2)  # [B, C, H, W]

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                res = time.time() - start
                validation_time.append(res)

                # Compute argmax per pixel
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
