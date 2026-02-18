from __future__ import print_function

import os
import copy
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.HDC_utils import set_model
from modules.ioueval import iouEval


def quantize_signed_nbit(x: torch.Tensor, n_bits: int = 6, eps: float = 1e-8):
    assert 2 <= n_bits <= 8, f"n_bits must be in [2,8], got {n_bits}"
    levels = 1 << (n_bits - 1)
    alpha = x.abs().max().clamp_min(eps)
    scale = alpha / float(levels)
    q = torch.round(x / scale)
    q = torch.clamp(q, -levels, levels)
    q = torch.where(q == 0, torch.where(x >= 0, torch.ones_like(q), -torch.ones_like(q)), q)
    return q.to(torch.int8), scale


class BasicHDOnline:
    """Unsupervised, scene-wise online HD adaptation.

    Design goals:
    - Keep CNN frozen (feature extractor only)
    - Update only HD prototypes with high-confidence pseudo labels
    - Reduce drift via EMA update + source anchor pullback
    - Prevent collapse via class-balanced sampling + memory replay
    - Guard performance with per-scene validation + rollback
    """

    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, logger=None):
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.logger = logger

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Quantized eval options
        self.hd_mode = str(self.ARCH.get("train", {}).get("hd_quant_mode", "float")).lower()
        self.hd_nbits = int(self.ARCH.get("train", {}).get("hd_quant_bits", 4))

        # Online adaptation config
        tr_cfg = self.ARCH.get("train", {})
        self.base_tau_prob = float(tr_cfg.get("online_tau_prob", 0.30))
        self.base_tau_margin = float(tr_cfg.get("online_tau_margin", 0.04))
        self.tau_prob = self.base_tau_prob
        self.tau_margin = self.base_tau_margin
        self.temp = float(tr_cfg.get("online_temperature", 0.7))
        self.min_tau_prob = float(tr_cfg.get("online_min_tau_prob", 0.15))
        self.min_tau_margin = float(tr_cfg.get("online_min_tau_margin", 0.01))
        self.use_teacher = bool(tr_cfg.get("online_use_teacher", True))
        self.teacher_momentum = float(tr_cfg.get("online_teacher_momentum", 0.995))
        self.teacher_tau_prob = float(tr_cfg.get("online_teacher_tau_prob", 0.35))
        self.teacher_tau_margin = float(tr_cfg.get("online_teacher_tau_margin", 0.05))
        self.teacher_base_tau_prob = float(tr_cfg.get("online_teacher_base_tau_prob", self.teacher_tau_prob))
        self.teacher_base_tau_margin = float(tr_cfg.get("online_teacher_base_tau_margin", self.teacher_tau_margin))
        self.teacher_min_tau_prob = float(tr_cfg.get("online_teacher_min_tau_prob", 0.20))
        self.teacher_min_tau_margin = float(tr_cfg.get("online_teacher_min_tau_margin", 0.02))
        self.zero_update_tau_prob_step = float(tr_cfg.get("online_zero_update_tau_prob_step", 0.03))
        self.zero_update_tau_margin_step = float(tr_cfg.get("online_zero_update_tau_margin_step", 0.01))
        self.fallback_topk_per_sample = int(tr_cfg.get("online_fallback_topk_per_sample", 1024))
        self.fallback_for_update = bool(tr_cfg.get("online_fallback_for_update", False))
        # Adaptive ranking selector (primary gate)
        self.select_top_ratio = float(tr_cfg.get("online_select_top_ratio", 0.02))
        self.select_min_k = int(tr_cfg.get("online_select_min_k", 256))
        self.select_max_k = int(tr_cfg.get("online_select_max_k", 4096))
        self.consistency_bonus = float(tr_cfg.get("online_consistency_bonus", 0.10))
        self.inconsistency_penalty = float(tr_cfg.get("online_inconsistency_penalty", 0.05))
        # Confidence-quality gate for update trigger.
        self.conf_pass_min_for_update = int(tr_cfg.get("online_conf_pass_min_for_update", 2048))
        self.confpass_top_ratio = float(tr_cfg.get("online_confpass_top_ratio", 0.01))
        self.confpass_min_k = int(tr_cfg.get("online_confpass_min_k", 256))
        self.confpass_max_k = int(tr_cfg.get("online_confpass_max_k", 4096))

        self.ema_alpha = float(tr_cfg.get("online_ema_alpha", 0.08))
        self.anchor_lambda = float(tr_cfg.get("online_anchor_lambda", 0.015))

        # <= 0 means no per-class cap (use all selected points).
        self.max_per_class = int(tr_cfg.get("online_max_per_class_per_scene", 0))
        self.min_per_class = int(tr_cfg.get("online_min_per_class", 32))
        self.min_update_points = int(tr_cfg.get("online_min_update_points", 5000))
        # If > 0, trigger one update every N streamed scenes (preferred for stability tests).
        self.update_every_n_scenes = int(tr_cfg.get("online_update_every_n_scenes", 5))

        self.eval_every_scene = bool(tr_cfg.get("online_eval_every_scene", True))
        self.max_drop = float(tr_cfg.get("online_guard_max_drop", 0.005))
        self.guard_use_best = bool(tr_cfg.get("online_guard_use_best", True))
        self.adapt_from_valid = bool(tr_cfg.get("online_adapt_from_valid", True))

        self.memory_cap_per_class = int(tr_cfg.get("online_memory_cap_per_class", 512))
        self.replay_per_class = int(tr_cfg.get("online_replay_per_class", 64))
        self.replay_alpha_scale = float(tr_cfg.get("online_replay_alpha_scale", 0.5))

        self.adapt_tau_step = float(tr_cfg.get("online_tau_step", 0.02))
        self.adapt_alpha_decay = float(tr_cfg.get("online_alpha_decay", 0.8))
        self.adapt_alpha_grow = float(tr_cfg.get("online_alpha_grow", 1.02))
        self.source_proto_path = tr_cfg.get("online_source_proto_path", None)
        self.require_source_proto = bool(tr_cfg.get("online_require_source_proto", True))
        self.require_source_projection = bool(tr_cfg.get("online_require_source_projection", True))
        self.resume_ckpt_path = tr_cfg.get("online_resume_ckpt_path", None)

        # Dataset parsers:
        # - train_parser: stream adaptation data, unlabeled (gt=False)
        # - val_parser: validation data, labeled (gt=True)
        from dataset.kitti.parser import Parser
        labels_for_training = self.DATA.get("labels_coarse", self.DATA["labels"])
        self.train_parser = Parser(
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
            gt=False,
            shuffle_train=False,
        )
        # Build an online stream loader without dropping tail samples.
        self.train_loader_online = torch.utils.data.DataLoader(
            self.train_parser.train_dataset,
            batch_size=self.ARCH["train"]["batch_size"],
            shuffle=False,
            num_workers=self.ARCH["train"]["workers"],
            drop_last=False,
        )
        # Online stream should be deterministic and augmentation-free.
        if hasattr(self.train_parser, "train_dataset"):
            self.train_parser.train_dataset.transform = False
        self.val_parser = Parser(
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
            shuffle_train=False,
        )
        # Build a full validation loader (no drop_last) to avoid metric bias.
        self.val_loader_online = torch.utils.data.DataLoader(
            self.val_parser.valid_dataset,
            batch_size=self.ARCH["train"]["batch_size"],
            shuffle=False,
            num_workers=self.ARCH["train"]["workers"],
            drop_last=False,
        )

        lmap = self.DATA.get("learning_map", {}) or {}
        tgt = [v for v in lmap.values() if isinstance(v, int) and v >= 0]
        if not tgt:
            raise RuntimeError("Cannot infer num_classes from data cfg")
        self.num_classes = max(tgt) + 1

        self.model = set_model(self.ARCH, self.modeldir, "rp", 0, 0, self.num_classes, self.device)
        self.model.eval()

        self.gpu = False
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            self.gpu = True
            self.model.cuda()

        # Ignore classes for IoU
        self.ignore_class = []
        for c, ign in self.DATA["learning_ignore"].items():
            if ign:
                self.ignore_class.append(int(c))

        self.evaluator = iouEval(self.num_classes, self.device, self.ignore_class)

        # Online state
        self.mask = None
        self.classify_weights_raw = self.model.classify_weights.detach().clone()
        self.teacher_weights_raw = self.model.classify_weights.detach().clone()
        self.source_anchor = copy.deepcopy(self.model.classify_weights.detach())
        self.best_valid_iou = -1.0
        self.last_valid_iou = -1.0
        self.best_state = None
        self.resume_scene_idx = 0
        self.resume_update_idx = 0
        self.resume_processed_scene_order = 0
        # Memory buffers by class (CPU tensors)
        self.mem_hv = {c: None for c in range(self.num_classes)}

        os.makedirs(self.logdir, exist_ok=True)
        ts = time.strftime("%Y%m%d-%H%M%S")
        base = f"online_log_{ts}"
        cand = os.path.join(self.logdir, f"{base}.txt")
        if os.path.exists(cand):
            suffix = 1
            while True:
                cand = os.path.join(self.logdir, f"{base}_{suffix:02d}.txt")
                if not os.path.exists(cand):
                    break
                suffix += 1
        self.log_path = cand

    # ------------------------------
    # Logging helpers
    # ------------------------------
    def _log(self, msg: str):
        print(msg)
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    # ------------------------------
    # Prototype helpers
    # ------------------------------
    def _normalize_rows(self, w):
        n = w.norm(dim=1, keepdim=True)
        return torch.where(n > 1e-12, w / n, w)

    def _init_random_prototypes_if_needed(self):
        with torch.no_grad():
            w = self.classify_weights_raw
            if torch.any(w.abs() > 0):
                return
            rnd = torch.randn_like(w)
            rnd = F.normalize(rnd, dim=1)
            for c in self.ignore_class:
                rnd[c].zero_()
            self.classify_weights_raw.copy_(rnd)

    def _sync_classifier_weight(self):
        with torch.no_grad():
            w = self.classify_weights_raw
            safe = self._normalize_rows(w)
            self.model.classify_weights.data.copy_(safe)
            self.model.classify.weight.data.copy_(safe)

    def _update_teacher_from_student(self):
        if not self.use_teacher:
            return
        with torch.no_grad():
            self.teacher_weights_raw.mul_(self.teacher_momentum)
            self.teacher_weights_raw.add_((1.0 - self.teacher_momentum) * self.classify_weights_raw)
            self.teacher_weights_raw.copy_(self._normalize_rows(self.teacher_weights_raw))

    def _resolve_source_proto_path(self):
        if self.source_proto_path is not None and str(self.source_proto_path).strip() != "":
            return str(self.source_proto_path)
        default_path = os.path.join(self.modeldir, "HD_source_prototypes.pt")
        if os.path.isfile(default_path):
            return default_path
        return None

    def _load_source_prototypes(self):
        path = self._resolve_source_proto_path()
        if path is None:
            if self.require_source_proto:
                raise RuntimeError(
                    "Source prototype file is required but not found. "
                    "Set train.online_source_proto_path or place HD_source_prototypes.pt under model dir."
                )
            return False

        if not os.path.isfile(path):
            if self.require_source_proto:
                raise RuntimeError(f"Source prototype file not found: {path}")
            return False

        ext = os.path.splitext(path)[1].lower()
        obj = None
        proj_obj = None
        if ext == ".npz":
            npz = np.load(path, allow_pickle=True)
            # Prefer normalized prototypes if available; raw sums are scale-sensitive.
            for k in ("classify_weights", "prototype", "prototypes", "w",
                      "classify_weights_raw", "prototype_raw", "prototypes_raw"):
                if k in npz:
                    obj = npz[k]
                    break
            for k in ("projection_weight", "projection", "proj_w"):
                if k in npz:
                    proj_obj = npz[k]
                    break
            if obj is None:
                raise RuntimeError(f"Unsupported npz keys in source prototype file: {list(npz.keys())}")
        else:
            ckpt = torch.load(path, map_location="cpu")
            if torch.is_tensor(ckpt):
                obj = ckpt
            elif isinstance(ckpt, dict):
                # Prefer normalized prototypes if available; raw sums are scale-sensitive.
                for k in ("classify_weights", "prototype", "prototypes", "w",
                          "classify_weights_raw", "prototype_raw", "prototypes_raw"):
                    if k in ckpt:
                        obj = ckpt[k]
                        break
                for k in ("projection_weight", "projection", "proj_w"):
                    if k in ckpt:
                        proj_obj = ckpt[k]
                        break
                if obj is None:
                    raise RuntimeError(f"Unsupported prototype checkpoint keys: {list(ckpt.keys())[:20]}")
            else:
                raise RuntimeError(f"Unsupported source prototype format type: {type(ckpt)}")

        if not torch.is_tensor(obj):
            obj = torch.tensor(obj)
        w = obj.to(torch.float32).to(self.device)
        if w.dim() != 2:
            raise RuntimeError(f"Source prototypes must be 2D [C, D], got shape={tuple(w.shape)}")
        if w.shape[0] != self.num_classes:
            raise RuntimeError(
                f"Source prototypes class dim mismatch: got {w.shape[0]}, expected {self.num_classes}"
            )
        if w.shape[1] != self.classify_weights_raw.shape[1]:
            raise RuntimeError(
                f"Source prototypes hd_dim mismatch: got {w.shape[1]}, expected {self.classify_weights_raw.shape[1]}"
            )

        if proj_obj is None:
            if self.require_source_projection:
                raise RuntimeError(
                    "Source prototype file missing projection_weight, which is required to keep "
                    "source/online HD space consistent."
                )
        else:
            if not torch.is_tensor(proj_obj):
                proj_obj = torch.tensor(proj_obj)
            proj_w = proj_obj.to(torch.float32).to(self.device)
            if not hasattr(self.model, "projection") or not hasattr(self.model.projection, "weight"):
                raise RuntimeError("Current HD model has no projection.weight but source projection was provided.")
            model_proj_w = self.model.projection.weight
            if proj_w.shape != model_proj_w.shape:
                raise RuntimeError(
                    f"Projection shape mismatch: source {tuple(proj_w.shape)} vs model {tuple(model_proj_w.shape)}"
                )

        with torch.no_grad():
            if proj_obj is not None:
                self.model.projection.weight.data.copy_(proj_w.to(self.model.projection.weight.dtype))
            # Always normalize to remove dependence on sample-count scale.
            n = w.norm(dim=1, keepdim=True)
            w_norm = torch.where(n > 1e-12, w / n, w)
            for c in self.ignore_class:
                w_norm[c].zero_()
            self.classify_weights_raw.copy_(w_norm)
            self._sync_classifier_weight()
            self.teacher_weights_raw.copy_(w_norm)
            self.source_anchor = copy.deepcopy(w_norm.detach())

        self._log(f"[BOOT] Loaded source prototypes from: {path}")
        return True

    def _snapshot_state(self):
        mem = {}
        for c in range(self.num_classes):
            t = self.mem_hv[c]
            mem[c] = None if t is None else t.clone()
        return {
            "classify_weights_raw": self.classify_weights_raw.detach().clone(),
            "teacher_weights_raw": self.teacher_weights_raw.detach().clone(),
            "tau_prob": self.tau_prob,
            "tau_margin": self.tau_margin,
            "teacher_tau_prob": self.teacher_tau_prob,
            "teacher_tau_margin": self.teacher_tau_margin,
            "ema_alpha": self.ema_alpha,
            "mem_hv": mem,
        }

    def _rollback_state(self, state):
        with torch.no_grad():
            self.classify_weights_raw.copy_(state["classify_weights_raw"])
            if "teacher_weights_raw" in state:
                self.teacher_weights_raw.copy_(state["teacher_weights_raw"])
            self._sync_classifier_weight()
        self.tau_prob = state["tau_prob"]
        self.tau_margin = state["tau_margin"]
        self.teacher_tau_prob = state.get("teacher_tau_prob", self.teacher_tau_prob)
        self.teacher_tau_margin = state.get("teacher_tau_margin", self.teacher_tau_margin)
        self.ema_alpha = state["ema_alpha"]
        if "mem_hv" in state:
            restored = {}
            for c in range(self.num_classes):
                t = state["mem_hv"].get(c, None)
                restored[c] = None if t is None else t.clone()
            self.mem_hv = restored

    def _quantize_class_hv_nbit(self):
        with torch.no_grad():
            w_fp32 = self.model.classify_weights
            w_norm = F.normalize(w_fp32, dim=1)
            q, scale = quantize_signed_nbit(w_norm, n_bits=self.hd_nbits)
            self.model.classify_weights_q = q
            self.model.classify_weights_scale = scale
            w_dequant = q.to(torch.float32) * scale
            self.model.classify_weights.copy_(w_dequant)
            self.model.classify.weight.data.copy_(w_dequant)

    # ------------------------------
    # Pseudo labels and selection
    # ------------------------------
    def _predict_logits(self, proj_in):
        samples_hv, _, _ = self.model.encode(proj_in, self.mask)
        logits_s = self.model.get_predictions(samples_hv, use_quantized=False)
        if self.use_teacher:
            w_t = self._normalize_rows(self.teacher_weights_raw).to(samples_hv.dtype)
            logits_t = torch.matmul(samples_hv, w_t.t())
        else:
            logits_t = logits_s
        return samples_hv, logits_s, logits_t

    def _select_high_confidence(self, probs_s, pseudo_s, probs_t, pseudo_t, valid_mask):
        top2_s = torch.topk(probs_s, k=2, dim=1).values
        conf_s = top2_s[:, 0]
        margin_s = top2_s[:, 0] - top2_s[:, 1]
        top2_t = torch.topk(probs_t, k=2, dim=1).values
        conf_t = top2_t[:, 0]
        margin_t = top2_t[:, 0] - top2_t[:, 1]

        cand = valid_mask.clone()
        for c in self.ignore_class:
            cand = cand & (pseudo_t != c)

        consistent = cand & (pseudo_s == pseudo_t)
        conf_pass = cand & (conf_s >= self.tau_prob) & (conf_t >= self.teacher_tau_prob)
        margin_pass = cand & (margin_s >= self.tau_margin) & (margin_t >= self.teacher_tau_margin)

        # Adaptive ranking score: confidence + margin as main signal,
        # teacher-student consistency used as auxiliary adjustment.
        score = 0.5 * (conf_s + conf_t) + 0.5 * (margin_s + margin_t)
        if self.use_teacher:
            score = score + self.consistency_bonus * consistent.float()
            score = score - self.inconsistency_penalty * (cand & (~consistent)).float()

        select = torch.zeros_like(cand, dtype=torch.bool)
        cand_idx = torch.nonzero(cand, as_tuple=False).squeeze(1)
        if cand_idx.numel() > 0:
            k_ratio = int(float(cand_idx.numel()) * self.select_top_ratio)
            k = max(self.select_min_k, k_ratio)
            k = min(k, self.select_max_k, int(cand_idx.numel()))
            top_local = torch.topk(score[cand_idx], k=k, dim=0).indices
            top_idx = cand_idx[top_local]
            select[top_idx] = True

            # If strict confidence gate is empty, build an adaptive confidence pass set
            # from top-confidence candidates so update gating can still work.
            if int(conf_pass.sum().item()) == 0:
                kc_ratio = int(float(cand_idx.numel()) * self.confpass_top_ratio)
                kc = max(self.confpass_min_k, kc_ratio)
                kc = min(kc, self.confpass_max_k, int(cand_idx.numel()))
                conf_rank = 0.5 * (conf_s[cand_idx] + conf_t[cand_idx])
                topc_local = torch.topk(conf_rank, k=kc, dim=0).indices
                conf_idx = cand_idx[topc_local]
                conf_pass = torch.zeros_like(cand, dtype=torch.bool)
                conf_pass[conf_idx] = True

        return select, conf_t, margin_t, cand, consistent, conf_pass, margin_pass

    # ------------------------------
    # Memory replay
    # ------------------------------
    def _memory_add(self, hv, labels):
        hv_cpu = hv.detach().cpu()
        lb_cpu = labels.detach().cpu()
        added = torch.zeros((self.num_classes,), dtype=torch.long)

        for c in range(self.num_classes):
            if c in self.ignore_class:
                continue
            idx = torch.nonzero(lb_cpu == c, as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            cur = hv_cpu[idx]
            old = self.mem_hv[c]
            added[c] = int(cur.shape[0])
            if old is None:
                merged = cur
            else:
                merged = torch.cat([old, cur], dim=0)

            if merged.shape[0] > self.memory_cap_per_class:
                keep = merged[-self.memory_cap_per_class :]
            else:
                keep = merged
            self.mem_hv[c] = keep
        return added

    def _memory_sample(self):
        hv_list = []
        lb_list = []
        for c in range(self.num_classes):
            if c in self.ignore_class:
                continue
            mem = self.mem_hv[c]
            if mem is None or mem.shape[0] == 0:
                continue
            n = min(self.replay_per_class, mem.shape[0])
            idx = torch.randperm(mem.shape[0])[:n]
            hv_list.append(mem[idx])
            lb_list.append(torch.full((n,), c, dtype=torch.long))

        if not hv_list:
            return None, None

        hv = torch.cat(hv_list, dim=0).to(self.device)
        lb = torch.cat(lb_list, dim=0).to(self.device)
        return hv, lb

    # ------------------------------
    # Prototype updates
    # ------------------------------
    def _apply_ema_anchor_update_from_stats(self, class_sum, class_count, alpha):
        updated_classes = 0
        with torch.no_grad():
            for c in range(self.num_classes):
                if c in self.ignore_class:
                    continue
                cnt = int(class_count[c].item())
                if cnt <= 0:
                    continue

                mean_hv = class_sum[c] / float(cnt)
                cur_raw = self.classify_weights_raw[c]
                new_raw = (1.0 - alpha) * cur_raw + alpha * mean_hv

                # Anchor pullback in direction space, then project back with raw magnitude.
                raw_norm = torch.norm(new_raw, p=2).clamp_min(1e-12)
                dir_new = new_raw / raw_norm
                anc = self.source_anchor[c].to(new_raw.device)
                dir_mix = (1.0 - self.anchor_lambda) * dir_new + self.anchor_lambda * anc
                dir_mix = F.normalize(dir_mix, dim=0)
                new_raw = dir_mix * raw_norm

                self.classify_weights_raw[c].copy_(new_raw)
                updated_classes += 1

            self._sync_classifier_weight()

        return updated_classes

    # ------------------------------
    # Scene adaptation
    # ------------------------------
    def _new_scene_accumulator(self):
        hd_dim = self.classify_weights_raw.shape[1]
        return {
            "total_valid": 0,
            "total_selected": 0,
            "total_consistent": 0,
            "total_conf_pass": 0,
            "total_margin_pass": 0,
            "total_candidate": 0,
            "fallback_selected": 0,
            "class_sum": torch.zeros(
                (self.num_classes, hd_dim),
                dtype=self.classify_weights_raw.dtype,
                device=self.device,
            ),
            "class_count": torch.zeros((self.num_classes,), dtype=torch.long, device=self.device),
        }

    def _merge_scene_into_pending(self, pending, scene_acc, scene_id):
        pending["total_valid"] += int(scene_acc["total_valid"])
        pending["total_selected"] += int(scene_acc["total_selected"])
        pending["total_consistent"] += int(scene_acc["total_consistent"])
        pending["total_conf_pass"] += int(scene_acc["total_conf_pass"])
        pending["total_margin_pass"] += int(scene_acc["total_margin_pass"])
        pending["total_candidate"] += int(scene_acc["total_candidate"])
        pending["fallback_selected"] += int(scene_acc["fallback_selected"])
        pending["class_sum"] += scene_acc["class_sum"]
        pending["class_count"] += scene_acc["class_count"]
        pending["scene_ids"].append(str(scene_id))

    def _pending_scene_str(self, scene_ids, max_show=8):
        if len(scene_ids) <= max_show:
            return ",".join(scene_ids)
        head = ",".join(scene_ids[: max_show // 2])
        tail = ",".join(scene_ids[-(max_show // 2) :])
        return f"{head},...,{tail}"

    def _make_pending_tag(self, scene_ids):
        if len(scene_ids) == 0:
            return "none"
        if len(scene_ids) <= 6:
            return "+".join(scene_ids)
        return f"{scene_ids[0]}-{scene_ids[-1]}-n{len(scene_ids)}"

    def _class_balanced_indices(self, pseudo, select, scene_class_count):
        selected = []
        for c in range(self.num_classes):
            if c in self.ignore_class:
                continue
            idx = torch.nonzero(select & (pseudo == c), as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                continue
            if self.max_per_class > 0:
                remaining = self.max_per_class - int(scene_class_count[c].item())
                if remaining <= 0:
                    continue
                if idx.numel() > remaining:
                    perm = torch.randperm(idx.numel(), device=idx.device)[:remaining]
                    idx = idx[perm]
            selected.append(idx)
        if not selected:
            return None
        return torch.cat(selected, dim=0)

    def _accumulate_one_sample(self, proj_in, proj_mask, acc):
        with torch.no_grad():
            if self.gpu:
                proj_in = proj_in.cuda(non_blocking=True)
                proj_mask = proj_mask.cuda(non_blocking=True)

            hv, logits_s, logits_t = self._predict_logits(proj_in)
            probs_s = torch.softmax(logits_s / self.temp, dim=1)
            probs_t = torch.softmax(logits_t / self.temp, dim=1)
            pseudo_s = probs_s.argmax(dim=1)
            pseudo_t = probs_t.argmax(dim=1)

            valid_mask = proj_mask.view(-1) > 0
            acc["total_valid"] += int(valid_mask.sum().item())

            select, conf, margin, cand, consistent, conf_pass, margin_pass = self._select_high_confidence(
                probs_s, pseudo_s, probs_t, pseudo_t, valid_mask
            )
            acc["total_candidate"] += int(cand.sum().item())
            acc["total_consistent"] += int(consistent.sum().item())
            acc["total_conf_pass"] += int(conf_pass.sum().item())
            acc["total_margin_pass"] += int(margin_pass.sum().item())
            idx = self._class_balanced_indices(pseudo_t, select, acc["class_count"])
            if (
                (idx is None or idx.numel() == 0)
                and self.fallback_for_update
                and self.fallback_topk_per_sample > 0
            ):
                cand_idx = torch.nonzero(consistent, as_tuple=False).squeeze(1)
                if cand_idx.numel() > 0:
                    k = min(int(self.fallback_topk_per_sample), int(cand_idx.numel()))
                    # Prefer high confidence and larger top-1/top-2 separation.
                    rank_score = conf[cand_idx] + 0.25 * margin[cand_idx]
                    top_local = torch.topk(rank_score, k=k, dim=0).indices
                    top_idx = cand_idx[top_local]
                    fb_select = torch.zeros_like(cand, dtype=torch.bool)
                    fb_select[top_idx] = True
                    idx = self._class_balanced_indices(pseudo_t, fb_select, acc["class_count"])
                    if idx is not None and idx.numel() > 0:
                        acc["fallback_selected"] += int(idx.numel())
            if idx is None or idx.numel() == 0:
                return

            hv_sel = hv[idx]
            lb_sel = pseudo_t[idx]
            acc["total_selected"] += int(idx.numel())

            acc["class_sum"].index_add_(0, lb_sel, hv_sel.to(acc["class_sum"].dtype))
            ones = torch.ones_like(lb_sel, dtype=torch.long)
            acc["class_count"].index_add_(0, lb_sel, ones)

            # Update replay memory online without caching full scene features.
            self._memory_add(hv_sel, lb_sel)

    def _finalize_scene_update(self, scene_id, acc):
        total_valid = acc["total_valid"]
        total_selected = acc["total_selected"]
        class_sum = acc["class_sum"]
        class_count = acc["class_count"]
        class_updates = 0

        with torch.no_grad():
            # Apply scene-level minimum support filtering (after full-scene aggregation).
            low_support = class_count < self.min_per_class
            if torch.any(low_support):
                class_sum[low_support] = 0
                class_count[low_support] = 0
            if int(class_count.sum().item()) > 0:
                class_updates += self._apply_ema_anchor_update_from_stats(class_sum, class_count, self.ema_alpha)

                # Replay update with smaller step.
                hv_rep, lb_rep = self._memory_sample()
                if hv_rep is not None:
                    rep_sum = torch.zeros_like(class_sum)
                    rep_cnt = torch.zeros_like(class_count)
                    rep_sum.index_add_(0, lb_rep, hv_rep.to(rep_sum.dtype))
                    rep_ones = torch.ones_like(lb_rep, dtype=torch.long)
                    rep_cnt.index_add_(0, lb_rep, rep_ones)
                    class_updates += self._apply_ema_anchor_update_from_stats(
                        rep_sum, rep_cnt, self.ema_alpha * self.replay_alpha_scale
                    )
            if class_updates > 0:
                self._update_teacher_from_student()

        keep_ratio = 0.0 if total_valid == 0 else total_selected / float(total_valid)
        return {
            "scene": scene_id,
            "valid_points": total_valid,
            "selected_points": total_selected,
            "keep_ratio": keep_ratio,
            "updated_classes": class_updates,
            "candidate_points": int(acc["total_candidate"]),
            "consistent_points": int(acc["total_consistent"]),
            "conf_pass_points": int(acc["total_conf_pass"]),
            "margin_pass_points": int(acc["total_margin_pass"]),
            "fallback_selected": int(acc["fallback_selected"]),
        }

    def _save_checkpoint(self, update_idx, scene_id, iou, acc, tag, processed_scene_order=None):
        mem_dump = {}
        for c in range(self.num_classes):
            t = self.mem_hv[c]
            mem_dump[c] = None if t is None else t.detach().float().cpu().clone()
        ckpt = {
            "scene_idx": int(update_idx),
            "update_idx": int(update_idx),
            "processed_scene_order": int(update_idx if processed_scene_order is None else processed_scene_order),
            "scene_id": str(scene_id),
            "valid_iou": float(iou),
            "valid_acc": float(acc),
            "classify_weights_raw": self.classify_weights_raw.detach().float().cpu(),
            "teacher_weights_raw": self.teacher_weights_raw.detach().float().cpu(),
            "classify_weights": self.model.classify_weights.detach().float().cpu(),
            "source_anchor": self.source_anchor.detach().float().cpu(),
            "tau_prob": float(self.tau_prob),
            "tau_margin": float(self.tau_margin),
            "teacher_tau_prob": float(self.teacher_tau_prob),
            "teacher_tau_margin": float(self.teacher_tau_margin),
            "ema_alpha": float(self.ema_alpha),
            "mem_hv": mem_dump,
            "best_valid_iou": float(self.best_valid_iou),
            "last_valid_iou": float(self.last_valid_iou),
            "arch_train": copy.deepcopy(self.ARCH.get("train", {})),
        }
        if hasattr(self.model, "projection") and hasattr(self.model.projection, "weight"):
            ckpt["projection_weight"] = self.model.projection.weight.detach().float().cpu()
        ckpt_name = f"online_{tag}_upd{int(update_idx):04d}_seq{scene_id}_miou{iou:.4f}.pt"
        ckpt_path = os.path.join(self.logdir, ckpt_name)
        torch.save(ckpt, ckpt_path)
        self._log(f"[CKPT] saved: {ckpt_path}")

    def _load_online_checkpoint(self, path):
        if path is None or str(path).strip() == "":
            return False
        if not os.path.isfile(path):
            raise RuntimeError(f"Resume checkpoint not found: {path}")

        ckpt = torch.load(path, map_location="cpu")
        if not isinstance(ckpt, dict):
            raise RuntimeError(f"Resume checkpoint must be dict, got {type(ckpt)}")

        if "classify_weights_raw" not in ckpt:
            raise RuntimeError("Resume checkpoint missing classify_weights_raw")

        w = ckpt["classify_weights_raw"]
        if not torch.is_tensor(w):
            w = torch.tensor(w)
        w = w.to(torch.float32).to(self.device)
        if w.shape != self.classify_weights_raw.shape:
            raise RuntimeError(
                f"Resume classify_weights_raw shape mismatch: got {tuple(w.shape)}, "
                f"expected {tuple(self.classify_weights_raw.shape)}"
            )

        with torch.no_grad():
            self.classify_weights_raw.copy_(w)
            self._sync_classifier_weight()
            tw = ckpt.get("teacher_weights_raw", None)
            if tw is not None:
                if not torch.is_tensor(tw):
                    tw = torch.tensor(tw)
                tw = tw.to(torch.float32).to(self.device)
                if tw.shape == self.teacher_weights_raw.shape:
                    self.teacher_weights_raw.copy_(tw)
                else:
                    self.teacher_weights_raw.copy_(self._normalize_rows(self.classify_weights_raw))
            else:
                self.teacher_weights_raw.copy_(self._normalize_rows(self.classify_weights_raw))

            if "projection_weight" in ckpt and ckpt["projection_weight"] is not None:
                if not hasattr(self.model, "projection") or not hasattr(self.model.projection, "weight"):
                    raise RuntimeError("Resume ckpt has projection_weight but model has no projection.")
                pw = ckpt["projection_weight"]
                if not torch.is_tensor(pw):
                    pw = torch.tensor(pw)
                pw = pw.to(torch.float32).to(self.device)
                if tuple(pw.shape) != tuple(self.model.projection.weight.shape):
                    raise RuntimeError(
                        f"Resume projection shape mismatch: got {tuple(pw.shape)}, "
                        f"expected {tuple(self.model.projection.weight.shape)}"
                    )
                self.model.projection.weight.data.copy_(pw.to(self.model.projection.weight.dtype))

            if "source_anchor" in ckpt and ckpt["source_anchor"] is not None:
                sa = ckpt["source_anchor"]
                if not torch.is_tensor(sa):
                    sa = torch.tensor(sa)
                sa = sa.to(torch.float32).to(self.device)
                if tuple(sa.shape) != tuple(self.source_anchor.shape):
                    raise RuntimeError(
                        f"Resume source_anchor shape mismatch: got {tuple(sa.shape)}, "
                        f"expected {tuple(self.source_anchor.shape)}"
                    )
                self.source_anchor = sa.detach().clone()
            else:
                self.source_anchor = self.model.classify_weights.detach().clone()

        self.tau_prob = float(ckpt.get("tau_prob", self.tau_prob))
        self.tau_margin = float(ckpt.get("tau_margin", self.tau_margin))
        self.teacher_tau_prob = float(ckpt.get("teacher_tau_prob", self.teacher_tau_prob))
        self.teacher_tau_margin = float(ckpt.get("teacher_tau_margin", self.teacher_tau_margin))
        self.ema_alpha = float(ckpt.get("ema_alpha", self.ema_alpha))
        self.last_valid_iou = float(ckpt.get("last_valid_iou", ckpt.get("valid_iou", self.last_valid_iou)))
        self.best_valid_iou = float(ckpt.get("best_valid_iou", ckpt.get("valid_iou", self.best_valid_iou)))
        self.resume_update_idx = int(ckpt.get("update_idx", ckpt.get("scene_idx", 0)))
        self.resume_scene_idx = self.resume_update_idx
        self.resume_processed_scene_order = int(
            ckpt.get("processed_scene_order", ckpt.get("scene_idx", 0))
        )

        mem = ckpt.get("mem_hv", None)
        if isinstance(mem, dict):
            restored = {}
            for c in range(self.num_classes):
                t = mem.get(c, mem.get(str(c), None))
                if t is None:
                    restored[c] = None
                else:
                    if not torch.is_tensor(t):
                        t = torch.tensor(t)
                    restored[c] = t.to(torch.float32).cpu().clone()
            self.mem_hv = restored

        self.best_state = self._snapshot_state()
        self._log(
            f"[RESUME] Loaded online checkpoint: {path} | update_idx={self.resume_update_idx} "
            f"processed_scenes={self.resume_processed_scene_order} "
            f"last_iou={self.last_valid_iou:.4f} best_iou={self.best_valid_iou:.4f}"
        )
        return True

    # ------------------------------
    # Source bootstrap
    # ------------------------------
    def _bootstrap_source_prototypes(self):
        """Initialize prototypes from precomputed source prototype file."""
        loaded = self._load_source_prototypes()
        if loaded:
            return

        # Fallback path only when explicitly allowed.
        self._init_random_prototypes_if_needed()
        self._sync_classifier_weight()
        self.teacher_weights_raw = copy.deepcopy(self.classify_weights_raw.detach())
        self.source_anchor = copy.deepcopy(self.model.classify_weights.detach())
        self._log("[BOOT] Fallback to random prototype initialization.")
        return

    # ------------------------------
    # Validation
    # ------------------------------
    def validate(self, val_loader, use_quantized=False):
        self.evaluator.reset()
        self.model.eval()

        if self.gpu:
            torch.cuda.empty_cache()

        validation_time = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                (
                    proj_in,
                    proj_mask,
                    proj_labels,
                    unproj_labels,
                    path_seq,
                    path_name,
                    p_x,
                    p_y,
                    proj_range,
                    unproj_range,
                    _,
                    _,
                    _,
                    _,
                    npoints,
                ) = batch

                B, C, H, W = proj_in.shape

                if self.gpu:
                    proj_in = proj_in.cuda(non_blocking=True)
                    proj_mask = proj_mask.cuda(non_blocking=True)
                    proj_labels = proj_labels.cuda(non_blocking=True).long()

                start = time.time()

                if not use_quantized:
                    pred_logits, _, _, _ = self.model(proj_in, self.mask)
                else:
                    hv, _, _ = self.model.encode(proj_in, self.mask)
                    pred_logits = self.model.get_predictions(hv, use_quantized=True)

                pred_logits = pred_logits.view(B, H, W, self.num_classes)
                pred_logits = pred_logits.permute(0, 3, 1, 2)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                validation_time.append(time.time() - start)

                argmax = pred_logits.argmax(dim=1)
                self.evaluator.addBatch(argmax, proj_labels)

        acc = self.evaluator.getacc().item()
        iou, class_iou = self.evaluator.getIoU()
        iou = iou.item()

        t_avg = float(np.mean(validation_time)) if validation_time else float("nan")
        t_std = float(np.std(validation_time)) if validation_time else float("nan")

        self._log(
            f"[VAL] use_quantized={use_quantized} | time={t_avg:.3f}±{t_std:.3f} | acc={acc:.4f} | iou={iou:.4f}"
        )
        # Keep output format close to the previous retrain pipeline for easier comparison.
        self._log(f"Mean HDC validation time:{t_avg}\t std:{t_std}")
        self._log(
            "Validation set:\n"
            f"Time avg per batch {t_avg:.3f} (std {t_std:.3f})\n"
            "Loss avg 0.0000\n"
            "Jaccard avg 0.0000\n"
            "WCE avg 0.0000\n"
            f"Acc avg {acc:.3f}\n"
            f"IoU avg {iou:.3f}"
        )
        self._log(f"Class Jaccard:  {class_iou}")

        return iou, acc

    # ------------------------------
    # Main entry
    # ------------------------------
    def start(self):
        self._log(f"[LOG] online log file: {self.log_path}")
        self._log("[ONLINE] Start unsupervised scene-stream HD adaptation")
        self._log(
            f"[ONLINE] Runtime eval mode: {'quantized' if self.hd_mode == 'nbit' else 'float'} "
            f"(hd_mode={self.hd_mode}, hd_bits={self.hd_nbits})"
        )
        self._log(
            f"[ONLINE] consistency gate: use_teacher={self.use_teacher} "
            f"tau_s={self.tau_prob:.3f}/{self.tau_margin:.3f} "
            f"tau_t={self.teacher_tau_prob:.3f}/{self.teacher_tau_margin:.3f}"
        )
        self._log(
            f"[ONLINE] adaptive selector: top_ratio={self.select_top_ratio:.4f} "
            f"min_k={self.select_min_k} max_k={self.select_max_k} "
            f"cons_bonus={self.consistency_bonus:.3f} incons_penalty={self.inconsistency_penalty:.3f}"
        )
        self._log(
            f"[ONLINE] update trigger: min_update_points={self.min_update_points}, "
            f"update_every_n_scenes={self.update_every_n_scenes}, "
            f"fallback_for_update={self.fallback_for_update}, fallback_topk={self.fallback_topk_per_sample}"
        )
        self._log(
            f"[ONLINE] confidence-quality gate: conf_pass_min_for_update={self.conf_pass_min_for_update} "
            f"confpass_top_ratio={self.confpass_top_ratio:.4f} "
            f"confpass_min_k={self.confpass_min_k} confpass_max_k={self.confpass_max_k}"
        )
        self._log(
            f"[ONLINE] per-class cap: {'off' if self.max_per_class <= 0 else self.max_per_class}"
        )

        resumed = False
        if self.resume_ckpt_path is not None and str(self.resume_ckpt_path).strip() != "":
            resumed = self._load_online_checkpoint(self.resume_ckpt_path)
        if not resumed:
            # 1) Optional source bootstrap to get stable initial prototypes.
            self._bootstrap_source_prototypes()

            # 2) Initial validation baseline before any target scene update.
            self._log("[ONLINE] Running baseline validation before streaming any new scene")
            base_iou, base_acc = self.validate(self.val_loader_online, use_quantized=False)
            self.last_valid_iou = base_iou
            self.best_valid_iou = base_iou
            self.best_state = self._snapshot_state()
            self._save_checkpoint(0, "baseline", base_iou, base_acc, "best", processed_scene_order=0)
            self._log(f"[ONLINE] Baseline valid iou={base_iou:.4f}, acc={base_acc:.4f}")

        # 3) Stream scenes, but update only when enough selected points are accumulated.
        update_idx = int(self.resume_update_idx) if resumed else 0
        resume_processed_scene_order = int(self.resume_processed_scene_order) if resumed else 0
        current_scene = None
        scene_acc = self._new_scene_accumulator()
        pending_acc = self._new_scene_accumulator()
        pending_acc["scene_ids"] = []
        current_scene_order = 0
        process_current = True

        def _finish_scene(scene_id, acc):
            nonlocal update_idx
            if scene_id is None:
                return

            scene_selected = int(acc["total_selected"])
            scene_valid = int(acc["total_valid"])
            keep_scene = 0.0 if scene_valid == 0 else scene_selected / float(scene_valid)
            self._log(
                f"[SCENE {current_scene_order}] id={scene_id} "
                f"selected={scene_selected}/{scene_valid} keep={keep_scene:.4f} "
                f"cand={int(acc['total_candidate'])} consistent={int(acc['total_consistent'])} "
                f"conf_pass={int(acc['total_conf_pass'])} "
                f"margin_pass={int(acc['total_margin_pass'])} fb_sel={int(acc['fallback_selected'])} "
                f"tau_prob={self.tau_prob:.3f} tau_margin={self.tau_margin:.3f} alpha={self.ema_alpha:.4f}"
            )

            if scene_selected == 0:
                self._log(
                    f"[SCENE {current_scene_order}] selected=0 under adaptive selector; continue accumulation"
                )

            self._merge_scene_into_pending(pending_acc, acc, scene_id)
            pending_scene_num = len(pending_acc["scene_ids"])
            pending_selected = int(pending_acc["total_selected"])
            pending_conf_pass = int(pending_acc["total_conf_pass"])
            pending_ids_str = self._pending_scene_str(pending_acc["scene_ids"])
            if self.update_every_n_scenes > 0:
                if pending_scene_num < self.update_every_n_scenes:
                    self._log(
                        f"[PENDING] scenes={pending_scene_num}/{self.update_every_n_scenes} "
                        f"selected={pending_selected} conf_pass={pending_conf_pass} ids=[{pending_ids_str}] "
                        f"(waiting for scene-count trigger)"
                    )
                    return
                if (pending_scene_num % self.update_every_n_scenes) != 0:
                    self._log(
                        f"[PENDING] scenes={pending_scene_num} "
                        f"selected={pending_selected} conf_pass={pending_conf_pass} ids=[{pending_ids_str}] (waiting for next "
                        f"{self.update_every_n_scenes}-scene boundary)"
                    )
                    return
                if pending_selected <= 0:
                    self._log(
                        f"[PENDING] scenes={pending_scene_num}/{self.update_every_n_scenes} "
                        f"selected=0 ids=[{pending_ids_str}] "
                        "(trigger reached but no high-confidence samples, continue accumulating)"
                    )
                    return
                if pending_conf_pass < self.conf_pass_min_for_update:
                    self._log(
                        f"[PENDING] scenes={pending_scene_num}/{self.update_every_n_scenes} "
                        f"selected={pending_selected} conf_pass={pending_conf_pass}/{self.conf_pass_min_for_update} "
                        f"ids=[{pending_ids_str}] (trigger reached but confidence-quality is low, continue accumulating)"
                    )
                    return
            else:
                if pending_selected < self.min_update_points:
                    self._log(
                        f"[PENDING] scenes={pending_scene_num} "
                        f"selected={pending_selected}/{self.min_update_points} conf_pass={pending_conf_pass} "
                        f"ids=[{pending_ids_str}] "
                        f"(waiting for selected-points trigger)"
                    )
                    return

            update_idx += 1
            pending_tag = self._make_pending_tag(pending_acc["scene_ids"])
            stats = self._finalize_scene_update(pending_tag, pending_acc)
            self._log(
                f"[UPDATE {update_idx}] scenes={len(pending_acc['scene_ids'])} "
                f"selected={stats['selected_points']}/{stats['valid_points']} "
                f"keep={stats['keep_ratio']:.4f} updated_classes={stats['updated_classes']} "
                f"cand={stats['candidate_points']} consistent={stats['consistent_points']} "
                f"conf_pass={stats['conf_pass_points']} ids=[{pending_ids_str}] "
                f"margin_pass={stats['margin_pass_points']} fb_sel={stats['fallback_selected']}"
            )

            pending_acc["scene_ids"] = []
            pending_acc["total_valid"] = 0
            pending_acc["total_selected"] = 0
            pending_acc["total_consistent"] = 0
            pending_acc["total_conf_pass"] = 0
            pending_acc["total_margin_pass"] = 0
            pending_acc["total_candidate"] = 0
            pending_acc["fallback_selected"] = 0
            pending_acc["class_sum"].zero_()
            pending_acc["class_count"].zero_()

            if not self.eval_every_scene:
                return

            new_iou, new_acc = self.validate(self.val_loader_online, use_quantized=False)

            # Guard and rollback on degradation.
            ref_iou = self.best_valid_iou if self.guard_use_best else self.last_valid_iou
            if ref_iou - new_iou > self.max_drop:
                if self.best_state is not None:
                    self._rollback_state(self.best_state)
                if self.adapt_from_valid:
                    # Keep adaptive selector stable; only reduce update aggressiveness.
                    self.ema_alpha = max(1e-4, self.ema_alpha * self.adapt_alpha_decay)
                self._log(
                    f"[GUARD] rollback update={update_idx} scenes={pending_tag} | iou {self.last_valid_iou:.4f}->{new_iou:.4f} "
                    f"ref={ref_iou:.4f} drop={ref_iou - new_iou:.4f} > {self.max_drop:.4f}"
                )
                # Re-evaluate rolled-back state for accurate tracking.
                new_iou, new_acc = self.validate(self.val_loader_online, use_quantized=False)
            elif self.adapt_from_valid:
                # No guard-triggered degradation: mildly increase update plasticity.
                self.ema_alpha = min(0.2, self.ema_alpha * self.adapt_alpha_grow)

            self.last_valid_iou = new_iou
            if new_iou > self.best_valid_iou:
                self.best_valid_iou = new_iou
                self.best_state = self._snapshot_state()
                self._save_checkpoint(
                    update_idx, pending_tag, new_iou, new_acc, "best",
                    processed_scene_order=current_scene_order
                )
            self._log(
                f"[UPDATE {update_idx}] post-valid iou={new_iou:.4f} acc={new_acc:.4f} "
                f"best_iou={self.best_valid_iou:.4f}"
            )

        for batch in tqdm(self.train_loader_online, desc="Streaming train scenes"):
            (
                proj_in,
                proj_mask,
                proj_labels,
                unproj_labels,
                path_seq,
                path_name,
                p_x,
                p_y,
                proj_range,
                unproj_range,
                _,
                _,
                _,
                _,
                npoints,
            ) = batch

            bsz = proj_in.shape[0]
            for b in range(bsz):
                seq = str(path_seq[b])
                if current_scene is None:
                    current_scene = seq
                    current_scene_order = 1
                    process_current = current_scene_order > resume_processed_scene_order
                    if not process_current:
                        self._log(
                            f"[RESUME] Skip scene order={current_scene_order} id={current_scene} "
                            f"(resume_processed_scene_order={resume_processed_scene_order})"
                        )
                if seq != current_scene:
                    if process_current:
                        _finish_scene(current_scene, scene_acc)
                    current_scene = seq
                    current_scene_order += 1
                    process_current = current_scene_order > resume_processed_scene_order
                    scene_acc = self._new_scene_accumulator()
                    if not process_current:
                        self._log(
                            f"[RESUME] Skip scene order={current_scene_order} id={current_scene} "
                            f"(resume_processed_scene_order={resume_processed_scene_order})"
                        )

                if process_current:
                    self._accumulate_one_sample(proj_in[b:b + 1], proj_mask[b:b + 1], scene_acc)

        if process_current:
            _finish_scene(current_scene, scene_acc)

        # Flush remaining pending buffer at the end.
        if len(pending_acc["scene_ids"]) > 0 and int(pending_acc["total_selected"]) > 0:
            update_idx += 1
            pending_tag = self._make_pending_tag(pending_acc["scene_ids"])
            pending_ids_str = self._pending_scene_str(pending_acc["scene_ids"])
            stats = self._finalize_scene_update(pending_tag, pending_acc)
            self._log(
                f"[UPDATE {update_idx}] (final flush) scenes={len(pending_acc['scene_ids'])} "
                f"selected={stats['selected_points']}/{stats['valid_points']} "
                f"keep={stats['keep_ratio']:.4f} updated_classes={stats['updated_classes']} "
                f"ids=[{pending_ids_str}]"
            )
            if self.eval_every_scene:
                new_iou, new_acc = self.validate(self.val_loader_online, use_quantized=False)
                self.last_valid_iou = new_iou
                if new_iou > self.best_valid_iou:
                    self.best_valid_iou = new_iou
                    self.best_state = self._snapshot_state()
                    self._save_checkpoint(
                        update_idx, pending_tag, new_iou, new_acc, "best",
                        processed_scene_order=current_scene_order
                    )
                self._log(
                    f"[UPDATE {update_idx}] post-valid iou={new_iou:.4f} acc={new_acc:.4f} "
                    f"best_iou={self.best_valid_iou:.4f}"
                )

        # 4) Optional quantized evaluation.
        if self.hd_mode == "nbit":
            self._log(f"[ONLINE] Quantized eval enabled: {self.hd_nbits}-bit")
            self._quantize_class_hv_nbit()
            iou_q, acc_q = self.validate(self.val_loader_online, use_quantized=True)
            self._log(f"[ONLINE] Quantized final iou={iou_q:.4f} acc={acc_q:.4f}")
            # Restore float classifier from raw prototypes after quantized eval.
            self._sync_classifier_weight()

        final_iou, final_acc = self.validate(self.val_loader_online, use_quantized=False)
        self._save_checkpoint(
            update_idx, "final", final_iou, final_acc, "last",
            processed_scene_order=current_scene_order
        )

        self._log("[ONLINE] Finished unsupervised scene-stream adaptation")
