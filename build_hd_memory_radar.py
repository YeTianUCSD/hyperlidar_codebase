import _init_path
import argparse
import datetime
import os
import re
from pathlib import Path
import warnings
import logging
import sys

warnings.filterwarnings("ignore")

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network
from pcdet.utils import common_utils


def parse_config():
    parser = argparse.ArgumentParser(description="Build HD memory (prototypes) on train split (cell-level)")

    parser.add_argument("--cfg_file", type=str, required=True,
                        help="cfg file, e.g. cfgs/kitti_models/pointpillar_vod_hd.yaml")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="checkpoint path, e.g. .../checkpoint_epoch_110.pth")

    parser.add_argument("--batch_size", type=int, default=None, help="batch size")
    parser.add_argument("--workers", type=int, default=0, help="dataloader workers")
    parser.add_argument("--extra_tag", type=str, default="default", help="extra tag for output dir")

    parser.add_argument("--launcher", choices=["none", "pytorch", "slurm"], default="none")
    parser.add_argument("--tcp_port", type=int, default=18888)
    parser.add_argument("--local_rank", type=int, default=0)

    parser.add_argument("--set", dest="set_cfgs", default=None, nargs=argparse.REMAINDER,
                        help="set extra config keys if needed")

    # Memory build options
    parser.add_argument("--save_path", type=str, default=None,
                        help="explicit path to save hd_memory.pth. If None, save under output/.../hd_memory/")
    parser.add_argument("--max_batches", type=int, default=-1,
                        help="limit number of batches for quick debug (-1 means full pass)")
    parser.add_argument("--use_fp16", action="store_true", default=False,
                        help="use autocast fp16 for forward (feature extraction)")
    parser.add_argument("--chunk_cells", type=int, default=65536,
                        help="chunk size for cell features when encoding (avoid peak memory)")

    # NEW: feature source for memory build (must match AnchorHeadSingle HD.FEAT_SOURCE)
    parser.add_argument("--feat_source", type=str, default=None, choices=["auto", "bev", "cls"],
                        help="Which feature to build HD memory from: "
                             "'bev'=spatial_features_2d, 'cls'=hd_cls_feat. "
                             "Default auto: use cfg.MODEL.DENSE_HEAD.HD.FEAT_SOURCE if exists else 'cls'.")

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = "/".join(args.cfg_file.split("/")[1:-1])

    np.random.seed(1024)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def _infer_epoch_id_from_ckpt(ckpt_path: str) -> str:
    num_list = re.findall(r"\d+", ckpt_path)
    return num_list[-1] if len(num_list) > 0 else "no_number"


def _get_num_anchors_per_loc(dense_head) -> int:
    if hasattr(dense_head, "num_anchors_per_location"):
        v = dense_head.num_anchors_per_location
        if isinstance(v, (list, tuple)):
            return int(sum(v))
        return int(v)
    raise RuntimeError("dense_head has no num_anchors_per_location attribute.")


@torch.no_grad()
def _ensure_hd_memory_ready(dense_head, logger):
    hd_core = getattr(dense_head, "hd_core", None)
    if hd_core is None:
        raise RuntimeError("dense_head.hd_core is None. Enable HD in cfg and rebuild model.")
    if not hasattr(hd_core, "embedder") or not hasattr(hd_core, "memory"):
        raise RuntimeError("hd_core missing embedder/memory.")
    mem = hd_core.memory
    if not hasattr(mem, "classify_weights") or not hasattr(mem, "prototypes"):
        raise RuntimeError("hd_core.memory missing classify_weights/prototypes.")

    logger.info(
        f"HD memory tensors: classify_weights={tuple(mem.classify_weights.shape)}, "
        f"prototypes={tuple(mem.prototypes.shape)}"
    )
    return hd_core


def _resolve_feat_source(args, cfg_local) -> str:
    """
    Priority:
      1) CLI --feat_source if provided (and not auto)
      2) cfg.MODEL.DENSE_HEAD.HD.FEAT_SOURCE if exists
      3) default 'cls'
    """
    if args.feat_source is not None and args.feat_source != "auto":
        return args.feat_source

    try:
        fs = str(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "FEAT_SOURCE", "cls")).lower()
        if fs in ("bev", "cls"):
            return fs
    except Exception:
        pass
    return "cls"


def _feat_key_from_source(feat_source: str) -> str:
    if feat_source == "cls":
        return "hd_cls_feat"
    return "spatial_features_2d"


@torch.no_grad()
def _get_feat_tensor_after_forward(dense_head, batch_dict, model_ret, feat_key: str, logger):
    """
    Correct feature fetch order:
      1) batch_dict mutated in-place by forward (best)
      2) dense_head.forward_ret_dict (fallback)
      3) search model_ret recursively ONLY if it actually contains feat_key tensor
    """
    # 1) batch_dict
    if isinstance(batch_dict, dict) and (feat_key in batch_dict) and torch.is_tensor(batch_dict[feat_key]):
        return batch_dict[feat_key]

    # 2) head forward_ret_dict
    frd = getattr(dense_head, "forward_ret_dict", None)
    if isinstance(frd, dict) and (feat_key in frd) and torch.is_tensor(frd[feat_key]):
        return frd[feat_key]

    # 3) search model_ret (careful: model_ret may be recall dict!)
    def _search(x):
        if isinstance(x, dict):
            if (feat_key in x) and torch.is_tensor(x[feat_key]):
                return x[feat_key]
            return None
        if isinstance(x, (tuple, list)):
            for it in x:
                y = _search(it)
                if y is not None:
                    return y
        return None

    t = _search(model_ret)
    if t is not None:
        return t

    # debug
    keys = list(batch_dict.keys()) if isinstance(batch_dict, dict) else []
    logger.error(
        f"[FEAT] Cannot find feature '{feat_key}' in batch_dict/forward_ret_dict/model_ret. "
        f"batch_dict keys(head)={keys[:60]}"
    )
    if isinstance(model_ret, dict):
        logger.error(f"[FEAT] model_ret keys(head)={list(model_ret.keys())[:60]}")
    return None


@torch.no_grad()
def _accumulate_from_batch_cell_level(
    dense_head,
    feat_map_2d: torch.Tensor,
    box_cls_labels: torch.Tensor,
    anchors_per_loc: int,
    num_classes: int,
    chunk_cells: int,
):
    """
    Accumulate prototypes using CELL features (NO anchor expansion).

    feat_map_2d:         [B, C, H, W]  (either spatial_features_2d or hd_cls_feat)
    box_cls_labels:      [B, H*W*A]  (A fastest)
      - background: 0
      - ignored:    -1
      - positives:  1..K
    """
    hd_core = dense_head.hd_core
    embedder = hd_core.embedder
    memory = hd_core.memory

    assert feat_map_2d.dim() == 4
    B, C, H, W = feat_map_2d.shape
    HW = H * W
    A = int(anchors_per_loc)
    K = int(num_classes)

    # [B, C, H, W] -> [B, H, W, C] -> [B, HW, C]
    cell_feat = feat_map_2d.permute(0, 2, 3, 1).contiguous().view(B, HW, C)

    # labels [B, HW*A] -> [B, HW, A]
    if box_cls_labels.shape[1] != HW * A:
        raise RuntimeError(
            f"Label/feature mismatch: box_cls_labels={tuple(box_cls_labels.shape)}, expected second dim={HW*A}. "
            f"Got H={H}, W={W}, A={A}. Check feature map stride / anchor order."
        )
    labels_hw_a = box_cls_labels.view(B, HW, A).long()

    total_pos = 0
    step = int(chunk_cells) if int(chunk_cells) > 0 else -1

    for a in range(A):
        lab = labels_hw_a[:, :, a]          # [B, HW]
        pos_mask = lab > 0                 # positives only (1..K)
        if not pos_mask.any():
            continue

        feat_sel = cell_feat[pos_mask]     # [Npos, C]
        lab_sel = lab[pos_mask] - 1        # -> [0..K-1]

        if (lab_sel.min() < 0) or (lab_sel.max() >= K):
            raise RuntimeError(
                f"Shifted labels out of range: min={lab_sel.min().item()}, max={lab_sel.max().item()}, K={K}"
            )

        Npos = feat_sel.shape[0]
        if step <= 0:
            step = Npos

        start = 0
        while start < Npos:
            end = min(start + step, Npos)
            f_chunk = feat_sel[start:end]
            y_chunk = lab_sel[start:end]
            a_chunk = torch.full_like(y_chunk, fill_value=int(a), dtype=torch.long)

            # Keep memory-build feature mapping consistent with inference-time HD logits.
            f_chunk = hd_core.inject_anchor_context(
                feat_mid=f_chunk,
                anchor_ids=a_chunk,
                num_anchors=A
            )

            hv = embedder(f_chunk)               # [n, HD_DIM]
            memory.add_(y_chunk, hv, alpha=1.0)

            start = end

        total_pos += int(Npos)

    memory.normalize_()
    return total_pos


def _tensor_fingerprint(x: torch.Tensor, n: int = 16):
    x = x.detach().float().flatten()
    if x.numel() == 0:
        return {"numel": 0}
    idx = torch.linspace(0, x.numel() - 1, steps=min(n, x.numel())).long()
    vals = x[idx].cpu().numpy().tolist()
    return {
        "numel": int(x.numel()),
        "mean": float(x.mean().item()),
        "std": float(x.std().item()),
        "samples": vals,
    }


def main():
    args, cfg_local = parse_config()

    # Distributed init (same as tools/test.py)
    if args.launcher == "none":
        dist = False
    else:
        _, cfg_local.LOCAL_RANK = getattr(common_utils, "init_dist_%s" % args.launcher)(
            args.tcp_port, args.local_rank, backend="nccl"
        )
        dist = True

    if args.batch_size is None:
        args.batch_size = cfg_local.OPTIMIZATION.BATCH_SIZE_PER_GPU

    output_dir = cfg_local.ROOT_DIR / "output" / cfg_local.EXP_GROUP_PATH / cfg_local.TAG / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    epoch_id = _infer_epoch_id_from_ckpt(args.ckpt)

    if args.save_path is None:
        mem_dir = output_dir / "hd_memory"
        mem_dir.mkdir(parents=True, exist_ok=True)
        save_path = mem_dir / f"hd_memory_epoch_{epoch_id}.pth"
    else:
        save_path = Path(args.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    # Logger
    log_file = output_dir / "hd_memory" / f"log_build_hd_memory_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = common_utils.create_logger(log_file, rank=cfg_local.LOCAL_RANK)

    # Also log to stdout
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    logger.addHandler(sh)
    logger.propagate = False
    logger.info("[STDOUT] logger stream handler attached")

    feat_source = _resolve_feat_source(args, cfg_local)
    feat_key = _feat_key_from_source(feat_source)

    logger.info("********************** Start build_hd_memory.py **********************")
    logger.info(f"cfg_file: {args.cfg_file}")
    logger.info(f"ckpt: {args.ckpt}")
    logger.info(f"save_path: {str(save_path)}")
    logger.info(f"batch_size: {args.batch_size}, workers: {args.workers}, launcher: {args.launcher}")
    logger.info(f"max_batches: {args.max_batches}, use_fp16: {args.use_fp16}, chunk_cells: {args.chunk_cells}")
    logger.info(f"feat_source: {feat_source}  -> feat_key='{feat_key}'")
    log_config_to_file(cfg_local, logger=logger)

    # Build TRAIN split dataloader
    train_set, train_loader, _sampler = build_dataloader(
        dataset_cfg=cfg_local.DATA_CONFIG,
        class_names=cfg_local.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist,
        workers=args.workers,
        logger=logger,
        training=True,
    )

    model = build_network(model_cfg=cfg_local.MODEL, num_class=len(cfg_local.CLASS_NAMES), dataset=train_set)
    model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=dist)
    model.cuda()
    model.eval()

    dense_head = getattr(model, "dense_head", None)
    if dense_head is None:
        raise RuntimeError("Model has no `dense_head`. Cannot build HD memory.")

    anchors_per_loc = _get_num_anchors_per_loc(dense_head)
    num_classes = len(cfg_local.CLASS_NAMES)
    logger.info(f"anchors_per_loc: {anchors_per_loc}, num_classes: {num_classes}")

    hd_core = _ensure_hd_memory_ready(dense_head, logger)
    hd_core.memory.reset()

    total_pos = 0
    total_seen = 0
    use_autocast = bool(args.use_fp16)

    with torch.no_grad():
        for it, batch_dict in enumerate(train_loader):
            if args.max_batches > 0 and it >= args.max_batches:
                break

            # Move to GPU
            try:
                from pcdet.models import load_data_to_gpu as _load_data_to_gpu
                _load_data_to_gpu(batch_dict)
            except Exception:
                for k, v in batch_dict.items():
                    if torch.is_tensor(v):
                        batch_dict[k] = v.cuda(non_blocking=True)

            # Forward once
            if use_autocast:
                with torch.cuda.amp.autocast(enabled=True):
                    model_ret = model(batch_dict)
            else:
                model_ret = model(batch_dict)

            # ---- fetch feature map correctly ----
            feat_map = _get_feat_tensor_after_forward(dense_head, batch_dict, model_ret, feat_key, logger)
            if feat_map is None:
                if cfg_local.LOCAL_RANK == 0 and it == 0:
                    logger.error(f"[FEAT] FAIL: feat_source={feat_source}, feat_key={feat_key}")
                    logger.error(f"[FEAT] batch_dict keys(head)={list(batch_dict.keys())[:80]}")
                    frd = getattr(dense_head, "forward_ret_dict", None)
                    if isinstance(frd, dict):
                        logger.error(f"[FEAT] dense_head.forward_ret_dict keys(head)={list(frd.keys())[:80]}")
                raise RuntimeError(f"Cannot find feature '{feat_key}' for feat_source={feat_source}.")

            # ---- get gt_boxes ----
            gt_boxes = batch_dict.get("gt_boxes", None)
            if gt_boxes is None:
                raise RuntimeError("Cannot find 'gt_boxes' in batch_dict. Cannot assign anchor labels.")

            # ---- assign targets ----
            targets = dense_head.assign_targets(gt_boxes=gt_boxes)
            if "box_cls_labels" not in targets:
                raise RuntimeError("assign_targets() output has no key 'box_cls_labels'.")

            box_cls_labels = targets["box_cls_labels"]  # [B, num_anchors]

            # ---- label sanity log: only first 2 iters ----
            if it < 2 and cfg_local.LOCAL_RANK == 0:
                lab = box_cls_labels
                u = torch.unique(lab)
                logger.info(
                    f"[LABEL] box_cls_labels shape={tuple(lab.shape)} "
                    f"min={lab.min().item()} max={lab.max().item()} unique={u.tolist()}"
                )
                logger.info(
                    f"[LABEL] counts: ignore(-1)={(lab==-1).sum().item()} "
                    f"bg(0)={(lab==0).sum().item()} pos(>0)={(lab>0).sum().item()}"
                )
                if (lab > 0).any():
                    lab_shift = lab[lab > 0] - 1
                    logger.info(
                        f"[LABEL] shifted_pos_labels (lab-1): "
                        f"min={lab_shift.min().item()} max={lab_shift.max().item()} "
                        f"expected_range=[0,{num_classes-1}]"
                    )
                logger.info(f"[FEAT] using {feat_key}: shape={tuple(feat_map.shape)} dtype={feat_map.dtype} device={feat_map.device}")

            cared = box_cls_labels.view(-1) >= 0
            total_seen += int(cared.sum().item())

            pos_added = _accumulate_from_batch_cell_level(
                dense_head=dense_head,
                feat_map_2d=feat_map,
                box_cls_labels=box_cls_labels,
                anchors_per_loc=anchors_per_loc,
                num_classes=num_classes,
                chunk_cells=int(args.chunk_cells),
            )
            total_pos += int(pos_added)

            if cfg_local.LOCAL_RANK == 0 and ((it + 1) % 200 == 0):
                logger.info(f"[Iter {it+1}] accumulated_pos={total_pos}, total_seen={total_seen}")

    # Final normalize + sanity
    hd_core.memory.normalize_()

    if cfg_local.LOCAL_RANK == 0:
        mem = hd_core.memory
        try:
            logger.info(
                f"[MEM] proto_norm_mean={mem.prototypes.norm(dim=1).mean().item():.6f}, "
                f"w_norm_mean={mem.classify_weights.norm(dim=1).mean().item():.6f}"
            )
        except Exception:
            pass

    # ---------------- Save FULL payload (embedder + memory) ----------------
    embedder_sd = None
    try:
        embedder_sd = hd_core.embedder.state_dict()
    except Exception as e:
        logger.warning(f"[SAVE] cannot get embedder.state_dict(): {repr(e)}")

    memory_sd = None
    try:
        memory_sd = hd_core.memory.state_dict()
    except Exception as e:
        logger.warning(f"[SAVE] cannot get memory.state_dict(): {repr(e)}")

    proj_fp = None
    try:
        if hasattr(hd_core.embedder, "projection") and hasattr(hd_core.embedder.projection, "weight"):
            proj_fp = _tensor_fingerprint(hd_core.embedder.projection.weight, n=16)
    except Exception:
        proj_fp = None

    save_obj = {
        "embedder": embedder_sd,
        "memory": memory_sd,
        "classify_weights": hd_core.memory.classify_weights.detach().float().cpu(),
        "prototypes": hd_core.memory.prototypes.detach().float().cpu(),
        "meta": {
            "source": f"hd_core.memory (cell-level accumulation, repeated to anchors) | feat_source={feat_source}",
            "feat_source": feat_source,
            "feat_key": feat_key,
            "num_classes": int(num_classes),
            "anchors_per_loc": int(anchors_per_loc),
            "ckpt": args.ckpt,
            "cfg_file": args.cfg_file,
            "epoch_id": epoch_id,
            "total_pos_anchors": int(total_pos),
            "total_seen_anchors": int(total_seen),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "hd_cfg": {
                "HD_DIM": int(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "HD_DIM", -1)),
                "ENCODER": str(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "ENCODER", "unknown")),
                "QUANTIZE": bool(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "QUANTIZE", False)),
                "TEMPERATURE": float(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "TEMPERATURE", 1.0)),
                "SEED": int(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "SEED", 0)),
                "FEAT_SOURCE": str(getattr(cfg_local.MODEL.DENSE_HEAD.HD, "FEAT_SOURCE", feat_source)),
            },
            "projection_fingerprint": proj_fp,
        }
    }

    torch.save(save_obj, str(save_path))
    logger.info(f"Saved HD memory to: {str(save_path)}")
    logger.info(f"meta: {save_obj['meta']}")
    logger.info("********************** Finished build_hd_memory.py **********************")


if __name__ == "__main__":
    main()
