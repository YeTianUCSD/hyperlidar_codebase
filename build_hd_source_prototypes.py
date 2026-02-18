#!/usr/bin/env python3

import argparse
import datetime
import os
import sys
import yaml
import torch
import torch.nn.functional as F
from tqdm import tqdm

from modules.HDC_utils import set_model


def _read_yaml(path: str, kind: str):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] cannot open {kind} yaml: {path}\n{e}")
        sys.exit(1)


def _parse_train_seq(train_seq: str):
    if train_seq is None:
        return None
    try:
        return [int(x.strip()) for x in train_seq.split(",") if x.strip() != ""]
    except Exception as e:
        raise ValueError(f"bad --train_seq '{train_seq}': {e}")


def _ensure_parent(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d != "":
        os.makedirs(d, exist_ok=True)


def _save_payload(path: str, payload: dict):
    _ensure_parent(path)
    torch.save(payload, path)


def _make_logger(log_path: str):
    _ensure_parent(log_path)

    def _log(msg: str):
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    return _log


def main():
    parser = argparse.ArgumentParser("Build source-domain HD prototypes from labeled train split")
    parser.add_argument("--dataset", "-d", required=True, type=str, help="Dataset root")
    parser.add_argument("--model", "-m", required=True, type=str, help="Model dir containing SENet_valid_best")
    parser.add_argument("--arch_cfg", "-ac", default="config/arch/senet-512.yml", type=str)
    parser.add_argument("--data_cfg", "-dc", default="config/labels/semantic-nuscenes_all_pretrain.yaml", type=str)
    parser.add_argument("--train_seq", "-t", default=None, type=str,
                        help="Optional override for split.train, e.g. '0,1,2'")

    parser.add_argument("--out", default=None, type=str,
                        help="Output .pt file path. Default: <model>/HD_source_prototypes.pt")
    parser.add_argument("--max_batches", default=-1, type=int,
                        help="For debug. -1 means full pass")
    parser.add_argument("--chunk_size", default=200000, type=int,
                        help="Chunk size when index_add over selected points")
    parser.add_argument("--seed", default=1024, type=int)

    args = parser.parse_args()

    if not os.path.isdir(args.dataset):
        print(f"[ERROR] dataset not found: {args.dataset}")
        sys.exit(1)
    if not os.path.isdir(args.model):
        print(f"[ERROR] model dir not found: {args.model}")
        sys.exit(1)
    if not os.path.isfile(args.arch_cfg):
        print(f"[ERROR] arch cfg not found: {args.arch_cfg}")
        sys.exit(1)
    if not os.path.isfile(args.data_cfg):
        print(f"[ERROR] data cfg not found: {args.data_cfg}")
        sys.exit(1)

    out_path = args.out if args.out else os.path.join(args.model, "HD_source_prototypes.pt")
    out_dir = os.path.dirname(os.path.abspath(out_path))
    log_path = os.path.join(
        out_dir,
        f"build_hd_source_prototypes_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
    )
    log = _make_logger(log_path)

    log(f"Start build_hd_source_prototypes.py")
    log(f"dataset={args.dataset}")
    log(f"model={args.model}")
    log(f"arch_cfg={args.arch_cfg}")
    log(f"data_cfg={args.data_cfg}")
    log(f"out={out_path}")
    log(f"max_batches={args.max_batches}, chunk_size={args.chunk_size}, seed={args.seed}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    arch = _read_yaml(args.arch_cfg, "arch")
    data = _read_yaml(args.data_cfg, "data")

    seq_override = _parse_train_seq(args.train_seq)
    if seq_override is not None:
        data.setdefault("split", {})
        data["split"]["train"] = seq_override
        log(f"override split.train -> {data['split']['train']}")

    # Build parser on labeled train split
    from dataset.kitti.parser import Parser

    labels_for_training = data.get("labels_coarse", data["labels"])
    parser_obj = Parser(
        root=args.dataset,
        train_sequences=data["split"]["train"],
        valid_sequences=data["split"]["valid"],
        test_sequences=None,
        labels=labels_for_training,
        color_map=data["color_map"],
        learning_map=data["learning_map"],
        learning_map_inv=data["learning_map_inv"],
        sensor=arch["dataset"]["sensor"],
        max_points=arch["dataset"]["max_points"],
        batch_size=arch["train"]["batch_size"],
        workers=arch["train"]["workers"],
        gt=True,
        shuffle_train=False,
    )
    if hasattr(parser_obj, "train_dataset"):
        parser_obj.train_dataset.transform = False

    # Infer class count from learning_map
    lmap = data.get("learning_map", {}) or {}
    tgt = [v for v in lmap.values() if isinstance(v, int) and v >= 0]
    if not tgt:
        raise RuntimeError("Cannot infer num_classes from learning_map")
    num_classes = max(tgt) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model (loads CNN weights from modeldir/SENet_valid_best)
    model = set_model(arch, args.model, "rp", 0, 0, num_classes, device)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    # Reset class prototype accumulator to zero explicitly
    with torch.no_grad():
        model.classify_weights.zero_()
        model.classify.weight.zero_()

    ignore_class = [int(c) for c, ign in data["learning_ignore"].items() if ign]

    train_loader = torch.utils.data.DataLoader(
        parser_obj.train_dataset,
        batch_size=arch["train"]["batch_size"],
        shuffle=False,
        num_workers=arch["train"]["workers"],
        drop_last=False,
    )

    total_points = 0
    used_points = 0
    chunk_size = max(1, int(args.chunk_size))

    with torch.no_grad():
        for bi, batch in enumerate(tqdm(train_loader, desc="Building source prototypes")):
            if args.max_batches > 0 and bi >= args.max_batches:
                break

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

            if torch.cuda.is_available():
                proj_in = proj_in.cuda(non_blocking=True)
                proj_mask = proj_mask.cuda(non_blocking=True)
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            samples_hv, _, _ = model.encode(proj_in, None)  # [B*H*W, D]
            labels = proj_labels.view(-1)
            valid = proj_mask.view(-1) > 0

            labels_v = labels[valid]
            hv_v = samples_hv[valid]
            total_points += int(valid.sum().item())

            if labels_v.numel() == 0:
                continue

            keep = torch.ones_like(labels_v, dtype=torch.bool)
            for c in ignore_class:
                keep = keep & (labels_v != c)

            labels_u = labels_v[keep]
            hv_u = hv_v[keep].to(model.classify_weights.dtype)

            if labels_u.numel() == 0:
                continue

            # Chunked accumulation to avoid temporary peak memory on huge batches
            n = labels_u.shape[0]
            for s in range(0, n, chunk_size):
                e = min(s + chunk_size, n)
                model.classify_weights.index_add_(0, labels_u[s:e], hv_u[s:e])

            used_points += n
            if (bi + 1) % 200 == 0:
                log(f"progress batch={bi+1}, total_valid_points={total_points}, used_points={used_points}")

    with torch.no_grad():
        w_raw = model.classify_weights.detach().clone()
        # Normalize each class vector for stable cosine/dot comparisons
        n = w_raw.norm(dim=1, keepdim=True)
        w = torch.where(n > 1e-12, w_raw / n, w_raw)
        for c in ignore_class:
            w[c].zero_()
            w_raw[c].zero_()
        model.classify_weights.copy_(w)
        model.classify.weight.copy_(w)

    projection_weight = None
    if hasattr(model, "projection") and hasattr(model.projection, "weight"):
        projection_weight = model.projection.weight.detach().float().cpu()

    payload = {
        "classify_weights_raw": w_raw.detach().float().cpu(),
        "prototypes_raw": w_raw.detach().float().cpu(),
        "classify_weights": model.classify_weights.detach().float().cpu(),
        "prototypes": model.classify_weights.detach().float().cpu(),
        "projection_weight": projection_weight,
        "meta": {
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": args.dataset,
            "model_dir": args.model,
            "arch_cfg": args.arch_cfg,
            "data_cfg": args.data_cfg,
            "train_sequences": data["split"]["train"],
            "num_classes": int(num_classes),
            "hd_dim": int(model.classify_weights.shape[1]),
            "input_dim": int(model.input_dim) if hasattr(model, "input_dim") else -1,
            "total_valid_points": int(total_points),
            "used_points": int(used_points),
            "ignore_class": ignore_class,
        },
    }

    _save_payload(out_path, payload)
    log(f"saved source prototypes: {out_path}")
    log(f"total_valid_points={total_points}, used_points={used_points}")
    log(f"log file: {log_path}")


if __name__ == "__main__":
    main()
