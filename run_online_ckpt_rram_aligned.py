#!/usr/bin/env python3

import argparse
import csv
import datetime
import glob
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from modules.HDC_utils import set_model


def _read_yaml(path: str, kind: str):
    try:
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"[ERROR] cannot open {kind} yaml: {path}\n{e}")
        sys.exit(1)


def _extract_update_idx(path: str) -> int:
    m = re.search(r"_upd(\d+)(?:_|\.)", os.path.basename(path))
    return int(m.group(1)) if m else 10**9


def _choose_w_tensor(ckpt: Dict) -> torch.Tensor:
    for k in ("classify_weights", "classify_weights_raw", "w", "prototypes", "prototype"):
        if k in ckpt and ckpt[k] is not None:
            w = ckpt[k]
            if not torch.is_tensor(w):
                w = torch.tensor(w)
            return w.to(torch.float32)
    raise RuntimeError(f"no prototype tensor found in checkpoint keys={list(ckpt.keys())[:30]}")


def _quantize_signed_nbit_nozero(x: torch.Tensor, n_bits: int, eps: float = 1e-8) -> np.ndarray:
    levels = 1 << (n_bits - 1)
    alpha = x.abs().max().clamp_min(eps)
    scale = alpha / float(levels)
    q = torch.round(x / scale)
    q = torch.clamp(q, -levels, levels)
    q = torch.where(q == 0, torch.where(x >= 0, torch.ones_like(q), -torch.ones_like(q)), q)
    return q.to(torch.int8).cpu().numpy()


def _projection_hash(weight: torch.Tensor) -> str:
    arr = weight.detach().cpu().numpy()
    return hashlib.md5(arr.tobytes()).hexdigest()[:12]


def _safe_tag(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", s)


def _run_cmd(cmd: List[str]) -> None:
    proc = subprocess.run(cmd, stdout=sys.stdout, stderr=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}")


def _iter_split_frames(loader: Iterable, split: str):
    for batch in loader:
        (
            proj_in,
            proj_mask,
            proj_labels,
            _unproj_labels,
            path_seq,
            path_name,
            _p_x,
            _p_y,
            _proj_range,
            _unproj_range,
            _a,
            _b,
            _c,
            _d,
            _npoints,
        ) = batch

        bsz = proj_in.shape[0]
        for b in range(bsz):
            seq = str(path_seq[b])
            name = str(path_name[b])
            frame_tag = f"{split}_seq{seq}_frame{name.replace('.label', '')}"
            yield frame_tag, proj_in[b:b + 1], proj_mask[b:b + 1], proj_labels[b:b + 1]


def _prepare_parser(arch: dict, data: dict, dataset_root: str, workers: int):
    from dataset.kitti.parser import Parser

    labels_for_training = data.get("labels_coarse", data["labels"])
    parser_obj = Parser(
        root=dataset_root,
        train_sequences=data["split"]["train"],
        valid_sequences=data["split"]["valid"],
        test_sequences=None,
        labels=labels_for_training,
        color_map=data["color_map"],
        learning_map=data["learning_map"],
        learning_map_inv=data["learning_map_inv"],
        sensor=arch["dataset"]["sensor"],
        max_points=arch["dataset"]["max_points"],
        batch_size=1,
        workers=int(workers),
        gt=True,
        shuffle_train=False,
    )
    if hasattr(parser_obj, "train_dataset"):
        parser_obj.train_dataset.transform = False
    return parser_obj


def main():
    ap = argparse.ArgumentParser(
        "Align per-ckpt projection -> dump x/y/w -> run RRAM sim -> summarize GT vs RRAM"
    )
    ap.add_argument("--dataset", "-d", required=True, type=str)
    ap.add_argument("--model", "-m", required=True, type=str,
                    help="CNN model dir containing SENet_valid_best")
    ap.add_argument("--ckpt_glob", required=True, type=str,
                    help="online ckpt glob, e.g. .../online_*_upd*.pt")
    ap.add_argument("--arch_cfg", "-ac", default="config/arch/senet-512.yml", type=str)
    ap.add_argument("--data_cfg", "-dc", default="config/labels/semantic-nuscenes_online-unsup-valid10.yaml", type=str)
    ap.add_argument("--split", choices=["valid", "train"], default="valid")
    ap.add_argument("--max_frames", type=int, default=1)
    ap.add_argument("--max_ckpts", type=int, default=None)
    ap.add_argument("--dump_points", type=int, default=2048)
    ap.add_argument("--hd_bits", type=int, default=4, choices=[1, 2, 4, 6, 8])
    ap.add_argument("--workers", type=int, default=0,
                    help="dataloader workers for this script (default 0 for max compatibility)")

    ap.add_argument("--runner_py", type=str, default="/home/Hyperlidar/bulk-RRAM-sim/sim/run_hyperlidar_npz.py")
    ap.add_argument("--n_sample", type=int, default=20)
    ap.add_argument("--n_WL_act", type=int, default=8, choices=[8, 16])
    ap.add_argument("--cell_type", type=str, default="large_dynamic",
                    choices=["large_dynamic", "fast_switching", "2d_rram"])
    ap.add_argument("--status_method", type=str, default="value",
                    choices=["value", "cycle", "opt"])
    ap.add_argument("--use_x_sign", action="store_true")

    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--keep_input_npz", action="store_true")
    args = ap.parse_args()

    if not os.path.isdir(args.dataset):
        raise RuntimeError(f"dataset not found: {args.dataset}")
    if not os.path.isdir(args.model):
        raise RuntimeError(f"model dir not found: {args.model}")
    if not os.path.isfile(args.arch_cfg):
        raise RuntimeError(f"arch cfg not found: {args.arch_cfg}")
    if not os.path.isfile(args.data_cfg):
        raise RuntimeError(f"data cfg not found: {args.data_cfg}")
    if not os.path.isfile(args.runner_py):
        raise RuntimeError(f"runner_py not found: {args.runner_py}")

    ckpts = sorted(glob.glob(args.ckpt_glob), key=lambda p: (_extract_update_idx(p), p))
    if not ckpts:
        raise RuntimeError(f"no checkpoint matched: {args.ckpt_glob}")
    if args.max_ckpts is not None:
        ckpts = ckpts[: int(args.max_ckpts)]

    arch = _read_yaml(args.arch_cfg, "arch")
    data = _read_yaml(args.data_cfg, "data")
    arch.setdefault("train", {})
    arch["train"]["hd_quant_mode"] = "nbit"
    arch["train"]["hd_quant_bits"] = int(args.hd_bits)
    arch["train"]["hd_encode_store_int8"] = True

    lmap = data.get("learning_map", {}) or {}
    tgt = [v for v in lmap.values() if isinstance(v, int) and v >= 0]
    if not tgt:
        raise RuntimeError("Cannot infer num_classes from learning_map")
    num_classes = max(tgt) + 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = set_model(arch, args.model, "rp", 0, 0, num_classes, device)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    parser_obj = _prepare_parser(arch, data, args.dataset, workers=int(args.workers))
    if args.split == "valid":
        loader = torch.utils.data.DataLoader(
            parser_obj.valid_dataset, batch_size=1, shuffle=False,
            num_workers=int(args.workers), drop_last=False
        )
    else:
        loader = torch.utils.data.DataLoader(
            parser_obj.train_dataset, batch_size=1, shuffle=False,
            num_workers=int(args.workers), drop_last=False
        )

    os.makedirs(args.out_dir, exist_ok=True)
    input_dir = os.path.join(args.out_dir, "input_npz")
    result_dir = os.path.join(args.out_dir, "result_npz")
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    per_frame_csv = os.path.join(args.out_dir, "aligned_rram_summary_per_frame.csv")
    avg_csv = os.path.join(args.out_dir, "aligned_rram_summary_avg_by_update.csv")
    if os.path.isfile(per_frame_csv):
        os.remove(per_frame_csv)
    if os.path.isfile(avg_csv):
        os.remove(avg_csv)

    rows = []
    fieldnames = []

    frame_cache = []
    for fi, item in enumerate(_iter_split_frames(loader, args.split)):
        if fi >= int(args.max_frames):
            break
        frame_cache.append(item)
    if not frame_cache:
        raise RuntimeError("no frames collected from split")

    print(f"[INFO] split={args.split}, frames={len(frame_cache)}, ckpts={len(ckpts)}")
    print(f"[INFO] out_dir={args.out_dir}")

    for ci, ckpt_path in enumerate(ckpts, start=1):
        ck_name = os.path.basename(ckpt_path)
        upd_idx = _extract_update_idx(ck_name)
        print(f"\n[CKPT {ci}/{len(ckpts)}] {ck_name}")

        ck = torch.load(ckpt_path, map_location="cpu")
        if not isinstance(ck, dict):
            raise RuntimeError(f"checkpoint is not a dict: {ckpt_path}")

        if "projection_weight" in ck and ck["projection_weight"] is not None:
            pw = ck["projection_weight"]
            if not torch.is_tensor(pw):
                pw = torch.tensor(pw)
            pw = pw.to(torch.float32).to(device)
            if tuple(pw.shape) != tuple(model.projection.weight.shape):
                raise RuntimeError(
                    f"projection shape mismatch for {ck_name}: {tuple(pw.shape)} vs {tuple(model.projection.weight.shape)}"
                )
            model.projection.weight.data.copy_(pw.to(model.projection.weight.dtype))

        proj_hash = _projection_hash(model.projection.weight)
        w_fp = _choose_w_tensor(ck).to(device)
        if w_fp.shape[0] != num_classes:
            raise RuntimeError(f"class dim mismatch in {ck_name}: got {w_fp.shape[0]}, expected {num_classes}")
        w_fp = F.normalize(w_fp, dim=1)
        w_q = _quantize_signed_nbit_nozero(w_fp, n_bits=int(args.hd_bits))

        for fi, (frame_tag, proj_in, proj_mask, proj_labels) in enumerate(frame_cache):
            if torch.cuda.is_available():
                proj_in = proj_in.cuda(non_blocking=True)
                proj_mask = proj_mask.cuda(non_blocking=True)
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            with torch.no_grad():
                hv, _, _ = model.encode(proj_in, None)  # [H*W, D], int8 expected
                bsz, _cin, h, wdim = proj_in.shape
                hv = hv.view(bsz, h, wdim, -1)

                mask_b = proj_mask[0].view(-1) > 0
                y_b = proj_labels[0].view(-1)[mask_b]
                x_b = hv[0].view(-1, hv.shape[-1])[mask_b]

                if x_b.shape[0] == 0:
                    print(f"[WARN] no valid points: ckpt={ck_name} frame={frame_tag}, skip")
                    continue

                n_keep = min(int(args.dump_points), int(x_b.shape[0]))
                x_np = x_b[:n_keep].to(torch.int8).cpu().numpy()
                y_np = y_b[:n_keep].to(torch.int16).cpu().numpy()

            frame_dir_tag = _safe_tag(f"{frame_tag}_{proj_hash}")
            in_frame_dir = os.path.join(input_dir, frame_dir_tag)
            out_frame_dir = os.path.join(result_dir, frame_dir_tag)
            os.makedirs(in_frame_dir, exist_ok=True)
            os.makedirs(out_frame_dir, exist_ok=True)

            if args.keep_input_npz:
                in_npz = os.path.join(in_frame_dir, f"rram_input_upd{upd_idx:04d}_{_safe_tag(ck_name)}.npz")
            else:
                fd, in_npz = tempfile.mkstemp(
                    prefix=f"rram_input_upd{upd_idx:04d}_{_safe_tag(ck_name)}_",
                    suffix=".npz",
                    dir=in_frame_dir,
                )
                os.close(fd)

            out_npz = os.path.join(out_frame_dir, f"rram_result_upd{upd_idx:04d}_{_safe_tag(ck_name)}.npz")

            np.savez_compressed(
                in_npz,
                x=x_np,
                w=w_q,
                y=y_np,
                hd_nbits=np.int32(args.hd_bits),
                num_classes=np.int32(w_q.shape[0]),
                hd_dim=np.int32(w_q.shape[1]),
                projection_hash=np.array([proj_hash], dtype=object),
                note=np.array([f"aligned_ckpt={ck_name}"], dtype=object),
            )

            cmd = [
                sys.executable,
                args.runner_py,
                "--npz", in_npz,
                "--n_sample", str(args.n_sample),
                "--n_WL_act", str(args.n_WL_act),
                "--cell_type", args.cell_type,
                "--status_method", args.status_method,
                "--out_npz", out_npz,
            ]
            if args.use_x_sign:
                cmd.append("--use_x_sign")

            status = "ok"
            err_msg = ""
            summary = {}
            try:
                _run_cmd(cmd)
                res = np.load(out_npz, allow_pickle=True)
                summary = json.loads(str(res["summary_json"][0]))
            except Exception as e:
                status = "failed"
                err_msg = repr(e)
                print(f"[WARN] failed ckpt={ck_name}, frame={frame_tag}: {err_msg}")
            finally:
                if (not args.keep_input_npz) and os.path.isfile(in_npz):
                    os.remove(in_npz)

            row = {
                "frame_idx": fi,
                "frame_tag": frame_tag,
                "projection_hash": proj_hash,
                "update_idx": upd_idx,
                "ckpt": ck_name,
                "status": status,
                "error": err_msg,
                "valid_iou_ckpt": float(ck.get("valid_iou", np.nan)),
                "valid_acc_ckpt": float(ck.get("valid_acc", np.nan)),
                "acc_gt": float(summary.get("acc_gt", np.nan)),
                "iou_gt": float(summary.get("iou_gt", np.nan)),
                "acc_rram": float(summary.get("acc_rram", np.nan)),
                "iou_rram": float(summary.get("iou_rram", np.nan)),
                "time_rram_infer_s": float(summary.get("time_rram_infer_s", np.nan)),
                "time_rram_build_s": float(summary.get("time_rram_build_s", np.nan)),
                "time_rram_end2end_s": float(summary.get("time_rram_end2end_s", np.nan)),
                "result_npz": out_npz if os.path.isfile(out_npz) else "",
            }
            rows.append(row)

            if not fieldnames:
                fieldnames = list(row.keys())
            with open(per_frame_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)

    # avg by update
    by_upd = {}
    for r in rows:
        key = (int(r["update_idx"]), str(r["ckpt"]))
        by_upd.setdefault(key, []).append(r)

    avg_rows = []
    for key in sorted(by_upd.keys(), key=lambda t: (t[0], t[1])):
        upd, ck_name = key
        rr = by_upd[key]
        out = {"update_idx": upd, "ckpt": ck_name, "n_rows": len(rr)}
        for k in ("valid_iou_ckpt", "valid_acc_ckpt", "acc_gt", "iou_gt", "acc_rram", "iou_rram",
                  "time_rram_infer_s", "time_rram_build_s", "time_rram_end2end_s"):
            vals = [float(x[k]) for x in rr if not np.isnan(float(x[k]))]
            out[f"{k}_mean"] = float(sum(vals) / len(vals)) if vals else float("nan")
        avg_rows.append(out)

    if avg_rows:
        with open(avg_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(avg_rows[0].keys()))
            writer.writeheader()
            writer.writerows(avg_rows)

    print(f"\n[OK] per-frame summary csv: {per_frame_csv}")
    print(f"[OK] avg-by-update summary csv: {avg_csv}")
    print(f"[OK] finished at {datetime.datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
