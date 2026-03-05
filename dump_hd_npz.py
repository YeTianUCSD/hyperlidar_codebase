import argparse
import datetime
import yaml
import os
import sys
from typing import Optional
import torch
import numpy as np
import torch.nn.functional as F




def _read_yaml(path: str, kind: str):
    try:
        print(f"Opening {kind} config file from {path}")
        with open(path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(e)
        print(f"Error opening {kind} yaml file: {path}")
        sys.exit(1)


def _resolve_cfg_paths(flags):
    model_arch = os.path.join(flags.model, "arch_cfg.yaml")
    model_data = os.path.join(flags.model, "data_cfg.yaml")

    default_arch = "config/arch/senet-512.yml"
    default_data = "config/labels/semantic-kitti.yaml"

    arch_path = flags.arch_cfg if flags.arch_cfg else (model_arch if os.path.isfile(model_arch) else default_arch)
    data_path = flags.data_cfg if flags.data_cfg else (model_data if os.path.isfile(model_data) else default_data)

    if not os.path.isfile(arch_path):
        print(f"[ERROR] arch cfg not found: {arch_path}")
        sys.exit(1)
    if not os.path.isfile(data_path):
        print(f"[ERROR] data cfg not found: {data_path}")
        sys.exit(1)

    return arch_path, data_path


def _apply_hd_overrides(ARCH: dict, flags):
    ARCH.setdefault("train", {})

    if flags.hd_bits is not None and flags.hd_mode is None:
        ARCH["train"]["hd_quant_mode"] = "nbit"
    if flags.hd_mode is not None:
        ARCH["train"]["hd_quant_mode"] = flags.hd_mode
    if flags.hd_bits is not None:
        ARCH["train"]["hd_quant_bits"] = int(flags.hd_bits)

    ARCH["train"].setdefault("hd_quant_mode", "nbit")
    ARCH["train"].setdefault("hd_quant_bits", 4)

    bits = int(ARCH["train"]["hd_quant_bits"])
    if bits < 1 or bits > 8:
        print(f"[ERROR] hd_quant_bits must be in [1,8], got {bits}")
        sys.exit(1)
    mode = str(ARCH["train"]["hd_quant_mode"]).lower()
    if mode not in ("float", "nbit"):
        print(f"[ERROR] hd_quant_mode must be 'float' or 'nbit', got {mode}")
        sys.exit(1)

    # dump control (optional)
    ARCH["train"].setdefault("hd_dump_enable", True)
    ARCH["train"].setdefault("hd_dump_points", 2048)
    ARCH["train"].setdefault("hd_dump_dirname", "rram_dump")

    return ARCH


def _quantize_signed_nbit(x: torch.Tensor, n_bits: int = 6, eps: float = 1e-8):
    # No-zero symmetric signed n-bit quantization.
    levels = 1 << (n_bits - 1)
    alpha = x.abs().max().clamp_min(eps)
    scale = alpha / float(levels)
    q = torch.round(x / scale)
    q = torch.clamp(q, -levels, levels)
    q = torch.where(q == 0, torch.where(x >= 0, torch.ones_like(q), -torch.ones_like(q)), q)
    return q.to(torch.int8), scale


def _build_export_weight(runner, hd_mode: str, hd_bits: int):
    with torch.no_grad():
        w = runner.model.classify_weights.detach().to(torch.float32)
        w = F.normalize(w, dim=1)

        if hd_mode == "nbit":
            w_q, _ = _quantize_signed_nbit(w, n_bits=hd_bits)
            return w_q.cpu().numpy().astype(np.int8)

        # Float mode fallback for export compatibility with simulator input type.
        w_s = torch.sign(w)
        w_s[w_s == 0] = 1
        return w_s.cpu().numpy().astype(np.int8)


def _dump_valid_frames(
    runner,
    ARCH,
    log_dir,
    only_one_batch: bool = False,
    max_frames: Optional[int] = None,
):
    dump_root = os.path.join(log_dir, ARCH["train"].get("hd_dump_dirname", "rram_dump"))
    os.makedirs(dump_root, exist_ok=True)

    hd_mode = str(ARCH["train"].get("hd_quant_mode", "nbit")).lower()
    hd_bits = int(ARCH["train"].get("hd_quant_bits", 6))
    dump_points = int(ARCH["train"].get("hd_dump_points", 2048))

    w_exp = _build_export_weight(runner, hd_mode=hd_mode, hd_bits=hd_bits)
    num_classes, hd_dim = int(w_exp.shape[0]), int(w_exp.shape[1])
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    dumped = 0
    runner.model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(runner.parser.get_valid_set()):
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

            if runner.gpu:
                proj_in = proj_in.cuda(non_blocking=True)
                proj_mask = proj_mask.cuda(non_blocking=True)
                proj_labels = proj_labels.cuda(non_blocking=True).long()

            bsz, _cin, h, w = proj_in.shape
            hv, _, _ = runner.model.encode(proj_in, runner.mask)         # [B*H*W, D], float in {-1,+1}
            hv = hv.view(bsz, h, w, hd_dim)

            for b in range(bsz):
                mask_b = proj_mask[b].view(-1) > 0
                y_b = proj_labels[b].view(-1)[mask_b]
                x_b = hv[b].view(-1, hd_dim)[mask_b]
                if x_b.shape[0] == 0:
                    continue

                n_keep = min(dump_points, int(x_b.shape[0]))
                x_np = x_b[:n_keep].to(torch.int8).cpu().numpy()
                y_np = y_b[:n_keep].to(torch.int16).cpu().numpy()

                seq = str(path_seq[b])
                name = str(path_name[b])  # e.g. 000000.label
                out_name = f"rram_dump_seq{seq}_frame{name}_nbits{hd_bits}_{ts}.npz"
                out_path = os.path.join(dump_root, out_name)

                np.savez_compressed(
                    out_path,
                    x=x_np,
                    w=w_exp,
                    y=y_np,
                    hd_nbits=np.int32(hd_bits),
                    num_classes=np.int32(num_classes),
                    hd_dim=np.int32(hd_dim),
                    note=np.array(["HyperLiDAR HD dump for offline RRAM sim"], dtype=object),
                )
                dumped += 1
                print(f"[DUMP] {out_path} | x={x_np.shape} w={w_exp.shape} y={y_np.shape}")
                if max_frames is not None and dumped >= int(max_frames):
                    print(f"[INFO] Reached max_frames={max_frames}, stop dumping.")
                    print(f"[INFO] Dumped {dumped} file(s) into: {dump_root}")
                    return dump_root

            if only_one_batch:
                break

    print(f"[INFO] Dumped {dumped} file(s) into: {dump_root}")
    return dump_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dump HD vectors for offline RRAM sim")
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--log', '-l', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--arch_cfg', '-ac', type=str, default=None)
    parser.add_argument('--data_cfg', '-dc', type=str, default=None)

    parser.add_argument('--hd_mode', type=str, default="nbit", choices=['float', 'nbit'])
    parser.add_argument('--hd_bits', type=int, default=6, choices=[1, 2, 4, 6, 8])

    # fast controls
    parser.add_argument('--dump_points', type=int, default=2048, help="how many pixels to dump")
    parser.add_argument('--only_one_batch', action='store_true', help="stop after dumping once")
    parser.add_argument('--max_frames', type=int, default=None, help="export at most N frames")
    #parser.add_argument('--val_batches', type=int, default=20, help="only run first N val batches")


    FLAGS = parser.parse_args()

    if not os.path.isdir(FLAGS.model):
        print(f"[ERROR] model folder doesn't exist: {FLAGS.model}")
        sys.exit(1)
    if not os.path.isdir(FLAGS.dataset):
        print(f"[ERROR] dataset folder doesn't exist: {FLAGS.dataset}")
        sys.exit(1)

    arch_path, data_path = _resolve_cfg_paths(FLAGS)
    ARCH = _read_yaml(arch_path, "arch")
    DATA = _read_yaml(data_path, "data")
    ARCH = _apply_hd_overrides(ARCH, FLAGS)

    # override dump points from CLI
    ARCH.setdefault("train", {})
    ARCH["train"]["hd_dump_points"] = int(FLAGS.dump_points)

    os.makedirs(FLAGS.log, exist_ok=True)

    print("[INFO] Effective HD config:",
          "hd_quant_mode =", ARCH["train"].get("hd_quant_mode"),
          "| hd_quant_bits =", ARCH["train"].get("hd_quant_bits"))
    print(f"[INFO] Start time: {datetime.datetime.now().isoformat()}")
    print(f"[INFO] log dir: {FLAGS.log}")

    from modules.Basic_HD import BasicHD

    runner = BasicHD(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)

    # 1) quick prototype accumulation (one pass) so classify_weights are meaningful
    print("[INFO] Running one-pass prototype accumulation on TRAIN set (fast)...")
    runner.model.eval()
    runner.train(runner.parser.get_train_set(), runner.model, None)

    # 2) dump VALID split as per-frame npz for offline RRAM simulator
    print("[INFO] Exporting VALID split frames into rram_dump/*.npz ...")
    dump_root = _dump_valid_frames(
        runner=runner,
        ARCH=ARCH,
        log_dir=FLAGS.log,
        only_one_batch=bool(FLAGS.only_one_batch),
        max_frames=FLAGS.max_frames,
    )

    print(f"[INFO] Finished at: {datetime.datetime.now().isoformat()}")
    print(f"[INFO] Check dump at: {dump_root}")
