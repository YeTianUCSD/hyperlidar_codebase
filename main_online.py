#!/usr/bin/env python3

import argparse
import datetime
import os
import shutil
import sys
import yaml
import io


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
    # 1) CLI
    if flags.arch_cfg and not os.path.isfile(flags.arch_cfg):
        print(f"[ERROR] --arch_cfg not found: {flags.arch_cfg}")
        sys.exit(1)
    if flags.data_cfg and not os.path.isfile(flags.data_cfg):
        print(f"[ERROR] --data_cfg not found: {flags.data_cfg}")
        sys.exit(1)

    # 2) Model dir snapshots (arch only; data cfg is fixed to online-unsup unless CLI overrides)
    model_arch = os.path.join(flags.model, "arch_cfg.yaml")

    # 3) Defaults
    default_arch = "config/arch/senet-512.yml"
    default_data_online = "config/labels/semantic-nuscenes_online-unsup.yaml"
    default_data = default_data_online

    arch_path = flags.arch_cfg if flags.arch_cfg else (model_arch if os.path.isfile(model_arch) else default_arch)
    data_path = flags.data_cfg if flags.data_cfg else default_data

    if not os.path.isfile(arch_path):
        print(f"[ERROR] arch cfg not found: {arch_path}")
        sys.exit(1)
    if not os.path.isfile(data_path):
        print(f"[ERROR] data cfg not found: {data_path}")
        sys.exit(1)

    return arch_path, data_path


def _apply_online_overrides(ARCH: dict, flags):
    if ARCH is None:
        raise RuntimeError("ARCH yaml parsed to None (empty file?)")

    ARCH.setdefault("train", {})

    # Optional HD quant mode controls for final quantized eval.
    if flags.hd_mode is not None:
        ARCH["train"]["hd_quant_mode"] = flags.hd_mode
    if flags.hd_bits is not None:
        ARCH["train"]["hd_quant_bits"] = int(flags.hd_bits)

    ARCH["train"].setdefault("hd_quant_mode", "float")
    ARCH["train"].setdefault("hd_quant_bits", 4)

    mode = str(ARCH["train"]["hd_quant_mode"]).lower()
    bits = int(ARCH["train"]["hd_quant_bits"])

    if mode not in ("float", "nbit"):
        print(f"[ERROR] hd_quant_mode must be 'float' or 'nbit', got {mode}")
        sys.exit(1)
    if bits < 2 or bits > 8:
        print(f"[ERROR] hd_quant_bits must be in [2,8], got {bits}")
        sys.exit(1)

    # Online source prototype path override.
    if flags.source_proto is not None:
        ARCH["train"]["online_source_proto_path"] = flags.source_proto
    if flags.resume_online_ckpt is not None:
        ARCH["train"]["online_resume_ckpt_path"] = flags.resume_online_ckpt

    # If user explicitly wants to allow random fallback.
    if flags.allow_random_fallback:
        ARCH["train"]["online_require_source_proto"] = False

    # Progress log frequency for builder script only; keep here for unified cfg tracking.
    ARCH["train"].setdefault("online_require_source_proto", True)
    ARCH["train"].setdefault("online_require_source_projection", True)

    return ARCH


def _maybe_snapshot_cfgs(log_dir: str, arch_path: str, data_path: str):
    try:
        shutil.copy2(arch_path, os.path.join(log_dir, "arch_cfg_used.yaml"))
        shutil.copy2(data_path, os.path.join(log_dir, "data_cfg_used.yaml"))
        print("[INFO] Saved cfg snapshots to log dir: arch_cfg_used.yaml / data_cfg_used.yaml")
    except Exception as e:
        print("[WARN] Failed to snapshot cfg files into log dir:", e)


class _Tee(io.TextIOBase):
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)

    def flush(self):
        for st in self.streams:
            st.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Online unsupervised HD adaptation entry")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset root (contains sequences/)")
    parser.add_argument("--log", "-l", type=str, required=True, help="Log/output directory")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model directory containing SENet_valid_best")

    parser.add_argument("--train_seq", "-t", type=str, default=None,
                        help="Override split.train, comma-separated, e.g. '1082,1093,...'")
    parser.add_argument("--arch_cfg", "-ac", type=str, default=None,
                        help="Path to arch yaml. If omitted, try <model>/arch_cfg.yaml, else default")
    parser.add_argument("--data_cfg", "-dc", type=str, default=None,
                        help="Path to data yaml. If omitted, try <model>/data_cfg.yaml, else default")

    parser.add_argument("--source_proto", type=str, default=None,
                        help="Path to prebuilt source prototypes (.pt/.pth/.npz)")
    parser.add_argument("--resume_online_ckpt", type=str, default=None,
                        help="Resume online state from checkpoint (.pt/.pth)")
    parser.add_argument("--allow_random_fallback", action="store_true",
                        help="Allow random prototype init when source prototype is missing")

    parser.add_argument("--hd_mode", type=str, default=None, choices=["float", "nbit"],
                        help="HD mode: float or nbit (for final quantized eval)")
    parser.add_argument("--hd_bits", type=int, default=None, choices=[2, 4, 6, 8],
                        help="HD quant bits when hd_mode=nbit")

    FLAGS, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Unknown args ignored:", unknown)

    print("----------")
    print("ONLINE INTERFACE:")
    print("dataset     :", FLAGS.dataset)
    print("log         :", FLAGS.log)
    print("model       :", FLAGS.model)
    print("train_seq   :", FLAGS.train_seq if FLAGS.train_seq else "(from yaml)")
    print("arch_cfg    :", FLAGS.arch_cfg if FLAGS.arch_cfg else "(auto)")
    print("data_cfg    :", FLAGS.data_cfg if FLAGS.data_cfg else "(auto)")
    print("source_proto:", FLAGS.source_proto if FLAGS.source_proto else "(auto: <model>/HD_source_prototypes.pt)")
    print("resume_ckpt :", FLAGS.resume_online_ckpt if FLAGS.resume_online_ckpt else "(none)")
    print("hd_mode     :", FLAGS.hd_mode if FLAGS.hd_mode else "(yaml/default)")
    print("hd_bits     :", FLAGS.hd_bits if FLAGS.hd_bits is not None else "(yaml/default)")
    print("----------\n")

    if not os.path.isdir(FLAGS.dataset):
        print(f"[ERROR] dataset folder doesn't exist: {FLAGS.dataset}")
        sys.exit(1)
    if not os.path.isdir(os.path.join(FLAGS.dataset, "sequences")):
        print(f"[ERROR] dataset root must contain 'sequences/' folder: {FLAGS.dataset}")
        sys.exit(1)
    if not os.path.isdir(FLAGS.model):
        print(f"[ERROR] model folder doesn't exist: {FLAGS.model}")
        sys.exit(1)
    if FLAGS.resume_online_ckpt is not None and not os.path.isfile(FLAGS.resume_online_ckpt):
        print(f"[ERROR] resume checkpoint not found: {FLAGS.resume_online_ckpt}")
        sys.exit(1)

    arch_path, data_path = _resolve_cfg_paths(FLAGS)
    if FLAGS.data_cfg is None:
        print(f"[INFO] Auto-selected data cfg: {data_path}")
    ARCH = _read_yaml(arch_path, "arch")
    DATA = _read_yaml(data_path, "data")

    ARCH = _apply_online_overrides(ARCH, FLAGS)

    if FLAGS.train_seq is not None:
        try:
            parsed = [int(s.strip()) for s in FLAGS.train_seq.split(",") if s.strip() != ""]
            DATA.setdefault("split", {})
            DATA["split"]["train"] = parsed
            print(f"[INFO] Overriding split.train: {DATA['split']['train']}")
        except ValueError as e:
            print(f"[ERROR] Failed to parse --train_seq: {e}")
            sys.exit(1)
    else:
        print("[INFO] Using split.train from data yaml")

    os.makedirs(FLAGS.log, exist_ok=True)
    run_log = os.path.join(
        FLAGS.log,
        f"main_online_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
    )
    run_fp = open(run_log, "a")
    sys.stdout = _Tee(sys.__stdout__, run_fp)
    sys.stderr = _Tee(sys.__stderr__, run_fp)
    print(f"[INFO] Main log file: {run_log}")
    _maybe_snapshot_cfgs(FLAGS.log, arch_path, data_path)

    print(f"[INFO] Effective online settings:")
    print("       online_require_source_proto      =", ARCH["train"].get("online_require_source_proto"))
    print("       online_require_source_projection =", ARCH["train"].get("online_require_source_projection"))
    print("       online_source_proto_path         =", ARCH["train"].get("online_source_proto_path", "(auto)"))
    print("       hd_quant_mode                    =", ARCH["train"].get("hd_quant_mode"))
    print("       hd_quant_bits                    =", ARCH["train"].get("hd_quant_bits"))

    print(f"[INFO] Start time: {datetime.datetime.now().isoformat()}")

    try:
        from modules.Basic_HD_online import BasicHDOnline
    except Exception as e:
        print(e)
        print("[ERROR] Cannot import modules.Basic_HD_online")
        sys.exit(1)

    runner = BasicHDOnline(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)
    runner.start()

    print(f"[INFO] Finished at: {datetime.datetime.now().isoformat()}")
