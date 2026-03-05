
import argparse
import datetime
import yaml
import os
import sys
import shutil


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
    """
    Resolve ARCH/DATA yaml paths by priority:
    1) CLI: --arch_cfg/--data_cfg
    2) Model dir: <model>/arch_cfg.yaml and <model>/data_cfg.yaml
    3) Project defaults: config/arch/senet-512.yml and config/labels/semantic-kitti.yaml
    """
    # 1) CLI
    if flags.arch_cfg and not os.path.isfile(flags.arch_cfg):
        print(f"[ERROR] --arch_cfg not found: {flags.arch_cfg}")
        sys.exit(1)
    if flags.data_cfg and not os.path.isfile(flags.data_cfg):
        print(f"[ERROR] --data_cfg not found: {flags.data_cfg}")
        sys.exit(1)

    # 2) Model dir
    model_arch = os.path.join(flags.model, "arch_cfg.yaml")
    model_data = os.path.join(flags.model, "data_cfg.yaml")

    # 3) Defaults (project relative)
    default_arch = "config/arch/senet-512.yml"
    default_data = "config/labels/semantic-kitti.yaml"

    arch_path = (
        flags.arch_cfg
        if flags.arch_cfg
        else (model_arch if os.path.isfile(model_arch) else default_arch)
    )
    data_path = (
        flags.data_cfg
        if flags.data_cfg
        else (model_data if os.path.isfile(model_data) else default_data)
    )

    if not os.path.isfile(arch_path):
        print(f"[ERROR] arch cfg not found: {arch_path}")
        sys.exit(1)
    if not os.path.isfile(data_path):
        print(f"[ERROR] data cfg not found: {data_path}")
        sys.exit(1)

    return arch_path, data_path


def _apply_hd_overrides(ARCH: dict, flags):
    """
    Apply CLI overrides to ARCH so that modules/Basic_HD.py and modules/HDC_utils.py can read:
      ARCH["train"]["hd_quant_mode"]  in {"float","nbit"}
      ARCH["train"]["hd_quant_bits"]  in {1,2,4,6,8}  (int8-backed)
    """
    if ARCH is None:
        raise RuntimeError("ARCH yaml parsed to None (empty file?)")

    ARCH.setdefault("train", {})

    # If user sets bits but not mode, default mode -> nbit
    if flags.hd_bits is not None and flags.hd_mode is None:
        ARCH["train"]["hd_quant_mode"] = "nbit"

    # Explicit mode override
    if flags.hd_mode is not None:
        ARCH["train"]["hd_quant_mode"] = flags.hd_mode

    # Explicit bits override
    if flags.hd_bits is not None:
        ARCH["train"]["hd_quant_bits"] = int(flags.hd_bits)

    # Set defaults if missing (keep old behavior)
    ARCH["train"].setdefault("hd_quant_mode", "nbit")
    ARCH["train"].setdefault("hd_quant_bits", 4)

    # Safety: int8 storage => bits must be <= 8
    bits = int(ARCH["train"]["hd_quant_bits"])
    if bits < 1 or bits > 8:
        print(f"[ERROR] hd_quant_bits must be in [1,8] for int8-backed quantization, got {bits}")
        sys.exit(1)

    mode = str(ARCH["train"]["hd_quant_mode"]).lower()
    if mode not in ("float", "nbit"):
        print(f"[ERROR] hd_quant_mode must be 'float' or 'nbit', got {mode}")
        sys.exit(1)

    return ARCH


def _maybe_snapshot_cfgs(log_dir: str, arch_path: str, data_path: str):
    """
    Optional: copy the exact cfgs used into the log folder for reproducibility.
    """
    try:
        shutil.copy2(arch_path, os.path.join(log_dir, "arch_cfg_used.yaml"))
        shutil.copy2(data_path, os.path.join(log_dir, "data_cfg_used.yaml"))
        print("[INFO] Saved cfg snapshots to log dir: arch_cfg_used.yaml / data_cfg_used.yaml")
    except Exception as e:
        print("[WARN] Failed to snapshot cfg files into log dir:", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Inference entry (main.py)")
    parser.add_argument('--dataset', '-d', type=str, required=True,
                        help='Dataset root (SemanticKITTI-style).')
    parser.add_argument('--log', '-l', type=str, required=True,
                        help='Directory to put the predictions.')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Directory or file to get the trained model from.')
    parser.add_argument('--train_seq', '-t', type=str, default=None,
                        help='Comma-separated list of sequences for processing (e.g. "5,6" or "61,553").')
    parser.add_argument('--arch_cfg', '-ac', type=str, default=None,
                        help='Path to arch yaml. If omitted, try <model>/arch_cfg.yaml, else project default.')
    parser.add_argument('--data_cfg', '-dc', type=str, default=None,
                        help='Path to data yaml. If omitted, try <model>/data_cfg.yaml, else project default.')

    # NEW: HD quant switches
    parser.add_argument('--hd_mode', type=str, default=None, choices=['float', 'nbit'],
                        help='HD mode: float (skip quantization + quantized eval) or nbit (quantized).')
    parser.add_argument('--hd_bits', type=int, default=None, choices=[1, 2, 4, 6, 8],
                        help='HD quant bits for nbit mode (1/2/4/6/8). If set without --hd_mode, defaults to nbit.')

    FLAGS, unknown = parser.parse_known_args()
    if unknown:
        print("[WARN] Unknown args ignored:", unknown)

    print("----------")
    print("INTERFACE:")
    print("dataset   :", FLAGS.dataset)
    print("log       :", FLAGS.log)
    print("model     :", FLAGS.model)
    print("train_seq :", FLAGS.train_seq if FLAGS.train_seq is not None else "(default from yaml)")
    print("arch_cfg  :", FLAGS.arch_cfg if FLAGS.arch_cfg else "(auto)")
    print("data_cfg  :", FLAGS.data_cfg if FLAGS.data_cfg else "(auto)")
    print("hd_mode   :", FLAGS.hd_mode if FLAGS.hd_mode else "(yaml/default)")
    print("hd_bits   :", FLAGS.hd_bits if FLAGS.hd_bits is not None else "(yaml/default)")
    print("----------\n")

    # Basic path checks
    if not os.path.isdir(FLAGS.model):
        print(f"[ERROR] model folder doesn't exist: {FLAGS.model}")
        sys.exit(1)
    if not os.path.isdir(FLAGS.dataset):
        print(f"[ERROR] dataset folder doesn't exist: {FLAGS.dataset}")
        sys.exit(1)

    # Resolve cfg paths + load yaml
    arch_path, data_path = _resolve_cfg_paths(FLAGS)
    ARCH = _read_yaml(arch_path, "arch")
    DATA = _read_yaml(data_path, "data")

    # Apply HD overrides into ARCH["train"]
    ARCH = _apply_hd_overrides(ARCH, FLAGS)

    print("[INFO] Effective HD config:",
          "hd_quant_mode =", ARCH["train"].get("hd_quant_mode"),
          "| hd_quant_bits =", ARCH["train"].get("hd_quant_bits"))

    # Override training sequences if provided
    if FLAGS.train_seq is not None:
        try:
            parsed = [int(s.strip()) for s in FLAGS.train_seq.split(",") if s.strip() != ""]
            DATA.setdefault("split", {})
            DATA["split"]["train"] = parsed
            print(f"[INFO] Overriding sequences: {DATA['split']['train']}")
        except ValueError as e:
            print(f"[ERROR] Failed to parse --train_seq: {e}")
            sys.exit(1)
    else:
        print("[INFO] Using default sequences from data config (split.train).")

    # Create log dir
    try:
        os.makedirs(FLAGS.log, exist_ok=True)
    except Exception as e:
        print(e)
        print("[ERROR] Error creating log directory. Check permissions!")
        sys.exit(1)

    # Snapshot used cfgs into log dir (optional but useful)
    _maybe_snapshot_cfgs(FLAGS.log, arch_path, data_path)

    print(f"[INFO] Start time: {datetime.datetime.now().isoformat()}")
    print(f"[INFO] Saving predictions to: {FLAGS.log}")

    # Import BasicHD
    try:
        from modules.Basic_HD import BasicHD
    except Exception as e:
        print(e)
        print("[ERROR] Cannot import modules.Basic_HD. Check PYTHONPATH and project structure.")
        sys.exit(1)

    # Run
    runner = BasicHD(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)
    runner.start()

    print(f"[INFO] Finished at: {datetime.datetime.now().isoformat()}")
