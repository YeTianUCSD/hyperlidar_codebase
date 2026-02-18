import argparse
import datetime
import yaml
import os
import sys
import shutil
import torch
from modules.ioueval import iouEval
from itertools import islice




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
    if bits < 2 or bits > 8:
        print(f"[ERROR] hd_quant_bits must be in [2,8], got {bits}")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Dump HD vectors for offline RRAM sim")
    parser.add_argument('--dataset', '-d', type=str, required=True)
    parser.add_argument('--log', '-l', type=str, required=True)
    parser.add_argument('--model', '-m', type=str, required=True)
    parser.add_argument('--arch_cfg', '-ac', type=str, default=None)
    parser.add_argument('--data_cfg', '-dc', type=str, default=None)

    parser.add_argument('--hd_mode', type=str, default="nbit", choices=['float', 'nbit'])
    parser.add_argument('--hd_bits', type=int, default=6, choices=[2, 4, 6, 8])

    # fast controls
    parser.add_argument('--dump_points', type=int, default=2048, help="how many pixels to dump")
    parser.add_argument('--only_one_batch', action='store_true', help="stop after dumping once")
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

    # 2) quantize class HV
    if ARCH["train"]["hd_quant_mode"] == "nbit":
        runner.quantize_class_hv_nbit()

    # 3) run validation with quantized path -> triggers dump inside validate()
    print("[INFO] Running VALID to trigger dump...")
    
    class _DummyEval:
        def reset(self): pass
        def addBatch(self, *args, **kwargs): pass
        def getacc(self):
            return torch.tensor(0.0)
        def getIoU(self):
            return torch.tensor(0.0), torch.zeros(runner.num_classes)

    dummy = _DummyEval()
    runner.validate(runner.parser.get_valid_set(), runner.model, dummy, use_quantized=True)
    '''
    dummy = _DummyEval()
    val_loader = islice(runner.parser.get_valid_set(), FLAGS.val_batches)
    runner.validate(val_loader, runner.model, dummy, use_quantized=True)
    '''


    print(f"[INFO] Finished at: {datetime.datetime.now().isoformat()}")
    print(f"[INFO] Check dump at: {os.path.join(FLAGS.log, ARCH['train'].get('hd_dump_dirname','rram_dump'))}")
