#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import yaml
import os
import sys
import shutil

def _read_yaml(path, kind):
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

    FLAGS, _ = parser.parse_known_args()

    
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("arch_cfg", FLAGS.arch_cfg if FLAGS.arch_cfg else "(auto)")
    print("data_cfg", FLAGS.data_cfg if FLAGS.data_cfg else "(auto)")
    print("----------\n")

    
    if not os.path.isdir(FLAGS.model):
        print(f"[ERROR] model folder doesnt exist: {FLAGS.model}")
        sys.exit(1)
    if not os.path.isdir(FLAGS.dataset):
        print(f"[ERROR] dataset folder doesnt exist: {FLAGS.dataset}")
        sys.exit(1)

    
    arch_path, data_path = _resolve_cfg_paths(FLAGS)
    ARCH = _read_yaml(arch_path, "arch")
    DATA = _read_yaml(data_path, "data")

  
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


    try:
        os.makedirs(FLAGS.log, exist_ok=True)
    except Exception as e:
        print(e)
        print("[ERROR] Error creating log directory. Check permissions!")
        sys.exit(1)


    print(f"[INFO] Start time: {datetime.datetime.now().isoformat()}")
    print(f"[INFO] Saving predictions to: {FLAGS.log}")


    try:
        from modules.Basic_HD import BasicHD
    except Exception as e:
        print(e)
        print("[ERROR] Cannot import modules.Basic_HD. Check PYTHONPATH and project structure.")
        sys.exit(1)

    # BasicHD(dataset_cfg, data_cfg, dataset_root, log_dir, model_dir, extra)
    BasicHD = BasicHD(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model, None)
    BasicHD.start()

    print(f"[INFO] Finished at: {datetime.datetime.now().isoformat()}")
