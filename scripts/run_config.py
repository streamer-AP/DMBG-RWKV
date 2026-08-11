#!/usr/bin/env python3
"""Run DMBG-RWKV train/eval commands from a JSON config file."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Config must be a JSON object: {path}")
    required_sections = {"dataset", "model", "training", "evaluation", "runtime"}
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"Config is missing sections {sorted(missing)}: {path}")
    for section in required_sections:
        if not isinstance(config[section], dict):
            raise ValueError(f"Config section '{section}' must be an object: {path}")
    if not isinstance(config["dataset"].get("splits"), dict):
        raise ValueError(f"Config field 'dataset.splits' must be an object: {path}")
    return config


def _merge_args(config: Dict[str, Any], action: str) -> Dict[str, Any]:
    dataset = config.get("dataset", {})
    model = config.get("model", {})
    training = config.get("training", {})
    evaluation = config.get("evaluation", {})
    splits = dataset.get("splits", {})

    merged: Dict[str, Any] = {
        "dataset": dataset.get("name"),
        "root_path": dataset.get("root_path"),
        "volume_path": dataset.get("volume_path"),
        "list_dir": dataset.get("list_dir"),
        "train_split": splits.get("train"),
        "val_split": splits.get("val"),
        "test_split": splits.get("test"),
        "num_classes": dataset.get("num_classes"),
        "seq_length": dataset.get("seq_length"),
        "z_spacing": dataset.get("z_spacing"),
        "img_size": model.get("img_size"),
    }
    merged.update(training)
    if action == "train":
        merged["pretrained_path"] = model.get("pretrained_path")
    else:
        merged.update(evaluation)
    return merged


def _to_cli_args(values: Dict[str, Any]) -> List[str]:
    args: List[str] = []
    for key, value in values.items():
        if value is None:
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            args.append(flag if value else f"--no-{key}")
            continue
        if isinstance(value, (list, tuple)):
            for item in value:
                args.extend([flag, str(item)])
            continue
        args.extend([flag, str(value)])
    return args


def _normalize_extra(extra: Iterable[str]) -> List[str]:
    extra = list(extra)
    if extra and extra[0] == "--":
        return extra[1:]
    return extra


def build_command(action: str, config_path: Path, extra_args: Iterable[str]) -> List[str]:
    config = _load_config(config_path)
    entry = PROJECT_ROOT / "src" / ("train.py" if action == "train" else "test.py")
    return [sys.executable, str(entry)] + _to_cli_args(_merge_args(config, action)) + _normalize_extra(extra_args)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "eval"])
    parser.add_argument("config", type=Path)
    parser.add_argument("--print-only", action="store_true", help="print the resolved command and exit")
    args, extra_args = parser.parse_known_args()

    config_path = args.config if args.config.is_absolute() else PROJECT_ROOT / args.config
    config = _load_config(config_path)
    command = build_command(args.action, config_path, extra_args)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{SRC_DIR}:{env.get('PYTHONPATH', '')}".rstrip(":")
    cuda_visible_devices = config.get("runtime", {}).get("cuda_visible_devices", "0")
    if cuda_visible_devices is not None and "CUDA_VISIBLE_DEVICES" not in env:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_visible_devices)

    if args.print_only:
        print(" ".join(command))
        return 0

    print("Running:", " ".join(command), flush=True)
    return subprocess.call(command, cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
