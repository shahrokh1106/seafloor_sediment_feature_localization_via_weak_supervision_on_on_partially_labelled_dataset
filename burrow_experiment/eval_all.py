#!/usr/bin/env python3
"""
Build simple evaluation JSONs in out_eval/ — one file per out_labels run.

Each JSON lists train box counts and val metrics at initial (pseudo labels)
and at each SSL iteration.

Usage:
    python burrow_experiment/eval_all.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_EVAL = SCRIPT_DIR / "out_eval"

RUNS = [
    ("labels_dino_global", "out_labels_dino_global", "python burrow_experiment/run_burrow.py labels_dino_global --iterations 10 --device 0"),
    ("labels_dino_local", "out_labels_dino_local", "python burrow_experiment/run_burrow.py labels_dino_local --iterations 10 --device 0"),
    ("labels_sam_global", "out_labels_sam_global", "python burrow_experiment/run_burrow.py labels_sam_global --iterations 10 --device 0"),
    ("labels_sam_local", "out_labels_sam_local", "python burrow_experiment/run_burrow.py labels_sam_local --iterations 10 --device 0"),
    ("base_gt", "out_labels_base_gt", "python burrow_experiment/train_base.py --device 0"),
]

METRICS = ("precision", "recall", "f1", "map50", "map50-95")


def count_boxes(dataset_dir: Path, split: str) -> int:
    split_txt = dataset_dir / f"{split}.txt"
    labels_dir = dataset_dir / "labels"
    total = 0
    for line in split_txt.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        stem = Path(line).stem
        label_file = labels_dir / f"{stem}.txt"
        if label_file.exists():
            total += sum(1 for row in label_file.read_text(encoding="utf-8").splitlines() if row.strip())
    return total


def count_train_boxes(dataset_dir: Path) -> int:
    return count_boxes(dataset_dir, "train")


def count_val_boxes(dataset_dir: Path) -> int:
    return count_boxes(dataset_dir, "val")


def pick_metrics(raw: dict | None) -> dict | None:
    if not raw:
        return None
    return {k: raw[k] for k in METRICS if k in raw}


def step(name: str, train_boxes: int, val_boxes: int, raw_metrics: dict | None) -> dict:
    out = {"step": name, "train_boxes": train_boxes, "val_boxes": val_boxes}
    metrics = pick_metrics(raw_metrics)
    if metrics:
        out.update(metrics)
    return out


def build_ssl_report(out_dir: Path) -> dict:
    training_log = json.loads((out_dir / "ssl" / "training_log.json").read_text(encoding="utf-8"))
    initial_metrics = json.loads((out_dir / "initial_val_metrics.json").read_text(encoding="utf-8"))
    dataset_dir = out_dir / "dataset"
    val_boxes = count_val_boxes(dataset_dir)

    steps = [step("initial", count_train_boxes(dataset_dir), val_boxes, initial_metrics)]

    for entry in training_log:
        n = entry["iteration"]
        boxes = entry["pseudo_stats"]["total_boxes"]
        student_metrics = entry["student"]["metrics"]
        steps.append(step(f"iter_{n}", boxes, val_boxes, student_metrics))

    return {"steps": steps}


def build_base_report(out_dir: Path) -> dict:
    evaluation = json.loads((out_dir / "evaluation.json").read_text(encoding="utf-8"))
    dataset_dir = out_dir / "dataset"
    train_boxes = evaluation.get("split_manifest", {}).get("train_boxes")
    if train_boxes is None:
        train_boxes = count_train_boxes(dataset_dir)
    val_boxes = evaluation.get("split_manifest", {}).get("val_boxes")
    if val_boxes is None:
        val_boxes = count_val_boxes(dataset_dir)
    return {"steps": [step("initial", train_boxes, val_boxes, evaluation.get("metrics"))]}


def is_ready(out_dir: Path, is_base: bool) -> bool:
    if not out_dir.is_dir():
        return False
    if is_base:
        return (out_dir / "evaluation.json").exists()
    return (
        (out_dir / "ssl" / "training_log.json").exists()
        and (out_dir / "initial_val_metrics.json").exists()
        and (out_dir / "dataset" / "train.txt").exists()
    )


def main() -> int:
    missing = []
    for name, folder, command in RUNS:
        out_dir = SCRIPT_DIR / folder
        is_base = name == "base_gt"
        if not is_ready(out_dir, is_base):
            missing.append((name, command))

    if missing:
        print("Missing runs. Create them first:\n")
        for name, command in missing:
            print(f"  # {name}")
            print(f"  {command}\n")
        return 1

    OUT_EVAL.mkdir(parents=True, exist_ok=True)

    for name, folder, _ in RUNS:
        out_dir = SCRIPT_DIR / folder
        if name == "base_gt":
            report = build_base_report(out_dir)
        else:
            report = build_ssl_report(out_dir)

        out_path = OUT_EVAL / f"{name}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
