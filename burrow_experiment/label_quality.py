#!/usr/bin/env python3
"""
Compare training labels against GT on the train split at each SSL step.

Writes out_eval/label_quality.json with one steps list per experiment.

Usage:
    python burrow_experiment/label_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
OUT_EVAL = SCRIPT_DIR / "out_eval"
GT_DIR = SCRIPT_DIR / "dataset_burrow" / "labels_gt"
PSEUDO_DIR = SCRIPT_DIR / "dataset_burrow"

SSL_RUNS = [
    ("labels_dino_global", "out_labels_dino_global"),
    ("labels_dino_local", "out_labels_dino_local"),
    ("labels_sam_global", "out_labels_sam_global"),
    ("labels_sam_local", "out_labels_sam_local"),
]

BASE_RUN = ("base_gt", "out_labels_base_gt")


def _import_evaluate_set():
    fm_dir = REPO_ROOT / "feature_matching_scripts"
    if str(fm_dir) not in sys.path:
        sys.path.insert(0, str(fm_dir))
    from pseudo_common import evaluate_set  # noqa: WPS433

    return evaluate_set


def train_stems(dataset_dir: Path) -> list[str]:
    stems = []
    for line in (dataset_dir / "train.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            stems.append(Path(line).stem)
    return stems


def flatten_eval(step: str, results: dict) -> dict:
    out = {
        "step": step,
        "images": results["images"],
        "pred_boxes": results["pred_boxes"],
        "gt_boxes": results["gt_boxes"],
        "avg_pred_per_img": results["avg_pred_per_img"],
        "avg_gt_per_img": results["avg_gt_per_img"],
        "map": results["map"],
        "map_50": results["map_50"],
        "map_75": results["map_75"],
        "mar_100": results["mar_100"],
        "mean_best_iou_gt": results["mean_best_iou_gt"],
        "mean_best_iou_pred": results["mean_best_iou_pred"],
        "mean_matched_iou_50": results["mean_matched_iou_50"],
    }
    for thr_key, thr_vals in results["by_thr"].items():
        tag = thr_key.replace(".", "")
        out[f"tp_{tag}"] = thr_vals["tp"]
        out[f"fp_{tag}"] = thr_vals["fp"]
        out[f"fn_{tag}"] = thr_vals["fn"]
        out[f"precision_{tag}"] = thr_vals["precision"]
        out[f"recall_{tag}"] = thr_vals["recall"]
        out[f"f1_{tag}"] = thr_vals["f1"]
    return out


def build_ssl_quality(out_dir: Path, label_source: str, evaluate_set) -> dict:
    stems = train_stems(out_dir / "dataset")
    steps = []

    initial_pred = PSEUDO_DIR / label_source
    steps.append(flatten_eval("initial", evaluate_set(initial_pred, GT_DIR, stems)))

    training_log = json.loads((out_dir / "ssl" / "training_log.json").read_text(encoding="utf-8"))
    for entry in training_log:
        n = entry["iteration"]
        pred_dir = out_dir / "ssl" / f"pseudo_labels_iter{n}"
        steps.append(flatten_eval(f"iter_{n}", evaluate_set(pred_dir, GT_DIR, stems)))

    return {"steps": steps}


def build_base_quality(out_dir: Path, evaluate_set) -> dict:
    stems = train_stems(out_dir / "dataset")
    steps = [flatten_eval("initial", evaluate_set(GT_DIR, GT_DIR, stems))]
    return {"steps": steps}


def is_ssl_ready(out_dir: Path) -> bool:
    return (
        out_dir.is_dir()
        and (out_dir / "dataset" / "train.txt").exists()
        and (out_dir / "ssl" / "training_log.json").exists()
    )


def is_base_ready(out_dir: Path) -> bool:
    return out_dir.is_dir() and (out_dir / "dataset" / "train.txt").exists()


def main() -> int:
    missing = []
    for name, folder in SSL_RUNS:
        if not is_ssl_ready(SCRIPT_DIR / folder):
            missing.append(name)
    if not is_base_ready(SCRIPT_DIR / BASE_RUN[1]):
        missing.append(BASE_RUN[0])

    if missing:
        print("Missing runs. Create them first:")
        print("  python burrow_experiment/eval_all.py")
        print("Missing:", ", ".join(missing))
        return 1

    evaluate_set = _import_evaluate_set()
    report: dict = {}

    for name, folder in SSL_RUNS:
        out_dir = SCRIPT_DIR / folder
        run_config = json.loads((out_dir / "run_config.json").read_text(encoding="utf-8"))
        label_source = run_config["label_source"]
        report[name] = build_ssl_quality(out_dir, label_source, evaluate_set)
        print(f"Computed label quality for {name}")

    report[BASE_RUN[0]] = build_base_quality(SCRIPT_DIR / BASE_RUN[1], evaluate_set)
    print(f"Computed label quality for {BASE_RUN[0]}")

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    out_path = OUT_EVAL / "label_quality.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
