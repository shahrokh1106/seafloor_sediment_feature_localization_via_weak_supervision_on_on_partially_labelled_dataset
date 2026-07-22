#!/usr/bin/env python3
"""
Plot burrow experiment metrics from out_eval/ JSON files.

One figure per metric: 4 SSL approach lines vs iteration, GT baseline as horizontal line.

Usage:
    python burrow_experiment/plot_eval.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_EVAL = SCRIPT_DIR / "out_eval"
PLOT_DIR = OUT_EVAL / "plots"

APPROACHES = [
    ("labels_dino_global", "DINO Global"),
    ("labels_dino_local", "DINO Local"),
    ("labels_sam_global", "SAM Global"),
    ("labels_sam_local", "SAM Local"),
]

BASE_FILE = ("base_gt", "GT Baseline")
METRICS = [
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("map50", "mAP50"),
    ("map50-95", "mAP50-95"),
]


def step_to_x(step: str) -> int:
    if step == "initial":
        return 0
    return int(step.split("_", 1)[1])


def load_steps(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["steps"]


def series_for_metric(steps: list[dict], metric: str) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    for entry in steps:
        if metric not in entry:
            continue
        xs.append(step_to_x(entry["step"]))
        ys.append(entry[metric])
    return xs, ys


def main() -> int:
    missing = [name for name, _ in APPROACHES if not (OUT_EVAL / f"{name}.json").exists()]
    if not (OUT_EVAL / f"{BASE_FILE[0]}.json").exists():
        missing.append(BASE_FILE[0])
    if missing:
        print("Missing eval JSON(s). Run first:")
        print("  python burrow_experiment/eval_all.py")
        print("Missing:", ", ".join(missing))
        return 1

    base_steps = load_steps(OUT_EVAL / f"{BASE_FILE[0]}.json")
    base_values = {m: base_steps[0][m] for m, _ in METRICS if m in base_steps[0]}

    approach_data = []
    for file_name, label in APPROACHES:
        steps = load_steps(OUT_EVAL / f"{file_name}.json")
        approach_data.append((label, steps))

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for metric_key, metric_title in METRICS:
        fig, ax = plt.subplots(figsize=(8, 5))

        all_x: list[int] = []
        for _, steps in approach_data:
            xs, _ = series_for_metric(steps, metric_key)
            all_x.extend(xs)

        if metric_key in base_values:
            xmin = min(all_x) if all_x else 0
            xmax = max(all_x) if all_x else 10
            ax.axhline(
                base_values[metric_key],
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=BASE_FILE[1],
            )
            ax.set_xlim(xmin - 0.25, xmax + 0.25)

        for label, steps in approach_data:
            xs, ys = series_for_metric(steps, metric_key)
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)

        ax.set_xlabel("Iteration")
        ax.set_ylabel(metric_title)
        ax.set_title(f"{metric_title} vs SSL iteration")
        ax.set_xticks(range(0, max(all_x) + 1 if all_x else 11))
        ax.set_xticklabels(["initial"] + [str(i) for i in range(1, max(all_x) + 1)] if all_x else [])
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        fig.tight_layout()

        out_name = metric_key.replace("-", "_")
        out_path = PLOT_DIR / f"{out_name}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
