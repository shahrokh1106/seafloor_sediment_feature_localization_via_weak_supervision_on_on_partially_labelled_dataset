#!/usr/bin/env python3
"""
Estimate burrow annotation time from a short timed drawing session.

Shows 10 random images (fixed seed). Draw one box per image; measures elapsed
time per box. Extrapolates to:
  - feature-matching approach: avg_time * number of seed boxes in the dataset
  - full manual annotation: avg_time * total GT boxes in the dataset

If out_eval/annotation_time.json already exists, prints saved results and exits.

Usage:
    python burrow_experiment/time.py
    python burrow_experiment/time.py --force   # re-run even if JSON exists
"""

from __future__ import annotations

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR = SCRIPT_DIR / "dataset_burrow"
IMAGES_DIR = DATASET_DIR / "images"
SEED_DIR = DATASET_DIR / "labels_seed"
GT_DIR = DATASET_DIR / "labels_gt"
OUT_JSON = SCRIPT_DIR / "out_eval" / "annotation_time.json"

SEED = 42
SAMPLE_SIZE = 10
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_gt_boxes() -> int:
    total = 0
    for path in GT_DIR.glob("*.txt"):
        total += sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def count_seed_boxes() -> int:
    return sum(1 for path in SEED_DIR.glob("*.txt") if path.read_text(encoding="utf-8").strip())


def list_images() -> list[Path]:
    return sorted(
        p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def sample_images(images: list[Path], n: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    if n >= len(images):
        return images.copy()
    return rng.sample(images, n)


def draw_one_box(image_bgr) -> tuple[str, float | None]:
    """
    Let the user draw one box. Returns ('save', seconds) or ('quit', None).
    Timer starts when the window opens; stops when the user presses s.
    """
    box_color = (0, 0, 255)
    drawing = False
    points: list[tuple[int, int]] = []
    img_orig = image_bgr.copy()
    img = img_orig.copy()
    t0 = time.perf_counter()

    def on_mouse(event, x, y, flags, params):
        nonlocal drawing, points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = img_orig.copy()
            cv2.rectangle(img, points[0], (x, y), box_color, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(points) == 1:
                points.append((x, y))
            else:
                points[1] = (x, y)
            img = img_orig.copy()
            cv2.rectangle(img, points[0], points[1], box_color, 2)

    window = "Annotate burrow (s=save, r=reset, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(window, on_mouse)

    while True:
        cv2.imshow(window, img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            img = img_orig.copy()
            points = []
            drawing = False
        elif key == ord("s"):
            if len(points) == 2:
                elapsed = time.perf_counter() - t0
                cv2.destroyAllWindows()
                return "save", elapsed
            print("  Draw a box before saving (s)")
        elif key == ord("q"):
            cv2.destroyAllWindows()
            return "quit", None

    cv2.destroyAllWindows()
    return "quit", None


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} s"
    if seconds < 3600:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def print_results(data: dict) -> None:
    avg = data["avg_seconds_per_box"]
    approach = data["estimated_approach_seconds"]
    manual = data["estimated_manual_seconds"]
    print("\n=== Annotation time estimate ===")
    print(f"Sample size:        {data['sample_size']} images (seed {data['seed']})")
    print(f"Avg time per box:   {avg:.2f} s")
    print(f"Seed boxes (dataset): {data['dataset']['total_seed_boxes']}")
    print(f"GT boxes (dataset):   {data['dataset']['total_gt_boxes']}")
    print(f"Approach (seeds):   {format_duration(approach)}  ({approach:.1f} s)")
    print(f"Manual (all GT):    {format_duration(manual)}  ({manual:.1f} s)")
    if manual > 0:
        print(f"Ratio manual/approach: {manual / approach:.1f}x")
    print(f"Saved: {OUT_JSON}")


def run_session(force: bool) -> int:
    if OUT_JSON.exists() and not force:
        data = json.loads(OUT_JSON.read_text(encoding="utf-8"))
        print_results(data)
        return 0

    if not IMAGES_DIR.is_dir():
        print(f"Images not found: {IMAGES_DIR}")
        return 1
    if not GT_DIR.is_dir():
        print(f"GT labels not found: {GT_DIR}")
        return 1

    images = list_images()
    if len(images) < SAMPLE_SIZE:
        print(f"Need at least {SAMPLE_SIZE} images, found {len(images)}")
        return 1

    chosen = sample_images(images, SAMPLE_SIZE, SEED)
    total_seeds = count_seed_boxes()
    total_gt = count_gt_boxes()

    print(f"Timed annotation — {SAMPLE_SIZE} random images (seed {SEED})")
    print("Draw one burrow box per image. Controls: drag | r=reset | s=save | q=quit")
    print(f"Dataset: {total_seeds} seed boxes, {total_gt} GT boxes\n")

    times: list[float] = []
    recorded_images: list[str] = []

    for i, img_path in enumerate(chosen, 1):
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[{i}/{SAMPLE_SIZE}] could not read {img_path.name}, skipping")
            continue

        h, w = image_bgr.shape[:2]
        print(f"[{i}/{SAMPLE_SIZE}] {img_path.name} ({w}x{h})")

        action, elapsed = draw_one_box(image_bgr)
        if action == "quit":
            return 0
        if elapsed is None:
            continue

        times.append(elapsed)
        recorded_images.append(img_path.name)
        print(f"  time: {elapsed:.2f} s")

    if len(times) != SAMPLE_SIZE:
        print(f"Completed {len(times)}/{SAMPLE_SIZE} — results not saved.")
        return 1

    avg = sum(times) / len(times)
    approach_seconds = avg * total_seeds
    manual_seconds = avg * total_gt

    data = {
        "seed": SEED,
        "sample_size": SAMPLE_SIZE,
        "sample_images": recorded_images,
        "times_seconds": times,
        "avg_seconds_per_box": avg,
        "dataset": {
            "total_images": len(images),
            "total_seed_boxes": total_seeds,
            "total_gt_boxes": total_gt,
        },
        "estimated_approach_seconds": approach_seconds,
        "estimated_manual_seconds": manual_seconds,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print_results(data)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate burrow annotation time from timed box drawing.")
    parser.add_argument("--force", action="store_true", help="Re-run timed session even if JSON exists")
    return parser.parse_args()


def main() -> int:
    return run_session(parse_args().force)


if __name__ == "__main__":
    raise SystemExit(main())
