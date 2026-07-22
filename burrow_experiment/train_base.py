#!/usr/bin/env python3
"""
Train a YOLO baseline on ground-truth burrow labels using the same train/val split
and YOLO configuration as burrow SSL initial training.

Usage (from repo root or this folder):
    python burrow_experiment/train_base.py
    python burrow_experiment/train_base.py --epochs 200 --device 0

Outputs are written under burrow_experiment/out_labels_base_gt/.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import yaml

# --- paths -------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SOURCE_DATASET = SCRIPT_DIR / "dataset_burrow"
GT_LABELS_DIR = SOURCE_DATASET / "labels_gt"
IMAGES_DIR = SOURCE_DATASET / "images"
OUTPUT_DIR = SCRIPT_DIR / "out_labels_base_gt"

SEED = 42
TRAIN_RATIO = 0.8

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_seeds() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)


def ensure_repo_on_path() -> None:
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def _ensure_dir_link(dst: Path, src: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        dst.symlink_to(src, target_is_directory=True)
    except (OSError, NotImplementedError):
        try:
            subprocess.check_call(
                ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            logger.warning("Could not link %s -> %s; copying instead", dst, src)
            shutil.copytree(src, dst, dirs_exist_ok=True)


def _copy_or_empty_label(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.exists():
        shutil.copy2(src, dst)
    else:
        dst.write_text("", encoding="utf-8")


def _image_paths(images: Iterable[Path], images_root: Path) -> list[str]:
    return [str(images_root / img.name) for img in images]


def _write_split_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _json_safe_metrics(metrics: dict) -> dict:
    out = dict(metrics)
    per_class = out.get("per_class_ap")
    if per_class is not None:
        out["per_class_ap"] = np.asarray(per_class).tolist()
    names = out.get("names")
    if names is not None:
        out["names"] = {str(k): v for k, v in names.items()}
    return out


def _count_boxes(label_dir: Path, image_stems: Iterable[str]) -> int:
    total = 0
    for stem in image_stems:
        label_path = label_dir / f"{stem}.txt"
        if label_path.exists():
            total += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return total


def prepare_gt_dataset(output_dir: Path, force: bool = False) -> Path:
    """
    Build a YOLO dataset under output_dir/dataset/.

    Train and val both use labels_gt (same split as burrow SSL experiments).
    """
    dataset_dir = output_dir / "dataset"
    manifest_path = dataset_dir / "split_manifest.json"
    data_yaml_path = dataset_dir / "data.yaml"

    if not force and manifest_path.exists() and data_yaml_path.exists():
        logger.info("Reusing existing prepared dataset at %s", dataset_dir)
        return dataset_dir

    if not GT_LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Ground-truth labels not found: {GT_LABELS_DIR}")
    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Images folder not found: {IMAGES_DIR}")

    image_files = sorted(
        p for p in IMAGES_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_files:
        raise RuntimeError(f"No images found in {IMAGES_DIR}")

    rng = random.Random(SEED)
    shuffled = image_files.copy()
    rng.shuffle(shuffled)

    n_train = int(len(shuffled) * TRAIN_RATIO)
    train_images = shuffled[:n_train]
    val_images = shuffled[n_train:]

    if dataset_dir.exists() and force:
        shutil.rmtree(dataset_dir)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    linked_images = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    _ensure_dir_link(linked_images, IMAGES_DIR)

    for img in train_images + val_images:
        _copy_or_empty_label(GT_LABELS_DIR / f"{img.stem}.txt", labels_dir / f"{img.stem}.txt")

    train_lines = _image_paths(train_images, linked_images)
    val_lines = _image_paths(val_images, linked_images)
    _write_split_file(dataset_dir / "train.txt", train_lines)
    _write_split_file(dataset_dir / "val.txt", val_lines)
    _write_split_file(dataset_dir / "test.txt", val_lines)

    yaml_content = {
        "path": dataset_dir.resolve().as_posix(),
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "nc": 1,
        "names": ["Burrow"],
    }
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)

    train_stems = [img.stem for img in train_images]
    val_stems = [img.stem for img in val_images]
    manifest = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "label_source": "labels_gt",
        "gt_labels_dir": str(GT_LABELS_DIR.resolve()),
        "total_images": len(image_files),
        "train_count": len(train_images),
        "val_count": len(val_images),
        "train_boxes": _count_boxes(labels_dir, train_stems),
        "val_boxes": _count_boxes(labels_dir, val_stems),
        "train_images": [p.name for p in train_images],
        "val_images": [p.name for p in val_images],
        "note": "Train and val labels are ground truth (labels_gt). Split matches burrow SSL runs.",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Prepared GT dataset: %d train, %d val (%d / %d boxes) -> %s",
        len(train_images),
        len(val_images),
        manifest["train_boxes"],
        manifest["val_boxes"],
        dataset_dir,
    )
    return dataset_dir


def run_base_training(
    dataset_dir: Path,
    output_dir: Path,
    model_path: Path,
    imgsz: int,
    epochs: int,
    batch_size: int,
    device,
) -> Path:
    ensure_repo_on_path()
    from initial_training import get_val_metrics, train_yolo, validate_dataset

    data_yaml = dataset_dir / "data.yaml"

    logger.info("=" * 60)
    logger.info("BASE GT TRAINING")
    logger.info("=" * 60)

    dataset_stats = validate_dataset(str(data_yaml))
    best_model = train_yolo(
        data_yaml_path=str(data_yaml),
        model_path=str(model_path),
        out_dir=str(output_dir),
        imgsz=imgsz,
        epochs=epochs,
        batchsize=batch_size,
        device=device,
        multi_scale=True,
        run_name="base",
    )
    best_model = Path(best_model)

    metrics = get_val_metrics(best_model, str(data_yaml), imgsz=imgsz, device=device, split="val")
    safe_metrics = _json_safe_metrics(metrics)

    manifest_path = dataset_dir / "split_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        split_manifest = json.load(f)

    evaluation = {
        "model": "base",
        "model_path": str(best_model.resolve()),
        "dataset_path": str(dataset_dir.resolve()),
        "data_yaml": str(data_yaml.resolve()),
        "eval_split": "val",
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch_size,
        "base_weights": str(model_path.resolve()),
        "device": device,
        "dataset_stats": dataset_stats,
        "split_manifest": {
            "train_count": split_manifest["train_count"],
            "val_count": split_manifest["val_count"],
            "train_boxes": split_manifest["train_boxes"],
            "val_boxes": split_manifest["val_boxes"],
        },
        "metrics": safe_metrics,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    evaluation_path = output_dir / "evaluation.json"
    with open(evaluation_path, "w", encoding="utf-8") as f:
        json.dump(evaluation, f, indent=2)

    summary_csv = output_dir / "training_summary.csv"
    summary_csv.write_text(
        "model,map50_95,map50,precision,recall,f1\n"
        f"base,{metrics['map50-95']:.6f},{metrics['map50']:.6f},"
        f"{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f}\n",
        encoding="utf-8",
    )

    logger.info(
        "Validation (GT): mAP50-95=%.4f  mAP50=%.4f  F1=%.4f",
        metrics["map50-95"],
        metrics["map50"],
        metrics["f1"],
    )
    logger.info("Best model: %s", best_model)
    logger.info("Evaluation saved: %s", evaluation_path)
    return best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO baseline on burrow ground-truth labels (same split/config as SSL initial).",
    )
    parser.add_argument("--epochs", type=int, default=200, help="Training epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size (default: 512)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--model",
        type=str,
        default=str(REPO_ROOT / "yolo11s.pt"),
        help="Base YOLO weights (default: repo yolo11s.pt)",
    )
    parser.add_argument("--device", default=None, help="CUDA device index or 'cpu' (default: auto)")
    parser.add_argument("--force-dataset", action="store_true", help="Rebuild dataset split even if it exists")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training if base/weights/best.pt exists; re-run evaluation only",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seeds()

    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device if args.device == "cpu" else int(args.device)

    output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "experiment": "base_gt",
        "output_dir": str(output_dir.resolve()),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "model": args.model,
        "device": device,
        "train_labels": "labels_gt",
        "val_labels": "labels_gt",
        "yolo_run_name": "base",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Burrow base GT training")
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)

    dataset_dir = prepare_gt_dataset(output_dir, force=args.force_dataset)

    best_model_path = output_dir / "base" / "weights" / "best.pt"
    if args.skip_training and best_model_path.exists():
        logger.info("Skipping training (--skip-training); using %s", best_model_path)
        ensure_repo_on_path()
        from initial_training import get_val_metrics

        metrics = get_val_metrics(best_model_path, str(dataset_dir / "data.yaml"), imgsz=args.imgsz, device=device, split="val")
        safe_metrics = _json_safe_metrics(metrics)
        with open(dataset_dir / "split_manifest.json", encoding="utf-8") as f:
            split_manifest = json.load(f)
        evaluation = {
            "model": "base",
            "model_path": str(best_model_path.resolve()),
            "dataset_path": str(dataset_dir.resolve()),
            "data_yaml": str((dataset_dir / "data.yaml").resolve()),
            "eval_split": "val",
            "seed": SEED,
            "metrics": safe_metrics,
            "split_manifest": {
                "train_count": split_manifest["train_count"],
                "val_count": split_manifest["val_count"],
                "train_boxes": split_manifest["train_boxes"],
                "val_boxes": split_manifest["val_boxes"],
            },
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        with open(output_dir / "evaluation.json", "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2)
    else:
        run_base_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_path=Path(args.model),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch_size=args.batch,
            device=device,
        )

    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Done. All artifacts under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
