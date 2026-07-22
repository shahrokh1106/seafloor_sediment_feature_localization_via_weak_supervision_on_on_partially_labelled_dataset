#!/usr/bin/env python3
"""
Burrow weakly-supervised YOLO experiment runner.

Prepares an 80/20 train/val split (fixed seed), trains on pseudo-labels, evaluates
on ground-truth labels, then runs the main-repo SSL pipeline to refine labels.

Usage (from repo root or this folder):
    python burrow_experiment/run_burrow.py labels_dino_global
    python burrow_experiment/run_burrow.py labels_sam_local --iterations 10

Outputs are written under burrow_experiment/out_<label_source>/.
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

VALID_LABEL_SOURCES = (
    "labels_dino_global",
    "labels_dino_local",
    "labels_sam_global",
    "labels_sam_local",
)

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
    """
    Paths for train/val.txt.
    """
    lines: list[str] = []
    for img in images:
        lines.append(str((images_root / img.name)))
    return lines


def _write_split_file(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _json_safe_metrics(metrics: dict) -> dict:
    """Convert get_val_metrics() output to JSON-serializable types."""
    out = dict(metrics)
    per_class = out.get("per_class_ap")
    if per_class is not None:
        out["per_class_ap"] = np.asarray(per_class).tolist()
    names = out.get("names")
    if names is not None:
        out["names"] = {str(k): v for k, v in names.items()}
    return out


def prepare_dataset(
    label_source: str,
    output_dir: Path,
    force: bool = False,
) -> Path:
    """
    Build a YOLO dataset under output_dir/dataset/.

    Train images use pseudo-labels from label_source; val/test images use labels_gt.
    """
    dataset_dir = output_dir / "dataset"
    manifest_path = dataset_dir / "split_manifest.json"
    data_yaml_path = dataset_dir / "data.yaml"

    if not force and manifest_path.exists() and data_yaml_path.exists():
        logger.info("Reusing existing prepared dataset at %s", dataset_dir)
        return dataset_dir

    pseudo_dir = SOURCE_DATASET / label_source
    if not pseudo_dir.is_dir():
        raise FileNotFoundError(f"Pseudo-label folder not found: {pseudo_dir}")
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

    for img in train_images:
        _copy_or_empty_label(pseudo_dir / f"{img.stem}.txt", labels_dir / f"{img.stem}.txt")
    for img in val_images:
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

    manifest = {
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "label_source": label_source,
        "pseudo_labels_dir": str(pseudo_dir.resolve()),
        "gt_labels_dir": str(GT_LABELS_DIR.resolve()),
        "total_images": len(image_files),
        "train_count": len(train_images),
        "val_count": len(val_images),
        "train_images": [p.name for p in train_images],
        "val_images": [p.name for p in val_images],
        "note": "Train labels are pseudo; val/test labels are ground truth (labels_gt).",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(
        "Prepared dataset: %d train (pseudo), %d val (GT) -> %s",
        len(train_images),
        len(val_images),
        dataset_dir,
    )
    return dataset_dir


def run_initial_training(
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
    initial_out = output_dir / "initial"
    initial_out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("INITIAL TRAINING")
    logger.info("=" * 60)

    validate_dataset(str(data_yaml))
    best_model = train_yolo(
        data_yaml_path=str(data_yaml),
        model_path=str(model_path),
        out_dir=str(initial_out),
        imgsz=imgsz,
        epochs=epochs,
        batchsize=batch_size,
        device=device,
        multi_scale=True,
    )

    metrics = get_val_metrics(best_model, str(data_yaml), imgsz=imgsz, device=device, split="val")
    metrics_path = output_dir / "initial_val_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(_json_safe_metrics(metrics), f, indent=2)

    logger.info("Initial validation (GT): mAP50-95=%.4f  F1=%.4f", metrics["map50-95"], metrics["f1"])
    logger.info("Initial best model: %s", best_model)
    return Path(best_model)


BURROW_CONF_TUNE_METRIC = "map50"
BURROW_CONF_TUNE_MIN = 0.05
BURROW_CONF_TUNE_MAX = 0.7
BURROW_CONF_TUNE_STEPS = 20


def run_ssl_pipeline(
    teacher_model: Path,
    dataset_dir: Path,
    output_dir: Path,
    iterations: int,
    imgsz: int,
    batch_size: int,
    device,
) -> None:
    ensure_repo_on_path()
    from ssl_weakly_supervised import WeaklySupervisedPipeline

    ssl_out = output_dir / "ssl"
    ssl_out.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("SSL WEAKLY SUPERVISED PIPELINE")
    logger.info("=" * 60)

    pipeline = WeaklySupervisedPipeline(
        initial_teacher_path=str(teacher_model),
        dataset_path=str(dataset_dir),
        output_dir=str(ssl_out),
        max_iterations=iterations,
        device=device,
        imgsz=imgsz,
        batch_size=batch_size,
        preserve_gt=False,
        conf_tune_metric=BURROW_CONF_TUNE_METRIC,
        training_seed=SEED,
        conf_tune_min=BURROW_CONF_TUNE_MIN,
        conf_tune_max=BURROW_CONF_TUNE_MAX,
        conf_tune_steps=BURROW_CONF_TUNE_STEPS,
    )
    pipeline.run()

    summary = {
        "best_f1": pipeline.best_f1,
        "best_model": str(pipeline.best_model_path),
        "iterations": iterations,
        "preserve_gt": False,
        "preserve_gt_mode": "teacher_only_iter1_then_accumulate",
        "conf_tune_metric": BURROW_CONF_TUNE_METRIC,
        "conf_tune_min": BURROW_CONF_TUNE_MIN,
        "conf_tune_max": BURROW_CONF_TUNE_MAX,
        "conf_tune_steps": BURROW_CONF_TUNE_STEPS,
        "training_seed": SEED,
    }
    with open(output_dir / "ssl_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("SSL complete. Best F1=%.4f  model=%s", pipeline.best_f1, pipeline.best_model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run burrow YOLO initial training + SSL using selected pseudo-labels.",
    )
    parser.add_argument(
        "label_source",
        choices=VALID_LABEL_SOURCES,
        help="Pseudo-label folder under dataset_burrow/ (e.g. labels_dino_global)",
    )
    parser.add_argument("--iterations", type=int, default=10, help="SSL iterations (default: 10)")
    parser.add_argument("--epochs", type=int, default=200, help="Initial training epochs (default: 200)")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size (default: 512)")
    parser.add_argument("--batch", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument(
        "--model",
        type=str,
        default=str(REPO_ROOT / "yolo11s.pt"),
        help="Base YOLO weights (default: repo yolo11s.pt)",
    )
    parser.add_argument("--device", default=None, help="CUDA device index or 'cpu' (default: auto)")
    parser.add_argument("--skip-initial", action="store_true", help="Skip initial training if best.pt exists")
    parser.add_argument("--skip-ssl", action="store_true", help="Skip SSL pipeline")
    parser.add_argument("--force-dataset", action="store_true", help="Rebuild dataset split even if it exists")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    set_seeds()

    if args.device is None:
        device = 0 if torch.cuda.is_available() else "cpu"
    else:
        device = args.device if args.device == "cpu" else int(args.device)

    output_dir = SCRIPT_DIR / f"out_{args.label_source}"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "label_source": args.label_source,
        "output_dir": str(output_dir.resolve()),
        "seed": SEED,
        "train_ratio": TRAIN_RATIO,
        "iterations": args.iterations,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "model": args.model,
        "device": device,
        "ssl_preserve_gt": False,
        "ssl_preserve_gt_mode": "teacher_only_iter1_then_accumulate",
        "ssl_conf_tune_metric": BURROW_CONF_TUNE_METRIC,
        "ssl_conf_tune_min": BURROW_CONF_TUNE_MIN,
        "ssl_conf_tune_max": BURROW_CONF_TUNE_MAX,
        "ssl_conf_tune_steps": BURROW_CONF_TUNE_STEPS,
        "ssl_training_seed": SEED,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Burrow experiment: %s", args.label_source)
    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)

    dataset_dir = prepare_dataset(args.label_source, output_dir, force=args.force_dataset)

    teacher_path = output_dir / "initial" / "full_initial_bce" / "weights" / "best.pt"
    if args.skip_initial and teacher_path.exists():
        logger.info("Skipping initial training (--skip-initial); using %s", teacher_path)
    else:
        teacher_path = run_initial_training(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            model_path=Path(args.model),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch_size=args.batch,
            device=device,
        )

    if not args.skip_ssl:
        run_ssl_pipeline(
            teacher_model=teacher_path,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            iterations=args.iterations,
            batch_size=args.batch,
            imgsz=args.imgsz,
            device=device,
        )
    else:
        logger.info("Skipping SSL (--skip-ssl)")

    run_config["finished_at"] = datetime.now(timezone.utc).isoformat()
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)

    logger.info("Done. All artifacts under %s", output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
