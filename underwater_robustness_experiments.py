#!/usr/bin/env python3
"""
Additional underwater robustness experiments (backscatter, non-uniform illumination).

Run via compare_models.py:
  python compare_models.py --experiment backscatter
  python compare_models.py --experiment illumination
  python compare_models.py --experiment combined
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from ultralytics import YOLO

from compare_models import (
    DATA_YAML_PATH,
    DEVICE,
    IMGSZ,
    RESULTS_DIR,
    get_cached_best_model_path,
    get_iterations_for_analysis,
    load_best_model_info,
    load_iteration_metrics,
)

# (beta, airlight BGR 0-255) — veiling-light backscatter: I = J*t + A*(1-t)
BACKSCATTER_CONFIGS: List[Tuple[float, Tuple[int, int, int]]] = [
    (0.0, (0, 0, 0)),
    (0.20, (140, 165, 115)),
    (0.40, (155, 180, 125)),
    (0.60, (170, 195, 135)),
    (0.85, (185, 210, 145)),
]

# (vignette_strength, gradient_strength) — multiply image by spatial illumination map
ILLUMINATION_CONFIGS: List[Tuple[float, float]] = [
    (0.0, 0.0),
    (0.20, 0.05),
    (0.35, 0.10),
    (0.50, 0.18),
    (0.65, 0.28),
]


def apply_backscatter(image: np.ndarray, beta: float, airlight_bgr: Tuple[int, int, int]) -> np.ndarray:
    """Veiling-light model: I = J*t + A*(1-t) with depth-proxy transmission map."""
    if beta <= 0:
        return image.copy()

    img = image.astype(np.float32) / 255.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    depth = 1.0 - gray
    depth = cv2.GaussianBlur(depth, (31, 31), 0)
    depth = depth / (float(depth.max()) + 1e-6)
    transmission = np.exp(-beta * 3.0 * depth)[..., np.newaxis]
    airlight = np.array(airlight_bgr, dtype=np.float32) / 255.0
    out = img * transmission + airlight * (1.0 - transmission)
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def apply_nonuniform_illumination(
    image: np.ndarray, vignette_strength: float, gradient_strength: float
) -> np.ndarray:
    """Spatial illumination: radial vignette plus vertical light falloff."""
    if vignette_strength <= 0 and gradient_strength <= 0:
        return image.copy()

    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    r_norm = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (np.sqrt(cx ** 2 + cy ** 2) + 1e-6)
    illumination = 1.0 - vignette_strength * np.clip(r_norm ** 2, 0.0, 1.0)
    if gradient_strength > 0:
        illumination *= np.clip(1.0 - gradient_strength * (yy / max(h - 1, 1)), 0.25, 1.0)
    illumination = np.clip(illumination, 0.2, 1.0)[..., np.newaxis]

    img = image.astype(np.float32) / 255.0
    out = img * illumination
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _resolve_label_path(img_path_str_orig: str, filename: str) -> Path | None:
    if "detector_dataset_simple" not in img_path_str_orig:
        return None
    parts = img_path_str_orig.split("detector_dataset_simple", 1)
    if len(parts) != 2:
        return None
    base_part = parts[0].rstrip("\\/")
    if base_part:
        dataset_root = (
            Path(base_part) / "detector_dataset_simple"
            if Path(base_part).is_absolute()
            else Path.cwd() / base_part / "detector_dataset_simple"
        )
    else:
        dataset_root = Path.cwd() / "detector_dataset_simple"
    label_path = dataset_root / "labels" / filename
    if label_path.suffix in {".png", ".jpg", ".jpeg"}:
        label_path = label_path.with_suffix(".txt")
    return label_path


def apply_combined_degradation(image: np.ndarray, level: int) -> np.ndarray:
    """Apply backscatter then non-uniform illumination at the same level index."""
    beta, airlight = BACKSCATTER_CONFIGS[level]
    vig, grad = ILLUMINATION_CONFIGS[level]
    out = apply_backscatter(image, beta, airlight)
    return apply_nonuniform_illumination(out, vig, grad)


class _RobustnessExperimentBase:
    title: str = "Robustness experiment"
    subdir: str = "experiment"

    def __init__(self, conf_threshold: float, iou_threshold: float, force_reeval: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.force_reeval = force_reeval
        self.results: Dict[int, Dict] = {}
        self.experiment_dir = RESULTS_DIR / "experiments" / self.subdir
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
            self.data_config = yaml.safe_load(f)
        self.class_names = {int(k): v for k, v in self.data_config["names"].items()}
        self.num_classes = self.data_config["nc"]

        self.best_model_path = self._find_best_model()
        self.model = YOLO(str(self.best_model_path))
        self.val_images = self._load_validation_images()

    def _find_best_model(self) -> Path:
        print("\nFinding best model...")
        for iteration in get_iterations_for_analysis():
            metrics = load_iteration_metrics(iteration)
            if metrics and isinstance(metrics.get("f1"), (int, float)):
                print(f"  Model {iteration}: F1 = {metrics['f1']:.4f}")
        best_path = get_cached_best_model_path(force=self.force_reeval)
        info = load_best_model_info()
        if info:
            iteration = info.get("iteration", "N/A")
            metrics = info.get("metrics", {})
            f1 = metrics.get("f1")
            if isinstance(f1, (int, float)):
                print(f"  → Best: Iteration {iteration} (F1 = {f1:.4f})")
        return best_path

    def _load_validation_images(self) -> List[str]:
        dataset_path = Path(self.data_config["path"])
        val_file = dataset_path / self.data_config["val"]
        val_images: List[str] = []
        dataset_folder_name = dataset_path.name
        for path_str in val_file.read_text(encoding="utf-8").splitlines():
            path_str = path_str.strip()
            if not path_str:
                continue
            path = Path(path_str)
            if not path.is_absolute():
                if path_str.startswith(dataset_folder_name):
                    abs_path = (dataset_path.parent / path).resolve()
                else:
                    abs_path = (dataset_path / path).resolve()
                if not abs_path.exists():
                    abs_path = (val_file.parent / path).resolve()
                if not abs_path.exists():
                    abs_path = path.resolve()
                path = abs_path
            else:
                path = path.resolve()
            if path.exists():
                val_images.append(str(path))
        print(f"Loaded {len(val_images)} validation images")
        return val_images

    def transform(self, image: np.ndarray, level: int) -> np.ndarray:
        raise NotImplementedError

    def level_count(self) -> int:
        raise NotImplementedError

    def create_dataset(self, level: int) -> Path:
        print(f"\nCreating dataset (level {level})...")
        level_dir = self.experiment_dir / f"level_{level}"
        if level_dir.exists():
            shutil.rmtree(level_dir)
        images_dir = level_dir / "images"
        labels_dir = level_dir / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)

        processed: List[Path] = []
        total = len(self.val_images)
        for idx, img_path_str_orig in enumerate(self.val_images):
            if (idx + 1) % 20 == 0 or idx == 0 or (idx + 1) == total:
                print(f"  Processing: {idx + 1}/{total}...")
            filename = Path(img_path_str_orig).name
            label_path = _resolve_label_path(img_path_str_orig, filename)
            img_path = Path(img_path_str_orig)
            if not img_path.is_absolute():
                img_path = img_path.resolve()
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            degraded = self.transform(img, level)
            output_path = images_dir / filename
            if not cv2.imwrite(str(output_path), degraded):
                continue
            processed.append(output_path)
            if label_path is None:
                label_path = img_path.parent.parent / "labels" / img_path.name
                if label_path.suffix in {".png", ".jpg", ".jpeg"}:
                    label_path = label_path.with_suffix(".txt")
                label_path_str = str(label_path)
                if "detector_dataset" in label_path_str and "detector_dataset_simple" not in label_path_str:
                    label_path = Path(label_path_str.replace("detector_dataset", "detector_dataset_simple"))
            if label_path.exists() and "detector_dataset_simple" in str(label_path):
                shutil.copy(str(label_path), str(labels_dir / label_path.name))

        if not processed:
            raise RuntimeError(f"No images processed for level {level}")

        val_txt = level_dir / "val.txt"
        val_txt.write_text("\n".join(str(p) for p in processed) + "\n", encoding="utf-8")
        data_yaml = level_dir / "data.yaml"
        yaml.safe_dump(
            {
                "path": str(level_dir.absolute()),
                "train": "images",
                "val": str(val_txt.absolute()),
                "test": "images",
                "nc": self.num_classes,
                "names": {i: self.class_names[i] for i in range(self.num_classes)},
            },
            data_yaml.open("w", encoding="utf-8"),
            default_flow_style=False,
        )
        return data_yaml

    def _metrics_path(self, level: int) -> Path:
        return self.experiment_dir / f"level_{level}_results" / "metrics.json"

    def evaluate_level(self, level: int, data_yaml: Path) -> Dict:
        print(f"\nEvaluating level {level} ({self.level_labels(level)})")
        output_dir = self.experiment_dir / f"level_{level}_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        results = self.model.val(
            data=str(data_yaml),
            imgsz=IMGSZ,
            device=DEVICE,
            split="val",
            verbose=False,
            save=False,
            plots=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        metrics = {
            "level": level,
            "map50-95": float(results.box.map),
            "map50": float(results.box.map50),
            "precision": float(getattr(results.box, "mp", float("nan"))),
            "recall": float(getattr(results.box, "mr", float("nan"))),
            "f1": float("nan"),
        }
        if metrics["precision"] == metrics["precision"] and metrics["recall"] == metrics["recall"]:
            if metrics["precision"] + metrics["recall"] > 0:
                metrics["f1"] = (
                    2 * metrics["precision"] * metrics["recall"] / (metrics["precision"] + metrics["recall"])
                )
        self._metrics_path(level).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"  mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}")
        return metrics

    def plot_results(self) -> None:
        levels = sorted(self.results)
        metrics = ["map50-95", "map50", "precision", "recall", "f1"]
        metric_labels = ["mAP50-95", "mAP50", "Precision", "Recall", "F1-Score"]
        data = {"levels": levels}
        for metric in metrics:
            data[metric] = [self.results[l][metric] for l in levels]
        (self.experiment_dir / "degradation_data.json").write_text(
            json.dumps(data, indent=2), encoding="utf-8"
        )

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ["#2E86AB", "#E15759", "#76B041", "#F28E2B", "#9467BD"]
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            ax.plot(levels, data[metric], marker="o", linewidth=2.5, markersize=8, label=label, color=colors[i])
        ax.set_xlabel("Degradation level", fontsize=13, fontweight="bold")
        ax.set_ylabel("Score", fontsize=13, fontweight="bold")
        ax.set_title(self.title, fontsize=15, fontweight="bold")
        ax.set_xticks(levels)
        ax.set_xticklabels([self.level_labels(l) for l in levels], rotation=15, ha="right")
        ax.legend(loc="best", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            np.array([data[m] for m in metrics]),
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            xticklabels=[self.level_labels(l) for l in levels],
            yticklabels=metric_labels,
            cbar_kws={"label": "Score"},
            vmin=0,
            vmax=1.0,
        )
        ax.set_title(f"{self.title} — heatmap", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    def run(self) -> None:
        print("\n" + "=" * 60)
        print(self.title.upper())
        print("=" * 60)
        n_levels = self.level_count()
        for level in range(n_levels):
            if not self.force_reeval and self._metrics_path(level).exists():
                self.results[level] = json.loads(self._metrics_path(level).read_text(encoding="utf-8"))
                print(f"Level {level} loaded: mAP50={self.results[level]['map50']:.4f}")
                continue
            data_yaml = self.create_dataset(level)
            self.results[level] = self.evaluate_level(level, data_yaml)
        self.plot_results()
        print(f"\nResults saved to: {self.experiment_dir}")


class BackscatterExperiment(_RobustnessExperimentBase):
    title = "Model performance under veiling-light backscatter"
    subdir = "backscatter"

    def level_count(self) -> int:
        return len(BACKSCATTER_CONFIGS)

    def level_labels(self, level: int) -> str:
        beta, _ = BACKSCATTER_CONFIGS[level]
        return "Original" if level == 0 else f"L{level} β={beta:.2f}"

    def transform(self, image: np.ndarray, level: int) -> np.ndarray:
        beta, airlight = BACKSCATTER_CONFIGS[level]
        return apply_backscatter(image, beta, airlight)


class IlluminationExperiment(_RobustnessExperimentBase):
    title = "Model performance under non-uniform illumination"
    subdir = "illumination"

    def level_count(self) -> int:
        return len(ILLUMINATION_CONFIGS)

    def level_labels(self, level: int) -> str:
        vig, grad = ILLUMINATION_CONFIGS[level]
        return "Original" if level == 0 else f"L{level} vig={vig:.2f}"

    def transform(self, image: np.ndarray, level: int) -> np.ndarray:
        vig, grad = ILLUMINATION_CONFIGS[level]
        return apply_nonuniform_illumination(image, vig, grad)


class CombinedExperiment(_RobustnessExperimentBase):
    title = "Model performance under combined backscatter and non-uniform illumination"
    subdir = "combined"

    def level_count(self) -> int:
        return len(BACKSCATTER_CONFIGS)

    def level_labels(self, level: int) -> str:
        if level == 0:
            return "Original"
        beta, _ = BACKSCATTER_CONFIGS[level]
        vig, _ = ILLUMINATION_CONFIGS[level]
        return f"L{level} β={beta:.2f}, vig={vig:.2f}"

    def transform(self, image: np.ndarray, level: int) -> np.ndarray:
        return apply_combined_degradation(image, level)
