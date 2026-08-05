#!/usr/bin/env python3
"""
Image-level bootstrap confidence intervals for validation metrics (seafloor dataset).

Runs YOLO.model.val() once, caches per-image predictions and ground truth, then
recomputes metrics on B bootstrap resamples of val images.

Outputs are written to bootstrap_val_results/.

Usage (from repo root):
    python bootstrap_val.py
    python bootstrap_val.py --n-bootstrap 1000 --force-predict
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import DetMetrics, box_iou

SEED = 42
DATA_YAML_PATH = Path("detector_dataset_simple/data.yaml")
BEST_MODEL_PATH = Path("trained_models/best_model/best.pt")
OUTPUT_DIR = Path("bootstrap_val_results")

CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.5
IOUV = torch.linspace(0.5, 0.95, 10)  # mAP@0.5:0.95 thresholds used by Ultralytics
DEVICE = 0
IMGSZ = 960
DEFAULT_N_BOOTSTRAP = 1000
CI_ALPHA = 0.05


@dataclass
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float = 1.0


@dataclass
class ImageRecord:
    stem: str
    image_path: str
    width: int
    height: int
    gt_boxes: List[Box] = field(default_factory=list)
    pred_boxes: List[Box] = field(default_factory=list)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def resolve_path(line: str, dataset_root: Path) -> Path:
    raw = Path(line.strip())
    candidates = [
        Path.cwd() / raw,
        dataset_root.parent / raw,
        dataset_root / "images" / raw.name,
        dataset_root / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"Could not resolve path for split entry: {line.strip()}")


def tensor_boxes_to_list(
    bboxes: torch.Tensor, cls: torch.Tensor, conf: Optional[torch.Tensor] = None
) -> List[Box]:
    boxes: List[Box] = []
    for i in range(int(cls.shape[0])):
        x1, y1, x2, y2 = bboxes[i].tolist()
        conf_val = float(conf[i].item()) if conf is not None else 1.0
        boxes.append(
            Box(
                class_id=int(cls[i].item()),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
                conf=conf_val,
            )
        )
    return boxes


def _make_caching_validator(image_cache: Dict[str, Dict]):
    """Factory: validator class compatible with YOLO.model.val(validator=...)."""

    class CachingDetectionValidator(DetectionValidator):
        def update_metrics(self, preds, batch) -> None:
            for si, pred in enumerate(preds):
                pbatch = self._prepare_batch(si, batch)
                predn = self._prepare_pred(pred)
                stem = Path(pbatch["im_file"]).stem
                gt_boxes = tensor_boxes_to_list(
                    pbatch["bboxes"].cpu(), pbatch["cls"].cpu()
                )
                if predn["cls"].shape[0]:
                    pred_boxes = tensor_boxes_to_list(
                        predn["bboxes"].cpu(),
                        predn["cls"].cpu(),
                        predn["conf"].cpu(),
                    )
                else:
                    pred_boxes = []
                image_cache[stem] = {
                    "im_file": pbatch["im_file"],
                    "pred_boxes": pred_boxes,
                    "gt_boxes": gt_boxes,
                }
            super().update_metrics(preds, batch)

    return CachingDetectionValidator


def extract_point_from_detmetrics(
    det_metrics: DetMetrics, num_classes: int, class_names: Dict[int, str]
) -> Dict:
    """Extract metrics from YOLO.model.val() return value."""
    box = det_metrics.box
    mp = float(getattr(box, "mp", float("nan")))
    mr = float(getattr(box, "mr", float("nan")))
    map50 = float(box.map50)
    map50_95 = float(box.map)
    overall_f1 = float(2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else float("nan")

    ap50 = np.full(num_classes, np.nan)
    ap50_95 = np.full(num_classes, np.nan)
    precision = np.full(num_classes, np.nan)
    recall = np.full(num_classes, np.nan)
    f1 = np.full(num_classes, np.nan)
    instances = np.zeros(num_classes, dtype=int)

    if det_metrics.nt_per_class is not None:
        n = min(num_classes, len(det_metrics.nt_per_class))
        instances[:n] = det_metrics.nt_per_class[:n].astype(int)

    for i, cls_id in enumerate(det_metrics.ap_class_index):
        p, r, ap50_val, ap_val = det_metrics.class_result(i)
        cls_id = int(cls_id)
        if cls_id >= num_classes:
            continue
        ap50[cls_id] = float(ap50_val)
        ap50_95[cls_id] = float(ap_val)
        precision[cls_id] = float(p)
        recall[cls_id] = float(r)
        f1[cls_id] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    return {
        "per_class": {
            "ap50": ap50,
            "ap50_95": ap50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "instances": instances,
        },
        "overall": {
            "map50": map50,
            "map50_95": map50_95,
            "precision": mp,
            "recall": mr,
            "f1": overall_f1,
        },
    }


def boxes_to_pred_tensors(boxes: List[Box]) -> Dict[str, torch.Tensor]:
    if not boxes:
        z4 = torch.zeros((0, 4), dtype=torch.float32)
        z1 = torch.zeros(0, dtype=torch.float32)
        return {"bboxes": z4, "cls": z1, "conf": z1}
    return {
        "bboxes": torch.tensor(
            [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32
        ),
        "cls": torch.tensor([b.class_id for b in boxes], dtype=torch.float32),
        "conf": torch.tensor([b.conf for b in boxes], dtype=torch.float32),
    }


def boxes_to_gt_tensors(boxes: List[Box]) -> Dict[str, torch.Tensor]:
    if not boxes:
        return {"bboxes": torch.zeros((0, 4), dtype=torch.float32), "cls": torch.zeros(0, dtype=torch.float32)}
    return {
        "bboxes": torch.tensor(
            [[b.x1, b.y1, b.x2, b.y2] for b in boxes], dtype=torch.float32
        ),
        "cls": torch.tensor([b.class_id for b in boxes], dtype=torch.float32),
    }


def match_predictions(
    pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor
) -> torch.Tensor:
    """Same matching logic as ultralytics.engine.validator.BaseValidator."""
    correct = np.zeros((pred_classes.shape[0], IOUV.shape[0]), dtype=bool)
    correct_class = true_classes[:, None] == pred_classes
    iou_np = (iou * correct_class).cpu().numpy()
    for i, threshold in enumerate(IOUV.cpu().tolist()):
        matches = np.nonzero(iou_np >= threshold)
        if matches[0].shape[0]:
            matches_arr = np.array(matches).T
            if matches_arr.shape[0] > 1:
                matches_arr = matches_arr[iou_np[matches_arr[:, 0], matches_arr[:, 1]].argsort()[::-1]]
                matches_arr = matches_arr[np.unique(matches_arr[:, 1], return_index=True)[1]]
                matches_arr = matches_arr[iou_np[matches_arr[:, 0], matches_arr[:, 1]].argsort()[::-1]]
                matches_arr = matches_arr[np.unique(matches_arr[:, 0], return_index=True)[1]]
            correct[matches_arr[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


def process_image_batch(
    preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
) -> Dict[str, np.ndarray]:
    """Same logic as ultralytics DetectionValidator._process_batch."""
    if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
        return {"tp": np.zeros((preds["cls"].shape[0], IOUV.shape[0]), dtype=bool)}
    iou = box_iou(batch["bboxes"], preds["bboxes"])
    return {"tp": match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}


def evaluate_with_detmetrics(
    records: List[ImageRecord],
    sampled_indices: Sequence[int],
    class_names: Dict[int, str],
    num_classes: int,
) -> Dict:
    """Evaluate a bootstrap sample using Ultralytics DetMetrics (no re-inference)."""
    metrics = DetMetrics(names=class_names)

    for idx in sampled_indices:
        rec = records[idx]
        preds = boxes_to_pred_tensors(rec.pred_boxes)
        batch = boxes_to_gt_tensors(rec.gt_boxes)
        tp_dict = process_image_batch(preds, batch)
        cls = batch["cls"].cpu().numpy()
        no_pred = preds["cls"].shape[0] == 0
        metrics.update_stats(
            {
                **tp_dict,
                "target_cls": cls,
                "target_img": np.unique(cls),
                "conf": np.zeros(0, dtype=np.float32) if no_pred else preds["conf"].cpu().numpy(),
                "pred_cls": np.zeros(0, dtype=np.float32) if no_pred else preds["cls"].cpu().numpy(),
                "im_name": rec.stem,
            }
        )

    if not metrics.stats["tp"]:
        nan = float("nan")
        empty = np.full(num_classes, np.nan)
        zeros = np.zeros(num_classes, dtype=int)
        return {
            "per_class": {
                "ap50": empty.copy(),
                "ap50_95": empty.copy(),
                "precision": empty.copy(),
                "recall": empty.copy(),
                "f1": empty.copy(),
                "instances": zeros,
            },
            "overall": {
                "map50": nan,
                "map50_95": nan,
                "precision": nan,
                "recall": nan,
                "f1": nan,
            },
        }

    metrics.process(plot=False)

    ap50 = np.full(num_classes, np.nan)
    ap50_95 = np.full(num_classes, np.nan)
    precision = np.full(num_classes, np.nan)
    recall = np.full(num_classes, np.nan)
    f1 = np.full(num_classes, np.nan)
    instances = np.zeros(num_classes, dtype=int)

    if metrics.nt_per_class is not None:
        n = min(num_classes, len(metrics.nt_per_class))
        instances[:n] = metrics.nt_per_class[:n].astype(int)

    for i, cls_id in enumerate(metrics.ap_class_index):
        p, r, ap50_val, ap_val = metrics.class_result(i)
        cls_id = int(cls_id)
        if cls_id >= num_classes:
            continue
        ap50[cls_id] = float(ap50_val)
        ap50_95[cls_id] = float(ap_val)
        precision[cls_id] = float(p)
        recall[cls_id] = float(r)
        f1[cls_id] = float(2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    mp, mr = float(metrics.box.mp), float(metrics.box.mr)
    overall_f1 = float(2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else float("nan")

    return {
        "per_class": {
            "ap50": ap50,
            "ap50_95": ap50_95,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "instances": instances,
        },
        "overall": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": mp,
            "recall": mr,
            "f1": overall_f1,
        },
    }


def evaluate_indices(
    records: List[ImageRecord],
    sampled_indices: Sequence[int],
    class_names: Dict[int, str],
    num_classes: int,
) -> Dict:
    return evaluate_with_detmetrics(records, sampled_indices, class_names, num_classes)


def box_to_dict(box: Box) -> Dict:
    return {
        "class_id": box.class_id,
        "xyxy": [box.x1, box.y1, box.x2, box.y2],
        "conf": box.conf,
    }


def box_from_dict(d: Dict) -> Box:
    x1, y1, x2, y2 = d["xyxy"]
    return Box(
        class_id=int(d["class_id"]),
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        conf=float(d.get("conf", 1.0)),
    )


def load_val_records(data_config: Dict) -> Tuple[List[ImageRecord], Dict[int, str]]:
    dataset_root = Path(data_config["path"])
    class_names = {int(k): v for k, v in data_config["names"].items()}

    split_file = dataset_root / data_config["val"]
    records: List[ImageRecord] = []

    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        image_path = resolve_path(line, dataset_root)
        records.append(
            ImageRecord(
                stem=image_path.stem,
                image_path=str(image_path),
                width=0,
                height=0,
            )
        )

    return records, class_names


def apply_cache_to_records(
    records: List[ImageRecord], pred_cache: Path, gt_cache: Path
) -> None:
    pred_data = json.loads(pred_cache.read_text(encoding="utf-8"))
    gt_data = json.loads(gt_cache.read_text(encoding="utf-8"))
    for rec in records:
        pred_entry = pred_data["images"][rec.stem]
        gt_entry = gt_data["images"][rec.stem]
        rec.pred_boxes = [box_from_dict(b) for b in pred_entry["predictions"]]
        rec.gt_boxes = [box_from_dict(b) for b in gt_entry["ground_truth"]]
        rec.image_path = pred_entry.get("image_path", rec.image_path)


def point_to_jsonable(point: Dict) -> Dict:
    pc = point["per_class"]
    return {
        "overall": point["overall"],
        "per_class": {
            key: val.tolist() if hasattr(val, "tolist") else val
            for key, val in pc.items()
        },
    }


def point_from_jsonable(data: Dict) -> Dict:
    pc = data["per_class"]
    return {
        "overall": data["overall"],
        "per_class": {
            key: np.array(val) if isinstance(val, list) else val
            for key, val in pc.items()
        },
    }


def save_prediction_caches(
    records: List[ImageRecord],
    pred_cache: Path,
    gt_cache: Path,
    meta: Dict,
    point: Dict,
) -> None:
    pred_payload = {
        **meta,
        "coordinate_space": "letterbox_imgsz",
        "point": point_to_jsonable(point),
        "images": {
            rec.stem: {
                "image_path": rec.image_path,
                "predictions": [box_to_dict(b) for b in rec.pred_boxes],
            }
            for rec in records
        },
    }
    gt_payload = {
        **meta,
        "coordinate_space": "letterbox_imgsz",
        "images": {
            rec.stem: {
                "image_path": rec.image_path,
                "ground_truth": [box_to_dict(b) for b in rec.gt_boxes],
            }
            for rec in records
        },
    }
    pred_cache.write_text(json.dumps(pred_payload, indent=2), encoding="utf-8")
    gt_cache.write_text(json.dumps(gt_payload, indent=2), encoding="utf-8")


def load_cached_point(pred_cache: Path) -> Optional[Dict]:
    pred_data = json.loads(pred_cache.read_text(encoding="utf-8"))
    if "point" in pred_data:
        return point_from_jsonable(pred_data["point"])
    return None


def run_val_and_cache(
    records: List[ImageRecord],
    model_path: Path,
    num_classes: int,
    class_names: Dict[int, str],
) -> Dict:
    """Run YOLO.model.val(), cache letterbox-space boxes, return point from results.box."""
    image_cache: Dict[str, Dict] = {}

    print("Running YOLO.model.val() and caching predictions...")
    model = YOLO(str(model_path))
    det_metrics = model.val(
        validator=_make_caching_validator(image_cache),
        data=str(DATA_YAML_PATH),
        split="val",
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        imgsz=IMGSZ,
        device=DEVICE,
        verbose=True,
        save=False,
        plots=False,
    )

    for rec in records:
        cached = image_cache.get(rec.stem)
        if cached is None:
            raise KeyError(f"Validation run did not produce cache for image: {rec.stem}")
        rec.pred_boxes = cached["pred_boxes"]
        rec.gt_boxes = cached["gt_boxes"]
        rec.image_path = cached["im_file"]

    return extract_point_from_detmetrics(det_metrics, num_classes, class_names)


def summarize_bootstrap(
    metric_name: str,
    values: np.ndarray,
    class_names: Dict[int, str],
    instances_full: np.ndarray,
) -> List[Dict]:
    rows = []
    n_classes = values.shape[1]
    for cls_id in range(n_classes):
        cls_vals = values[:, cls_id]
        valid = cls_vals[~np.isnan(cls_vals)]
        n_valid = len(valid)
        row = {
            "class_id": cls_id,
            "class_name": class_names[cls_id],
            "instances_val": int(instances_full[cls_id]),
            "metric": metric_name,
            "n_bootstrap_valid": n_valid,
            "bootstrap_fraction_valid": n_valid / values.shape[0] if values.shape[0] else 0.0,
        }
        if n_valid == 0:
            row.update(
                {
                    "point": float("nan"),
                    "median": float("nan"),
                    "mean": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "std": float("nan"),
                }
            )
        else:
            row.update(
                {
                    "median": float(np.median(valid)),
                    "mean": float(np.mean(valid)),
                    "ci_low": float(np.percentile(valid, 100 * CI_ALPHA / 2)),
                    "ci_high": float(np.percentile(valid, 100 * (1 - CI_ALPHA / 2))),
                    "std": float(np.std(valid, ddof=1)) if n_valid > 1 else 0.0,
                }
            )
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fast bootstrap CIs for seafloor validation metrics"
    )
    parser.add_argument("--n-bootstrap", type=int, default=DEFAULT_N_BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--force-predict",
        action="store_true",
        help="Re-run model.val() and refresh cached predictions",
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not DATA_YAML_PATH.exists():
        raise FileNotFoundError(f"Dataset config not found: {DATA_YAML_PATH}")
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")

    with open(DATA_YAML_PATH, "r", encoding="utf-8") as f:
        data_config = yaml.safe_load(f)

    records, class_names = load_val_records(data_config)
    num_classes = int(data_config.get("nc", len(class_names)))
    n_images = len(records)

    pred_cache = args.output_dir / "val_predictions.json"
    gt_cache = args.output_dir / "val_ground_truth.json"
    cache_meta = {
        "model_path": str(BEST_MODEL_PATH),
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": IMGSZ,
    }

    print(f"Validation images: {n_images}")
    print(f"Model: {BEST_MODEL_PATH}")
    print(f"conf={CONF_THRESHOLD}, iou={IOU_THRESHOLD}, imgsz={IMGSZ}")

    if pred_cache.exists() and gt_cache.exists() and not args.force_predict:
        print("Loading cached val predictions from prior run...")
        apply_cache_to_records(records, pred_cache, gt_cache)
        point = load_cached_point(pred_cache)
        if point is None:
            print("Cached point missing; re-running YOLO.model.val()...")
            point = run_val_and_cache(records, BEST_MODEL_PATH, num_classes, class_names)
            save_prediction_caches(records, pred_cache, gt_cache, cache_meta, point)
    else:
        point = run_val_and_cache(records, BEST_MODEL_PATH, num_classes, class_names)
        save_prediction_caches(records, pred_cache, gt_cache, cache_meta, point)

    pc = point["per_class"]
    ov = point["overall"]

    point_payload = {
        "split": "val",
        "n_images": n_images,
        "evaluation_settings": {
            "conf_threshold": CONF_THRESHOLD,
            "iou_threshold": IOU_THRESHOLD,
            "imgsz": IMGSZ,
            "method": "YOLO.model.val() point estimate + image-level bootstrap",
        },
        "overall": ov,
        "per_class": {},
    }
    for cls_id in range(num_classes):
        point_payload["per_class"][str(cls_id)] = {
            "name": class_names[cls_id],
            "instances": int(pc["instances"][cls_id]),
            "ap50": float(pc["ap50"][cls_id]) if not np.isnan(pc["ap50"][cls_id]) else None,
            "ap50_95": float(pc["ap50_95"][cls_id])
            if not np.isnan(pc["ap50_95"][cls_id])
            else None,
            "precision": float(pc["precision"][cls_id])
            if not np.isnan(pc["precision"][cls_id])
            else None,
            "recall": float(pc["recall"][cls_id])
            if not np.isnan(pc["recall"][cls_id])
            else None,
            "f1": float(pc["f1"][cls_id]) if not np.isnan(pc["f1"][cls_id]) else None,
        }

    (args.output_dir / "point_estimate.json").write_text(
        json.dumps(point_payload, indent=2), encoding="utf-8"
    )

    # Bootstrap resampling
    rng = np.random.default_rng(args.seed)
    metric_names = ["ap50", "ap50_95", "precision", "recall", "f1"]
    bootstrap_arrays = {
        name: np.full((args.n_bootstrap, num_classes), np.nan) for name in metric_names
    }
    bootstrap_overall = {
        "map50": np.full(args.n_bootstrap, np.nan),
        "map50_95": np.full(args.n_bootstrap, np.nan),
        "precision": np.full(args.n_bootstrap, np.nan),
        "recall": np.full(args.n_bootstrap, np.nan),
        "f1": np.full(args.n_bootstrap, np.nan),
    }

    for b in tqdm(range(args.n_bootstrap), desc="Bootstrap resamples"):
        sampled = rng.integers(0, n_images, size=n_images, endpoint=False)
        metrics = evaluate_indices(records, sampled.tolist(), class_names, num_classes)
        m_pc = metrics["per_class"]
        m_ov = metrics["overall"]
        for name in metric_names:
            bootstrap_arrays[name][b, :] = m_pc[name]
        bootstrap_overall["map50"][b] = m_ov["map50"]
        bootstrap_overall["map50_95"][b] = m_ov["map50_95"]
        bootstrap_overall["precision"][b] = m_ov["precision"]
        bootstrap_overall["recall"][b] = m_ov["recall"]
        bootstrap_overall["f1"][b] = m_ov["f1"]

    np.savez_compressed(
        args.output_dir / "bootstrap_distributions.npz",
        **bootstrap_arrays,
        map50=bootstrap_overall["map50"],
        map50_95=bootstrap_overall["map50_95"],
    )

    # Summary tables
    summary_rows: List[Dict] = []
    for metric_name in metric_names:
        for row in summarize_bootstrap(
            metric_name,
            bootstrap_arrays[metric_name],
            class_names,
            point["per_class"]["instances"],
        ):
            cls_id = row["class_id"]
            point_val = pc[metric_name][cls_id]
            row["point"] = float(point_val) if not np.isnan(point_val) else float("nan")
            summary_rows.append(row)

    summary_json = {
        "n_bootstrap": args.n_bootstrap,
        "ci_alpha": CI_ALPHA,
        "ci_percent": int((1 - CI_ALPHA) * 100),
        "seed": args.seed,
        "per_class": summary_rows,
        "overall": {},
    }

    for metric in ["map50", "map50_95", "precision", "recall", "f1"]:
        vals = bootstrap_overall[metric]
        valid = vals[~np.isnan(vals)]
        point_val = ov[metric]
        summary_json["overall"][metric] = {
            "point": float(point_val) if not np.isnan(point_val) else float("nan"),
            "median": float(np.median(valid)) if len(valid) else float("nan"),
            "ci_low": float(np.percentile(valid, 100 * CI_ALPHA / 2)) if len(valid) else float("nan"),
            "ci_high": float(np.percentile(valid, 100 * (1 - CI_ALPHA / 2)))
            if len(valid)
            else float("nan"),
        }

    (args.output_dir / "bootstrap_summary.json").write_text(
        json.dumps(summary_json, indent=2), encoding="utf-8"
    )

    csv_path = args.output_dir / "bootstrap_per_class.csv"
    fieldnames = [
        "class_id",
        "class_name",
        "instances_val",
        "metric",
        "point",
        "median",
        "mean",
        "ci_low",
        "ci_high",
        "std",
        "n_bootstrap_valid",
        "bootstrap_fraction_valid",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    config = {
        "data_yaml": str(DATA_YAML_PATH),
        "model_path": str(BEST_MODEL_PATH),
        "output_dir": str(args.output_dir),
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "imgsz": IMGSZ,
        "ci_alpha": CI_ALPHA,
        "method": "YOLO.model.val() point estimate + image-level bootstrap",
        "note": (
            "Point estimate from YOLO.model.val(); bootstrap resamples recompute "
            "metrics on cached letterbox-space boxes."
        ),
        "files": {
            "val_predictions": str(pred_cache.name),
            "val_ground_truth": str(gt_cache.name),
            "point_estimate": "point_estimate.json",
            "bootstrap_summary": "bootstrap_summary.json",
            "bootstrap_distributions": "bootstrap_distributions.npz",
            "bootstrap_per_class": str(csv_path.name),
        },
    }
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\nDone.")
    print(f"Results: {args.output_dir.resolve()}")
    print(f"  point_estimate.json")
    print(f"  bootstrap_summary.json")
    print(f"  bootstrap_per_class.csv")
    print(f"  bootstrap_distributions.npz")
    print(f"  val_predictions.json / val_ground_truth.json")
    print(
        f"\nOverall mAP@50={ov['map50']:.4f}, mAP@50:95={ov['map50_95']:.4f}, "
        f"F1={ov['f1']:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
