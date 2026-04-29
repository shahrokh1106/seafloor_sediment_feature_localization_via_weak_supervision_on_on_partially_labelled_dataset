#!/usr/bin/env python3
"""
Test Set Evaluation Script

Evaluates the best model from trained_models on the test set from detector_dataset_simple.
Computes comprehensive metrics and saves them to a JSON file.
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO
import os
import torch
import random
import yaml
import json
from typing import Dict, Optional

# Set seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

# Configuration
DATA_YAML_PATH = "detector_dataset_simple/data.yaml"
TRAINED_MODELS_DIR = Path("trained_models")
BEST_MODEL_DIR = TRAINED_MODELS_DIR / "best_model"
BEST_MODEL_PATH = BEST_MODEL_DIR / "best.pt"
BEST_INFO_PATH = BEST_MODEL_DIR / "best_info.json"
OUTPUT_JSON = Path("test_evaluation_results.json")
DEVICE = 0
IMGSZ = 960

# Default thresholds (can be overridden)
CONF_THRESHOLD = 0.001
IOU_THRESHOLD = 0.7


def load_best_model_info() -> Optional[Dict]:
    """Load best model information."""
    if not BEST_INFO_PATH.exists():
        return None
    
    with open(BEST_INFO_PATH, 'r') as f:
        return json.load(f)


def extract_metrics(results, class_names: Dict) -> Dict:
    """
    Extract comprehensive metrics from validation results.
    
    Args:
        results: Ultralytics validation results object
        class_names: Dictionary mapping class IDs to names
    
    Returns:
        Dictionary containing all metrics
    """
    # Overall metrics
    metrics = {
        'map50-95': float(results.box.map),
        'map50': float(results.box.map50),
        'map75': float(getattr(results.box, 'map75', float('nan'))),
        'precision': float(getattr(results.box, 'mp', float('nan'))),
        'recall': float(getattr(results.box, 'mr', float('nan'))),
        'f1': float('nan'),
    }
    
    # Calculate F1 score
    if metrics['precision'] == metrics['precision'] and metrics['recall'] == metrics['recall']:
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    
    # Per-class metrics
    per_class_metrics = {}
    
    # Get number of classes
    num_classes = getattr(results.box, 'nc', len(class_names) if isinstance(class_names, dict) else 10)
    
    # Helper function to convert numpy array to list
    def to_list(arr):
        if arr is None:
            return []
        if hasattr(arr, 'tolist'):
            return arr.tolist()
        elif hasattr(arr, '__iter__') and not isinstance(arr, str):
            return list(arr)
        return []
    
    # AP50-95 per class (maps - Average Precision at IoU 0.5-0.95)
    if hasattr(results.box, 'maps') and results.box.maps is not None:
        maps_per_class = to_list(results.box.maps)
        for cls_idx in range(len(maps_per_class)):
            # Get class name - try both string and int keys
            class_name = class_names.get(str(cls_idx), class_names.get(cls_idx, f"Class_{cls_idx}"))
            per_class_metrics[cls_idx] = {
                'name': class_name,
                'ap50-95': float(maps_per_class[cls_idx]),  # AP50-95 (not mAP)
            }
    
    # AP50 per class (ap50 - Average Precision at IoU=0.5)
    if hasattr(results.box, 'ap50') and results.box.ap50 is not None:
        ap50_per_class = to_list(results.box.ap50)
        for cls_idx in range(len(ap50_per_class)):
            if cls_idx not in per_class_metrics:
                class_name = class_names.get(str(cls_idx), class_names.get(cls_idx, f"Class_{cls_idx}"))
                per_class_metrics[cls_idx] = {'name': class_name}
            per_class_metrics[cls_idx]['ap50'] = float(ap50_per_class[cls_idx])  # AP50 (not mAP50)
    
    # Precision per class
    if hasattr(results.box, 'p') and results.box.p is not None:
        precision_per_class = to_list(results.box.p)
        for cls_idx in range(len(precision_per_class)):
            if cls_idx not in per_class_metrics:
                class_name = class_names.get(str(cls_idx), class_names.get(cls_idx, f"Class_{cls_idx}"))
                per_class_metrics[cls_idx] = {'name': class_name}
            per_class_metrics[cls_idx]['precision'] = float(precision_per_class[cls_idx])
    
    # Recall per class
    if hasattr(results.box, 'r') and results.box.r is not None:
        recall_per_class = to_list(results.box.r)
        for cls_idx in range(len(recall_per_class)):
            if cls_idx not in per_class_metrics:
                class_name = class_names.get(str(cls_idx), class_names.get(cls_idx, f"Class_{cls_idx}"))
                per_class_metrics[cls_idx] = {'name': class_name}
            per_class_metrics[cls_idx]['recall'] = float(recall_per_class[cls_idx])
    
    # Calculate F1 per class
    for cls_idx in per_class_metrics:
        prec = per_class_metrics[cls_idx].get('precision', 0.0)
        rec = per_class_metrics[cls_idx].get('recall', 0.0)
        if prec + rec > 0:
            per_class_metrics[cls_idx]['f1'] = 2 * prec * rec / (prec + rec)
        else:
            per_class_metrics[cls_idx]['f1'] = 0.0
    
    metrics['per_class'] = per_class_metrics
    
    return metrics


def main():
    print("=" * 70)
    print("Test Set Evaluation")
    print("=" * 70)
    
    # Check if best model exists
    if not BEST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Best model not found: {BEST_MODEL_PATH}")
    
    # Load best model info
    best_info = load_best_model_info()
    if best_info:
        iteration = best_info.get('iteration', 'Unknown')
        print(f"Best model: Iteration {iteration}")
        if 'metrics' in best_info:
            val_f1 = best_info['metrics'].get('f1', 'N/A')
            print(f"Validation F1: {val_f1}")
    else:
        print("Best model info not found, using best.pt")
    
    # Load data config
    if not Path(DATA_YAML_PATH).exists():
        raise FileNotFoundError(f"Data YAML not found: {DATA_YAML_PATH}")
    
    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    class_names = data_config.get('names', {})
    num_classes = data_config.get('nc', len(class_names))
    
    print(f"Dataset: {data_config.get('path', 'Unknown')}")
    print(f"Number of classes: {num_classes}")
    print(f"Test set: {data_config.get('test', 'Unknown')}")
    print()
    
    # Load model
    print(f"Loading model: {BEST_MODEL_PATH}")
    model = YOLO(str(BEST_MODEL_PATH))
    print("Model loaded successfully\n")
    
    # Run evaluation on test set
    print("Running evaluation on test set...")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print()
    
    results = model.val(
        data=str(DATA_YAML_PATH),
        split='test',  # Use test split
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=True,
        save=False,
        plots=False,
        save_json=False,
    )
    
    print("\nExtracting metrics...")
    
    # Extract all metrics
    metrics = extract_metrics(results, class_names)
    
    # Add metadata
    evaluation_results = {
        'model_path': str(BEST_MODEL_PATH),
        'model_info': best_info if best_info else None,
        'dataset': {
            'path': data_config.get('path', 'Unknown'),
            'test_split': data_config.get('test', 'Unknown'),
            'num_classes': num_classes,
            'class_names': class_names,
        },
        'evaluation_settings': {
            'image_size': IMGSZ,
            'confidence_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'device': DEVICE,
        },
        'metrics': metrics,
    }
    
    # Save results to JSON
    print(f"\nSaving results to: {OUTPUT_JSON}")
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 70)
    print("Test Set Evaluation Results")
    print("=" * 70)
    print(f"mAP50-95: {metrics['map50-95']:.4f}")
    print(f"mAP50:    {metrics['map50']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1 Score:  {metrics['f1']:.4f}")
    print(f"\nPer-class metrics:")
    print("-" * 70)
    print(f"{'Class':<20} {'AP50-95':<12} {'AP50':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 70)
    
    for cls_idx in sorted(metrics['per_class'].keys()):
        cls_metrics = metrics['per_class'][cls_idx]
        name = cls_metrics.get('name', f'Class_{cls_idx}')
        ap50_95 = cls_metrics.get('ap50-95', 0.0)  # AP50-95 (Average Precision, not mean)
        ap50 = cls_metrics.get('ap50', 0.0)  # AP50 (Average Precision, not mean)
        precision = cls_metrics.get('precision', 0.0)
        recall = cls_metrics.get('recall', 0.0)
        f1 = cls_metrics.get('f1', 0.0)
        print(f"{name:<20} {ap50_95:<12.4f} {ap50:<12.4f} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("=" * 70)
    print(f"\nFull results saved to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()

