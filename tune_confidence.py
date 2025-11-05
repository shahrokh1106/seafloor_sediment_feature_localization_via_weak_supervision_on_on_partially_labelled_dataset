#!/usr/bin/env python3
"""
Confidence Threshold Tuning for Weakly Supervised Learning

This script finds the optimal confidence threshold for pseudo-label generation
by evaluating different thresholds on the validation set.

Usage:
    python tune_confidence.py --model path/to/model.pt --data path/to/data.yaml
    
Author: Refined from original ts_train_detr.py
"""

import os
import json
import numpy as np
from pathlib import Path
import argparse
from ultralytics import YOLO
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def tune_conf_threshold(
        model_or_path,
        data_yaml,
        conf_values=None,
        metric="f1",  # Default to F1 for pseudo-labeling
        imgsz=960,
        iou=0.5,
        device="auto",
        batch=8,
        workers=0,
        verbose=False,
        out_path=None,
        save_plot=True):
    """
    Find optimal confidence threshold by maximizing a chosen metric on validation set.
    
    Args:
        model_or_path: YOLO model instance or path to model weights
        data_yaml: Path to data.yaml file with validation split
        conf_values: List/array of confidence values to test (default: 0.05 to 0.95 in 19 steps)
        metric: Metric to optimize - "f1", "map50", "map50-95", "precision", "recall"
        imgsz: Image size for validation
        iou: IoU threshold for NMS
        device: Device to use ("auto", "cpu", "cuda", 0, 1, etc.)
        batch: Batch size for validation
        workers: Number of data loading workers
        verbose: Whether to print verbose output
        out_path: Path to save results JSON (if None, uses default)
        save_plot: Whether to save visualization plot
    
    Returns:
        best_conf: Optimal confidence threshold
        results: List of dicts with conf, score, and raw metrics for each tested value
    """
    
    def extract_score(results_obj, which):
        """Extract metric score from YOLO validation results."""
        # Map metric names to YOLO result attributes
        if which == "f1":
            # Calculate F1 from precision and recall
            try:
                p = float(results_obj.box.mp) if hasattr(results_obj.box, 'mp') else None
                r = float(results_obj.box.mr) if hasattr(results_obj.box, 'mr') else None
                if p is not None and r is not None and (p + r) > 0:
                    return 2 * p * r / (p + r)
                return 0.0
            except:
                return 0.0
        
        elif which == "precision":
            try:
                return float(results_obj.box.mp) if hasattr(results_obj.box, 'mp') else 0.0
            except:
                return 0.0
        
        elif which == "recall":
            try:
                return float(results_obj.box.mr) if hasattr(results_obj.box, 'mr') else 0.0
            except:
                return 0.0
        
        elif which == "map50":
            try:
                return float(results_obj.box.map50) if hasattr(results_obj.box, 'map50') else 0.0
            except:
                return 0.0
        
        elif which == "map50-95":
            try:
                return float(results_obj.box.map) if hasattr(results_obj.box, 'map') else 0.0
            except:
                return 0.0
        
        else:
            raise ValueError(f"Unknown metric: {which}")
    
    # Default confidence values to test
    if conf_values is None:
        conf_values = np.round(np.linspace(0.001, 0.7, 20), 3)
    
    logger.info("="*60)
    logger.info("CONFIDENCE THRESHOLD TUNING")
    logger.info("="*60)
    logger.info(f"Model: {model_or_path}")
    logger.info(f"Data: {data_yaml}")
    logger.info(f"Optimizing for: {metric.upper()}")
    logger.info(f"Testing {len(conf_values)} confidence values: {conf_values[0]:.2f} to {conf_values[-1]:.2f}")
    logger.info(f"Image size: {imgsz}")
    logger.info(f"Device: {device}")
    
    # Handle device selection
    if device == "auto":
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"
        logger.info(f"Auto device selection: {device}")
    
    # Load model
    if isinstance(model_or_path, YOLO):
        model = model_or_path
        logger.info("Using provided YOLO model instance")
    else:
        model = YOLO(str(model_or_path))
        logger.info(f"Loaded model from {model_or_path}")
    
    # Test each confidence value
    results = []
    logger.info("\nTesting confidence thresholds...")
    
    for i, c in enumerate(conf_values):
        logger.info(f"  [{i+1}/{len(conf_values)}] Testing conf={c:.2f}...")
        
        try:
            # Run validation with this confidence threshold
            val_results = model.val(
                data=str(data_yaml),
                split="val",
                imgsz=imgsz,
                conf=float(c),
                iou=iou,
                device=device,
                batch=batch,
                workers=workers,
                save=False,
                save_json=False,
                plots=False,
                verbose=verbose
            )
            
            # Extract metrics
            score = extract_score(val_results, metric)
            
            # Collect all metrics for analysis
            raw_metrics = {
                'precision': float(val_results.box.mp) if hasattr(val_results.box, 'mp') else 0.0,
                'recall': float(val_results.box.mr) if hasattr(val_results.box, 'mr') else 0.0,
                'map50': float(val_results.box.map50) if hasattr(val_results.box, 'map50') else 0.0,
                'map50-95': float(val_results.box.map) if hasattr(val_results.box, 'map') else 0.0,
            }
            raw_metrics['f1'] = (2 * raw_metrics['precision'] * raw_metrics['recall'] / 
                                (raw_metrics['precision'] + raw_metrics['recall'] + 1e-6))
            
            results.append({
                "conf": float(c),
                "score": float(score),
                "metrics": raw_metrics
            })
            
            logger.info(f"      Score: {score:.4f} | P: {raw_metrics['precision']:.4f} | "
                       f"R: {raw_metrics['recall']:.4f} | F1: {raw_metrics['f1']:.4f}")
            
        except Exception as e:
            logger.error(f"      Error at conf={c:.2f}: {e}")
            results.append({
                "conf": float(c),
                "score": 0.0,
                "metrics": {'precision': 0.0, 'recall': 0.0, 'map50': 0.0, 'map50-95': 0.0, 'f1': 0.0}
            })
    
    # Find best confidence
    best = max(results, key=lambda x: x["score"])
    best_conf = best["conf"]
    best_score = best["score"]
    best_metrics = best["metrics"]
    
    logger.info("\n" + "="*60)
    logger.info("TUNING RESULTS")
    logger.info("="*60)
    logger.info(f"Best confidence threshold: {best_conf:.2f}")
    logger.info(f"Best {metric.upper()} score: {best_score:.4f}")
    logger.info(f"Metrics at best confidence:")
    logger.info(f"  Precision: {best_metrics['precision']:.4f}")
    logger.info(f"  Recall: {best_metrics['recall']:.4f}")
    logger.info(f"  F1: {best_metrics['f1']:.4f}")
    logger.info(f"  mAP50: {best_metrics['map50']:.4f}")
    logger.info(f"  mAP50-95: {best_metrics['map50-95']:.4f}")
    
    # Save results
    if out_path is None:
        out_path = "confidence_tuning_results.json"
    
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "best_conf": best_conf,
        "best_score": best_score,
        "metric_optimized": metric,
        "best_metrics": best_metrics,
        "all_results": results,
        "config": {
            "model": str(model_or_path),
            "data_yaml": str(data_yaml),
            "imgsz": imgsz,
            "iou": iou,
            "conf_range": [float(conf_values[0]), float(conf_values[-1])],
            "num_tests": len(conf_values)
        }
    }
    
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    logger.info(f"\nResults saved to: {out_path}")
    
    # Create visualization plot (with iteration number if in path)
    if save_plot:
        # Extract iteration number from output path if present
        plot_name = "confidence_tuning_plot.png"
        if "iter" in str(out_path):
            # Extract iteration number from path like "conf_tuning_iter3.json"
            try:
                import re
                match = re.search(r'iter(\d+)', str(out_path))
                if match:
                    iter_num = match.group(1)
                    plot_name = f"confidence_tuning_plot_{iter_num}.png"
            except:
                pass
        
        plot_path = out_path.parent / plot_name
        create_tuning_plot(results, best_conf, metric, plot_path)
        logger.info(f"Plot saved to: {plot_path}")
    
    return best_conf, results


def create_tuning_plot(results, best_conf, metric, save_path):
    """Create visualization of confidence tuning results."""
    
    confs = [r['conf'] for r in results]
    scores = [r['score'] for r in results]
    precisions = [r['metrics']['precision'] for r in results]
    recalls = [r['metrics']['recall'] for r in results]
    f1s = [r['metrics']['f1'] for r in results]
    map50s = [r['metrics']['map50'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Confidence Threshold Tuning (Optimized: {metric.upper()})', fontsize=14, fontweight='bold')
    
    # Plot 1: Optimized metric
    axes[0, 0].plot(confs, scores, 'b-o', linewidth=2, markersize=4)
    axes[0, 0].axvline(best_conf, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_conf:.2f}')
    axes[0, 0].set_xlabel('Confidence Threshold', fontsize=10)
    axes[0, 0].set_ylabel(f'{metric.upper()} Score', fontsize=10)
    axes[0, 0].set_title(f'{metric.upper()} vs Confidence', fontsize=11, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Precision & Recall
    axes[0, 1].plot(confs, precisions, 'g-o', linewidth=2, markersize=4, label='Precision')
    axes[0, 1].plot(confs, recalls, 'm-o', linewidth=2, markersize=4, label='Recall')
    axes[0, 1].axvline(best_conf, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_conf:.2f}')
    axes[0, 1].set_xlabel('Confidence Threshold', fontsize=10)
    axes[0, 1].set_ylabel('Score', fontsize=10)
    axes[0, 1].set_title('Precision & Recall vs Confidence', fontsize=11, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: F1 Score
    axes[1, 0].plot(confs, f1s, 'orange', linewidth=2, marker='o', markersize=4)
    axes[1, 0].axvline(best_conf, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_conf:.2f}')
    axes[1, 0].set_xlabel('Confidence Threshold', fontsize=10)
    axes[1, 0].set_ylabel('F1 Score', fontsize=10)
    axes[1, 0].set_title('F1 Score vs Confidence', fontsize=11, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: mAP50
    axes[1, 1].plot(confs, map50s, 'purple', linewidth=2, marker='o', markersize=4)
    axes[1, 1].axvline(best_conf, color='r', linestyle='--', linewidth=2, 
                       label=f'Best: {best_conf:.2f}')
    axes[1, 1].set_xlabel('Confidence Threshold', fontsize=10)
    axes[1, 1].set_ylabel('mAP50', fontsize=10)
    axes[1, 1].set_title('mAP50 vs Confidence', fontsize=11, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Tune confidence threshold for pseudo-label generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model weights (.pt file)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to data.yaml file')
    parser.add_argument('--metric', type=str, default='f1',
                       choices=['f1', 'precision', 'recall', 'map50', 'map50-95'],
                       help='Metric to optimize')
    parser.add_argument('--min-conf', type=float, default=0.001,
                       help='Minimum confidence threshold to test')
    parser.add_argument('--max-conf', type=float, default=0.7,
                       help='Maximum confidence threshold to test')
    parser.add_argument('--steps', type=int, default=20,
                       help='Number of confidence values to test')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='Image size for validation')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size for validation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--workers', type=int, default=0,
                       help='Number of data loading workers')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS')
    parser.add_argument('--output', type=str, default='confidence_tuning_results.json',
                       help='Path to save results JSON')
    parser.add_argument('--no-plot', action='store_true',
                       help='Do not save visualization plot')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose validation output')
    
    args = parser.parse_args()
    
    # Generate confidence values
    conf_values = np.round(np.linspace(args.min_conf, args.max_conf, args.steps), 3)
    
    # Run tuning
    best_conf, results = tune_conf_threshold(
        model_or_path=args.model,
        data_yaml=args.data,
        conf_values=conf_values,
        metric=args.metric,
        imgsz=args.imgsz,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        workers=args.workers,
        verbose=args.verbose,
        out_path=args.output,
        save_plot=not args.no_plot
    )
    
    logger.info("\n" + "="*60)
    logger.info("TUNING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Use this confidence threshold for pseudo-labeling: {best_conf:.2f}")


if __name__ == "__main__":
    main()
