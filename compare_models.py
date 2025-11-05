#!/usr/bin/env python3
"""
Model Comparison Tool

Compares multiple trained models (initial + SSL iterations) on validation set.
Evaluates each model and generates comparison plots and metrics.
"""

from pathlib import Path
import numpy as np
from ultralytics import YOLO
import os 
import torch
import random 
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from typing import Dict, List
from math import pi
import cv2
import shutil

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
RESULTS_DIR = TRAINED_MODELS_DIR / "results"
DEVICE = 0
IMGSZ = 960

# Model folders to compare
MODEL_FOLDERS = {
    0: "0",  # Initial training
    1: "1",  # SSL iteration 1
    2: "2",  # SSL iteration 2
    3: "3",  # SSL iteration 3
}

# Gaussian blur configurations for robustness experiment
# (kernel_size, sigma) - increasing blur levels
BLUR_CONFIGS = [
    (0, 0),      # Version 0: Original (no blur)
    (5, 1.0),    # Version 1: Light blur
    (11, 2.0),   # Version 2: Moderate blur
    (21, 4.0),   # Version 3: Heavy blur
    (31, 6.0),   # Version 4: Very heavy blur
]

# Underwater effect configurations for robustness experiment
# (blue_intensity, green_intensity, haze_intensity, darkness) - increasing underwater effect
UNDERWATER_CONFIGS = [
    (0.0, 0.0, 0.0, 0.0),      # Level 0: Original (no effect)
    (0.15, 0.10, 0.15, 0.10),  # Level 1: Shallow water (light effect)
    (0.30, 0.20, 0.30, 0.20),  # Level 2: Medium depth (moderate effect)
    (0.50, 0.35, 0.50, 0.35),  # Level 3: Deep water (strong effect)
    (0.70, 0.50, 0.70, 0.50),  # Level 4: Very deep/murky (intense effect)
]

class ModelComparator:
    def __init__(self, conf_threshold: float, iou_threshold: float, force_reeval: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.force_reeval = force_reeval
        self.all_results = {}
        
        # Create results directory
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(DATA_YAML_PATH, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        # Convert string keys to integer keys
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.num_classes = self.data_config['nc']
        
        print(f"Model Comparison Tool")
        print(f"Conf threshold: {conf_threshold}")
        print(f"IoU threshold: {iou_threshold}")
        print(f"Dataset: {DATA_YAML_PATH}")
        print(f"Classes: {self.num_classes}")
        print(f"Results directory: {RESULTS_DIR}")
        print(f"Force re-evaluation: {force_reeval}")
        
    def find_model_path(self, iteration: int) -> Path:
        """Find the best.pt model for a given iteration"""
        model_folder = TRAINED_MODELS_DIR / MODEL_FOLDERS[iteration]
        
        # Try common paths
        possible_paths = [
            model_folder / "weights" / "best.pt",
            model_folder / "best.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find best.pt in {model_folder}")
    
    def check_results_exist(self, iteration: int) -> bool:
        """Check if results already exist for a given iteration"""
        output_dir = RESULTS_DIR / f"model_{iteration}"
        metrics_file = output_dir / "metrics.json"
        return metrics_file.exists()
    
    def load_existing_results(self, iteration: int) -> Dict:
        """Load existing results from JSON file"""
        output_dir = RESULTS_DIR / f"model_{iteration}"
        metrics_file = output_dir / "metrics.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"No existing results found for iteration {iteration}")
        
        print(f"Loading existing results for Model {iteration} from: {metrics_file}")
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    def evaluate_model(self, iteration: int) -> Dict:
        """Evaluate a single model and return metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluating Model {iteration} (Iteration {iteration})")
        print(f"{'='*60}")
        
        # Find model path
        model_path = self.find_model_path(iteration)
        print(f"Model path: {model_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Create output directory for this model
        output_dir = RESULTS_DIR / f"model_{iteration}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run validation
        print(f"Running validation...")
        results = model.val(
            data=DATA_YAML_PATH,
            imgsz=IMGSZ,
            device=DEVICE,
            split='val',
            verbose=True,
            save=True,
            plots=True,
            save_json=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            project=str(output_dir),
            name="plots",
            exist_ok=True,
        )
        
        # Extract metrics
        metrics = self.extract_metrics(results, iteration)
        
        # Save metrics for this model
        self.save_model_metrics(metrics, output_dir, iteration)
        
        return metrics
    
    def extract_metrics(self, results, iteration: int) -> Dict:
        """Extract comprehensive metrics from validation results"""
        metrics = {
            'iteration': iteration,
            'map50-95': float(results.box.map),
            'map50': float(results.box.map50),
            'precision': float(getattr(results.box, 'mp', float('nan'))),
            'recall': float(getattr(results.box, 'mr', float('nan'))),
            'f1': float('nan'),
            'per_class_ap': None,
            'per_class_precision': None,
            'per_class_recall': None,
            'per_class_f1': None,
        }
        
        # Calculate F1
        if metrics['precision'] == metrics['precision'] and metrics['recall'] == metrics['recall']:
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        
        # Per-class metrics
        per_class_precision = getattr(results.box, 'p', None)
        per_class_recall = getattr(results.box, 'r', None)
        per_class_ap = getattr(results.box, 'maps', None)
        
        if per_class_ap is not None:
            metrics['per_class_ap'] = per_class_ap.tolist() if hasattr(per_class_ap, 'tolist') else list(per_class_ap)
        
        if per_class_precision is not None:
            metrics['per_class_precision'] = per_class_precision.tolist() if hasattr(per_class_precision, 'tolist') else list(per_class_precision)
        
        if per_class_recall is not None:
            metrics['per_class_recall'] = per_class_recall.tolist() if hasattr(per_class_recall, 'tolist') else list(per_class_recall)
        
        # Calculate per-class F1
        if per_class_precision is not None and per_class_recall is not None:
            per_class_f1 = []
            for p, r in zip(per_class_precision, per_class_recall):
                if p + r > 0:
                    f1 = 2 * p * r / (p + r)
                else:
                    f1 = 0.0
                per_class_f1.append(f1)
            metrics['per_class_f1'] = per_class_f1
        
        return metrics
    
    def save_model_metrics(self, metrics: Dict, output_dir: Path, iteration: int):
        """Save metrics for a single model"""
        # Save as JSON
        json_path = output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to: {json_path}")
        
        # Save overall metrics as text
        text_path = output_dir / "overall_metrics.txt"
        with open(text_path, 'w') as f:
            f.write(f"=== Model {iteration} Validation Results ===\n\n")
            f.write(f"mAP50-95: {metrics['map50-95']:.6f}\n")
            f.write(f"mAP50: {metrics['map50']:.6f}\n")
            f.write(f"Precision: {metrics['precision']:.6f}\n")
            f.write(f"Recall: {metrics['recall']:.6f}\n")
            f.write(f"F1-Score: {metrics['f1']:.6f}\n")
        
        # Save per-class metrics as CSV
        if metrics['per_class_ap'] is not None:
            per_class_data = []
            for class_id in range(len(metrics['per_class_ap'])):
                per_class_data.append({
                    'class_id': class_id,
                    'class_name': self.class_names[class_id],
                    'precision': metrics['per_class_precision'][class_id] if metrics['per_class_precision'] else None,
                    'recall': metrics['per_class_recall'][class_id] if metrics['per_class_recall'] else None,
                    'f1': metrics['per_class_f1'][class_id] if metrics['per_class_f1'] else None,
                    'ap50_95': metrics['per_class_ap'][class_id],
                })
            
            df = pd.DataFrame(per_class_data)
            csv_path = output_dir / "per_class_metrics.csv"
            df.to_csv(csv_path, index=False)
            print(f"Saved per-class metrics to: {csv_path}")
    
    def compare_all_models(self):
        """Create comparison plots and tables for all models"""
        print(f"\n{'='*60}")
        print("Creating Comparison Plots and Tables")
        print(f"{'='*60}")
        
        comparison_dir = RESULTS_DIR / "all"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract data for comparison
        iterations = sorted(self.all_results.keys())
        
        # Overall metrics comparison
        self.plot_overall_metrics_comparison(comparison_dir, iterations)
        
        # Per-class metrics comparison
        self.plot_per_class_comparison(comparison_dir, iterations)
        
        # Combined per-class metrics in one figure (2x2)
        self.plot_combined_per_class_radar(comparison_dir, iterations)
        
        # Create summary table
        self.create_summary_table(comparison_dir, iterations)
        
        print(f"\nAll comparison results saved to: {comparison_dir}")
    
    def plot_overall_metrics_comparison(self, output_dir: Path, iterations: List[int]):
        """Plot overall metrics comparison across models using radar chart"""
        metrics_to_plot = ['map50-95', 'map50', 'precision', 'recall', 'f1']
        
        # Prepare data
        data = {f"iter_{i}": {} for i in iterations}
        data['metrics'] = metrics_to_plot
        
        for iteration in iterations:
            for metric in metrics_to_plot:
                data[f"iter_{iteration}"][metric] = self.all_results[iteration][metric]
        
        # Save data as JSON
        json_path = output_dir / "overall_metrics_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved overall metrics data to: {json_path}")
        
        # Create ONE radar plot with all iterations overlaid
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='polar')
        
        # Colors for each iteration
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Number of variables
        num_vars = len(metrics_to_plot)
        
        # Compute angle for each axis
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        
        # Metric labels for display
        metric_labels = ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'F1-Score']
        
        # Plot each iteration on the same radar
        for idx, iteration in enumerate(iterations):
            values = [data[f"iter_{iteration}"][metric] for metric in metrics_to_plot]
            values_plot = values + values[:1]  # Complete the circle
            angles_plot = angles + angles[:1]
            
            # Plot line and fill
            ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, 
                   color=colors[idx % len(colors)], label=f"Iteration {iteration}",
                   markersize=6)
            ax.fill(angles_plot, values_plot, alpha=0.15, color=colors[idx % len(colors)])
        
        # Set metric names as labels with better positioning
        ax.set_xticks(angles)
        ax.set_xticklabels(metric_labels, size=12, weight='bold')
        
        # Increase label distance from plot
        ax.tick_params(pad=25)
        
        # Set radial limits and ticks
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11, weight='bold')
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.2)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12, frameon=True)
        
        # Title
        plt.title('Overall Metrics Comparison\nAcross All Iterations', 
                 size=15, fontweight='bold', pad=35)
        
        plt.tight_layout()
        
        plot_path = output_dir / "overall_metrics_comparison.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved overall metrics radar plot to: {plot_path}")
        
        # Keep the line plot to show progression
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, (metric, label) in enumerate(zip(metrics_to_plot, metric_labels)):
            values = [data[f"iter_{iteration}"][metric] for iteration in iterations]
            ax.plot(iterations, values, marker='o', linewidth=2, 
                   label=label, color=colors[i % len(colors)])
        
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Metrics Progression Across Iterations', fontsize=14, fontweight='bold')
        ax.set_xticks(iterations)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        line_plot_path = output_dir / "metrics_progression.png"
        plt.savefig(line_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved metrics progression plot to: {line_plot_path}")
    
    def plot_combined_per_class_radar(self, output_dir: Path, iterations: List[int]):
        """Plot all per-class metrics in one combined 2x2 radar chart"""
        metrics = ['precision', 'recall', 'f1', 'ap50_95']
        metric_titles = ['Precision', 'Recall', 'F1-Score', 'AP50-95']
        
        # Colors for each iteration
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        # Create 2x2 subplot grid with polar projection
        fig = plt.figure(figsize=(16, 16))
        
        # Store handles and labels for combined legend
        legend_handles = []
        legend_labels = []
        
        for idx, (metric, title) in enumerate(zip(metrics, metric_titles)):
            # Create subplot
            ax = fig.add_subplot(2, 2, idx + 1, projection='polar')
            
            # Load data for this metric
            data = {f"iter_{i}": [] for i in iterations}
            data['class_names'] = [self.class_names[i] for i in range(self.num_classes)]
            
            for iteration in iterations:
                metric_key = f'per_class_{metric}' if metric != 'ap50_95' else 'per_class_ap'
                values = self.all_results[iteration].get(metric_key, [])
                if values:
                    data[f"iter_{iteration}"] = values
                else:
                    data[f"iter_{iteration}"] = [0.0] * self.num_classes
            
            # Number of variables
            num_vars = len(data['class_names'])
            
            # Compute angle for each axis
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            
            # Plot each iteration on the same radar
            for iter_idx, iteration in enumerate(iterations):
                values = data[f"iter_{iteration}"]
                values_plot = values + values[:1]  # Complete the circle
                angles_plot = angles + angles[:1]
                
                # Plot line and fill
                line, = ax.plot(angles_plot, values_plot, 'o-', linewidth=2, 
                       color=colors[iter_idx % len(colors)], label=f"Iteration {iteration}",
                       markersize=5)
                ax.fill(angles_plot, values_plot, alpha=0.12, color=colors[iter_idx % len(colors)])
                
                # Collect handles and labels from first subplot only (all iterations)
                if idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(f"Iteration {iteration}")
            
            # Set class names as labels (larger, not bold)
            ax.set_xticks(angles)
            ax.set_xticklabels(data['class_names'], size=12)
            
            # Increase label distance from plot
            ax.tick_params(pad=18)
            
            # Set radial limits and ticks (larger font)
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=11, weight='bold')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=1)
            
            # Subplot title (larger)
            ax.set_title(title, size=15, fontweight='bold', pad=25)
        
        # Reserve more space at top for title and legend
        plt.tight_layout(rect=[0, 0, 1, 0.88])
        
        # Main title at the very top
        fig.suptitle('Per-Class Metrics Comparison - All Metrics Overview', 
                    fontsize=18, fontweight='bold', y=0.97)
        
        # Add single legend at center top (between title and plots)
        fig.legend(legend_handles, legend_labels, 
                  loc='upper center', 
                  bbox_to_anchor=(0.5, 0.93),
                  fontsize=16,
                  frameon=True,
                  shadow=True,
                  fancybox=True,
                  title='Models',
                  title_fontsize=18,
                  ncol=len(iterations))
        
        plot_path = output_dir / "per_class_all_metrics_combined_radar.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved combined per-class metrics radar plot to: {plot_path}")
    
    def plot_per_class_comparison(self, output_dir: Path, iterations: List[int]):
        """Plot per-class metrics comparison across models using radar charts"""
        metrics = ['precision', 'recall', 'f1', 'ap50_95']
        
        # Colors for each iteration
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for metric in metrics:
            # Prepare data
            data = {f"iter_{i}": [] for i in iterations}
            data['class_names'] = [self.class_names[i] for i in range(self.num_classes)]
            data['class_ids'] = list(range(self.num_classes))
            
            for iteration in iterations:
                metric_key = f'per_class_{metric}' if metric != 'ap50_95' else 'per_class_ap'
                values = self.all_results[iteration].get(metric_key, [])
                if values:
                    data[f"iter_{iteration}"] = values
                else:
                    data[f"iter_{iteration}"] = [0.0] * self.num_classes
            
            # Save data as JSON
            json_path = output_dir / f"per_class_{metric}_comparison.json"
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved per-class {metric} data to: {json_path}")
            
            # Create ONE radar plot with all iterations overlaid
            fig = plt.figure(figsize=(10, 10))
            ax = fig.add_subplot(111, projection='polar')
            
            # Number of variables
            num_vars = len(data['class_names'])
            
            # Compute angle for each axis
            angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
            
            # Plot each iteration on the same radar
            for idx, iteration in enumerate(iterations):
                values = data[f"iter_{iteration}"]
                values_plot = values + values[:1]  # Complete the circle
                angles_plot = angles + angles[:1]
                
                # Plot line and fill
                ax.plot(angles_plot, values_plot, 'o-', linewidth=2.5, 
                       color=colors[idx % len(colors)], label=f"Iteration {iteration}",
                       markersize=6)
                ax.fill(angles_plot, values_plot, alpha=0.15, color=colors[idx % len(colors)])
            
            # Set class names as labels with better positioning
            ax.set_xticks(angles)
            ax.set_xticklabels(data['class_names'], size=11, weight='bold')
            
            # Increase label distance from plot
            ax.tick_params(pad=20)
            
            # Set radial limits and ticks
            ax.set_ylim(0, 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=10, weight='bold')
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.4, linewidth=1.2)
            
            # Add legend
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11, frameon=True)
            
            # Title
            metric_name = metric.upper().replace('_', ' ')
            if metric == 'ap50_95':
                metric_name = 'AP50-95'
            
            plt.title(f'Per-Class {metric_name} Comparison\nAcross All Iterations', 
                     size=14, fontweight='bold', pad=30)
            
            plt.tight_layout()
            
            plot_path = output_dir / f"per_class_{metric}_radar.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved per-class {metric} radar plot to: {plot_path}")
    
    def create_summary_table(self, output_dir: Path, iterations: List[int]):
        """Create summary table comparing all models"""
        # Overall metrics table
        overall_data = []
        for iteration in iterations:
            metrics = self.all_results[iteration]
            overall_data.append({
                'Model': f"Iteration {iteration}",
                'Iteration': iteration,
                'mAP50-95': f"{metrics['map50-95']:.4f}",
                'mAP50': f"{metrics['map50']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1']:.4f}",
            })
        
        df_overall = pd.DataFrame(overall_data)
        csv_path = output_dir / "overall_comparison.csv"
        df_overall.to_csv(csv_path, index=False)
        print(f"Saved overall comparison table to: {csv_path}")
        
        # Save overall comparison as JSON too
        json_path = output_dir / "overall_comparison.json"
        with open(json_path, 'w') as f:
            json.dump(overall_data, f, indent=2)
        print(f"Saved overall comparison JSON to: {json_path}")
    
    def run(self):
        """Main execution: evaluate all models and create comparisons"""
        print("\n" + "="*60)
        print("STARTING MODEL COMPARISON")
        print("="*60)
        
        # Check if all results already exist
        all_exist = all(self.check_results_exist(i) for i in sorted(MODEL_FOLDERS.keys()))
        
        if all_exist and not self.force_reeval:
            print("\n✓ All model results already exist!")
            print("Loading existing results (use --force to re-evaluate)...\n")
            
            # Load existing results
            for iteration in sorted(MODEL_FOLDERS.keys()):
                try:
                    metrics = self.load_existing_results(iteration)
                    self.all_results[iteration] = metrics
                    
                    print(f"Model {iteration}: mAP50={metrics['map50']:.4f}, "
                          f"F1={metrics['f1']:.4f}")
                    
                except Exception as e:
                    print(f"\n⚠️  Error loading Model {iteration}: {e}")
                    print("Will re-evaluate this model...")
                    # Fall back to evaluation
                    try:
                        metrics = self.evaluate_model(iteration)
                        self.all_results[iteration] = metrics
                    except Exception as e2:
                        print(f"⚠️  Also failed to evaluate: {e2}")
                        continue
        else:
            # Evaluate each model
            if self.force_reeval:
                print("\n🔄 Force re-evaluation enabled. Re-running all models...\n")
            else:
                print("\n📊 Some results missing. Evaluating models...\n")
            
            for iteration in sorted(MODEL_FOLDERS.keys()):
                # Check if this specific model needs evaluation
                if not self.force_reeval and self.check_results_exist(iteration):
                    print(f"\n✓ Model {iteration} results already exist, loading...")
                    try:
                        metrics = self.load_existing_results(iteration)
                        self.all_results[iteration] = metrics
                        
                        print(f"  mAP50-95: {metrics['map50-95']:.4f}")
                        print(f"  mAP50:    {metrics['map50']:.4f}")
                        print(f"  F1-Score:  {metrics['f1']:.4f}")
                        continue
                    except Exception as e:
                        print(f"  ⚠️ Error loading existing results: {e}")
                        print("  Will re-evaluate...")
                
                # Evaluate model
                try:
                    metrics = self.evaluate_model(iteration)
                    self.all_results[iteration] = metrics
                    
                    print(f"\nModel {iteration} Results:")
                    print(f"  mAP50-95: {metrics['map50-95']:.4f}")
                    print(f"  mAP50:    {metrics['map50']:.4f}")
                    print(f"  Precision: {metrics['precision']:.4f}")
                    print(f"  Recall:    {metrics['recall']:.4f}")
                    print(f"  F1-Score:  {metrics['f1']:.4f}")
                    
                except Exception as e:
                    print(f"\n⚠️  Error evaluating Model {iteration}: {e}")
                    print("Skipping this model...")
                    continue
        
        if not self.all_results:
            print("\n❌ No models were successfully evaluated or loaded!")
            return
        
        # Create comparison plots and tables
        print("\n" + "="*60)
        print("Creating comparison plots and tables...")
        print("="*60)
        self.compare_all_models()
        
        print("\n" + "="*60)
        print("MODEL COMPARISON COMPLETE!")
        print("="*60)
        print(f"\nResults saved to: {RESULTS_DIR}")
        print(f"\nIndividual model results:")
        for iteration in sorted(self.all_results.keys()):
            print(f"  - Model {iteration}: {RESULTS_DIR}/model_{iteration}/")
        print(f"\nComparison results: {RESULTS_DIR}/all/")


class AugmentationConsistencyExperiment:
    """Test-Time Augmentation (TTA) Consistency Experiment using model.val()"""
    
    def __init__(self, conf_threshold: float, iou_threshold: float, force_reeval: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.force_reeval = force_reeval
        self.experiment_dir = RESULTS_DIR / "experiments" / "tta_consistency"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(DATA_YAML_PATH, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.num_classes = self.data_config['nc']
        
        # Find best model
        self.best_model_path = self.find_best_model()
        self.model = YOLO(str(self.best_model_path))
    
    def find_best_model(self) -> Path:
        """Find the best model based on F1 score"""
        best_f1 = -1
        best_iteration = None
        
        print("\nFinding best model...")
        for i in range(4):
            metrics_file = RESULTS_DIR / f"model_{i}" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    f1 = metrics.get('f1', 0)
                    print(f"  Model {i}: F1 = {f1:.4f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_iteration = i
        
        if best_iteration is None:
            raise FileNotFoundError("No model results found. Run model comparison first!")
        
        print(f"  -> Best: Iteration {best_iteration} (F1 = {best_f1:.4f})")
        
        model_folder = TRAINED_MODELS_DIR / str(best_iteration)
        possible_paths = [
            model_folder / "weights" / "best.pt",
            model_folder / "best.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find best.pt for iteration {best_iteration}")
    
    def run_validation(self, use_augment: bool) -> Dict:
        """Run validation with or without test-time augmentation"""
        mode = "with_augment" if use_augment else "no_augment"
        metrics_file = self.experiment_dir / f"metrics_{mode}.json"
        
        if metrics_file.exists() and not self.force_reeval:
            print(f"Loading cached {mode} metrics...")
            with open(metrics_file, 'r') as f:
                return json.load(f)
        
        print(f"\nRunning validation {'WITH' if use_augment else 'WITHOUT'} augmentation...")
        print(f"Using conf={self.conf_threshold}, iou={self.iou_threshold}")
        
        # Run validation
        results = self.model.val(
            data=str(DATA_YAML_PATH),
            imgsz=IMGSZ,
            device=DEVICE,
            split='val',
            verbose=False,
            save=False,
            plots=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            augment=use_augment,  # Key parameter!
        )
        
        # Extract overall metrics
        metrics = {
            'mode': mode,
            'augment': use_augment,
            'map50-95': float(results.box.map),
            'map50': float(results.box.map50),
            'map75': float(results.box.map75),
            'precision': float(getattr(results.box, 'mp', float('nan'))),
            'recall': float(getattr(results.box, 'mr', float('nan'))),
        }
        
        # Calculate F1
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1'] = 0.0
        
        # Extract per-class metrics
        metrics['per_class'] = {}
        if hasattr(results.box, 'maps') and results.box.maps is not None:
            maps_per_class = results.box.maps  # mAP50-95 per class
            for cls_idx in range(self.num_classes):
                if cls_idx < len(maps_per_class):
                    metrics['per_class'][cls_idx] = {
                        'name': self.class_names[cls_idx],
                        'map50-95': float(maps_per_class[cls_idx]) if cls_idx < len(maps_per_class) else 0.0,
                    }
        
        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved {mode} metrics")
        return metrics
    
    def compute_consistency_metrics(self, metrics_no_aug: Dict, metrics_with_aug: Dict) -> Dict:
        """Compute consistency between metrics (percentage difference)"""
        print("\nComputing consistency metrics...")
        
        # Overall metrics comparison (excluding map75)
        overall_metrics = ['map50-95', 'map50', 'precision', 'recall', 'f1']
        
        consistency_summary = {
            'no_augment': {},
            'with_augment': {},
            'consistency_percentage': {},  # 100% means identical
            'absolute_difference': {},
        }
        
        for metric in overall_metrics:
            val_no_aug = metrics_no_aug.get(metric, 0)
            val_with_aug = metrics_with_aug.get(metric, 0)
            
            consistency_summary['no_augment'][metric] = val_no_aug
            consistency_summary['with_augment'][metric] = val_with_aug
            consistency_summary['absolute_difference'][metric] = abs(val_with_aug - val_no_aug)
            
            # Consistency percentage: 100% = identical, 0% = completely different
            if val_no_aug > 0:
                # Percentage change relative to no_aug baseline
                pct_change = abs(val_with_aug - val_no_aug) / val_no_aug
                consistency_pct = max(0, 100 * (1 - pct_change))
            else:
                consistency_pct = 100.0 if val_with_aug == 0 else 0.0
            
            consistency_summary['consistency_percentage'][metric] = consistency_pct
        
        # Average consistency across all metrics
        avg_consistency = np.mean(list(consistency_summary['consistency_percentage'].values()))
        consistency_summary['average_consistency'] = avg_consistency
        
        # Per-class consistency
        per_class_consistency = {}
        for cls_idx in range(self.num_classes):
            if str(cls_idx) in metrics_no_aug.get('per_class', {}) and str(cls_idx) in metrics_with_aug.get('per_class', {}):
                map_no = metrics_no_aug['per_class'][str(cls_idx)].get('map50-95', 0)
                map_aug = metrics_with_aug['per_class'][str(cls_idx)].get('map50-95', 0)
                
                if map_no > 0:
                    pct_change = abs(map_aug - map_no) / map_no
                    consistency_pct = max(0, 100 * (1 - pct_change))
                else:
                    consistency_pct = 100.0 if map_aug == 0 else 0.0
                
                per_class_consistency[cls_idx] = {
                    'name': self.class_names[cls_idx],
                    'map_no_aug': map_no,
                    'map_with_aug': map_aug,
                    'consistency_percentage': consistency_pct,
                }
        
        consistency_summary['per_class'] = per_class_consistency
        
        # Save consistency metrics
        output_file = self.experiment_dir / "consistency_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(consistency_summary, f, indent=2)
        
        print(f"Average consistency: {avg_consistency:.2f}%")
        return consistency_summary
    
    def plot_analysis(self, consistency_data: Dict):
        """Generate consistency analysis plots"""
        print("\nCreating analysis plots...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics to plot (excluding map75)
        metrics = ['map50-95', 'map50', 'precision', 'recall', 'f1']
        metric_labels = ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'F1']
        
        # Plot 1: Metrics comparison - bar chart
        ax = axes[0]
        no_aug_values = [consistency_data['no_augment'][m] for m in metrics]
        with_aug_values = [consistency_data['with_augment'][m] for m in metrics]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, no_aug_values, width, label='No Augment', color='steelblue', edgecolor='black')
        bars2 = ax.bar(x + width/2, with_aug_values, width, label='With Augment', color='coral', edgecolor='black')
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Metrics Comparison', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.05)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 2: Consistency percentage - bar chart
        ax = axes[1]
        consistency_values = [consistency_data['consistency_percentage'][m] for m in metrics]
        
        bars = ax.bar(metric_labels, consistency_values, color='mediumseagreen', edgecolor='black')
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Consistency (%)', fontsize=12, fontweight='bold')
        ax.set_title('Metric Consistency (100% = Identical)', fontsize=13, fontweight='bold')
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_ylim(0, 105)
        ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Perfect')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(fontsize=10)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "consistency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved consistency analysis plots")
    
    def create_summary_report(self, consistency_data: Dict):
        """Create text summary report"""
        report_path = self.experiment_dir / "consistency_report.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("TEST-TIME AUGMENTATION CONSISTENCY ANALYSIS\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_path}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n\n")
            
            f.write("Overall Metrics Comparison:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Metric':<12} {'No Aug':<12} {'With Aug':<12} {'Diff':<12} {'Consistency':<12}\n")
            f.write("-"*70 + "\n")
            
            metrics = ['map50-95', 'map50', 'precision', 'recall', 'f1']
            metric_labels = {'map50-95': 'mAP50-95', 'map50': 'mAP50',
                           'precision': 'Precision', 'recall': 'Recall', 'f1': 'F1'}
            
            for m in metrics:
                no_aug = consistency_data['no_augment'][m]
                with_aug = consistency_data['with_augment'][m]
                diff = consistency_data['absolute_difference'][m]
                cons_pct = consistency_data['consistency_percentage'][m]
                
                f.write(f"{metric_labels[m]:<12} {no_aug:<12.4f} {with_aug:<12.4f} "
                       f"{diff:<12.4f} {cons_pct:>10.2f}%\n")
            
            f.write("-"*70 + "\n\n")
            
            f.write(f"Average Consistency: {consistency_data['average_consistency']:.2f}%\n\n")
            
            f.write("Per-Class Consistency:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Class':<15} {'No Aug mAP':<15} {'With Aug mAP':<15} {'Consistency':<15}\n")
            f.write("-"*70 + "\n")
            
            for cls_data in consistency_data['per_class'].values():
                f.write(f"{cls_data['name']:<15} {cls_data['map_no_aug']:<15.4f} "
                       f"{cls_data['map_with_aug']:<15.4f} {cls_data['consistency_percentage']:>13.2f}%\n")
            
            f.write("-"*70 + "\n\n")
            
            # Interpretation
            f.write("Interpretation:\n")
            f.write("-"*70 + "\n")
            
            avg_consistency = consistency_data['average_consistency']
            if avg_consistency >= 95:
                f.write("[EXCELLENT] Very high consistency (>= 95%)\n")
                f.write("            Model produces nearly identical metrics with/without TTA\n")
                f.write("            Predictions are highly stable\n")
            elif avg_consistency >= 90:
                f.write("[GOOD] Good consistency (>= 90%)\n")
                f.write("       Model metrics are reasonably stable with TTA\n")
            elif avg_consistency >= 85:
                f.write("[MODERATE] Acceptable consistency (>= 85%)\n")
                f.write("           TTA causes some metric variation\n")
            else:
                f.write("[WARN] Low consistency (< 85%)\n")
                f.write("       TTA causes significant metric changes\n")
                f.write("       Consider investigating why augmentation affects results\n")
        
        print("Saved consistency report")
    
    def run(self):
        """Run the augmentation consistency experiment"""
        print("\n" + "="*60)
        print("TEST-TIME AUGMENTATION CONSISTENCY EXPERIMENT")
        print("="*60)
        
        # Run validation without augmentation
        metrics_no_aug = self.run_validation(use_augment=False)
        
        # Run validation with augmentation
        metrics_with_aug = self.run_validation(use_augment=True)
        
        # Compute consistency metrics
        consistency_data = self.compute_consistency_metrics(metrics_no_aug, metrics_with_aug)
        
        # Generate plots
        self.plot_analysis(consistency_data)
        
        # Create summary report
        self.create_summary_report(consistency_data)
        
        print("\n" + "="*60)
        print("TTA CONSISTENCY EXPERIMENT COMPLETE!")
        print("="*60)
        print(f"Results: {self.experiment_dir}")


class GaussianBlurExperiment:
    """Gaussian blur robustness experiment - tests model under degraded conditions"""
    
    def __init__(self, conf_threshold: float, iou_threshold: float, force_reeval: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.force_reeval = force_reeval
        self.results = {}
        self.experiment_dir = RESULTS_DIR / "experiments" / "gaussian"
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(DATA_YAML_PATH, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.num_classes = self.data_config['nc']
        
        # Find best model
        self.best_model_path = self.find_best_model()
        print(f"Best model for experiment: {self.best_model_path}")
        
        # Load validation images
        self.load_validation_images()
    
    def find_best_model(self) -> Path:
        """Find the best model based on F1 score from existing results"""
        best_f1 = -1
        best_iteration = None
        
        for i in range(4):
            metrics_file = RESULTS_DIR / f"model_{i}" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    f1 = metrics.get('f1', 0)
                    print(f"  Model {i}: F1 = {f1:.4f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_iteration = i
        
        if best_iteration is None:
            raise FileNotFoundError("No model results found. Run model comparison first!")
        
        print(f"  → Best: Iteration {best_iteration} (F1 = {best_f1:.4f})")
        
        model_folder = TRAINED_MODELS_DIR / str(best_iteration)
        possible_paths = [
            model_folder / "weights" / "best.pt",
            model_folder / "best.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find best.pt for iteration {best_iteration}")
    
    def load_validation_images(self):
        """Load validation image paths"""
        dataset_path = Path(self.data_config['path'])
        val_file = dataset_path / self.data_config['val']
        
        with open(val_file, 'r') as f:
            self.val_images = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.val_images)} validation images")
    
    def check_version_results_exist(self, version: int) -> bool:
        """Check if results already exist for a version"""
        output_dir = self.experiment_dir / f"version_{version}_results"
        metrics_file = output_dir / "metrics.json"
        return metrics_file.exists()
    
    def load_existing_version_results(self, version: int) -> Dict:
        """Load existing results from JSON file"""
        output_dir = self.experiment_dir / f"version_{version}_results"
        metrics_file = output_dir / "metrics.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"No existing results found for version {version}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    def apply_gaussian_blur(self, image: np.ndarray, kernel_size: int, sigma: float) -> np.ndarray:
        """Apply Gaussian blur to an image"""
        if kernel_size == 0:
            return image
        
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def create_blurred_dataset(self, version: int, kernel_size: int, sigma: float) -> Path:
        """Create a blurred version of the validation dataset"""
        print(f"\nCreating Version {version} (k={kernel_size}, s={sigma})")
        
        if kernel_size == 0:
            blur_dir = self.experiment_dir / f"version_{version}_original"
        else:
            blur_dir = self.experiment_dir / f"version_{version}_k{kernel_size}_s{sigma}"
        
        images_dir = blur_dir / "images"
        labels_dir = blur_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        total = len(self.val_images)
        for idx, img_path in enumerate(self.val_images):
            if (idx + 1) % 50 == 0 or idx == 0 or (idx + 1) == total:
                print(f"  Processing: {idx + 1}/{total} images...")
            
            img_path = Path(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            blurred_img = self.apply_gaussian_blur(img, kernel_size, sigma)
            output_img_path = images_dir / img_path.name
            cv2.imwrite(str(output_img_path), blurred_img)
            
            label_path = str(img_path).replace('images', 'labels').replace('.png', '.txt')
            label_path = Path(label_path)
            
            if label_path.exists():
                output_label_path = labels_dir / label_path.name
                shutil.copy2(label_path, output_label_path)
        
        # Create data.yaml
        data_yaml = blur_dir / "data.yaml"
        dataset_config = {
            'path': str(blur_dir.absolute()),
            'train': 'labels',
            'val': str((blur_dir / 'val.txt').absolute()),
            'test': 'labels',
            'nc': self.num_classes,
            'names': {i: self.class_names[i] for i in range(self.num_classes)}
        }
        
        with open(data_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        val_txt = blur_dir / "val.txt"
        with open(val_txt, 'w') as f:
            for img_path in self.val_images:
                img_name = Path(img_path).name
                abs_path = (images_dir / img_name).absolute()
                f.write(f"{abs_path}\n")
        
        return data_yaml
    
    def evaluate_on_blur_level(self, version: int, data_yaml: Path) -> Dict:
        """Evaluate model on a specific blur level"""
        kernel_size, sigma = BLUR_CONFIGS[version]
        print(f"Evaluating Version {version} (k={kernel_size}, s={sigma})")
        
        model = YOLO(str(self.best_model_path))
        output_dir = self.experiment_dir / f"version_{version}_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = model.val(
            data=str(data_yaml),
            imgsz=IMGSZ,
            device=DEVICE,
            split='val',
            verbose=False,
            save=False,
            plots=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        
        # Extract metrics
        metrics = {
            'version': version,
            'kernel_size': kernel_size,
            'sigma': sigma,
            'map50-95': float(results.box.map),
            'map50': float(results.box.map50),
            'precision': float(getattr(results.box, 'mp', float('nan'))),
            'recall': float(getattr(results.box, 'mr', float('nan'))),
            'f1': float('nan'),
        }
        
        if metrics['precision'] == metrics['precision'] and metrics['recall'] == metrics['recall']:
            if metrics['precision'] + metrics['recall'] > 0:
                metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        
        # Save metrics
        json_path = output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}")
        return metrics
    
    def plot_degradation_analysis(self):
        """Generate all degradation analysis plots"""
        print("\nCreating degradation analysis plots...")
        
        versions = sorted(self.results.keys())
        blur_labels = []
        
        for v in versions:
            k, s = BLUR_CONFIGS[v]
            if k == 0:
                blur_labels.append('Original')
            else:
                blur_labels.append(f'k={k},s={s}')
        
        metrics = ['map50-95', 'map50', 'precision', 'recall', 'f1']
        metric_labels = ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'F1-Score']
        
        degradation_data = {
            'versions': versions,
            'blur_configs': [{'kernel': k, 'sigma': s} for k, s in BLUR_CONFIGS],
            'blur_labels': blur_labels,
        }
        
        for metric in metrics:
            degradation_data[metric] = [self.results[v][metric] for v in versions]
        
        # Save JSON
        json_path = self.experiment_dir / "degradation_data.json"
        with open(json_path, 'w') as f:
            json.dump(degradation_data, f, indent=2)
        
        # Plot 1: Degradation curve
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = degradation_data[metric]
            ax.plot(versions, values, marker='o', linewidth=2.5, markersize=8,
                   label=label, color=colors[i])
        
        ax.set_xlabel('Blur Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Degradation Under Gaussian Blur', 
                    fontsize=15, fontweight='bold')
        ax.set_xticks(versions)
        ax.set_xticklabels(blur_labels, rotation=0, fontsize=11)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([degradation_data[metric] for metric in metrics])
        
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=blur_labels, yticklabels=metric_labels,
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1.0,
                   annot_kws={'size': 10, 'weight': 'bold'})
        
        ax.set_title('Performance Heatmap Across Blur Levels', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Blur Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Retention percentage
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = degradation_data[metric]
            original = values[0]
            if original > 0:
                retention = [(v / original) * 100 for v in values]
            else:
                retention = [100] * len(values)
            
            ax.plot(versions, retention, marker='o', linewidth=2.5, markersize=8,
                   label=label, color=colors[i])
        
        ax.set_xlabel('Blur Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance Retention (%)', fontsize=13, fontweight='bold')
        ax.set_title('Performance Retention Relative to Original Images', 
                    fontsize=15, fontweight='bold')
        ax.set_xticks(versions)
        ax.set_xticklabels(blur_labels, rotation=0, fontsize=11)
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Baseline')
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_retention.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved all analysis plots")
    
    def create_summary_report(self):
        """Create text summary of experiment results"""
        report_path = self.experiment_dir / "experiment_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("GAUSSIAN BLUR ROBUSTNESS EXPERIMENT - SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_path}\n")
            f.write(f"Validation Images: {len(self.val_images)}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n\n")
            
            f.write("Results Across Blur Levels:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Version':<10} {'Blur':<20} {'mAP50':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-"*70 + "\n")
            
            for version in sorted(self.results.keys()):
                metrics = self.results[version]
                k, s = BLUR_CONFIGS[version]
                blur_str = "Original" if k == 0 else f"k={k}, s={s}"
                
                f.write(f"{version:<10} {blur_str:<20} {metrics['map50']:<10.4f} "
                       f"{metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                       f"{metrics['f1']:<10.4f}\n")
            
            f.write("-"*70 + "\n\n")
            
            f.write("Performance Degradation Analysis:\n")
            f.write("-"*70 + "\n")
            
            original_metrics = self.results[0]
            worst_metrics = self.results[len(BLUR_CONFIGS) - 1]
            
            for metric, label in [('map50', 'mAP50'), ('precision', 'Precision'), 
                                 ('recall', 'Recall'), ('f1', 'F1-Score')]:
                orig = original_metrics[metric]
                worst = worst_metrics[metric]
                
                if orig > 0:
                    drop = ((orig - worst) / orig) * 100
                    retention = (worst / orig) * 100
                    f.write(f"{label}:\n")
                    f.write(f"  Original: {orig:.4f}\n")
                    f.write(f"  Worst:    {worst:.4f}\n")
                    f.write(f"  Drop:     {drop:.2f}%\n")
                    f.write(f"  Retention: {retention:.2f}%\n\n")
        
        print(f"✓ Saved experiment summary")
    
    def run(self):
        """Run the Gaussian blur robustness experiment"""
        print("\n" + "="*60)
        print("GAUSSIAN BLUR ROBUSTNESS EXPERIMENT")
        print("="*60)
        
        all_exist = all(self.check_version_results_exist(v) for v in range(len(BLUR_CONFIGS)))
        
        if all_exist and not self.force_reeval:
            print("\n✓ All version results already exist!")
            print("Loading existing results (use --force to re-evaluate)...\n")
            
            for version in range(len(BLUR_CONFIGS)):
                try:
                    metrics = self.load_existing_version_results(version)
                    self.results[version] = metrics
                    k, s = BLUR_CONFIGS[version]
                    blur_str = "Original" if k == 0 else f"k={k}, s={s}"
                    print(f"Version {version} ({blur_str}): mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}")
                except Exception as e:
                    kernel_size, sigma = BLUR_CONFIGS[version]
                    data_yaml = self.create_blurred_dataset(version, kernel_size, sigma)
                    metrics = self.evaluate_on_blur_level(version, data_yaml)
                    self.results[version] = metrics
        else:
            if self.force_reeval:
                print("\n🔄 Force re-evaluation enabled\n")
            else:
                print("\n📊 Some results missing\n")
            
            for version, (kernel_size, sigma) in enumerate(BLUR_CONFIGS):
                if not self.force_reeval and self.check_version_results_exist(version):
                    print(f"✓ Version {version} exists, loading...")
                    try:
                        metrics = self.load_existing_version_results(version)
                        self.results[version] = metrics
                        continue
                    except Exception as e:
                        print(f"  Error loading: {e}, will re-process...")
                
                data_yaml = self.create_blurred_dataset(version, kernel_size, sigma)
                metrics = self.evaluate_on_blur_level(version, data_yaml)
                self.results[version] = metrics
        
        if not self.results:
            print("\n❌ No versions processed!")
            return
        
        self.plot_degradation_analysis()
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("GAUSSIAN BLUR EXPERIMENT COMPLETE!")
        print("="*60)
        print(f"Results: {self.experiment_dir}")


class UnderwaterEffectExperiment:
    """Underwater effect robustness experiment - tests model under simulated underwater conditions"""
    
    def __init__(self, conf_threshold: float, iou_threshold: float, force_reeval: bool = False):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.force_reeval = force_reeval
        self.results = {}
        self.experiment_dir = RESULTS_DIR / "experiments" / "underwater"
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(DATA_YAML_PATH, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.num_classes = self.data_config['nc']
        
        # Find best model
        self.best_model_path = self.find_best_model()
        self.model = YOLO(str(self.best_model_path))
        
        # Load validation images
        self.load_validation_images()
    
    def find_best_model(self) -> Path:
        """Find the best model based on F1 score"""
        best_f1 = -1
        best_iteration = None
        
        print("\nFinding best model...")
        for i in range(4):
            metrics_file = RESULTS_DIR / f"model_{i}" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    f1 = metrics.get('f1', 0)
                    print(f"  Model {i}: F1 = {f1:.4f}")
                    
                    if f1 > best_f1:
                        best_f1 = f1
                        best_iteration = i
        
        if best_iteration is None:
            raise FileNotFoundError("No model results found. Run model comparison first!")
        
        print(f"  -> Best: Iteration {best_iteration} (F1 = {best_f1:.4f})")
        
        model_folder = TRAINED_MODELS_DIR / str(best_iteration)
        possible_paths = [
            model_folder / "weights" / "best.pt",
            model_folder / "best.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find best.pt for iteration {best_iteration}")
    
    def load_validation_images(self):
        """Load validation image paths"""
        dataset_path = Path(self.data_config['path'])
        val_file = dataset_path / self.data_config['val']
        
        with open(val_file, 'r') as f:
            self.val_images = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.val_images)} validation images")
    
    def apply_underwater_effect(self, image: np.ndarray, blue_intensity: float, green_intensity: float, 
                                haze_intensity: float, darkness: float) -> np.ndarray:
        """
        Apply underwater effect to image
        
        Args:
            image: Input image (BGR format)
            blue_intensity: Blue channel boost (0.0 - 1.0)
            green_intensity: Green channel boost (0.0 - 1.0)
            haze_intensity: Haze/fog effect (0.0 - 1.0)
            darkness: Overall darkening (0.0 - 1.0)
        
        Returns:
            Image with underwater effect applied
        """
        if blue_intensity == 0 and green_intensity == 0 and haze_intensity == 0 and darkness == 0:
            return image.copy()
        
        img_float = image.astype(np.float32) / 255.0
        
        # 1. Apply blue-green color cast (underwater light absorption)
        # Red light is absorbed most in water, blue/green penetrate deeper
        if blue_intensity > 0 or green_intensity > 0:
            # Reduce red channel
            img_float[:, :, 2] = img_float[:, :, 2] * (1.0 - blue_intensity * 0.5)
            # Boost blue channel
            img_float[:, :, 0] = np.clip(img_float[:, :, 0] + blue_intensity * 0.3, 0, 1)
            # Boost green channel
            img_float[:, :, 1] = np.clip(img_float[:, :, 1] + green_intensity * 0.3, 0, 1)
        
        # 2. Apply haze effect (reduced visibility/contrast)
        if haze_intensity > 0:
            # Create haze color (blue-green tint)
            haze_color = np.array([0.4 + blue_intensity * 0.2, 
                                   0.5 + green_intensity * 0.2, 
                                   0.3], dtype=np.float32)
            # Blend with haze
            img_float = img_float * (1.0 - haze_intensity) + haze_color * haze_intensity
        
        # 3. Reduce contrast (scattering effect)
        if haze_intensity > 0:
            mean_val = np.mean(img_float)
            img_float = mean_val + (img_float - mean_val) * (1.0 - haze_intensity * 0.5)
        
        # 4. Apply darkness (depth/light absorption)
        if darkness > 0:
            img_float = img_float * (1.0 - darkness * 0.6)
        
        # Convert back to uint8
        img_result = np.clip(img_float * 255, 0, 255).astype(np.uint8)
        return img_result
    
    def create_underwater_dataset(self, level: int, blue_int: float, green_int: float, 
                                   haze_int: float, darkness: float) -> Path:
        """Create dataset with underwater effect applied"""
        print(f"\nCreating underwater effect dataset (Level {level})...")
        
        # Create directory structure
        level_dir = self.experiment_dir / f"level_{level}"
        images_dir = level_dir / "images"
        labels_dir = level_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Process images
        total = len(self.val_images)
        for idx, img_path in enumerate(self.val_images):
            if (idx + 1) % 20 == 0 or idx == 0 or (idx + 1) == total:
                print(f"  Processing: {idx + 1}/{total}...")
            
            # Read and apply effect
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_underwater = self.apply_underwater_effect(img, blue_int, green_int, haze_int, darkness)
            
            # Save processed image
            img_name = Path(img_path).name
            output_path = images_dir / img_name
            cv2.imwrite(str(output_path), img_underwater)
            
            # Copy label file
            label_path = str(img_path).replace('images', 'labels').replace('.png', '.txt')
            if Path(label_path).exists():
                label_name = Path(label_path).name
                shutil.copy(label_path, labels_dir / label_name)
        
        # Create val.txt
        val_txt = level_dir / "val.txt"
        with open(val_txt, 'w') as f:
            for img_path in self.val_images:
                img_name = Path(img_path).name
                new_path = images_dir / img_name
                f.write(f"{new_path}\n")
        
        # Create data.yaml
        data_yaml = level_dir / "data.yaml"
        dataset_config = {
            'path': str(level_dir.absolute()),
            'train': 'images',
            'val': str(val_txt.absolute()),
            'test': 'images',
            'nc': self.num_classes,
            'names': {i: self.class_names[i] for i in range(self.num_classes)}
        }
        
        with open(data_yaml, 'w') as f:
            yaml.dump(dataset_config, f, default_flow_style=False)
        
        print(f"Created underwater dataset: {level_dir}")
        return data_yaml
    
    def check_version_results_exist(self, level: int) -> bool:
        """Check if results already exist for a level"""
        output_dir = self.experiment_dir / f"level_{level}_results"
        metrics_file = output_dir / "metrics.json"
        return metrics_file.exists()
    
    def load_existing_version_results(self, level: int) -> Dict:
        """Load existing results from JSON file"""
        output_dir = self.experiment_dir / f"level_{level}_results"
        metrics_file = output_dir / "metrics.json"
        
        if not metrics_file.exists():
            raise FileNotFoundError(f"No existing results found for level {level}")
        
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        return metrics
    
    def evaluate_on_underwater_level(self, level: int, data_yaml: Path) -> Dict:
        """Evaluate model on a specific underwater effect level"""
        blue_int, green_int, haze_int, darkness = UNDERWATER_CONFIGS[level]
        effect_str = "Original" if level == 0 else f"blue={blue_int:.2f}, haze={haze_int:.2f}"
        print(f"\nEvaluating Level {level} ({effect_str})")
        
        output_dir = self.experiment_dir / f"level_{level}_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = self.model.val(
            data=str(data_yaml),
            imgsz=IMGSZ,
            device=DEVICE,
            split='val',
            verbose=False,
            save=False,
            plots=False,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
        )
        
        # Extract metrics
        metrics = {
            'level': level,
            'blue_intensity': blue_int,
            'green_intensity': green_int,
            'haze_intensity': haze_int,
            'darkness': darkness,
            'map50-95': float(results.box.map),
            'map50': float(results.box.map50),
            'precision': float(getattr(results.box, 'mp', float('nan'))),
            'recall': float(getattr(results.box, 'mr', float('nan'))),
            'f1': float('nan'),
        }
        
        if metrics['precision'] > 0 and metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
        
        # Save metrics
        json_path = output_dir / "metrics.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"  mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}")
        return metrics
    
    def plot_degradation_analysis(self):
        """Generate degradation analysis plots"""
        print("\nCreating degradation analysis plots...")
        
        levels = sorted(self.results.keys())
        metrics = ['map50-95', 'map50', 'precision', 'recall', 'f1']
        metric_labels = ['mAP50-95', 'mAP50', 'Precision', 'Recall', 'F1-Score']
        
        # Prepare data
        degradation_data = {
            'levels': levels,
            'underwater_configs': [UNDERWATER_CONFIGS[l] for l in levels],
        }
        
        for metric in metrics:
            degradation_data[metric] = [self.results[l][metric] for l in levels]
        
        # Save JSON
        json_path = self.experiment_dir / "degradation_data.json"
        with open(json_path, 'w') as f:
            json.dump(degradation_data, f, indent=2)
        
        # Plot 1: Degradation curves
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = degradation_data[metric]
            ax.plot(levels, values, marker='o', linewidth=2.5, markersize=8,
                   label=label, color=colors[i])
        
        ax.set_xlabel('Underwater Effect Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Model Performance Under Simulated Underwater Conditions', 
                    fontsize=15, fontweight='bold')
        ax.set_xticks(levels)
        ax.set_xticklabels([f'L{l}' for l in levels])
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_curve.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Performance retention
        fig, ax = plt.subplots(figsize=(12, 7))
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = degradation_data[metric]
            baseline = values[0]
            if baseline > 0:
                retention = [(v / baseline) * 100 for v in values]
            else:
                retention = [100] * len(values)
            
            ax.plot(levels, retention, marker='o', linewidth=2.5, markersize=8,
                   label=label, color=colors[i])
        
        ax.set_xlabel('Underwater Effect Level', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance Retention (%)', fontsize=13, fontweight='bold')
        ax.set_title('Performance Retention Relative to Original Images', 
                    fontsize=15, fontweight='bold')
        ax.set_xticks(levels)
        ax.set_xticklabels([f'L{l}' for l in levels])
        ax.legend(loc='best', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=100, color='black', linestyle='--', alpha=0.5)
        ax.set_ylim(0, 110)
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "performance_retention.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Heatmap
        fig, ax = plt.subplots(figsize=(10, 6))
        data_matrix = np.array([degradation_data[metric] for metric in metrics])
        
        sns.heatmap(data_matrix, annot=True, fmt='.3f', cmap='RdYlGn',
                   xticklabels=[f'Level {l}' for l in levels],
                   yticklabels=metric_labels,
                   cbar_kws={'label': 'Score'}, vmin=0, vmax=1.0,
                   annot_kws={'size': 10, 'weight': 'bold'})
        
        ax.set_title('Performance Heatmap Across Underwater Effect Levels', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Underwater Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.experiment_dir / "degradation_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Saved all analysis plots")
    
    def create_summary_report(self):
        """Create text summary of experiment results"""
        report_path = self.experiment_dir / "experiment_summary.txt"
        
        with open(report_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("UNDERWATER EFFECT ROBUSTNESS EXPERIMENT - SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Best Model: {self.best_model_path}\n")
            f.write(f"Validation Images: {len(self.val_images)}\n")
            f.write(f"Confidence Threshold: {self.conf_threshold}\n")
            f.write(f"IoU Threshold: {self.iou_threshold}\n\n")
            
            f.write("Underwater Effect Levels:\n")
            f.write("-"*70 + "\n")
            for level in sorted(self.results.keys()):
                blue, green, haze, dark = UNDERWATER_CONFIGS[level]
                desc = "Original" if level == 0 else f"blue={blue:.2f}, haze={haze:.2f}, dark={dark:.2f}"
                f.write(f"  Level {level}: {desc}\n")
            f.write("\n")
            
            f.write("Performance Across Underwater Levels:\n")
            f.write("-"*70 + "\n")
            f.write(f"{'Level':<8} {'mAP50':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}\n")
            f.write("-"*70 + "\n")
            
            for level in sorted(self.results.keys()):
                metrics = self.results[level]
                f.write(f"{level:<8} {metrics['map50']:<10.4f} {metrics['precision']:<10.4f} "
                       f"{metrics['recall']:<10.4f} {metrics['f1']:<10.4f}\n")
            
            f.write("-"*70 + "\n\n")
            
            # Performance degradation analysis
            baseline = self.results[0]
            worst = self.results[len(UNDERWATER_CONFIGS) - 1]
            
            f.write("Robustness Analysis:\n")
            f.write("-"*70 + "\n")
            
            for metric, label in [('map50', 'mAP50'), ('precision', 'Precision'), 
                                 ('recall', 'Recall'), ('f1', 'F1-Score')]:
                base_val = baseline[metric]
                worst_val = worst[metric]
                
                if base_val > 0:
                    drop_pct = ((base_val - worst_val) / base_val) * 100
                    retention_pct = (worst_val / base_val) * 100
                    
                    f.write(f"{label}:\n")
                    f.write(f"  Original: {base_val:.4f}\n")
                    f.write(f"  Deep Underwater: {worst_val:.4f}\n")
                    f.write(f"  Performance Drop: {drop_pct:.2f}%\n")
                    f.write(f"  Retention: {retention_pct:.2f}%\n\n")
            
            # Overall robustness assessment
            f1_retention = (worst['f1'] / baseline['f1'] * 100) if baseline['f1'] > 0 else 0
            
            f.write("Overall Robustness:\n")
            f.write("-"*70 + "\n")
            if f1_retention >= 80:
                f.write("[EXCELLENT] Model maintains >= 80% F1 under deep underwater conditions\n")
                f.write("            Highly robust to underwater effects\n")
            elif f1_retention >= 60:
                f.write("[GOOD] Model retains >= 60% F1 under deep underwater conditions\n")
                f.write("       Reasonably robust\n")
            elif f1_retention >= 40:
                f.write("[MODERATE] Model retains >= 40% F1 under deep underwater conditions\n")
                f.write("           Some degradation under extreme conditions\n")
            else:
                f.write("[POOR] Model retains < 40% F1 under deep underwater conditions\n")
                f.write("       Significant degradation - consider training with underwater augmentation\n")
        
        print("Saved experiment summary")
    
    def run(self):
        """Run the underwater effect experiment"""
        print("\n" + "="*60)
        print("UNDERWATER EFFECT ROBUSTNESS EXPERIMENT")
        print("="*60)
        
        # Check if all results exist
        all_exist = all(self.check_version_results_exist(v) for v in range(len(UNDERWATER_CONFIGS)))
        
        if all_exist and not self.force_reeval:
            print("\n✓ All level results already exist!")
            print("Loading existing results (use --force to re-evaluate)...\n")
            
            for level in range(len(UNDERWATER_CONFIGS)):
                try:
                    metrics = self.load_existing_version_results(level)
                    self.results[level] = metrics
                    blue, green, haze, dark = UNDERWATER_CONFIGS[level]
                    effect_str = "Original" if level == 0 else f"blue={blue:.2f}, haze={haze:.2f}"
                    print(f"Level {level} ({effect_str}): mAP50={metrics['map50']:.4f}, F1={metrics['f1']:.4f}")
                except Exception as e:
                    blue, green, haze, dark = UNDERWATER_CONFIGS[level]
                    data_yaml = self.create_underwater_dataset(level, blue, green, haze, dark)
                    metrics = self.evaluate_on_underwater_level(level, data_yaml)
                    self.results[level] = metrics
        else:
            if self.force_reeval:
                print("\n🔄 Force re-evaluation enabled\n")
            
            # Create and evaluate each level
            for level in range(len(UNDERWATER_CONFIGS)):
                if not self.force_reeval and self.check_version_results_exist(level):
                    print(f"✓ Level {level} exists, loading...")
                    try:
                        metrics = self.load_existing_version_results(level)
                        self.results[level] = metrics
                        continue
                    except Exception as e:
                        print(f"  Error loading: {e}, will re-process...")
                
                blue, green, haze, dark = UNDERWATER_CONFIGS[level]
                data_yaml = self.create_underwater_dataset(level, blue, green, haze, dark)
                metrics = self.evaluate_on_underwater_level(level, data_yaml)
                self.results[level] = metrics
        
        if not self.results:
            print("\n❌ No levels processed!")
            return
        
        self.plot_degradation_analysis()
        self.create_summary_report()
        
        print("\n" + "="*60)
        print("UNDERWATER EFFECT EXPERIMENT COMPLETE!")
        print("="*60)
        print(f"Results: {self.experiment_dir}")


class Visualizer:
    """Generate visualizations of predictions and ground truth"""
    
    def __init__(self, show_conf: float, show_iou: float, force_reeval: bool = False):
        self.show_conf = show_conf
        self.show_iou = show_iou
        self.force_reeval = force_reeval
        self.vis_dir = RESULTS_DIR / "vis"
        self.preds_dir = self.vis_dir / "PREDs"
        self.gts_dir = self.vis_dir / "GTs"
        self.params_file = self.vis_dir / "parameters.json"
        
        # Create directories
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        self.preds_dir.mkdir(parents=True, exist_ok=True)
        self.gts_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data config
        with open(DATA_YAML_PATH, 'r') as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config['names']
        self.class_names = {int(k): v for k, v in self.class_names.items()}
        self.num_classes = self.data_config['nc']
        
        # Find best model
        self.best_model_path = self.find_best_model()
        self.model = YOLO(str(self.best_model_path))
        
        # Load validation images
        self.load_validation_images()
        
        # Colors for classes (BGR format for OpenCV)
        self.colors = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 0, 255),
            (255, 255, 0), (0, 165, 255), (128, 0, 128), (203, 192, 255), (0, 0, 139),
            (128, 128, 128), (0, 0, 0), (255, 255, 255), (0, 128, 0), (128, 0, 0),
            (0, 215, 255), (180, 105, 255), (50, 205, 50), (255, 20, 147), (255, 140, 0),
            (0, 191, 255), (255, 69, 0), (138, 43, 226), (220, 20, 60), (0, 250, 154)
        ]
    
    def find_best_model(self) -> Path:
        """Find the best model based on F1 score"""
        best_f1 = -1
        best_iteration = None
        
        print("\nFinding best model for visualization...")
        for i in range(4):
            metrics_file = RESULTS_DIR / f"model_{i}" / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    metrics = json.load(f)
                    f1 = metrics.get('f1', 0)
                    if f1 > best_f1:
                        best_f1 = f1
                        best_iteration = i
        
        if best_iteration is None:
            raise FileNotFoundError("No model results found. Run model comparison first!")
        
        print(f"  -> Best: Iteration {best_iteration} (F1 = {best_f1:.4f})")
        
        model_folder = TRAINED_MODELS_DIR / str(best_iteration)
        possible_paths = [
            model_folder / "weights" / "best.pt",
            model_folder / "best.pt",
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        raise FileNotFoundError(f"Could not find best.pt for iteration {best_iteration}")
    
    def load_validation_images(self):
        """Load validation image paths"""
        dataset_path = Path(self.data_config['path'])
        val_file = dataset_path / self.data_config['val']
        
        with open(val_file, 'r') as f:
            self.val_images = [line.strip() for line in f if line.strip()]
        
        print(f"Loaded {len(self.val_images)} validation images")
    
    def check_parameters_match(self) -> bool:
        """Check if current parameters match saved parameters"""
        if not self.params_file.exists():
            return False
        
        try:
            with open(self.params_file, 'r') as f:
                saved_params = json.load(f)
            
            current_params = {
                'show_conf': self.show_conf,
                'show_iou': self.show_iou,
                'best_model': str(self.best_model_path),
                'augment': True,
            }
            
            return saved_params == current_params
        except:
            return False
    
    def save_parameters(self):
        """Save visualization parameters"""
        params = {
            'show_conf': self.show_conf,
            'show_iou': self.show_iou,
            'best_model': str(self.best_model_path),
            'augment': True,
        }
        
        with open(self.params_file, 'w') as f:
            json.dump(params, f, indent=2)
    
    def draw_boxes_on_image(self, img, boxes, is_prediction=True):
        """Draw bounding boxes on image"""
        img_copy = img.copy()
        h, w = img_copy.shape[:2]
        
        for box in boxes:
            if is_prediction:
                # Prediction: (class_id, x1, y1, x2, y2, confidence)
                class_id, x1, y1, x2, y2, conf = box
                x1_px, y1_px, x2_px, y2_px = int(x1), int(y1), int(x2), int(y2)
                label = f"{self.class_names.get(class_id, f'Class_{class_id}')} ({conf:.2f})"
            else:
                # GT: (class_id, x_center, y_center, width, height) - normalized
                class_id, x_center, y_center, width, height = box
                x1_px = int((x_center - width/2) * w)
                y1_px = int((y_center - height/2) * h)
                x2_px = int((x_center + width/2) * w)
                y2_px = int((y_center + height/2) * h)
                label = self.class_names.get(class_id, f'Class_{class_id}')
            
            # Ensure coordinates are within bounds
            x1_px = max(0, min(x1_px, w-1))
            y1_px = max(0, min(y1_px, h-1))
            x2_px = max(0, min(x2_px, w-1))
            y2_px = max(0, min(y2_px, h-1))
            
            if x2_px <= x1_px or y2_px <= y1_px:
                continue
            
            # Get color
            color = self.colors[class_id % len(self.colors)]
            
            # Draw rectangle
            cv2.rectangle(img_copy, (x1_px, y1_px), (x2_px, y2_px), color, 2)
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img_copy, (x1_px, y1_px - text_height - 10),
                         (x1_px + text_width, y1_px), color, -1)
            
            # Draw label text
            cv2.putText(img_copy, label, (x1_px, y1_px - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img_copy
    
    def run(self):
        """Generate visualizations"""
        print("\n" + "="*60)
        print("GENERATING VISUALIZATIONS")
        print("="*60)
        print(f"Best model: {self.best_model_path}")
        print(f"Show conf: {self.show_conf}")
        print(f"Show IoU: {self.show_iou}")
        
        # Check if parameters match
        if not self.force_reeval and self.check_parameters_match():
            print("\n✓ Visualizations already exist with same parameters!")
            print("  (use --force to regenerate)")
            return
        
        if self.force_reeval:
            print("\n🔄 Force re-generation enabled")
        else:
            print("\n📊 Generating new visualizations...")
        
        # Save parameters
        self.save_parameters()
        
        total = len(self.val_images)
        for idx, img_path in enumerate(self.val_images):
            if (idx + 1) % 20 == 0 or idx == 0 or (idx + 1) == total:
                print(f"  Processing: {idx + 1}/{total}...")
            
            img_path = Path(img_path)
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            img_name = img_path.name
            
            # Generate predictions with augment=True
            results = self.model.predict(
                source=str(img_path),
                conf=self.show_conf,
                iou=self.show_iou,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False,
                augment=True,  # Always use augmentation
            )
            
            # Extract predictions
            predictions = []
            if results and len(results) > 0:
                result = results[0]
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for i in range(len(boxes)):
                        predictions.append((
                            class_ids[i],
                            boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3],
                            confidences[i]
                        ))
            
            # Draw predictions
            img_preds = self.draw_boxes_on_image(img, predictions, is_prediction=True)
            
            # Add header
            header_text = f"Predictions ({len(predictions)} detections)"
            cv2.putText(img_preds, header_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Save predictions visualization
            cv2.imwrite(str(self.preds_dir / img_name), img_preds)
            
            # Load ground truth
            label_path = str(img_path).replace('images', 'labels').replace('.png', '.txt')
            gt_boxes = []
            
            if Path(label_path).exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            gt_boxes.append((class_id, x_center, y_center, width, height))
            
            # Draw ground truth
            img_gts = self.draw_boxes_on_image(img, gt_boxes, is_prediction=False)
            
            # Add header
            header_text = f"Ground Truth ({len(gt_boxes)} annotations)"
            cv2.putText(img_gts, header_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Save GT visualization
            cv2.imwrite(str(self.gts_dir / img_name), img_gts)
        
        print("\n" + "="*60)
        print("VISUALIZATION COMPLETE!")
        print("="*60)
        print(f"Predictions: {self.preds_dir}")
        print(f"Ground Truth: {self.gts_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare trained models and run experiments")
    parser.add_argument("--conf", type=float, default=0.001, 
                       help="Confidence threshold for predictions (default: 0.001)")
    parser.add_argument("--iou", type=float, default=0.5, 
                       help="IoU threshold for validation (default: 0.5)")
    parser.add_argument("--show_conf", type=float, default=0.2,
                       help="Confidence threshold for visualization (default: 0.2)")
    parser.add_argument("--show_iou", type=float, default=0.5,
                       help="IoU threshold for visualization NMS (default: 0.5)")
    parser.add_argument("--force", action="store_true",
                       help="Force re-evaluation even if results exist")
    parser.add_argument("--experiment", choices=['gaussian', 'underwater', 'tta', 'all'], default=None,
                       help="Run experiment: 'gaussian'=blur test, 'underwater'=underwater effect, 'tta'=augmentation consistency, 'all'=comparison+all experiments")
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("MODEL EVALUATION & EXPERIMENTS TOOL")
    print(f"{'='*60}")
    print(f"Confidence threshold: {args.conf}")
    print(f"IoU threshold: {args.iou}")
    print(f"Visualization conf: {args.show_conf}")
    print(f"Visualization IoU: {args.show_iou}")
    if args.force:
        print(f"Force re-evaluation: YES")
    
    try:
        # Always run visualizations first (with parameter checking)
        visualizer = Visualizer(show_conf=args.show_conf, show_iou=args.show_iou, 
                               force_reeval=args.force)
        visualizer.run()
        
        # Run model comparison if no experiment specified or if 'all'
        if args.experiment is None or args.experiment == 'all':
            print("\n" + "="*60)
            print("RUNNING MODEL COMPARISON")
            print("="*60)
            comparator = ModelComparator(conf_threshold=args.conf, iou_threshold=args.iou, 
                                        force_reeval=args.force)
            comparator.run()
        
        # Run Gaussian blur experiment if specified or if 'all'
        if args.experiment == 'gaussian' or args.experiment == 'all':
            print("\n" + "="*60)
            print("RUNNING GAUSSIAN BLUR EXPERIMENT")
            print("="*60)
            experiment = GaussianBlurExperiment(conf_threshold=args.conf, iou_threshold=args.iou,
                                               force_reeval=args.force)
            experiment.run()
        
        # Run Underwater effect experiment if specified or if 'all'
        if args.experiment == 'underwater' or args.experiment == 'all':
            print("\n" + "="*60)
            print("RUNNING UNDERWATER EFFECT EXPERIMENT")
            print("="*60)
            underwater_exp = UnderwaterEffectExperiment(conf_threshold=args.conf, iou_threshold=args.iou,
                                                       force_reeval=args.force)
            underwater_exp.run()
        
        # Run TTA consistency experiment if specified or if 'all'
        if args.experiment == 'tta' or args.experiment == 'all':
            print("\n" + "="*60)
            print("RUNNING TTA CONSISTENCY EXPERIMENT")
            print("="*60)
            tta_exp = AugmentationConsistencyExperiment(conf_threshold=args.conf, iou_threshold=args.iou,
                                                       force_reeval=args.force)
            tta_exp.run()
        
        print("\n" + "="*60)
        print("ALL TASKS COMPLETE!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())

