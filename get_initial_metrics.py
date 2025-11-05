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
import textwrap
from collections import Counter

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
BEST_MODEL_PATH = "training_results_simple/full_initial_vfl/weights/best.pt"
DATA_YAML_PATH = "detector_dataset_simple/data.yaml"
OUTPUT_DIR = "initial_metrics"
DEVICE = 0
IMGSZ = 960

# Validation thresholds
IOU_THRESHOLD = 0.7  
CONF_THRESHOLD = 0.001    

def load_model_and_validate():
    """
    Load the best trained patch model and validate it on the patch dataset.
    
    Returns:
        dict: Validation metrics
    """
    print("Loading best trained patch model...")
    
    # Check if model exists
    if not Path(BEST_MODEL_PATH).exists():
        raise FileNotFoundError(f"Best model not found at: {BEST_MODEL_PATH}")
    
    # Load the model
    model = YOLO(BEST_MODEL_PATH)
    print(f"Model loaded successfully from: {BEST_MODEL_PATH}")
    
    # Check if data yaml exists
    if not Path(DATA_YAML_PATH).exists():
        raise FileNotFoundError(f"Data YAML not found at: {DATA_YAML_PATH}")
    
    print(f"Using dataset configuration from: {DATA_YAML_PATH}")
    
    # Load dataset info
    with open(DATA_YAML_PATH, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_nc = data_config.get('nc', 0)
    print(f"Dataset classes: {dataset_nc}")
    print(f"Class names: {list(data_config['names'].values())}")
    
    # Check model vs dataset class count
    try:
        model_nc = model.model.nc if hasattr(model.model, 'nc') else None
        if model_nc is not None:
            print(f"Model classes: {model_nc}")
            if model_nc != dataset_nc:
                print(f"⚠️  WARNING: Model has {model_nc} classes but dataset has {dataset_nc} classes!")
                print("   This may cause mismatches in confusion matrix and per-class metrics.")
                print("   Custom confusion matrix plotting will handle this automatically.")
        else:
            print("Model class count not available (nc attribute missing)")
    except Exception as e:
        print(f"Could not verify model class count: {e}")
    
    # Validate the model with default parameters
    print("\nStarting validation with default parameters...")
    print(f"Image size: {IMGSZ}")
    print(f"Device: {DEVICE}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    
    # Run validation
    results = model.val(
        data=DATA_YAML_PATH,
        imgsz=IMGSZ,
        device=DEVICE,
        split='val',
        verbose=True,
        save=True,
        plots=True,
        save_json=True,
        save_txt=True,
        save_conf=True,
        save_crop=False,
        show=False,
        project=OUTPUT_DIR,
        name="plots",
        exist_ok=True,
        iou=IOU_THRESHOLD,      # IoU threshold for Precision and Recall computation
        conf=CONF_THRESHOLD,    # Confidence threshold for predictions
        # max_det=MAX_DETECTIONS  # Maximum detections per image
    )
    
    return results

def extract_metrics(results):
    """
    Extract comprehensive metrics from validation results.
    
    Args:
        results: YOLO validation results object
        
    Returns:
        dict: Extracted metrics
    """
    # Get per-class metrics if available
    per_class_precision = getattr(results.box, 'p', None)  # Precision per class
    per_class_recall = getattr(results.box, 'r', None)     # Recall per class
    
    metrics = {
        'map50-95': float(results.box.map),
        'map50': float(results.box.map50),
        'precision': float(getattr(results.box, 'mp', float('nan'))),
        'recall': float(getattr(results.box, 'mr', float('nan'))),
        'f1': float('nan'),
        'per_class_ap': getattr(results.box, 'maps', None),
        'per_class_precision': per_class_precision,
        'per_class_recall': per_class_recall,
        'names': getattr(results, 'names', None)
    }
    
    # Calculate overall F1 score
    if metrics['precision'] == metrics['precision'] and metrics['recall'] == metrics['recall']:
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall'])
    
    # Calculate per-class F1 scores
    if per_class_precision is not None and per_class_recall is not None:
        metrics['per_class_f1'] = []
        for p, r in zip(per_class_precision, per_class_recall):
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
            metrics['per_class_f1'].append(f1)
    else:
        metrics['per_class_f1'] = None
    
    return metrics

def _extract_confusion_matrix(results):
    """Best-effort extraction of confusion matrix (numpy array) from ultralytics results."""
    # Common attributes seen in ultralytics
    cm = None
    # 1) results.confusion_matrix.matrix
    cm_obj = getattr(results, 'confusion_matrix', None)
    if cm_obj is not None:
        cm = getattr(cm_obj, 'matrix', None)
        if cm is None and isinstance(cm_obj, np.ndarray):
            cm = cm_obj
    # 2) results.box.confusion_matrix
    if cm is None and hasattr(results, 'box'):
        cm2 = getattr(results.box, 'confusion_matrix', None)
        if cm2 is not None:
            cm = getattr(cm2, 'matrix', None)
            if cm is None and isinstance(cm2, np.ndarray):
                cm = cm2
    # Ensure numpy array
    if cm is not None:
        cm = np.array(cm)
    return cm


def _plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path, normalize: bool = True):
    """Plot and save confusion matrix with readability for many classes."""
    if cm is None:
        return
    
    title = "Confusion Matrix (Normalized)" if normalize else "Confusion Matrix (Counts)"
    
    # Create working copy for plotting
    cm_plot = cm.astype(np.float32)
    
    if normalize:
        # Normalize the plot data
        with np.errstate(all='ignore'):
            row_sums = cm_plot.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1.0
            cm_plot = cm_plot / row_sums
        annot_data = cm_plot  # Show normalized values
        fmt = '.2f'
    else:
        # For counts, convert to integers for annotation
        annot_data = cm.astype(np.int32)  # Integer annotations
        fmt = 'd'
    
    # Wrap long class names to improve readability
    def wrap_label(s: str, width: int = 12):
        # Ensure we have a string
        label = str(s) if s is not None else ""
        return "\n".join(textwrap.wrap(label, width=width)) if label else ""
    
    wrapped_names = [wrap_label(n) for n in class_names]
    n = len(wrapped_names)
    
    # Adjust figure size based on number of classes
    figsize = (max(12, n * 0.65), max(10, n * 0.6))
    plt.figure(figsize=figsize)
    
    # Scale font size based on number of classes
    annot_fontsize = max(6, min(10, 100 // n))
    
    ax = sns.heatmap(
        cm_plot,
        annot=annot_data,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=wrapped_names,
        yticklabels=wrapped_names,
        cbar_kws={"shrink": 0.6},
        annot_kws={"size": annot_fontsize}
    )
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ax.tick_params(axis='x', labelrotation=90, labelsize=8, pad=2)
    ax.tick_params(axis='y', labelrotation=0, labelsize=8, pad=2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()

def save_metrics(metrics, output_dir):
    """
    Save validation metrics to files.
    
    Args:
        metrics: Dictionary containing validation metrics
        output_dir: Output directory path
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save overall metrics
    overall_metrics_path = output_path / "overall_metrics.txt"
    with open(overall_metrics_path, "w") as f:
        f.write("=== Patch Model Validation Results ===\n\n")
        f.write(f"mAP50-95: {metrics['map50-95']:.6f}\n")
        f.write(f"mAP50: {metrics['map50']:.6f}\n")
        f.write(f"Precision: {metrics['precision']:.6f}\n")
        f.write(f"Recall: {metrics['recall']:.6f}\n")
        f.write(f"F1-Score: {metrics['f1']:.6f}\n")
    
    # Save metrics as CSV
    csv_path = output_path / "validation_metrics.csv"
    df = pd.DataFrame([{
        'metric': 'mAP50-95',
        'value': metrics['map50-95']
    }, {
        'metric': 'mAP50', 
        'value': metrics['map50']
    }, {
        'metric': 'Precision',
        'value': metrics['precision']
    }, {
        'metric': 'Recall',
        'value': metrics['recall']
    }, {
        'metric': 'F1-Score',
        'value': metrics['f1']
    }])
    df.to_csv(csv_path, index=False)
    
    # Save per-class metrics if available
    if metrics['per_class_ap'] is not None and metrics['names'] is not None:
        per_class_path = output_path / "per_class_metrics.csv"
        per_class_data = []
        
        num_classes = len(metrics['per_class_ap'])
        
        for class_id in range(num_classes):
            class_name = metrics['names'].get(class_id, f"Class_{class_id}")
            
            ap = metrics['per_class_ap'][class_id]
            
            # Get per-class precision, recall, and F1
            precision = metrics['per_class_precision'][class_id] if metrics['per_class_precision'] is not None else None
            recall = metrics['per_class_recall'][class_id] if metrics['per_class_recall'] is not None else None
            f1 = metrics['per_class_f1'][class_id] if metrics['per_class_f1'] is not None else None
            
            per_class_data.append({
                'class_id': class_id,
                'class_name': class_name,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'ap50_95': ap
            })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_df.to_csv(per_class_path, index=False)
        
        print(f"Per-class metrics saved to: {per_class_path}")
    
    print(f"Overall metrics saved to: {overall_metrics_path}")
    print(f"Metrics CSV saved to: {csv_path}")

def print_results(metrics):
    """
    Print validation results to console.
    
    Args:
        metrics: Dictionary containing validation metrics
    """
    print("\n" + "="*50)
    print("PATCH MODEL VALIDATION RESULTS")
    print("="*50)
    print(f"mAP50-95: {metrics['map50-95']:.6f}")
    print(f"mAP50: {metrics['map50']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall: {metrics['recall']:.6f}")
    print(f"F1-Score: {metrics['f1']:.6f}")
    
    if metrics['per_class_ap'] is not None and metrics['names'] is not None:
        print(f"\nPer-class Metrics:")
        print("-" * 90)
        print(f"{'Class':<25} {'Precision':<12} {'Recall':<12} {'F1':<12} {'mAP50-95':<12}")
        print("-" * 90)
        
        for class_id in range(len(metrics['per_class_ap'])):
            class_name = metrics['names'].get(class_id, f"Class_{class_id}")
            ap = metrics['per_class_ap'][class_id]
            
            # Get per-class precision, recall, and F1
            precision = metrics['per_class_precision'][class_id] if metrics['per_class_precision'] is not None else None
            recall = metrics['per_class_recall'][class_id] if metrics['per_class_recall'] is not None else None
            f1 = metrics['per_class_f1'][class_id] if metrics['per_class_f1'] is not None else None
            
            p_str = f"{precision:.4f}" if precision is not None else "N/A"
            r_str = f"{recall:.4f}" if recall is not None else "N/A"
            f_str = f"{f1:.4f}" if f1 is not None else "N/A"
            
            print(f"{class_name:<25} {p_str:<12} {r_str:<12} {f_str:<12} {ap:.4f}")
        
        print("-" * 90)
    
    print("="*50)

def _derive_label_path_from_image(img_path: Path) -> Path:
    """Derive YOLO label .txt path from an image path by swapping folders and extension."""
    p = Path(img_path)
    # Swap 'images' with 'labels' in parents
    parts = list(p.parts)
    try:
        idx = parts.index('images')
        parts[idx] = 'labels'
        lbl_dir_swap = Path(*parts[:-1])
    except ValueError:
        # If 'images' not present, assume labels folder is sibling named 'labels'
        lbl_dir_swap = p.parent.parent / 'labels' / p.parent.name
    # Change extension to .txt
    return (lbl_dir_swap / p.stem).with_suffix('.txt')


def _plot_val_label_frequency(data_yaml_path: str, output_dir: str):
    """Compute and plot frequency of labels in the validation split."""
    out_dir = Path(output_dir) / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    # Load data.yaml
    with open(data_yaml_path, 'r') as f:
        data_cfg = yaml.safe_load(f)
    names_map = data_cfg.get('names', {})
    num_classes = data_cfg.get('nc', len(names_map) if isinstance(names_map, dict) else 0)
    # Resolve validation list (txt with image paths)
    val_entry = data_cfg.get('val')
    if val_entry is None:
        print("No 'val' entry in data.yaml; skipping label frequency plot.")
        return
    val_path = Path(val_entry)
    if not val_path.exists():
        # If relative to data.yaml parent
        val_path = Path(data_yaml_path).parent / val_entry
    if not val_path.exists():
        print(f"Validation split file not found: {val_entry}; skipping label frequency plot.")
        return
    # Gather label files
    image_paths = []
    with open(val_path, 'r') as f:
        for line in f:
            s = line.strip()
            if s:
                image_paths.append(Path(s))
    label_paths = [_derive_label_path_from_image(p) for p in image_paths]
    # Count class ids
    counts = Counter()
    for lp in label_paths:
        if not lp.exists():
            continue
        try:
            with open(lp, 'r') as f:
                for ln in f:
                    parts = ln.strip().split()
                    if not parts:
                        continue
                    cid = int(float(parts[0]))
                    counts[cid] += 1
        except Exception:
            continue
    # Build arrays
    class_ids = list(range(num_classes))
    class_names = [names_map.get(i, str(i)) if isinstance(names_map, dict) else str(i) for i in class_ids]
    freq = [counts.get(i, 0) for i in class_ids]
    # Plot
    plt.figure(figsize=(max(14, num_classes * 0.7), 6))
    ax = sns.barplot(x=class_names, y=freq, color='#4C78A8')
    ax.set_title('Validation Label Frequency', pad=8)
    ax.set_xlabel('Class')
    ax.set_ylabel('Box Count')
    ax.tick_params(axis='x', rotation=60, labelsize=8)
    for i, v in enumerate(freq):
        if v > 0:
            ax.text(i, v, str(v), ha='center', va='bottom', fontsize=7, rotation=0)
    plt.tight_layout()
    save_path = out_dir / 'val_label_frequency.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Validation label frequency plot saved to: {save_path}")

def main():
    """
    Main function to load model and validate with default parameters.
    """
    print("Starting patch model validation...")
    print(f"Model path: {BEST_MODEL_PATH}")
    print(f"Data YAML: {DATA_YAML_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    try:
        # Load model and validate
        results = load_model_and_validate()
        
        # Extract metrics
        metrics = extract_metrics(results)
        
        # Print results
        print_results(metrics)
        
        # Save metrics
        save_metrics(metrics, OUTPUT_DIR)
        
        # Confusion matrices (counts and normalized)
        print("Generating confusion matrices...")
        cm = _extract_confusion_matrix(results)
        
        # Load names list in index order from dataset
        with open(DATA_YAML_PATH, 'r') as f:
            data_config = yaml.safe_load(f)
        names_dict = data_config.get('names', {})
        nc = data_config.get('nc', len(names_dict) if isinstance(names_dict, dict) else 0)
        
        # Build class names list based on dataset nc
        # Handle both string keys ('0', '1') and int keys (0, 1)
        class_names = []
        for i in range(nc):
            # Try int key first, then string key
            name = names_dict.get(i) or names_dict.get(str(i)) or f'Class_{i}'
            class_names.append(name)
        
        print(f"Class names for confusion matrix: {class_names}")
        
        plots_dir = Path(OUTPUT_DIR) / 'plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        if cm is not None:
            cm_rows, cm_cols = cm.shape
            print(f"Confusion matrix shape: {cm.shape}, Dataset classes: {nc}")
            
            # Handle shape mismatch - slice to dataset nc if needed
            if cm_rows > nc or cm_cols > nc:
                print(f"⚠️  Confusion matrix ({cm_rows}×{cm_cols}) larger than dataset nc={nc}")
                print(f"   Slicing to {nc}×{nc} to match dataset classes")
                cm = cm[:nc, :nc]
            elif cm_rows < nc or cm_cols < nc:
                print(f"⚠️  Confusion matrix ({cm_rows}×{cm_cols}) smaller than dataset nc={nc}")
                print(f"   Padding to {nc}×{nc} to match dataset classes")
                cm_padded = np.zeros((nc, nc), dtype=cm.dtype)
                cm_padded[:cm_rows, :cm_cols] = cm
                cm = cm_padded
            
            # Now cm should match nc
            if cm.shape[0] == nc and cm.shape[1] == nc:
                _plot_confusion_matrix(cm, class_names, plots_dir / 'confusion_matrix_without_bg_counts.png', normalize=False)
                _plot_confusion_matrix(cm, class_names, plots_dir / 'confusion_matrix_without_bg_normalized.png', normalize=True)
                print("✓ Custom confusion matrices (with background) saved to:")
                print(f"  {plots_dir / 'confusion_matrix_without_bg_counts.png'}")
                print(f"  {plots_dir / 'confusion_matrix_without_bg_normalized.png'}")
            else:
                print(f"⚠️  Confusion matrix shape {cm.shape} still doesn't match nc={nc} after adjustment")
                print(f"   Skipping custom confusion matrix plotting")
        else:
            print("⚠️  Confusion matrix not available from results object")
            print("   Relying on Ultralytics built-in plots (may have incorrect class count)")
        
        # Note about built-in plots
        print("\nUltralytics built-in plots (with background class) also available at:")
        print(f"  {plots_dir / 'confusion_matrix.png'}")
        print(f"  {plots_dir / 'confusion_matrix_normalized.png'}")
        
        # Validation label frequency plot
        _plot_val_label_frequency(DATA_YAML_PATH, OUTPUT_DIR)
        
        print(f"\nValidation completed successfully!")
        print(f"All results saved to: {OUTPUT_DIR}/")
        print(f"Validation plots and predictions saved to: {OUTPUT_DIR}/plots/")
        
    except Exception as e:
        print(f"Error during validation: {e}")
        raise

if __name__ == "__main__":
    main()
