from pathlib import Path
import numpy as np
from ultralytics import YOLO
import os 
import torch
import random 

SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   

# Configuration for full dataset training with VarifocalLoss
DATASET_PATH = "detector_dataset_simple"
DATA_YAML_PATH = DATASET_PATH+"/data.yaml"
OUTDIR = "training_results_simple"
MODEL_PATH = "yolo11s.pt"  
DEVICE = 0
EPOCHS = 200  # From args.yaml
IMGSZ = 960
BATCHSIZE = 8
MULTISCALE = True

# Augmentation settings from full_initial/args.yaml
AUGMENTATION_CONFIG = {
    # Geometric augmentations (from args.yaml)
    'degrees': 10.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 2.0,
    'perspective': 0.0,
    
    # Flipping
    'fliplr': 0.5,
    'flipud': 0.0,
    
    # Color augmentations
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    
    # Advanced augmentations (from args.yaml)
    'mosaic': 0.5,
    'mixup': 0.0,
    'copy_paste': 0.05,
    'erasing': 0.4,
    
    # Training parameters (from args.yaml)
    'lr0': 0.01,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    
    # Loss weights (from args.yaml)
    'box': 7.5,
    'cls': 1.0,
    'dfl': 1.5,
}

def train_yolo(data_yaml_path, model_path, out_dir, imgsz, epochs, batchsize, device, multi_scale):
    """
    Train YOLO model with comprehensive augmentations for marine sediment detection.
    
    Args:
        data_yaml_path: Path to data.yaml file
        model_path: Path to pre-trained model
        out_dir: Output directory for training results
        imgsz: Image size for training
        epochs: Number of training epochs
        batchsize: Batch size
        device: Device to use (0 for GPU, 'cpu' for CPU)
        multi_scale: Whether to use multi-scale training
    
    Returns:
        Path to best model weights
    """
    model = YOLO(model_path)
    
    # Comprehensive augmentation parameters for marine sediment detection
    results = model.train(
        data=data_yaml_path,
        imgsz=imgsz,
        rect=False,  
        
        # Multi-scale training
        multi_scale=multi_scale,
        
        # Training parameters
        batch=batchsize,
        device=device,
        workers=0,  
        epochs=epochs,
        
        # Geometric augmentations
        degrees=AUGMENTATION_CONFIG['degrees'],
        translate=AUGMENTATION_CONFIG['translate'],
        scale=AUGMENTATION_CONFIG['scale'],
        shear=AUGMENTATION_CONFIG['shear'],
        perspective=AUGMENTATION_CONFIG['perspective'],
        
        # Flipping augmentations
        fliplr=AUGMENTATION_CONFIG['fliplr'],
        flipud=AUGMENTATION_CONFIG['flipud'],
        
        # Color space augmentations (important for underwater images)
        hsv_h=AUGMENTATION_CONFIG['hsv_h'],
        hsv_s=AUGMENTATION_CONFIG['hsv_s'],
        hsv_v=AUGMENTATION_CONFIG['hsv_v'],
        
        # Advanced augmentations
        mosaic=AUGMENTATION_CONFIG['mosaic'],
        mixup=AUGMENTATION_CONFIG['mixup'],
        
        # Mosaic settings
        close_mosaic=10,     # Disable mosaic in last 10 epochs
        
        # Learning rate and optimization
        lr0=AUGMENTATION_CONFIG['lr0'],
        lrf=AUGMENTATION_CONFIG['lrf'],
        momentum=AUGMENTATION_CONFIG['momentum'],
        weight_decay=AUGMENTATION_CONFIG['weight_decay'],
        warmup_epochs=AUGMENTATION_CONFIG['warmup_epochs'],
        warmup_momentum=0.8, # Warmup momentum
        warmup_bias_lr=0.1,  # Warmup bias learning rate
        
        # Loss function weights
        box=AUGMENTATION_CONFIG['box'],
        cls=AUGMENTATION_CONFIG['cls'],
        dfl=AUGMENTATION_CONFIG['dfl'],
        
        # Training settings
        project=str(out_dir),
        name="full_initial_bce",
        patience=70,         # Early stopping patience (from args.yaml)
        exist_ok=True,
        resume=False,
        
        # Additional settings (from args.yaml)
        amp=True,
        fraction=1.0,
        profile=False,
        freeze=None,
        cache=False,
        plots=True,
        val=True,
        seed=SEED,
        
        # Additional augmentations (from args.yaml)
        copy_paste=AUGMENTATION_CONFIG['copy_paste'],
        erasing=AUGMENTATION_CONFIG['erasing'],
        auto_augment='randaugment',
        augment=True,
        
        # Validation settings (from args.yaml)
        iou=0.7
    )
    
    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        raise FileNotFoundError(f"Best weights not found at {best}")
    return best

def validate_dataset(data_yaml_path):
    """
    Validate the dataset before training.
    
    Args:
        data_yaml_path: Path to data.yaml file
    
    Returns:
        dict: Dataset statistics
    """
    import yaml
    from pathlib import Path
    
    print("Validating the dataset...")
    
    # Load data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    dataset_path = Path(data_config['path'])
    
    # Check if files exist
    train_file = dataset_path / data_config['train']
    val_file = dataset_path / data_config['val']
    test_file = dataset_path / data_config['test']
    
    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not val_file.exists():
        raise FileNotFoundError(f"Validation file not found: {val_file}")
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    
    # Count images in each split
    with open(train_file, 'r') as f:
        train_images = [line.strip() for line in f if line.strip()]
    with open(val_file, 'r') as f:
        val_images = [line.strip() for line in f if line.strip()]
    with open(test_file, 'r') as f:
        test_images = [line.strip() for line in f if line.strip()]
    
    # Check if images exist
    missing_images = []
    for img_path in train_images[:10]:  # Check first 10 images
        if not Path(img_path).exists():
            missing_images.append(img_path)
    
    if missing_images:
        print(f"Warning: {len(missing_images)} images not found in first 10 train images")
    
    stats = {
        'num_classes': data_config['nc'],
        'class_names': data_config['names'],
        'train_images': len(train_images),
        'val_images': len(val_images),
        'test_images': len(test_images),
        'total_images': len(train_images) + len(val_images) + len(test_images)
    }
    
    print(f"Dataset validation completed:")
    print(f"  Classes: {stats['num_classes']}")
    print(f"  Train images: {stats['train_images']}")
    print(f"  Validation images: {stats['val_images']}")
    print(f"  Test images: {stats['test_images']}")
    print(f"  Total images: {stats['total_images']}")
    
    return stats

def get_val_metrics(model_or_path, data_yaml, imgsz=960, device=0, split='val'):
    """
    Return:
      {
        'map50-95': float,
        'map50': float,
        'precision': float,   # mean over classes
        'recall': float,      # mean over classes
        'f1': float,          # 2PR/(P+R)
        'per_class_ap': list|None,  # AP50-95 per class
        'names': dict|None    # class id -> name
      }
    """
    model = model_or_path if isinstance(model_or_path, YOLO) else YOLO(str(model_or_path))
    res = model.val(data=str(data_yaml), imgsz=imgsz, device=device, split=split,
                    verbose=False, save=False, plots=False, save_json=False)

    m5095 = float(res.box.map)       # mAP50-95 (mean over classes)
    m50   = float(res.box.map50)     # mAP50
    mp    = float(getattr(res.box, 'mp', float('nan')))  # mean precision
    mr    = float(getattr(res.box, 'mr', float('nan')))  # mean recall
    f1    = (2*mp*mr/(mp+mr)) if (mp == mp and mr == mr and (mp+mr) > 0) else float('nan')

    per_class_ap = getattr(res.box, 'maps', None)  # list of AP50-95 per class (len = nc)
    names = getattr(res, 'names', None)           # dict: id -> name

    return {
        'map50-95': m5095,
        'map50': m50,
        'precision': mp,
        'recall': mr,
        'f1': f1,
        'per_class_ap': per_class_ap,
        'names': names
    }

if __name__ == "__main__":
    print("Starting YOLO training on full dataset with VarifocalLoss...")
    print(f"Dataset: {DATASET_PATH}")
    print(f"Data YAML: {DATA_YAML_PATH}")
    print(f"Model: {MODEL_PATH}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCHSIZE}")
    print(f"Image size: {IMGSZ}")
    print(f"Multi-scale: {MULTISCALE}")
    print(f"Output: {OUTDIR}/full_initial_bce")
    
    # Validate dataset before training
    try:
        dataset_stats = validate_dataset(DATA_YAML_PATH)
    except Exception as e:
        print(f"Dataset validation failed: {e}")
        exit(1)
    
    # Train the model
    print("\nStarting training with VarifocalLoss...")
    best_model_path = train_yolo(DATA_YAML_PATH, MODEL_PATH, OUTDIR, IMGSZ, EPOCHS, BATCHSIZE, DEVICE, MULTISCALE)
    print(f"Training completed! Best model saved at: {best_model_path}")
    
    # Evaluate the model
    print("Evaluating model performance...")
    metrics = get_val_metrics(best_model_path, DATA_YAML_PATH, imgsz=IMGSZ, device=DEVICE, split='val')
    
    # Print metrics
    print("\n=== Validation Results ===")
    print(f"mAP50-95: {metrics['map50-95']:.4f}")
    print(f"mAP50: {metrics['map50']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save metrics to log file
    log_path = Path(OUTDIR) / "full_initial_bce" / "val_metrics_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not log_path.exists()
    
    with open(log_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write("model,map50_95,map50,precision,recall,f1\n")
        f.write(f"full_initial_bce,{metrics['map50-95']:.6f},{metrics['map50']:.6f},"
                f"{metrics['precision']:.6f},{metrics['recall']:.6f},{metrics['f1']:.6f}\n")
    
    # Save per-class metrics
    pc = metrics.get('per_class_ap')
    names = metrics.get('names') or {}
    if pc is not None:
        per_class_path = Path(OUTDIR) / "full_initial_bce" / "val_per_class_initial.csv"
        with open(per_class_path, "w", encoding="utf-8") as g:
            g.write("class_id,class_name,ap50_95\n")
            for cid, ap in enumerate(pc):
                cname = names.get(cid, str(cid))
                g.write(f"{cid},{cname},{ap:.6f}\n")
        
        print(f"\nPer-class metrics saved to: {per_class_path}")
    
    print(f"\nAll results saved to: {OUTDIR}/full_initial_bce/")
    print("Training completed successfully!")