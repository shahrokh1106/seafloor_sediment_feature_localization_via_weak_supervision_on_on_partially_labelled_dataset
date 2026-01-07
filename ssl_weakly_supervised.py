#!/usr/bin/env python3
"""
Weakly Supervised Learning Pipeline using Teacher-Student with Pseudo-Labeling

This script implements a refined SSL pipeline for marine sediment detection with:
- Automatic confidence threshold tuning per iteration
- Ground truth preservation + filtered pseudo-labels
- NMS and duplicate removal
- Progressive learning with teacher updates

Author: Refined from ts_train_yolo.py
"""

import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
import logging
from tqdm import tqdm

# Import our refined modules
from tune_confidence import tune_conf_threshold
from pseudo_label_generator import PseudoLabelGenerator
from dataset_builder import SnapshotDatasetBuilder

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)


class WeaklySupervisedPipeline:
    """Main pipeline for weakly supervised learning with pseudo-labeling."""
    
    def __init__(self,
                 initial_teacher_path: str,
                 dataset_path: str = "detector_dataset",
                 output_dir: str = "ssl_results",
                 max_iterations: int = 30,
                 device: str = "auto",
                 imgsz: int = 960,
                 batch_size: int = 8):
        """
        Initialize the weakly supervised learning pipeline.
        
        Args:
            initial_teacher_path: Path to initial teacher model (e.g., patch-trained model)
            dataset_path: Path to dataset directory
            output_dir: Output directory for all results
            max_iterations: Maximum number of SSL iterations
            device: Device for training/inference
            imgsz: Image size
            batch_size: Batch size for training
        """
        self.initial_teacher_path = Path(initial_teacher_path)
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.max_iterations = max_iterations
        self.imgsz = imgsz
        self.batch_size = batch_size
        
        # Handle device selection
        if device == "auto":
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto device selection: {self.device}")
        else:
            self.device = device
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data paths
        self.data_yaml = self.dataset_path / "data.yaml"
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"data.yaml not found: {self.data_yaml}")
        
        # Load split paths
        self.train_paths, self.val_paths, self.test_paths = self._load_splits()
        
        # Initialize teacher
        self.current_teacher_path = self.initial_teacher_path
        self.best_f1 = 0.0
        self.best_model_path = self.initial_teacher_path
        
        # Performance tracking
        self.iteration_history = []
        
        logger.info("="*60)
        logger.info("WEAKLY SUPERVISED LEARNING PIPELINE INITIALIZED")
        logger.info("="*60)
        logger.info(f"Initial teacher: {self.initial_teacher_path}")
        logger.info(f"Dataset: {self.dataset_path}")
        logger.info(f"Train images: {len(self.train_paths)}")
        logger.info(f"Val images: {len(self.val_paths)}")
        logger.info(f"Test images: {len(self.test_paths)}")
        logger.info(f"Max iterations: {self.max_iterations}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Image size: {self.imgsz}")
    
    def _load_splits(self) -> Tuple[List[str], List[str], List[str]]:
        """Load train/val/test splits."""
        splits = {}
        for split_name in ['train', 'val', 'test']:
            split_file = self.dataset_path / f"{split_name}.txt"
            if split_file.exists():
                with open(split_file, 'r') as f:
                    splits[split_name] = [line.strip() for line in f if line.strip()]
            else:
                raise FileNotFoundError(f"Split file not found: {split_file}")
        
        return splits['train'], splits['val'], splits['test']
    
    def tune_confidence_for_iteration(self, iteration: int) -> float:
        """
        Tune confidence threshold for current teacher model.
        
        Args:
            iteration: Current iteration number
            
        Returns:
            best_conf: Optimal confidence threshold
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}: CONFIDENCE TUNING")
        logger.info(f"{'='*60}")
        
        output_path = self.output_dir / f"conf_tuning_iter{iteration}.json"
        
        # Check if tuning results already exist
        if output_path.exists():
            logger.info(f"Loading existing confidence tuning results from {output_path}")
            import json
            with open(output_path, 'r') as f:
                tuning_data = json.load(f)
            best_conf = tuning_data['best_conf']
            logger.info(f"✓ Loaded best confidence threshold: {best_conf:.3f}")
            return best_conf
        
        # Define confidence range (start from very low for partially labeled data)
        conf_values = np.round(np.linspace(0.2, 0.7, 20), 3)
        
        # Run tuning
        best_conf, results = tune_conf_threshold(
            model_or_path=str(self.current_teacher_path),
            data_yaml=str(self.data_yaml),
            conf_values=conf_values,
            metric="f1", 
            imgsz=self.imgsz,
            iou=0.7,  # High IoU for NMS to keep more boxes initially
            device=self.device,
            batch=self.batch_size,
            workers=0,
            verbose=False,
            out_path=str(output_path),
            save_plot=True
        )
        
        logger.info(f"✓ Best confidence threshold: {best_conf:.3f}")
        
        return best_conf
    
    def generate_pseudo_labels(self, iteration: int, best_conf: float):
        """
        Generate pseudo-labels using teacher model.
        
        Args:
            iteration: Current iteration number
            best_conf: Confidence threshold to use
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}: PSEUDO-LABEL GENERATION")
        logger.info(f"{'='*60}")
        
        output_labels_dir = self.output_dir / f"pseudo_labels_iter{iteration}"
        
        # Check if pseudo-labels already exist
        if output_labels_dir.exists() and any(output_labels_dir.glob("*.txt")):
            logger.info(f"Loading existing pseudo-labels from {output_labels_dir}")
            
            # Count existing labels for stats
            label_files = list(output_labels_dir.glob("*.txt"))
            stats = {
                'gt_boxes': 0,
                'pseudo_boxes': 0,
                'images_with_pseudo': 0,
                'total_boxes': 0
            }
            
            for label_file in label_files:
                with open(label_file, 'r') as f:
                    lines = f.readlines()
                    stats['total_boxes'] += len(lines)
            
            logger.info(f"✓ Loaded existing pseudo-labels:")
            logger.info(f"  Total label files: {len(label_files)}")
            logger.info(f"  Total boxes: {stats['total_boxes']}")
            
            return output_labels_dir, stats
        
        output_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize pseudo-label generator
        generator = PseudoLabelGenerator(
            teacher_model_path=str(self.current_teacher_path),
            dataset_path=str(self.dataset_path),
            imgsz=self.imgsz,
            device=self.device
        )
        
        # Generate merged labels (GT + filtered pseudo-labels)
        stats = generator.generate_merged_labels(
            image_paths=self.train_paths,
            output_dir=output_labels_dir,
            best_conf=best_conf,
            topk_per_class=min(5 + iteration, 15+iteration), 
            dup_iou=0.25,  # Lower than 0.5 - allow learning from partial overlaps
            iou_nms=0.5,
            consistency_check=True,  # Enable multi-augmentation consistency filtering
            consistency_iou=0.7  # High IoU threshold for consistency (stable predictions only)
        )
        
        logger.info(f"✓ Pseudo-labels generated:")
        logger.info(f"  GT boxes preserved: {stats['gt_boxes']}")
        logger.info(f"  Pseudo-labels added: {stats['pseudo_boxes']}")
        logger.info(f"  Images with new labels: {stats['images_with_pseudo']}")
        logger.info(f"  Total merged boxes: {stats['total_boxes']}")
        logger.info(f"  Filtered by consistency: {stats['filtered_by_consistency']}")
        
        return output_labels_dir, stats
    
    def build_snapshot_dataset(self, iteration: int, merged_labels_dir: Path) -> Path:
        """
        Build snapshot dataset for current iteration.
        
        Args:
            iteration: Current iteration number
            merged_labels_dir: Directory with merged labels
            
        Returns:
            snapshot_yaml: Path to snapshot data.yaml
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}: BUILDING SNAPSHOT DATASET")
        logger.info(f"{'='*60}")
        
        snapshot_dir = self.output_dir / f"snapshot_iter{iteration}"
        snapshot_yaml = snapshot_dir / "data.yaml"
        
        # Check if snapshot already exists
        if snapshot_yaml.exists():
            logger.info(f"Loading existing snapshot dataset from {snapshot_yaml}")
            return snapshot_yaml
        
        builder = SnapshotDatasetBuilder(
            base_data_yaml=str(self.data_yaml),
            output_root=self.output_dir
        )
        
        snapshot_yaml = builder.build_snapshot(
            iteration=iteration,
            train_paths=self.train_paths,
            val_paths=self.val_paths,
            test_paths=self.test_paths,
            merged_labels_dir=merged_labels_dir
        )
        
        logger.info(f"✓ Snapshot dataset created: {snapshot_yaml}")
        
        return snapshot_yaml
    
    def train_student(self, iteration: int, snapshot_yaml: Path) -> Path:
        """
        Train student model with class imbalance handling via focal loss.
        
        Args:
            iteration: Current iteration number
            snapshot_yaml: Path to snapshot data.yaml
            
        Returns:
            best_model_path: Path to best student model weights
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}: TRAINING STUDENT MODEL")
        logger.info(f"{'='*60}")
        
        # VarifocalLoss enabled in training loss - no per-class weighting needed
        
        # Initialize student from current teacher
        logger.info(f"Initializing student model from: {self.current_teacher_path}")
        logger.info(f"  Teacher path exists: {Path(self.current_teacher_path).exists()}")
        logger.info(f"  Absolute path: {Path(self.current_teacher_path).absolute()}")
        
        student_model = YOLO(str(self.current_teacher_path))
        
        # Training parameters (progressive epochs)
        base_epochs = 50
        epochs = min(base_epochs + iteration * 10, 100)
    
        
        logger.info(f"Training for {epochs} epochs with class-weighted loss...")
        
        results = student_model.train(
            data=str(snapshot_yaml),
            imgsz=self.imgsz,
            rect=False,
            multi_scale=True,
            batch=self.batch_size,
            device=self.device,
            workers=0,
            epochs=epochs,
            
            # Conservative augmentation for SSL
            degrees=5.0,
            translate=0.1,
            scale=0.5,
            shear=1.0,
            perspective=0.0,
            fliplr=0.5,
            flipud=0.0,
            hsv_h=0.015,
            hsv_s=0.70,
            hsv_v=0.40,
            mosaic=0.5,
            close_mosaic=10,
            
            # Loss weights (VarifocalLoss used internally)
            box=5.0,
            cls=1.0,
            dfl=1.5,
            
            # Optimizer (auto)
            optimizer='auto',
            lr0=0.01,
            lrf=0.01,
            
            # Output settings
            project=str(self.output_dir),
            name=f"student_iter{iteration}",
            patience=30,
            exist_ok=True,
            resume=False,
            plots=True,
            save_period=5
        )
        
        save_dir = Path(results.save_dir)
        best_model_path = save_dir / "weights" / "best.pt"
        
        if not best_model_path.exists():
            raise FileNotFoundError(f"Student best weights not found at {best_model_path}")
        
        logger.info(f"✓ Student training complete: {best_model_path}")
        
        return best_model_path
    
    def evaluate_models(self, iteration: int, student_path: Path) -> Dict:
        """
        Evaluate both teacher and student models.
        
        Args:
            iteration: Current iteration number
            student_path: Path to student model
            
        Returns:
            evaluation_results: Dictionary with metrics
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}: MODEL EVALUATION")
        logger.info(f"{'='*60}")
        
        def get_metrics(model_path, split='val'):
            """Get validation metrics."""
            logger.info(f"Loading model for evaluation: {model_path}")
            model = YOLO(str(model_path))
            res = model.val(
                data=str(self.data_yaml),
                imgsz=self.imgsz,
                device=self.device,
                split=split,
                verbose=False,
                save=False,
                plots=False,
                save_json=False
            )
            
            mp = float(getattr(res.box, 'mp', 0.0))
            mr = float(getattr(res.box, 'mr', 0.0))
            f1 = (2 * mp * mr / (mp + mr)) if (mp + mr) > 0 else 0.0
            
            return {
                'map50-95': float(res.box.map),
                'map50': float(res.box.map50),
                'precision': mp,
                'recall': mr,
                'f1': f1
            }
        
        # Evaluate both models
        teacher_metrics = get_metrics(self.current_teacher_path)
        student_metrics = get_metrics(student_path)
        
        results = {
            'iteration': iteration,
            'teacher': {
                'path': str(self.current_teacher_path),
                'metrics': teacher_metrics
            },
            'student': {
                'path': str(student_path),
                'metrics': student_metrics
            }
        }
        
        logger.info("Teacher metrics:")
        logger.info(f"  mAP50-95: {teacher_metrics['map50-95']:.4f}")
        logger.info(f"  mAP50: {teacher_metrics['map50']:.4f}")
        logger.info(f"  F1: {teacher_metrics['f1']:.4f}")
        
        logger.info("Student metrics:")
        logger.info(f"  mAP50-95: {student_metrics['map50-95']:.4f}")
        logger.info(f"  mAP50: {student_metrics['map50']:.4f}")
        logger.info(f"  F1: {student_metrics['f1']:.4f}")
        
        return results
    
    def update_teacher(self, iteration: int, evaluation_results: Dict) -> bool:
        """
        Update teacher model if student is better.
        
        Args:
            iteration: Current iteration number
            evaluation_results: Evaluation results dictionary
            
        Returns:
            updated: True if teacher was updated
        """
        student_f1 = evaluation_results['student']['metrics']['f1']
        student_path = evaluation_results['student']['path']
        
        # Check if student is better than best so far
        if student_f1 > self.best_f1:
            logger.info(f"\n✓ Student F1 ({student_f1:.4f}) > Best F1 ({self.best_f1:.4f})")
            logger.info("  Updating teacher model!")
            
            self.best_f1 = student_f1
            self.best_model_path = Path(student_path)
            self.current_teacher_path = self.best_model_path
            
            return True
        else:
            logger.info(f"\n✗ Student F1 ({student_f1:.4f}) <= Best F1 ({self.best_f1:.4f})")
            logger.info("  Keeping current teacher")
            
            return False
    
    def save_iteration_log(self, iteration_data: Dict):
        """Save iteration results to log file."""
        log_file = self.output_dir / "training_log.json"
        
        # Load existing log
        if log_file.exists():
            with open(log_file, 'r') as f:
                history = json.load(f)
        else:
            history = []
        
        history.append(iteration_data)
        
        # Save updated log
        with open(log_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        # Also save CSV summary
        csv_file = self.output_dir / "training_summary.csv"
        is_new = not csv_file.exists()
        
        with open(csv_file, 'a') as f:
            if is_new:
                f.write("iteration,model,map50_95,map50,precision,recall,f1\n")
            
            teacher_m = iteration_data['teacher']['metrics']
            student_m = iteration_data['student']['metrics']
            
            f.write(f"{iteration_data['iteration']},teacher,"
                   f"{teacher_m['map50-95']:.6f},{teacher_m['map50']:.6f},"
                   f"{teacher_m['precision']:.6f},{teacher_m['recall']:.6f},"
                   f"{teacher_m['f1']:.6f}\n")
            
            f.write(f"{iteration_data['iteration']},student,"
                   f"{student_m['map50-95']:.6f},{student_m['map50']:.6f},"
                   f"{student_m['precision']:.6f},{student_m['recall']:.6f},"
                   f"{student_m['f1']:.6f}\n")
    
    def run(self):
        """Run the complete SSL pipeline."""
        logger.info("\n" + "="*60)
        logger.info("STARTING WEAKLY SUPERVISED LEARNING PIPELINE")
        logger.info("="*60)
        
        for iteration in range(1, self.max_iterations + 1):
            logger.info(f"\n{'#'*60}")
            logger.info(f"# ITERATION {iteration}/{self.max_iterations}")
            logger.info(f"{'#'*60}")
            
            try:
                # Check if current iteration already completed
                current_iter_path = self.output_dir / f"student_iter{iteration}" / "weights" / "best.pt"
                if current_iter_path.exists():
                    logger.info(f"✓ Iteration {iteration} already completed (best.pt exists)")
                    logger.info(f"  Skipping to next iteration...")
                    # Load existing results if available
                    continue
                
                # Check if previous iteration exists (prerequisite)
                if iteration > 1:
                    prev_iter_path = self.output_dir / f"student_iter{iteration-1}" / "weights" / "best.pt"
                    if not prev_iter_path.exists():
                        logger.warning(f"⚠ Iteration {iteration-1} not complete (best.pt missing)")
                        logger.warning(f"  Cannot train iteration {iteration} without previous teacher")
                        logger.warning(f"  Skipping iteration {iteration}...")
                        continue
                    else:
                        logger.info(f"✓ Previous iteration {iteration-1} completed, can proceed")
                
                # Step 1: Tune confidence threshold
                best_conf = self.tune_confidence_for_iteration(iteration)
                
                # Step 2: Generate pseudo-labels
                pseudo_labels_dir, pseudo_stats = self.generate_pseudo_labels(iteration, best_conf)
                
                # Step 3: Build snapshot dataset
                snapshot_yaml = self.build_snapshot_dataset(iteration, pseudo_labels_dir)
                
                # Step 4: Train student
                student_path = self.train_student(iteration, snapshot_yaml)
                
                # Step 5: Evaluate models
                eval_results = self.evaluate_models(iteration, student_path)
                
                # Step 6: Update teacher if student is better
                teacher_updated = self.update_teacher(iteration, eval_results)
                
                # Step 7: Save iteration data
                iteration_data = {
                    **eval_results,
                    'best_conf': best_conf,
                    'pseudo_stats': pseudo_stats,
                    'teacher_updated': teacher_updated,
                    'best_f1_so_far': self.best_f1,
                    'best_model_path': str(self.best_model_path)
                }
                
                self.save_iteration_log(iteration_data)
                self.iteration_history.append(iteration_data)
                
                logger.info(f"\n✓ Iteration {iteration} complete!")
                
            except Exception as e:
                logger.error(f"\n✗ Iteration {iteration} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Final summary
        self.print_final_summary()
    
    def print_final_summary(self):
        """Print final training summary."""
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE - FINAL SUMMARY")
        logger.info("="*60)
        
        logger.info(f"Total iterations: {len(self.iteration_history)}")
        logger.info(f"Best F1 score: {self.best_f1:.4f}")
        logger.info(f"Best model: {self.best_model_path}")
        
        logger.info("\nIteration summary:")
        for iter_data in self.iteration_history:
            student_f1 = iter_data['student']['metrics']['f1']
            updated = "✓ UPDATED" if iter_data['teacher_updated'] else "✗ kept"
            logger.info(f"  Iter {iter_data['iteration']}: F1={student_f1:.4f} {updated}")
        
        logger.info(f"\nResults saved to: {self.output_dir}")
        logger.info(f"  - training_log.json (detailed results)")
        logger.info(f"  - training_summary.csv (metrics summary)")
        logger.info(f"  - Best model: {self.best_model_path}")


def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Weakly Supervised Learning Pipeline for Marine Sediment Detection"
    )
    parser.add_argument('--teacher', type=str, required=True,
                       help='Path to initial teacher model (e.g., patch-trained model)')
    parser.add_argument('--dataset', type=str, default='detector_dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='ssl_results',
                       help='Output directory for results')
    parser.add_argument('--iterations', type=int, default=30,
                       help='Maximum number of SSL iterations')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for training (auto, cpu, cuda, 0, 1, etc.)')
    parser.add_argument('--imgsz', type=int, default=960,
                       help='Image size')
    parser.add_argument('--batch', type=int, default=8,
                       help='Batch size')
    
    args = parser.parse_args()
    
    # Initialize and run pipeline
    pipeline = WeaklySupervisedPipeline(
        initial_teacher_path=args.teacher,
        dataset_path=args.dataset,
        output_dir=args.output,
        max_iterations=args.iterations,
        device=args.device,
        imgsz=args.imgsz,
        batch_size=args.batch
    )
    
    pipeline.run()


if __name__ == "__main__":
    main()
