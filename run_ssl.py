#!/usr/bin/env python3
"""
Simple launcher for SSL Weakly Supervised Pipeline

Usage:
    python run_ssl.py
"""

import sys
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run SSL pipeline with default settings."""
    
    # Configuration
    TEACHER_MODEL = "training_results_simple/full_initial_bce/weights/best.pt"
    DATASET = "detector_dataset_simple"
    OUTPUT = "ssl_simple_results_bce"
    ITERATIONS = 30
    
    # Auto-detect device
    import torch
    DEVICE = 0 if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {DEVICE} ({'GPU' if DEVICE == 0 else 'CPU'})")
    
    IMGSZ = 960
    BATCH = 8
    
    logger.info("="*60)
    logger.info("SSL WEAKLY SUPERVISED PIPELINE LAUNCHER")
    logger.info("="*60)
    
    # Verify prerequisites
    if not Path(TEACHER_MODEL).exists():
        logger.error(f"Teacher model not found: {TEACHER_MODEL}")
        logger.error("Please complete full initial training with VarifocalLoss first:")
        logger.error("  python initial_training.py")
        logger.error(f"  Expected: {TEACHER_MODEL}")
        return False
    
    if not Path(DATASET).exists():
        logger.error(f"Dataset not found: {DATASET}")
        return False
    
    logger.info(f"Teacher model: {TEACHER_MODEL}")
    logger.info(f"Dataset: {DATASET}")
    logger.info(f"Output: {OUTPUT}")
    logger.info(f"Iterations: {ITERATIONS}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Image size: {IMGSZ}")
    logger.info(f"Batch size: {BATCH}")
    
    # Import and run pipeline
    try:
        from ssl_weakly_supervised import WeaklySupervisedPipeline
        
        logger.info("\nInitializing pipeline...")
        pipeline = WeaklySupervisedPipeline(
            initial_teacher_path=TEACHER_MODEL,
            dataset_path=DATASET,
            output_dir=OUTPUT,
            max_iterations=ITERATIONS,
            device=DEVICE,
            imgsz=IMGSZ,
            batch_size=BATCH
        )
        
        logger.info("Starting SSL training...")
        pipeline.run()
        
        logger.info("\n" + "="*60)
        logger.info("SSL PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Best F1 score: {pipeline.best_f1:.4f}")
        logger.info(f"Best model: {pipeline.best_model_path}")
        logger.info(f"Results directory: {OUTPUT}")
        
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

