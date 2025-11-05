#!/usr/bin/env python3
"""
Pseudo-Label Generator with Ground Truth Preservation

This module generates pseudo-labels from a teacher model while:
1. Preserving all ground truth labels
2. Filtering overlapping pseudo-labels with NMS
3. Removing duplicates with ground truth boxes

Author: Refined from generate_train_boxes.py
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PseudoLabelGenerator:
    """Generate pseudo-labels while preserving ground truth."""
    
    def __init__(self,
                 teacher_model_path: str,
                 dataset_path: str,
                 imgsz: int = 960,
                 device: str = "auto"):
        """
        Initialize pseudo-label generator.
        
        Args:
            teacher_model_path: Path to teacher model
            dataset_path: Path to dataset directory
            imgsz: Image size for inference
            device: Device to use
        """
        self.teacher_model_path = Path(teacher_model_path)
        self.dataset_path = Path(dataset_path)
        self.imgsz = imgsz
        
        # Handle device selection
        if device == "auto":
            import torch
            self.device = 0 if torch.cuda.is_available() else "cpu"
            logger.info(f"Auto device selection: {self.device}")
        else:
            self.device = device
        
        # Load teacher model
        self.teacher_model = YOLO(str(self.teacher_model_path))
        logger.info(f"Loaded teacher model: {self.teacher_model_path}")
    
    def generate_merged_labels(self,
                              image_paths: List[str],
                              output_dir: Path,
                              best_conf: float,
                              topk_per_class: int = 5,
                              dup_iou: float = 0.25,
                              iou_nms: float = 0.5,
                              consistency_check: bool = True,
                              consistency_iou: float = 0.7) -> Dict:
        """
        Generate merged labels: GT + filtered pseudo-labels.
        
        Args:
            image_paths: List of image paths to process
            output_dir: Output directory for labels
            best_conf: Confidence threshold for pseudo-labels
            topk_per_class: Maximum pseudo-labels per class
            dup_iou: IoU threshold for duplicate removal with GT
            iou_nms: IoU threshold for NMS
            consistency_check: If True, verify predictions with augmented image
            consistency_iou: IoU threshold for consistency check (default: 0.7)
            
        Returns:
            stats: Statistics dictionary
            
        Note:
            Consistency check runs predictions on both original and augmented versions
            of the image. Only predictions that match (same class, IoU > consistency_iou)
            are kept, ensuring robust pseudo-labels.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        stats = {
            'total_images': len(image_paths),
            'images_processed': 0,
            'images_with_pseudo': 0,
            'gt_boxes': 0,
            'pseudo_boxes': 0,
            'total_boxes': 0,
            'filtered_by_dup': 0,
            'filtered_by_topk': 0,
            'filtered_by_consistency': 0
        }
        
        if consistency_check:
            logger.info(f"Multi-augmentation consistency check enabled (IoU threshold: {consistency_iou})")
        
        for img_path in tqdm(image_paths, desc="Generating pseudo-labels"):
            img_path = Path(img_path)
            
            # Read ground truth labels
            gt_label_path = self._derive_label_path(img_path)
            gts = self._read_gt_yolo_norm_xyxy(gt_label_path)
            stats['gt_boxes'] += len(gts)
            
            # Get predictions from teacher (disable NMS, we'll do it ourselves)
            preds = self.teacher_model.predict(
                source=str(img_path),
                imgsz=self.imgsz,
                conf=best_conf,
                iou=0.0,  # Disable model NMS, we'll apply our own
                device=self.device,
                save=False,
                verbose=False,
                augment=False  # Original image, no augmentation
            )
            
            # Extract predictions
            new_preds, filtered_by_dup = self._extract_predictions(preds, gts, dup_iou)
            stats['filtered_by_dup'] += filtered_by_dup
            
            # Consistency check: verify predictions with augmented image
            if consistency_check and len(new_preds) > 0:
                new_preds, filtered_by_consistency = self._consistency_filter(
                    img_path,
                    new_preds,
                    best_conf,
                    consistency_iou
                )
                stats['filtered_by_consistency'] += filtered_by_consistency
            
            # Apply NMS first, then top-k filtering
            new_preds = self._apply_per_class_nms(new_preds, iou_nms)
            initial_preds = len(new_preds)
            
            # Apply top-k filtering
            new_preds = self._filter_pseudo_boxes(
                new_preds,
                topk_per_class=topk_per_class
            )
            
            stats['filtered_by_topk'] += (initial_preds - len(new_preds))
            stats['pseudo_boxes'] += len(new_preds)
            
            if len(new_preds) > 0:
                stats['images_with_pseudo'] += 1
            
            # Write merged labels (GT + pseudo)
            self._write_merged_labels(
                gt_label_path,
                new_preds,
                output_dir / f"{img_path.stem}.txt"
            )
            
            stats['images_processed'] += 1
        
        stats['total_boxes'] = stats['gt_boxes'] + stats['pseudo_boxes']
        
        return stats
    
    def _derive_label_path(self, img_path: Path) -> Path:
        """Derive label path from image path."""
        parts = list(img_path.parts)
        if "images" in parts:
            i = parts.index("images")
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
        return self.dataset_path / "labels" / f"{img_path.stem}.txt"
    
    def _read_gt_yolo_norm_xyxy(self, label_path: Path) -> List[Tuple]:
        """
        Read ground truth labels in normalized xyxy format.
        
        Returns:
            List of (class_id, (x1, y1, x2, y2)) tuples
        """
        gts = []
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    sp = line.strip().split()
                    if not sp:
                        continue
                    c = int(float(sp[0]))
                    cx, cy, w, h = map(float, sp[1:5])
                    # Convert from cxcywh to xyxy
                    x1 = cx - w / 2.0
                    y1 = cy - h / 2.0
                    x2 = cx + w / 2.0
                    y2 = cy + h / 2.0
                    gts.append((c, (x1, y1, x2, y2)))
        return gts
    
    def _extract_predictions(self,
                           preds,
                           gts: List[Tuple],
                           dup_iou: float) -> Tuple[List[Tuple], int]:
        """
        Extract predictions and filter duplicates with GT.
        
        Args:
            preds: YOLO prediction results
            gts: Ground truth boxes [(class_id, (x1,y1,x2,y2))]
            dup_iou: IoU threshold for duplicate removal
            
        Returns:
            Tuple of (filtered_predictions, count_filtered_by_dup)
        """
        new_preds = []
        filtered_by_dup = 0
        
        if preds and len(preds) > 0:
            r = preds[0]
            if r.boxes is not None and len(r.boxes) > 0:
                # Get normalized xyxy coordinates
                xyxyn = r.boxes.xyxyn.cpu().numpy()
                cls = r.boxes.cls.cpu().numpy().astype(int)
                conf = r.boxes.conf.cpu().numpy()
                
                for j in range(len(cls)):
                    c = int(cls[j])
                    sc = float(conf[j])
                    x1, y1, x2, y2 = map(float, xyxyn[j])
                    pred_box = (x1, y1, x2, y2)
                    
                    # Check if overlaps with any GT above dup_iou
                    is_dup = False
                    for (gc, gt_box) in gts:
                        # Check IoU
                        if self._bbox_iou_xyxy(pred_box, gt_box) >= dup_iou:
                            is_dup = True
                            break
                        
                        # Check containment (one box inside another)
                        if self._is_contained(pred_box, gt_box) or self._is_contained(gt_box, pred_box):
                            is_dup = True
                            break
                    
                    if not is_dup:
                        new_preds.append((c, pred_box, sc))
                    else:
                        filtered_by_dup += 1
        
        return new_preds, filtered_by_dup
    
    def _bbox_iou_xyxy(self, a: Tuple, b: Tuple, eps=1e-9) -> float:
        """Calculate IoU between two boxes in xyxy format."""
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])
        
        iw = max(0.0, x2 - x1)
        ih = max(0.0, y2 - y1)
        inter = iw * ih
        
        ua = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
        ub = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
        
        return inter / (ua + ub - inter + eps) if (ua + ub) > 0 else 0.0
    
    def _is_contained(self, a: Tuple, b: Tuple) -> bool:
        """Check if box a is contained in box b."""
        return (a[0] >= b[0] and a[1] >= b[1] and 
                a[2] <= b[2] and a[3] <= b[3])
    
    def _filter_pseudo_boxes(self,
                           preds: List[Tuple],
                           topk_per_class: int = 5) -> List[Tuple]:
        """
        Filter pseudo-labels by keeping top-k per class.
        
        Args:
            preds: List of (class_id, box, confidence)
            topk_per_class: Maximum predictions per class
            
        Returns:
            Filtered predictions
        """
        # Sort by confidence
        preds.sort(key=lambda x: x[2], reverse=True)
        
        # Keep top-k per class
        kept = []
        class_counts = {}
        
        for c, b, s in preds:
            count = class_counts.get(c, 0)
            if count < topk_per_class:
                kept.append((c, b, s))
                class_counts[c] = count + 1
        
        # Apply NMS per class (use default IoU threshold)
        kept = self._apply_per_class_nms(kept, 0.5)
        
        return kept
    
    def _apply_per_class_nms(self,
                            dets: List[Tuple],
                            iou_thr: float) -> List[Tuple]:
        """
        Apply Non-Maximum Suppression per class.
        
        Args:
            dets: List of (class_id, (x1,y1,x2,y2), confidence)
            iou_thr: IoU threshold for NMS
            
        Returns:
            Filtered detections after NMS
        """
        if not dets:
            return []
        
        # Group by class
        by_class = {}
        for c, b, s in dets:
            if c not in by_class:
                by_class[c] = []
            by_class[c].append((c, b, s))
        
        # Apply NMS per class
        result = []
        for c, group in by_class.items():
            result.extend(self._nms_single_class(group, iou_thr))
        
        return result
    
    def _nms_single_class(self,
                         dets: List[Tuple],
                         iou_thr: float) -> List[Tuple]:
        """Apply NMS to single class detections."""
        if not dets:
            return []
        
        # Sort by confidence
        dets.sort(key=lambda x: x[2], reverse=True)
        
        keep = []
        while dets:
            # Keep highest confidence
            best = dets.pop(0)
            keep.append(best)
            
            # Filter remaining
            filtered = []
            for det in dets:
                if self._bbox_iou_xyxy(best[1], det[1]) < iou_thr:
                    filtered.append(det)
            
            dets = filtered
        
        return keep
    
    def _write_merged_labels(self,
                           gt_label_path: Path,
                           pseudo_labels: List[Tuple],
                           output_path: Path):
        """
        Write merged labels (GT + pseudo) to file.
        
        Args:
            gt_label_path: Path to ground truth labels
            pseudo_labels: List of (class_id, (x1,y1,x2,y2), confidence)
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        final_lines = []
        
        # 1. Add all ground truth labels (as-is)
        if gt_label_path.exists():
            with open(gt_label_path, 'r') as f:
                gt_lines = [line.strip() for line in f if line.strip()]
            final_lines.extend(gt_lines)
        
        # 2. Add pseudo-labels (convert xyxy to cxcywh)
        for (c, (x1, y1, x2, y2), _) in pseudo_labels:
            # Convert to cxcywh
            w = max(0.0, x2 - x1)
            h = max(0.0, y2 - y1)
            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            
            # Clip to [0,1]
            cx = float(np.clip(cx, 0.0, 1.0))
            cy = float(np.clip(cy, 0.0, 1.0))
            w = float(np.clip(w, 0.0, 1.0))
            h = float(np.clip(h, 0.0, 1.0))
            
            if w > 0 and h > 0:
                final_lines.append(f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        
        # Write to file
        with open(output_path, 'w') as f:
            f.write("\n".join(final_lines) + ("\n" if final_lines else ""))
    
    def _consistency_filter(self,
                           img_path: Path,
                           predictions: List[Tuple],
                           conf_threshold: float,
                           consistency_iou: float) -> Tuple[List[Tuple], int]:
        """
        Filter predictions by consistency check with horizontally flipped image.
        
        Simple approach: Predict on original and flipped image, keep only predictions
        that appear in both versions (same class, same location after unflipping).
        
        Args:
            img_path: Path to image
            predictions: List of (class, box_xyxy, conf) from original image
            conf_threshold: Confidence threshold for flipped predictions
            consistency_iou: IoU threshold to consider boxes as matching
            
        Returns:
            consistent_preds: Filtered predictions that are consistent
            filtered_count: Number of predictions filtered out
        """
        if len(predictions) == 0:
            return predictions, 0
        
        # Load and flip image
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            # Can't load image, return original predictions
            return predictions, 0
        
        img_flipped = cv2.flip(img, 1)  # Flip horizontally (1 = horizontal)
        
        # Save flipped image temporarily
        temp_flipped = img_path.parent / f"_temp_flip_{img_path.name}"
        cv2.imwrite(str(temp_flipped), img_flipped)
        
        try:
            # Get predictions from flipped image
            preds_flip = self.teacher_model.predict(
                source=str(temp_flipped),
                imgsz=self.imgsz,
                conf=conf_threshold,
                iou=0.0,  # Disable NMS
                device=self.device,
                save=False,
                verbose=False,
                augment=False  # No TTA, just flipped image
            )
            
            # Extract flipped predictions and un-flip boxes
            flip_boxes = []
            if len(preds_flip) > 0 and preds_flip[0].boxes is not None:
                boxes = preds_flip[0].boxes
                if len(boxes) > 0:
                    for i in range(len(boxes)):
                        # Get normalized xyxy coordinates
                        box_xyxy = boxes.xyxyn[i].cpu().numpy()
                        cls = int(boxes.cls[i].item())
                        conf = float(boxes.conf[i].item())
                        
                        # Un-flip box coordinates: x' = 1 - x
                        x1, y1, x2, y2 = box_xyxy
                        unflipped_box = (1.0 - x2, y1, 1.0 - x1, y2)  # Flip x coordinates
                        
                        flip_boxes.append((cls, unflipped_box, conf))
        finally:
            # Clean up temporary file
            if temp_flipped.exists():
                temp_flipped.unlink()
        
        # Match original predictions with flipped predictions
        consistent_preds = []
        
        for orig_cls, orig_box, orig_conf in predictions:
            # Find matching boxes in flipped predictions
            found_match = False
            
            for flip_cls, flip_box, flip_conf in flip_boxes:
                # Check if same class
                if orig_cls != flip_cls:
                    continue
                
                # Check if boxes overlap sufficiently (after unflipping)
                iou = self._bbox_iou_xyxy(orig_box, flip_box)
                
                if iou >= consistency_iou:
                    # Found consistent match!
                    # Use average confidence from both predictions
                    avg_conf = (orig_conf + flip_conf) / 2.0
                    consistent_preds.append((orig_cls, orig_box, avg_conf))
                    found_match = True
                    break  # Only need one match
            
            # If no match found, this prediction is inconsistent (filtered out)
        
        filtered_count = len(predictions) - len(consistent_preds)
        
        return consistent_preds, filtered_count


def main():
    """Test pseudo-label generation."""
    print("Pseudo-Label Generator Module")
    print("This module is imported by ssl_weakly_supervised.py")
    print("\nKey features:")
    print("  - Preserves ground truth labels")
    print("  - Filters duplicate predictions with GT")
    print("  - Applies per-class NMS (configurable IoU threshold)")
    print("  - Limits pseudo-labels per class")
    print("  - Tracks comprehensive statistics")
    print("  - Simplified and optimized processing pipeline")


if __name__ == "__main__":
    main()

