#!/usr/bin/env python3
"""
Snapshot Dataset Builder

This module builds snapshot datasets for each SSL iteration by:
1. Linking to original images
2. Using merged labels (GT + pseudo) for train
3. Using original labels for val/test
4. Creating data.yaml for the snapshot

Author: Refined from get_student_yaml.py
"""

import yaml
import shutil
import subprocess
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class SnapshotDatasetBuilder:
    """Build snapshot datasets for SSL iterations."""
    
    def __init__(self, base_data_yaml: str, output_root: Path):
        """
        Initialize snapshot dataset builder.
        
        Args:
            base_data_yaml: Path to base data.yaml
            output_root: Root directory for outputs
        """
        self.base_data_yaml = Path(base_data_yaml)
        self.output_root = Path(output_root)
        
        # Load base metadata
        with open(self.base_data_yaml, 'r') as f:
            self.base_meta = yaml.safe_load(f)
        
        self.names = self.base_meta.get('names')
        self.nc = self.base_meta.get('nc', len(self.names or []))
    
    def build_snapshot(self,
                      iteration: int,
                      train_paths: List[str],
                      val_paths: List[str],
                      test_paths: List[str],
                      merged_labels_dir: Path) -> Path:
        """
        Build snapshot dataset for current iteration.
        
        Args:
            iteration: Current iteration number
            train_paths: Training image paths
            val_paths: Validation image paths
            test_paths: Test image paths
            merged_labels_dir: Directory with merged labels
            
        Returns:
            snapshot_yaml: Path to snapshot data.yaml
            
        Note:
            No class balancing is applied - VFL loss handles class imbalance naturally.
        """
        logger.info(f"Building snapshot dataset for iteration {iteration}...")
        
        # Create snapshot directory
        snap_root = self.output_root / f"snapshot_iter{iteration}"
        snap_root.mkdir(parents=True, exist_ok=True)
        
        # Detect images root
        imgs_src = self._detect_images_root(train_paths)
        imgs_dst = snap_root / "images"
        labs_dst = snap_root / "labels"
        
        # 1. Link images directory
        self._ensure_dir_link(imgs_dst, imgs_src)
        
        # 2. Build labels directory
        labs_dst.mkdir(parents=True, exist_ok=True)
        
        # Train labels from merged (GT + pseudo)
        for img_path in train_paths:
            img_path = Path(img_path)
            dst_label = labs_dst / f"{img_path.stem}.txt"
            src_label = merged_labels_dir / f"{img_path.stem}.txt"
            self._copy_label(src_label, dst_label)
        
        # Val labels from original GT
        for img_path in val_paths:
            img_path = Path(img_path)
            dst_label = labs_dst / f"{img_path.stem}.txt"
            src_label = self._derive_label_path(img_path)
            self._copy_label(src_label, dst_label)
        
        # Test labels from original GT
        for img_path in test_paths:
            img_path = Path(img_path)
            dst_label = labs_dst / f"{img_path.stem}.txt"
            src_label = self._derive_label_path(img_path)
            self._copy_label(src_label, dst_label)
        
        # 3. Write split files (remapped to snapshot images/)
        # Note: No class balancing - VFL loss handles class imbalance naturally
        train_txt = snap_root / "train.txt"
        val_txt = snap_root / "val.txt"
        test_txt = snap_root / "test.txt"
        
        self._write_split_list_remapped(train_paths, imgs_src, imgs_dst, train_txt)
        self._write_split_list_remapped(val_paths, imgs_src, imgs_dst, val_txt)
        self._write_split_list_remapped(test_paths, imgs_src, imgs_dst, test_txt)
        
        # 4. Create data.yaml
        snap_yaml = snap_root / "data.yaml"
        yaml_content = {
            'path': str(snap_root.as_posix()),
            'train': 'train.txt',
            'val': 'val.txt',
            'test': 'test.txt',
            'names': self.names,
            'nc': self.nc
        }
        
        with open(snap_yaml, 'w') as f:
            yaml.safe_dump(yaml_content, f, sort_keys=False)
        
        logger.info(f"✓ Snapshot created at: {snap_root}")
        logger.info(f"  Train: {len(train_paths)} images")
        logger.info(f"  Val: {len(val_paths)} images")
        logger.info(f"  Test: {len(test_paths)} images")
        logger.info(f"  No class balancing applied (VFL handles imbalance)")
        
        return snap_yaml
    
    def _detect_images_root(self, image_paths: List[str]) -> Path:
        """Detect root images/ directory from image paths."""
        for p in image_paths:
            parts = list(Path(p).resolve().parts)
            if "images" in parts:
                i = parts.index("images")
                return Path(*parts[:i + 1])  # .../images
        raise RuntimeError("Couldn't find 'images' root in image paths")
    
    def _ensure_dir_link(self, dst: Path, src: Path):
        """Create directory link/symlink."""
        if dst.exists():
            return
        
        dst.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Try symlink first (Unix-like systems)
            dst.symlink_to(src, target_is_directory=True)
            logger.debug(f"Created symlink: {dst} -> {src}")
        except (OSError, NotImplementedError):
            # Windows fallback: try junction
            try:
                subprocess.check_call(
                    ["cmd", "/c", "mklink", "/J", str(dst), str(src)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.debug(f"Created junction: {dst} -> {src}")
            except:
                # Last resort: copy directory
                logger.warning(f"Could not create link, copying directory instead")
                shutil.copytree(src, dst, dirs_exist_ok=True)
    
    def _derive_label_path(self, img_path: Path) -> Path:
        """Derive label path from image path."""
        parts = list(img_path.parts)
        if "images" in parts:
            i = parts.index("images")
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")
        return img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    
    def _copy_label(self, src_label: Path, dst_label: Path):
        """Copy label file, create empty if source doesn't exist."""
        dst_label.parent.mkdir(parents=True, exist_ok=True)
        
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            # Create empty label file
            dst_label.write_text("", encoding="utf-8")
    
    def _write_split_list_remapped(self,
                                   image_paths: List[str],
                                   images_root_src: Path,
                                   images_root_dst: Path,
                                   out_txt: Path):
        """Write split list with remapped paths."""
        out_txt.parent.mkdir(parents=True, exist_ok=True)
        
        lines = []
        for p in image_paths:
            rp = Path(p).resolve()
            try:
                rel = rp.relative_to(images_root_src)  # path under .../images/...
                lines.append((images_root_dst / rel).as_posix())
            except ValueError:
                # If path is not relative to images_root_src, use as-is
                logger.warning(f"Could not make relative path for {p}")
                lines.append(p)
        
        out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    """Test dataset builder."""
    print("Snapshot Dataset Builder Module")
    print("This module is imported by ssl_weakly_supervised.py")
    print("\nKey features:")
    print("  - Links to original images")
    print("  - Uses merged labels for train")
    print("  - Uses original GT for val/test")
    print("  - Creates data.yaml for snapshot")


if __name__ == "__main__":
    main()

