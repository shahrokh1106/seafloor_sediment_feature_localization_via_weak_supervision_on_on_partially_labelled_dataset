#!/usr/bin/env python3
"""
Create Simplified Dataset with Merged Classes

This script creates a simplified version of detector_dataset by:
1. Removing "Whelks" class entirely
2. Merging "Sediment Divet" and "Biotic Divet" into "Divet"
3. Grouping sponges into 2 categories (Sponges_1 and Sponges_2)

Output: detector_dataset_simple/
"""

import json
import shutil
import yaml
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


# Original dataset
SOURCE_DATASET = "detector_dataset"
OUTPUT_DATASET = "detector_dataset_simple"

# Class mapping configuration
CLASS_MAPPING = {
    # Classes to remove (map to -1)
    "Whelks": -1,
    
    # Divet merging
    "Sediment Divet": "Divet",
    "Biotic Divet": "Divet",
    
    # Sponges Group 1 (Encrusting, Amorphous, Ball types)
    "Sponges, Amorphous, Yellow": "Sponges_1",
    "Sponges, Encrusting, Yellow": "Sponges_1",
    "Sponges, Ball, Orange": "Sponges_1",
    "Sponges, Amorphous, Orange": "Sponges_1",
    "Sponges, Encrusting, Orange": "Sponges_1",
    "Sponges, Encrusting, Pink": "Sponges_1",
    "Sponge, Ball, Red": "Sponges_1",
    
    # Sponges Group 2 (Fingers, Tube, Tree, Plate types)
    "Sponges, Fingers, Yellow": "Sponges_2",
    "Sponges, Tube Cluster, Yellow": "Sponges_2",
    "Sponges, Fingers, Orange": "Sponges_2",
    "Sponges, Tree, Orange": "Sponges_2",
    "Sponges, Plate, Black": "Sponges_2",
    "Sponges, Tube Cluster, White": "Sponges_2",
    "Sponge, Hand, White": "Sponges_2",
}


def load_original_label_map(source_dir):
    """Load original label_map.json."""
    label_map_path = Path(source_dir) / "label_map.json"
    with open(label_map_path, 'r') as f:
        return json.load(f)


def create_new_label_map(original_map):
    """
    Create new label map with merged/removed classes.
    
    Returns:
        new_label_map: New class_id -> class_name mapping
        old_to_new_id: Old class_id -> new class_id mapping (-1 for removed)
    """
    # Build reverse mapping (class_name -> old_id)
    name_to_old_id = {name: int(cid) for cid, name in original_map.items()}
    
    # Determine new classes
    new_classes = set()
    for old_name in original_map.values():
        if old_name in CLASS_MAPPING:
            new_name = CLASS_MAPPING[old_name]
            if new_name != -1:  # Not removed
                new_classes.add(new_name)
        else:
            # Keep original name
            new_classes.add(old_name)
    
    # Create new label map (sorted for consistency)
    new_classes_sorted = sorted(new_classes)
    new_label_map = {str(i): name for i, name in enumerate(new_classes_sorted)}
    
    # Create old_to_new_id mapping
    old_to_new_id = {}
    for old_id, old_name in original_map.items():
        if old_name in CLASS_MAPPING:
            mapped = CLASS_MAPPING[old_name]
            if mapped == -1:
                old_to_new_id[int(old_id)] = -1  # Remove
            else:
                # Find new ID for the mapped class
                new_id = new_classes_sorted.index(mapped)
                old_to_new_id[int(old_id)] = new_id
        else:
            # Keep with new ID
            new_id = new_classes_sorted.index(old_name)
            old_to_new_id[int(old_id)] = new_id
    
    return new_label_map, old_to_new_id


def process_label_file(src_label, dst_label, old_to_new_id):
    """
    Process a single label file: remap class IDs and remove deleted classes.
    
    Returns:
        stats: Dict with processing statistics
    """
    stats = {'original_boxes': 0, 'removed_boxes': 0, 'remapped_boxes': 0, 'kept_boxes': 0}
    
    if not src_label.exists():
        # Create empty label file
        dst_label.touch()
        return stats
    
    new_lines = []
    with open(src_label, 'r') as f:
        for line in f:
            stats['original_boxes'] += 1
            parts = line.strip().split()
            if not parts:
                continue
            
            old_class_id = int(float(parts[0]))
            
            # Map to new class ID
            new_class_id = old_to_new_id.get(old_class_id, old_class_id)
            
            if new_class_id == -1:
                # Remove this box
                stats['removed_boxes'] += 1
                continue
            elif new_class_id != old_class_id:
                # Remap to new ID
                stats['remapped_boxes'] += 1
            else:
                # Keep as is
                stats['kept_boxes'] += 1
            
            # Write with new class ID
            new_line = f"{new_class_id} {' '.join(parts[1:])}\n"
            new_lines.append(new_line)
    
    # Write new label file
    with open(dst_label, 'w') as f:
        f.writelines(new_lines)
    
    return stats


def plot_class_frequencies(output_dir, new_label_map):
    """
    Plot class frequencies for all splits and save to output directory.
    
    Args:
        output_dir: Output dataset directory
        new_label_map: New label map dictionary
    """
    print("\n[Step 9] Creating class frequency plots...")
    
    output_dir = Path(output_dir)
    labels_dir = output_dir / "labels"
    
    # Read splits
    splits = {}
    for split_name in ['train', 'val', 'test']:
        split_file = output_dir / f"{split_name}.txt"
        if split_file.exists():
            with open(split_file, 'r') as f:
                splits[split_name] = [Path(line.strip()).stem for line in f if line.strip()]
    
    # Count classes per split
    split_counts = {split_name: defaultdict(int) for split_name in splits.keys()}
    
    for split_name, image_stems in splits.items():
        for stem in image_stems:
            label_file = labels_dir / f"{stem}.txt"
            if not label_file.exists():
                continue
            
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(float(parts[0]))
                        split_counts[split_name][class_id] += 1
    
    # Get class names
    num_classes = len(new_label_map)
    class_names = [new_label_map[str(i)] for i in range(num_classes)]
    
    # Create combined plot
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    for idx, (split_name, ax) in enumerate(zip(['train', 'val', 'test'], axes)):
        counts = split_counts[split_name]
        frequencies = [counts.get(i, 0) for i in range(num_classes)]
        
        # Create bar plot
        x_pos = range(num_classes)
        bars = ax.bar(x_pos, frequencies, color='#4C78A8', alpha=0.8)
        
        # Add value labels on bars
        for i, v in enumerate(frequencies):
            if v > 0:
                ax.text(i, v, str(v), ha='center', va='bottom', fontsize=9)
        
        ax.set_title(f'{split_name.capitalize()} Split - Class Frequency', fontsize=12, pad=10)
        ax.set_xlabel('Class', fontsize=10)
        ax.set_ylabel('Box Count', fontsize=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        
        # Add total count
        total = sum(frequencies)
        ax.text(0.98, 0.98, f'Total: {total} boxes\n{len(splits[split_name])} images',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / "class_frequency_all_splits.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {plot_path}")
    
    # Also create individual plots for each split
    for split_name in ['train', 'val', 'test']:
        counts = split_counts[split_name]
        frequencies = [counts.get(i, 0) for i in range(num_classes)]
        
        plt.figure(figsize=(12, 6))
        x_pos = range(num_classes)
        bars = plt.bar(x_pos, frequencies, color='#4C78A8', alpha=0.8)
        
        # Add value labels
        for i, v in enumerate(frequencies):
            if v > 0:
                plt.text(i, v, str(v), ha='center', va='bottom', fontsize=10)
        
        plt.title(f'{split_name.capitalize()} Split - Class Distribution', fontsize=14, pad=10)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Box Count', fontsize=12)
        plt.xticks(x_pos, class_names, rotation=45, ha='right', fontsize=10)
        plt.grid(axis='y', alpha=0.3)
        
        total = sum(frequencies)
        plt.text(0.98, 0.98, f'Total: {total} boxes\n{len(splits[split_name])} images',
                transform=plt.gca().transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=10)
        
        plt.tight_layout()
        
        plot_path = output_dir / f"class_frequency_{split_name}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  Saved: {plot_path}")


def main():
    """Main function to create simplified dataset."""
    print("="*70)
    print("Creating Simplified Dataset")
    print("="*70)
    
    source_dir = Path(SOURCE_DATASET)
    output_dir = Path(OUTPUT_DATASET)
    
    # Verify source exists
    if not source_dir.exists():
        print(f"Error: Source dataset not found: {source_dir}")
        return
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    
    # Load original label map
    print("\n[Step 1] Loading original label map...")
    original_map = load_original_label_map(source_dir)
    print(f"  Original classes: {len(original_map)}")
    
    # Create new label map
    print("\n[Step 2] Creating new label map...")
    new_label_map, old_to_new_id = create_new_label_map(original_map)
    print(f"  New classes: {len(new_label_map)}")
    
    # Print mapping details
    print("\n  Class mapping:")
    print("  " + "-"*66)
    print(f"  {'Old ID':<8} {'Old Name':<30} {'New ID':<8} {'New Name':<30}")
    print("  " + "-"*66)
    for old_id, old_name in sorted(original_map.items(), key=lambda x: int(x[0])):
        new_id = old_to_new_id[int(old_id)]
        if new_id == -1:
            new_name = "[REMOVED]"
        else:
            new_name = new_label_map[str(new_id)]
        print(f"  {old_id:<8} {old_name:<30} {new_id:<8} {new_name:<30}")
    print("  " + "-"*66)
    
    # Create output directory structure
    print("\n[Step 3] Creating output directory...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create images symlink
    print("\n[Step 4] Creating images link...")
    src_images = source_dir / "images"
    dst_images = output_dir / "images"
    
    if dst_images.exists():
        print(f"  Images link already exists: {dst_images}")
    else:
        try:
            # Try symlink
            dst_images.symlink_to(src_images.resolve(), target_is_directory=True)
            print(f"  Created symlink: {dst_images} -> {src_images}")
        except:
            # Try junction on Windows
            import subprocess
            try:
                subprocess.check_call(
                    ["cmd", "/c", "mklink", "/J", str(dst_images), str(src_images)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  Created junction: {dst_images} -> {src_images}")
            except:
                print(f"  Warning: Could not create link, images will need to be accessed via absolute paths")
    
    # Process split files will be done AFTER label processing to filter empty images
    
    # Process labels
    print("\n[Step 5] Processing labels...")
    src_labels = source_dir / "labels"
    dst_labels = output_dir / "labels"
    dst_labels.mkdir(parents=True, exist_ok=True)
    
    total_stats = defaultdict(int)
    images_with_boxes = set()  # Track which images have boxes after processing
    
    # Get all label files
    label_files = list(src_labels.glob("*.txt"))
    print(f"  Processing {len(label_files)} label files...")
    
    for src_label in tqdm(label_files, desc="  Processing labels"):
        dst_label = dst_labels / src_label.name
        stats = process_label_file(src_label, dst_label, old_to_new_id)
        
        for key, val in stats.items():
            total_stats[key] += val
        
        # Check if this label file has any boxes after processing
        if dst_label.exists() and dst_label.stat().st_size > 0:
            # Image has boxes - track it
            images_with_boxes.add(src_label.stem)
    
    print(f"\n  Label processing statistics:")
    print(f"    Original boxes: {total_stats['original_boxes']}")
    print(f"    Removed boxes (Whelks): {total_stats['removed_boxes']}")
    print(f"    Remapped boxes (merged classes): {total_stats['remapped_boxes']}")
    print(f"    Kept boxes (unchanged): {total_stats['kept_boxes']}")
    print(f"    Final boxes: {total_stats['original_boxes'] - total_stats['removed_boxes']}")
    print(f"    Images with boxes: {len(images_with_boxes)}")
    print(f"    Images with no boxes: {len(label_files) - len(images_with_boxes)}")
    
    # Copy and fix split files (filter out images with no boxes from train.txt)
    print("\n[Step 6] Creating split files with filtered train set...")
    
    for split_name in ['train', 'val', 'test']:
        src_split = source_dir / f"{split_name}.txt"
        dst_split = output_dir / f"{split_name}.txt"
        
        if not src_split.exists():
            print(f"  Warning: {split_name}.txt not found")
            continue
        
        # Read original split
        with open(src_split, 'r') as f:
            lines = f.readlines()
        
        # Fix paths and filter
        filtered_lines = []
        removed_count = 0
        
        for line in lines:
            # Fix path
            fixed_line = line.replace('detector_dataset/', 'detector_dataset_simple/')
            fixed_line = fixed_line.replace('detector_dataset\\', 'detector_dataset_simple\\')
            
            # Extract image stem
            img_path = Path(line.strip())
            img_stem = img_path.stem
            
            # For train split: filter out images with no boxes
            if split_name == 'train':
                if img_stem in images_with_boxes:
                    filtered_lines.append(fixed_line)
                else:
                    removed_count += 1
            else:
                # Keep all val/test images (even if no boxes)
                filtered_lines.append(fixed_line)
        
        # Write filtered split
        with open(dst_split, 'w') as f:
            f.writelines(filtered_lines)
        
        if split_name == 'train':
            print(f"  {split_name}.txt: {len(filtered_lines)} images (removed {removed_count} background images)")
        else:
            print(f"  {split_name}.txt: {len(filtered_lines)} images")
    
    # Save new label map
    print("\n[Step 7] Saving new label_map.json...")
    new_label_map_path = output_dir / "label_map.json"
    with open(new_label_map_path, 'w') as f:
        json.dump(new_label_map, f, indent=2)
    print(f"  Saved: {new_label_map_path}")
    
    # Create data.yaml
    print("\n[Step 8] Creating data.yaml...")
    data_yaml_path = output_dir / "data.yaml"
    
    yaml_content = {
        'path': str(output_dir.resolve().as_posix()),
        'train': 'train.txt',
        'val': 'val.txt',
        'test': 'test.txt',
        'nc': len(new_label_map),
        'names': new_label_map
    }
    
    with open(data_yaml_path, 'w') as f:
        yaml.safe_dump(yaml_content, f, sort_keys=False)
    
    print(f"  Saved: {data_yaml_path}")
    
    # Final summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Original classes: {len(original_map)}")
    print(f"New classes: {len(new_label_map)}")
    print(f"Classes removed: 1 (Whelks)")
    print(f"Classes merged: 2 -> 1 (Divets), 14 -> 2 (Sponges)")
    print(f"\nNew classes:")
    for cid, cname in new_label_map.items():
        print(f"  {cid}: {cname}")
    
    print(f"\nDataset created at: {output_dir}")
    print(f"  Images: {dst_images} (linked to original)")
    print(f"  Labels: {dst_labels} (remapped)")
    print(f"  Total boxes: {total_stats['original_boxes'] - total_stats['removed_boxes']}")
    print(f"\nSplit files:")
    with open(output_dir / 'train.txt', 'r') as f:
        train_count = len(f.readlines())
    with open(output_dir / 'val.txt', 'r') as f:
        val_count = len(f.readlines())
    with open(output_dir / 'test.txt', 'r') as f:
        test_count = len(f.readlines())
    print(f"  Train: {train_count} images (backgrounds filtered out)")
    print(f"  Val: {val_count} images")
    print(f"  Test: {test_count} images")
    
    # Create class frequency plots
    plot_class_frequencies(output_dir, new_label_map)
    
    print("="*70)
    print("Done!")


if __name__ == "__main__":
    main()

