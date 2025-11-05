#!/usr/bin/env python3
"""
YOLO Class Weights Modifier

This script modifies the YOLO loss function to support per-class weights.

Usage:
    python change_yolo.py --enable     # Enable per-class weights
    python change_yolo.py --disable    # Restore original (reinstall ultralytics)
"""

import sys
import shutil
import subprocess
from pathlib import Path


def find_files():
    """Find the YOLO files that need modification."""
    loss_file = Path("detenv/Lib/site-packages/ultralytics/utils/loss.py")
    config_file = Path("detenv/Lib/site-packages/ultralytics/cfg/default.yaml")
    
    if not loss_file.exists():
        print(f"Error: Could not find loss.py at {loss_file}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    if not config_file.exists():
        print(f"Error: Could not find default.yaml at {config_file}")
        print("Make sure you're running this from the project root directory.")
        sys.exit(1)
    
    return loss_file, config_file


def backup_original(loss_file):
    """Create a backup of the original loss.py file."""
    backup_file = loss_file.with_suffix('.py.original')
    
    if not backup_file.exists():
        print(f"Creating backup: {backup_file}")
        shutil.copy2(loss_file, backup_file)
    else:
        print(f"Backup already exists: {backup_file}")
    
    return backup_file


def enable_class_weights(loss_file, config_file):
    """Modify loss.py and default.yaml to support per-class weights."""
    print("Enabling per-class weights in YOLO...")
    
    # Backup files
    backup_file = backup_original(loss_file)
    backup_config = config_file.with_suffix('.yaml.original')
    if not backup_config.exists():
        print(f"Creating config backup: {backup_config}")
        shutil.copy2(config_file, backup_config)
    
    # ========== Step 1: Add class_weights to default.yaml ==========
    print("\n[1/2] Adding class_weights parameter to default.yaml...")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config_content = f.read()
    
    # Check if already added
    if 'class_weights:' in config_content:
        print("OK: class_weights already in default.yaml")
    else:
        # Add after the cls parameter
        marker = "cls: 0.5 # (float) classification loss gain"
        addition = "\nclass_weights: # (list | dict, optional) per-class weights for handling class imbalance"
        
        if marker in config_content:
            config_content = config_content.replace(marker, marker + addition)
            with open(config_file, 'w', encoding='utf-8') as f:
                f.write(config_content)
            print("OK: Added class_weights to default.yaml")
        else:
            print("WARNING: Could not find insertion point in default.yaml")
    
    # ========== Step 2: Modify loss.py ==========
    print("\n[2/2] Modifying loss.py to use class_weights...")
    
    with open(loss_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if already modified
    if '[CUSTOM] Using per-class weights' in content:
        print("OK: Per-class weights already enabled!")
        return
    
    # Modification 1: Add class weights loading in __init__
    init_marker = "self.nc = m.nc  # number of classes"
    init_addition = """
        
        # ============ CUSTOM: Per-Class Weights Support ============
        # Load per-class weights if available
        self.class_weights = None
        if hasattr(h, 'class_weights') and h.class_weights is not None:
            import torch
            # Convert to tensor and move to device
            if isinstance(h.class_weights, (list, tuple)):
                self.class_weights = torch.tensor(h.class_weights, device=device, dtype=torch.float32)
            elif isinstance(h.class_weights, dict):
                # Convert dict {0: weight0, 1: weight1, ...} to tensor
                weights_list = [h.class_weights.get(i, 1.0) for i in range(self.nc)]
                self.class_weights = torch.tensor(weights_list, device=device, dtype=torch.float32)
            print(f"[CUSTOM] Using per-class weights: {self.class_weights.cpu().tolist()}")
        # ==========================================================="""
    
    if init_marker in content:
        content = content.replace(init_marker, init_marker + init_addition)
        print("OK: Added class weights loading to __init__")
    else:
        print("WARNING: Could not find __init__ modification point")
    
    # Modification 2: Modify classification loss computation
    old_cls_loss = "loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE"
    new_cls_loss = """# Apply per-class weights if available
        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # Shape: [batch, num_anchors, num_classes]
        
        if self.class_weights is not None:
            # Apply class weights: multiply each class's loss by its weight
            # class_weights shape: [num_classes]
            # cls_loss shape: [batch, num_anchors, num_classes]
            cls_loss = cls_loss * self.class_weights.view(1, 1, -1)  # Broadcast weights
        
        loss[1] = cls_loss.sum() / target_scores_sum  # Weighted BCE"""
    
    if old_cls_loss in content:
        content = content.replace(old_cls_loss, new_cls_loss)
        print("OK: Modified classification loss computation")
    else:
        print("WARNING: Could not find classification loss modification point")
    
    # Write modified content
    with open(loss_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"\nSUCCESS: Per-class weights enabled!")
    print(f"   Modified: {loss_file}")
    print(f"   Modified: {config_file}")
    print(f"   Backups created with .original extension")


def disable_class_weights(loss_file, config_file):
    """Restore original files by using backups or reinstalling ultralytics."""
    print("Disabling per-class weights (restoring original YOLO)...")
    
    backup_file = loss_file.with_suffix('.py.original')
    backup_config = config_file.with_suffix('.yaml.original')
    
    # Option 1: Restore from backups if they exist
    restored = False
    
    if backup_file.exists():
        print(f"Restoring loss.py from backup: {backup_file}")
        shutil.copy2(backup_file, loss_file)
        print("OK: Restored loss.py")
        restored = True
    
    if backup_config.exists():
        print(f"Restoring default.yaml from backup: {backup_config}")
        shutil.copy2(backup_config, config_file)
        print("OK: Restored default.yaml")
        restored = True
    
    if not restored:
        # Option 2: Reinstall ultralytics
        print("No backup found. Reinstalling ultralytics...")
        print("This may take a minute...")
        
        try:
            # Use the virtual environment's pip
            pip_path = Path("detenv/Scripts/pip.exe")
            if not pip_path.exists():
                pip_path = "pip"  # Fallback to system pip
            
            subprocess.run([
                str(pip_path),
                "install",
                "--force-reinstall",
                "--no-deps",
                "ultralytics==8.3.203"
            ], check=True)
            
            print("OK: Ultralytics reinstalled")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error reinstalling ultralytics: {e}")
            print("You may need to run: pip install --force-reinstall ultralytics")
            sys.exit(1)
    
    print(f"\nSUCCESS: Per-class weights disabled!")
    print(f"   YOLO restored to original")


def enable_varifocal(loss_file: Path):
    """Switch YOLO classification loss to VarifocalLoss and neutralize class-weights logic."""
    print("Enabling VarifocalLoss for classification...")

    with open(loss_file, 'r', encoding='utf-8') as f:
        content = f.read()

    modified = False

    # 1) Replace weighted BCE classification loss with VarifocalLoss
    bce_block = (
        "# Apply per-class weights if available\n"
        "        cls_loss = self.bce(pred_scores, target_scores.to(dtype))  # Shape: [batch, num_anchors, num_classes]\n\n"
        "        if self.class_weights is not None:\n"
        "            # Apply class weights: multiply each class's loss by its weight\n"
        "            # class_weights shape: [num_classes]\n"
        "            # cls_loss shape: [batch, num_anchors, num_classes]\n"
        "            cls_loss = cls_loss * self.class_weights.view(1, 1, -1)  # Broadcast weights\n\n"
        "        loss[1] = cls_loss.sum() / target_scores_sum  # Weighted BCE"
    )

    vfl_block = (
        "# Cls loss - VarifocalLoss (more robust with partial labels)\n"
        "        vfl = VarifocalLoss(gamma=2.0, alpha=0.75)\n"
        "        loss[1] = vfl(pred_scores, target_scores.to(dtype), (target_scores > 0).to(dtype))"
    )

    if bce_block in content:
        content = content.replace(bce_block, vfl_block)
        modified = True
        print("OK: Replaced weighted BCE with VarifocalLoss")
    else:
        # Fallback: try replacing the simpler BCE line if present
        simple_bce = "loss[1] = self.bce(pred_scores, target_scores.to(dtype)).sum() / target_scores_sum  # BCE"
        if simple_bce in content:
            content = content.replace(simple_bce, vfl_block)
            modified = True
            print("OK: Replaced BCE with VarifocalLoss")
        else:
            # Already using VFL or manual patch applied
            print("OK: Checking if VarifocalLoss is already enabled...")
    
    # 2) Neutralize class-weights use if any remains
    if "class_weights" in content:
        # Ensure any multiplication by self.class_weights is no-op
        content = content.replace(" * self.class_weights.view(1, 1, -1)", "")
        modified = True
    
    # Check if VFL is already in use
    if "VarifocalLoss(gamma=2.0, alpha=0.75)" in content:
        print("OK: VarifocalLoss already enabled in loss.py")
        return
    
    if modified:
        with open(loss_file, 'w', encoding='utf-8') as f:
            f.write(content)
        print("SUCCESS: VarifocalLoss enabled.")
    else:
        print("OK: No changes needed (VarifocalLoss already enabled).")


def main():
    """Main entry point."""
    if len(sys.argv) != 2 or sys.argv[1] not in ['--enable', '--disable', '--enable-vfl']:
        print(__doc__)
        sys.exit(1)
    
    action = sys.argv[1]
    loss_file, config_file = find_files()
    
    print("="*60)
    print("YOLO Per-Class Weights Modifier")
    print("="*60)
    print(f"Loss file: {loss_file}")
    print(f"Config file: {config_file}")
    print(f"Action: {action}")
    print("="*60)
    print()
    
    if action == '--enable':
        enable_class_weights(loss_file, config_file)
    elif action == '--enable-vfl':
        enable_varifocal(loss_file)
    else:
        disable_class_weights(loss_file, config_file)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()

