import os
import glob
import json
import random
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import matplotlib.pyplot as plt

def read_yolo_labels(txt_path):
    """Return a sorted list of unique class ids in a YOLO label file; empty list if file missing/empty."""
    classes = []
    if not os.path.isfile(txt_path):
        return classes
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                cls_id = int(float(parts[0]))
            except ValueError:
                continue
            if 0 <= cls_id < num_classes:
                classes.append(cls_id)
    return sorted(set(classes))

def build_samples(image_paths, labels_dir):
    """Return (imgs, labels_list, y_matrix) where:
       - imgs: list of image paths
       - labels_list: list of per-image lists of class ids
       - y_matrix: list of per-image binary vectors length num_classes
    """
    imgs = []
    labels_list = []
    for img in image_paths:
        stem = Path(img).stem
        txt = os.path.join(labels_dir, f"{stem}.txt")
        cls_ids = read_yolo_labels(txt)
        imgs.append(img)
        labels_list.append(cls_ids)
    # Build multi-hot
    y = []
    for cls_ids in labels_list:
        row = [0] * num_classes
        for c in cls_ids:
            row[c] = 1
        y.append(row)
    return imgs, labels_list, y

def _target_distribution(y_rows):
    """Compute per-class prevalence as float list."""
    n = max(1, len(y_rows))
    sums = [0] * num_classes
    for row in y_rows:
        for j, v in enumerate(row):
            sums[j] += v
    return [s / n for s in sums]

def _popcount(row):
    return sum(row)

def _iterative_stratify_indices(y_rows, target_size, rng):
    """
    Greedy Iterative Stratification (Sechidis et al.) returning a set of chosen indices of length target_size.
    Works without external libs. For multi-label; tries to match class proportions.
    """
    n = len(y_rows)
    remaining = set(range(n))
    chosen = set()

    # Desired counts per class in the split
    class_totals = [sum(row[j] for row in y_rows) for j in range(num_classes)]
    desired = [min(class_totals[j], round(class_totals[j] * (target_size / max(1, n)))) for j in range(num_classes)]

    # Keep per-class outstanding demand
    outstanding = desired[:]

    # Precompute which indices contain each class
    class_to_indices = [set() for _ in range(num_classes)]
    for i, row in enumerate(y_rows):
        for j, v in enumerate(row):
            if v:
                class_to_indices[j].add(i)

    # Simple heuristic tie-breaking: prioritize rare classes, then examples with higher label cardinality
    def need_order():
        # classes sorted by (outstanding demand ascending? we need largest outstanding first)
        return sorted(range(num_classes), key=lambda j: (-outstanding[j], class_totals[j]))

    while len(chosen) < target_size:
        if all(d == 0 for d in outstanding):
            # No outstanding demand; fill randomly
            pool = list(remaining)
            rng.shuffle(pool)
            for i in pool:
                chosen.add(i)
                remaining.remove(i)
                if len(chosen) >= target_size:
                    break
            break

        picked_any = False
        for cls in need_order():
            if outstanding[cls] <= 0:
                continue
            # Candidates not yet chosen/removed that contain cls
            candidates = [i for i in class_to_indices[cls] if i in remaining]
            if not candidates:
                outstanding[cls] = 0
                continue
            # Among candidates, prefer those that help other outstanding classes too (higher overlap with positive outstanding)
            def score(i):
                row = y_rows[i]
                return sum(row[j] and outstanding[j] > 0 for j in range(num_classes)), _popcount(row)
            candidates.sort(key=lambda i: (score(i)[0], score(i)[1]), reverse=True)
            # Select the best
            i_sel = candidates[0]
            chosen.add(i_sel)
            remaining.remove(i_sel)
            # Decrement outstanding for classes covered
            for j, v in enumerate(y_rows[i_sel]):
                if v and outstanding[j] > 0:
                    outstanding[j] -= 1
            picked_any = True
            if len(chosen) >= target_size:
                break
        if not picked_any:
            # Fallback: cannot satisfy further; fill randomly
            pool = list(remaining)
            rng.shuffle(pool)
            for i in pool:
                chosen.add(i)
                remaining.remove(i)
                if len(chosen) >= target_size:
                    break
    return chosen

def multilabel_iterative_three_way_split(y_rows, train_ratio, val_ratio, test_ratio, seed=42):
    """
    Returns (train_idx, val_idx, test_idx) as lists using iterative stratification.
    Strategy: first reserve test, then reserve val from remaining, rest is train.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-8, "Ratios must sum to 1.0"
    n = len(y_rows)
    rng = random.Random(seed)

    test_size = round(n * test_ratio)
    val_size  = round(n * val_ratio)
    # Ensure sizes sum <= n; adjust train implicitly
    test_idx = _iterative_stratify_indices(y_rows, test_size, rng)
    remaining = [i for i in range(n) if i not in test_idx]
    y_rem = [y_rows[i] for i in remaining]

    # Reindex within remaining for the second split
    val_sel_local = _iterative_stratify_indices(y_rem, min(val_size, len(remaining)), rng)
    val_idx = { remaining[i] for i in val_sel_local }

    train_idx = [i for i in range(n) if i not in test_idx and i not in val_idx]
    val_idx = sorted(list(val_idx))
    test_idx = sorted(list(test_idx))
    train_idx = sorted(train_idx)
    return train_idx, val_idx, test_idx

def write_split_files(imgs, train_idx, val_idx, test_idx, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_file = os.path.join(out_dir, "train.txt")
    val_file   = os.path.join(out_dir, "val.txt")
    test_file  = os.path.join(out_dir, "test.txt")
    with open(train_file, "w", encoding="utf-8") as f:
        for i in train_idx:
            f.write(f"{imgs[i]}\n")
    with open(val_file, "w", encoding="utf-8") as f:
        for i in val_idx:
            f.write(f"{imgs[i]}\n")
    with open(test_file, "w", encoding="utf-8") as f:
        for i in test_idx:
            f.write(f"{imgs[i]}\n")
    return train_file, val_file, test_file

def print_class_frequencies_from_split_lists(split_files, labels_dir, label_map):
    """
    split_files: dict like {"train": "train.txt", "val": "val.txt", "test": "test.txt"}
    Prints per-class instance counts for each split.
    """
    class_names = [label_map.get(i, f"class_{i}") for i in range(max(label_map.keys())+1)]
    for split_name, list_path in split_files.items():
        with open(list_path, "r", encoding="utf-8") as f:
            imgs = [line.strip() for line in f if line.strip()]
        counts = Counter()
        for img in imgs:
            stem = Path(img).stem
            txt = os.path.join(labels_dir, f"{stem}.txt")
            if not os.path.isfile(txt):
                continue
            with open(txt, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cls_id = int(float(parts[0]))
                    except ValueError:
                        continue
                    counts[cls_id] += 1
        total = sum(counts.values())
        print(f"\n=== {split_name.upper()} ===")
        print(f"Images: {len(imgs)} | Total instances: {total}")
        for cid in range(len(class_names)):
            print(f"{cid:>2} ({class_names[cid]}): {counts[cid]}")

def plot_class_frequencies_from_split_lists(
    split_files,
    labels_dir,
    label_map,
    normalize=False,
    save_path=None,
    show=True,
):
    """
    Args:
        split_files: dict like {"train": "detector_dataset/train.txt", "val": "...", "test": "..."}
        labels_dir: path to YOLO labels folder (txt files)
        label_map: dict {class_id(int)->class_name(str)}
        normalize: if True, plot per-split proportions instead of raw counts
        save_path: if given, saves the figure there (e.g., 'detector_dataset/class_freqs.png')
        show: if True, calls plt.show()

    Returns:
        {
          "class_names": [str,...],           # ordered by class_id
          "counts": {split_name: [int,...]},  # counts per class_id
          "totals": {split_name: int}         # total instances per split
        }
    """
    num_classes = max(label_map.keys()) + 1
    class_names = [label_map.get(i, f"class_{i}") for i in range(num_classes)]

    def _count_instances_for_list(list_path, labels_dir, num_classes):
        counts = [0] * num_classes
        if not os.path.isfile(list_path):
            return counts
        with open(list_path, "r", encoding="utf-8") as f:
            imgs = [line.strip() for line in f if line.strip()]
        for img in imgs:
            stem = Path(img).stem
            txt = os.path.join(labels_dir, f"{stem}.txt")
            if not os.path.isfile(txt):
                continue
            with open(txt, "r", encoding="utf-8") as lf:
                for line in lf:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    try:
                        cid = int(float(parts[0]))
                    except ValueError:
                        continue
                    if 0 <= cid < num_classes:
                        counts[cid] += 1
        return counts

    # Compute counts for each split
    counts_per_split = {}
    totals_per_split = {}
    for split_name, list_path in split_files.items():
        c = _count_instances_for_list(list_path, labels_dir, num_classes)
        counts_per_split[split_name] = c
        totals_per_split[split_name] = int(sum(c))

    # Prepare data for plotting (grouped bars: classes on x, bars = splits)
    splits = list(counts_per_split.keys())
    data = np.array([counts_per_split[s] for s in splits])  # shape: (num_splits, num_classes)

    if normalize:
        # Convert to proportions per split; avoid div by zero
        denom = data.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1
        data = data / denom

    num_splits, num_classes = data.shape
    x = np.arange(num_classes)
    width = 0.8 / max(1, num_splits)  # total width ~0.8

    plt.figure(figsize=(max(8, num_classes * 0.7), 5))
    for i, split in enumerate(splits):
        plt.bar(x + i * width - (num_splits - 1) * width / 2.0, data[i], width=width, label=split)

    plt.xticks(x, class_names, rotation=90, ha="right")
    plt.ylabel("Proportion" if normalize else "Instance count")
    plt.xlabel("Class")
    plt.title("Class frequencies per split" + (" (normalized)" if normalize else ""))
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        Path(os.path.dirname(save_path) or ".").mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()

    return {
        "class_names": class_names,
        "counts": counts_per_split,
        "totals": totals_per_split,
    }

if __name__ == "__main__":
    dataset_path = "detector_dataset"
    # train_ratio = 0.8
    # val_ratio   = 0.1
    # test_ratio  = 0.1
    # seed = 42



    # image_paths = glob.glob(os.path.join(dataset_path, "images", "*"))
    # labels_folder_path = os.path.join(dataset_path, "labels")
    # label_map_path = os.path.join(dataset_path, "label_map.json")
    # with open(label_map_path, "r", encoding="utf-8") as f:
    #     raw = json.load(f)
    #     label_map = {int(k): v for k, v in raw.items()}
    # num_classes = max(label_map.keys()) + 1  

    # imgs, labels_list, y = build_samples(image_paths, labels_folder_path)
    # if len(imgs) == 0:
    #     raise RuntimeError("No images found under dataset_path/images")
    
    # train_idx, val_idx, test_idx = multilabel_iterative_three_way_split(y, train_ratio, val_ratio, test_ratio, seed=seed)

    # out_dir = dataset_path  # write into dataset root
    # train_file, val_file, test_file = write_split_files(imgs, train_idx, val_idx, test_idx, out_dir)
    # print(f"Wrote splits:\n  {train_file}\n  {val_file}\n  {test_file}")

    
    # split_files = {"train": train_file, "val": val_file, "test": test_file}
    # print_class_frequencies_from_split_lists(split_files, labels_folder_path, label_map)

    # split_files = {
    #     "train": os.path.join(dataset_path, "train.txt"),
    #     "val":   os.path.join(dataset_path, "val.txt"),
    #     "test":  os.path.join(dataset_path, "test.txt"),
    # }
    # _ = plot_class_frequencies_from_split_lists(
    #         split_files,
    #         labels_folder_path,
    #         label_map,
    #         normalize=False,
    #         save_path=os.path.join(dataset_path, "class_freqs.png"),
    #         show=False
    #     )


    

    outpath = dataset_path
    train_split = os.path.join(dataset_path, "org_split", "train.txt")
    test_split = os.path.join(dataset_path, "org_split", "test.txt")
    valid_split = os.path.join(dataset_path, "org_split", "val.txt")

    with open(os.path.join(valid_split), "r") as f:
        valid_paths = [line.strip() for line in f if line.strip()]

    with open(os.path.join(train_split), "r") as f:
        train_paths = [line.strip() for line in f if line.strip()]

    with open(os.path.join(test_split), "r") as f:
        test_paths = [line.strip() for line in f if line.strip()]

    print(len(valid_paths))
    print(len(train_paths))
    print(len(test_paths))

    random.seed(42)
    random.shuffle(valid_paths)
    new_valid_paths = valid_paths[:100]
    extra_train_paths = valid_paths[100:]
    new_train_paths = train_paths + extra_train_paths
    save_dir = os.path.join(outpath, "new_split")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train.txt"), "w") as f:
        f.write("\n".join(new_train_paths) + "\n")
    with open(os.path.join(save_dir, "val.txt"), "w") as f:
        f.write("\n".join(new_valid_paths) + "\n")
    with open(os.path.join(save_dir, "test.txt"), "w") as f:
        f.write("\n".join(test_paths) + "\n")

    print(len(new_valid_paths))
    print(len(new_train_paths))
    print(len(test_paths))



    


