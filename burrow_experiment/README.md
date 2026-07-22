# Burrow experiment

## Dataset

`dataset_burrow/` comes from the feature-matching annotation experiment. It contains burrow images, ground-truth labels (`labels_gt/`), seed boxes, and pseudo-labels produced by DINOv3 and SAM3 (local and global variants).

For how the dataset was built, seed drawing, pseudo-label generation, and evaluation --> see the feature-matching setup guide:

**[feature_matching_scripts/setup.md](../feature_matching_scripts/setup.md)**

---

## Weakly supervised training — `run_burrow.py`

`run_burrow.py` runs the full burrow detection pipeline:

1. **Prepare dataset** — fixed 80/20 train/val split (seed 42). Train images use pseudo-labels; val uses ground truth.
2. **Initial training** — train YOLO on pseudo-labels.
3. **SSL refinement** — run the weakly supervised loop: pseudo-label generation, student training, confidence tuning, and teacher updates.

Outputs go to `burrow_experiment/out_<label_source>/` (e.g. `out_labels_dino_global/`).

### Usage

From the repo root:

```bash
python burrow_experiment/run_burrow.py labels_dino_global
python burrow_experiment/run_burrow.py labels_dino_local --iterations 10 --device 0
```

**Label sources** (folders under `dataset_burrow/`):

- `labels_dino_global`
- `labels_dino_local`
- `labels_sam_global`
- `labels_sam_local`

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--iterations` | 10 | SSL iterations |
| `--epochs` | 200 | Initial training epochs |
| `--imgsz` | 512 | Training image size |
| `--batch` | 8 | Batch size |
| `--device` | auto | GPU index or `cpu` |
| `--skip-initial` | — | Skip initial training if weights exist |
| `--skip-ssl` | — | Run initial training only |
| `--force-dataset` | — | Rebuild the train/val split |

Key outputs per run: `dataset/`, `initial/`, `ssl/`, `initial_val_metrics.json`, `ssl_summary.json`, `run_config.json`.

---

## Supervised baseline — `train_base.py`

`train_base.py` trains a YOLO model on **ground-truth labels only**, using the **same train/val split** and **same YOLO configuration** as the initial stage of `run_burrow.py`. Use this as a supervised upper bound on the same validation set.

Outputs go to `burrow_experiment/out_labels_base_gt/`.

### Usage

From the repo root:

```bash
python burrow_experiment/train_base.py
python burrow_experiment/train_base.py --epochs 200 --device 0
```

**Common options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--epochs` | 200 | Training epochs |
| `--imgsz` | 512 | Training image size |
| `--batch` | 8 | Batch size |
| `--device` | auto | GPU index or `cpu` |
| `--force-dataset` | — | Rebuild the train/val split |
| `--skip-training` | — | Re-run evaluation only if `base/weights/best.pt` exists |

Key outputs: `dataset/`, `base/` (YOLO weights and training artifacts), `evaluation.json`, `training_summary.csv`, `run_config.json`.
