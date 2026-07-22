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

---

## Evaluation — `eval_all.py`

`eval_all.py` aggregates **model validation metrics** from all completed runs into simple JSON files under `out_eval/`. Each file lists train/val box counts and metrics (P, R, F1, mAP50, mAP50-95) at initial pseudo-label training and at each SSL iteration.

Requires all five output folders: the four `out_labels_*` SSL runs plus `out_labels_base_gt`. If any are missing, it prints the commands to create them.

### Usage

```bash
python burrow_experiment/eval_all.py
```

**Outputs** (`burrow_experiment/out_eval/`):

- `labels_dino_global.json`, `labels_dino_local.json`, `labels_sam_global.json`, `labels_sam_local.json`
- `base_gt.json`

---

## Label quality — `label_quality.py`

`label_quality.py` measures how well **training labels match ground truth** on the train split at each step — feature-matching pseudos at initial, then SSL pseudo labels at each iteration. Uses the same box-matching metrics as the feature-matching eval (`pseudo_common.evaluate_set`).

### Usage

```bash
python burrow_experiment/label_quality.py
```

**Output:** `out_eval/label_quality.json` — one `steps` list per experiment with pred/GT box counts, mAP, and P/R/F1 at IoU 0.50 and 0.75.

---

## Plots — `plot_eval.py`

`plot_eval.py` reads the JSON files produced by `eval_all.py` and plots **model validation metrics** across SSL iterations. One figure per metric; four approach lines plus a dashed horizontal line for the GT baseline.

### Usage

```bash
python burrow_experiment/plot_eval.py
```

Run `eval_all.py` first. **Outputs:** `out_eval/plots/` (`precision.png`, `recall.png`, `f1.png`, `map50.png`, `map50_95.png`).

---

## Annotation time — `time.py`

`time.py` estimates how long burrow annotation takes. It shows **10 random images** (seed 42), you draw **one box per image**, and it records the elapsed time per box. From the average it extrapolates:

- **Feature-matching approach** — `avg_time ×` number of seed boxes in the dataset (one seed per image)
- **Full manual annotation** — `avg_time ×` total GT boxes in the dataset

If `out_eval/annotation_time.json` already exists, the script prints the saved estimate and exits. Use `--force` to run the timed session again.

### Usage

```bash
python burrow_experiment/time.py
python burrow_experiment/time.py --force
```

**Controls:** drag box · `r` reset · `s` save and next · `q` quit

**Output:** `out_eval/annotation_time.json`

---

## Typical workflow

```bash
# 1. Run all experiments
python burrow_experiment/run_burrow.py labels_dino_global --iterations 10 --device 0
python burrow_experiment/run_burrow.py labels_dino_local --iterations 10 --device 0
python burrow_experiment/run_burrow.py labels_sam_global --iterations 10 --device 0
python burrow_experiment/run_burrow.py labels_sam_local --iterations 10 --device 0
python burrow_experiment/train_base.py --device 0

# 2. Aggregate metrics and label quality
python burrow_experiment/eval_all.py
python burrow_experiment/label_quality.py

# 3. Plot model metrics
python burrow_experiment/plot_eval.py
```
