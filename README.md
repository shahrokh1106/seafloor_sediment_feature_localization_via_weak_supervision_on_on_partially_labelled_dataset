# Reducing Annotation Burden in Benthic Ecology: Weak Supervision for Automated Detection of Seafloor Sediment-Linked Features

Sediment-linked benthic features provide evidence of ecological processes but remain difficult to analyse at scale because exhaustive box-level annotation is time-consuming and requires ecological expertise. We present a two-phase weakly supervised object detection framework that reduces reliance on dense manual annotations. In the first phase, one seed bounding box for each class present in an image is represented using dense DINOv3 features. Seed-conditioned similarity and affinity-based random-walk refinement identify related regions within the same image and convert them into initial pseudo-labelled boxes. In the second phase, an object detector iteratively expands the training set using validation-calibrated confidence thresholds, augmentation-consistency filtering, duplicate removal, non-maximum suppression, and class-wise top-$k$ selection. This combination of feature-guided seed propagation and controlled pseudo-label selection is the main methodological contribution.

We first evaluated the framework on a controlled binary-burrow benchmark with 10,072 exhaustive annotations. Using 350 seed boxes, 96.5% fewer manually provided boxes than full supervision, the best weakly supervised model achieved an mAP50 of 0.821, retaining 86.5% of the fully supervised baseline performance. We then applied the framework to a more complex ten-class sediment-feature dataset, where the generated annotations underwent one-time expert verification before iterative training. On its fully annotated test set, the final detector achieved an mAP50 of 0.838, precision of 0.762, recall of 0.746, and an F1 score of 0.754. These results show that the framework can support annotation-efficient detector development for visually repetitive benthic features, while expert review remains important in complex multi-class imagery.

![Weak supervision framework overview](figs/g-abstract.PNG)

---

## 1. Clone the repository

```bash
git clone https://github.com/shahrokh1106/seafloor_sediment_feature_localization_via_weak_supervision_on_on_partially_labelled_dataset.git
cd <repo-dir>
```

---

## 2. Create a Python environment

Use **Python 3.10+**.

```powershell
python -m venv detenv
detenv\Scripts\activate
```

---

## 3. Install dependencies

Install packages in this order:

### 3.1 PyTorch

Go to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/), choose your platform and CUDA version (or CPU), copy the command shown there, and run it in the activated environment.

### 3.2 Ultralytics

```bash
pip install ultralytics
```

### 3.3 Remaining packages

From the repository root:

```bash
pip install -r requirements.txt
```

---

## Annotation acceleration framework

The annotation acceleration pipeline generates pseudo-labels from a single seed box per image using dense feature matching (DINOv3 and SAM3). It is used to build and evaluate labels on the **burrow dataset** before weakly supervised YOLO training.

**Setup, demos, and pseudo-label workflow** (DINOv3/SAM3 install, seed drawing, local/global pseudos, visualization):

→ **[feature_matching_scripts/setup.md](feature_matching_scripts/setup.md)**

That guide covers installing DINOv3 and SAM3, running single-image demos in `test/`, drawing one seed box per burrow image (`get_seed.py`), generating pseudo-labels with local and global prototypes for both backbones (`get_pseudos_local.py`, `get_pseudos_global.py`), and visualizing the results against ground truth (`show_pseudos.py`). All steps run from `feature_matching_scripts/` on `dataset_burrow/`.

---

## Weak supervision on the burrow dataset

After pseudo-labels are generated with the annotation acceleration framework, the **burrow experiment** trains and evaluates YOLO detectors under weak supervision on the burrow dataset. Training uses pseudo-labels on the train split and ground-truth labels on validation; the pipeline includes initial training, iterative refinement, a supervised GT baseline, and aggregated evaluation.

**Run from the repository root** (with the Python environment activated):

```bash
# Weakly supervised training (pick one pseudo-label source)
python burrow_experiment/run_burrow.py labels_dino_global
python burrow_experiment/run_burrow.py labels_sam_local --iterations 10 --device 0

# Supervised GT baseline (same split and config as initial training)
python burrow_experiment/train_base.py

# Aggregate metrics, label quality, and plots (after runs finish)
python burrow_experiment/eval_all.py
python burrow_experiment/label_quality.py
```

**Label sources** for `run_burrow.py`: `labels_dino_global`, `labels_dino_local`, `labels_sam_global`, `labels_sam_local` (folders under `burrow_experiment/dataset_burrow/`).

**Full details** (options, outputs, typical workflow, annotation-time estimate):

→ **[burrow_experiment/README.md](burrow_experiment/README.md)**

## Weak supervision on the seafloor sediment-linked feature dataset

The sections below describe how to **reproduce our evaluation** on the seafloor detector (validation metrics, robustness experiments, bootstrap confidence intervals, and held-out test results). All commands assume the repository root as the working directory and an activated Python environment (see sections 1–3 above).

### Dataset and pretrained weights

Place the **`detector_dataset_simple/`** folder at the **repository root** (alongside `compare_models.py`). It must be self-contained: `data.yaml`, `train.txt`, `val.txt`, `test.txt`, `images/` (actual image files), and `labels/` (10-class YOLO labels).

After unpacking, open `detector_dataset_simple/data.yaml` and set `path` to your local dataset directory, for example:

```yaml
path: ./detector_dataset_simple
```

Download the pretrained **`trained_models/`** archive (`trained_models.rar`) from:

**[Google Drive — trained_models.rar](https://drive.google.com/file/d/1KCnjA_Mg9GWkZ9ZzJsq7rhwKhblo1HKF/view?usp=sharing)**

Extract it at the repository root so you have `trained_models/` and iteration folders `trained_models/0/` … `trained_models/4/`.


---

### 1. Model comparison (`compare_models.py`)

Run the main comparison first. With no `--experiment` flag, this evaluates each iteration (0–4) on the validation set, selects the best model, and writes comparison tables and plots.

```bash
python compare_models.py
```

### 2. Robustness experiments (validation set)

Each experiment evaluates **`trained_models/best_model/best.pt`** under synthetic degradations. Level 0 is the **original validation baseline** (should match the best-model validation metrics from step 1). 

Run **one** experiment at a time:

```bash
python compare_models.py --experiment gaussian      # Gaussian blur
python compare_models.py --experiment underwater    # Simulated underwater colour/haze
python compare_models.py --experiment combined      # Combined backscatter + non-illumination
python compare_models.py --experiment tta           # Test-time augmentation consistency
```

### 3. Bootstrap confidence intervals (validation set)

Image-level bootstrap CIs for validation metrics, aligned with the `compare_models.py` point estimates (`B=1000` by default):

```bash
python bootstrap_val.py
```

**Outputs** (`bootstrap_val_results/`)

### 4. Test set evaluation

Held-out **test set** evaluation (`test.txt`) using the best model. 
```bash
python test_evaluation.py
```
**Output:** `test_evaluation_results.json` at the repository root (overall metrics, per-class AP50 / AP50-95, precision, recall, F1, and test-set instance counts).
