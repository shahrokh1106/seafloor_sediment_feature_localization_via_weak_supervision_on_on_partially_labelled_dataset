# Reducing Annotation Burden in Benthic Ecology: Weak Supervision for Automated Detection of Seafloor Sediment-Linked Features



---

## 1. Clone the repository

```bash
git clone https://github.com/shahrokh1106/seafloor_sediment_feature_localization_via_weak_supervision_on_on_partially_labelled_dataset.git
cd <"repo-dir>
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

