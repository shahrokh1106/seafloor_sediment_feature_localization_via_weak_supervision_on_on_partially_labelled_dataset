# Annotation acceleration — setup & workflow

The working directory should be `feature_matching_scripts/`.
Burrow dataset: `dataset_burrow/`

---

## DINOv3 setup

We use the official Meta repository as a local clone plus downloaded ViT-L/16 weights.

Repository: https://github.com/facebookresearch/dinov3

1. Clone into `feature_matching_scripts/dinov3`:

```bash
cd feature_matching_scripts
git clone https://github.com/facebookresearch/dinov3.git
```

2. Follow the [DINOv3 README](https://github.com/facebookresearch/dinov3) for dependencies.

3. Download the ViT-L/16 pretrained weights and place them at:

```
feature_matching_scripts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

---

## SAM3 setup

SAM3 is loaded from Hugging Face via `transformers`.

1. Install a recent transformers build:

```bash
pip install "transformers>=5.0.0"
```

2. Log in to Hugging Face:

```bash
hf auth login
```

3. Request model access: https://huggingface.co/facebook/sam3

The model downloads automatically on first use.

---

## Demos (single-image tests)

Interactive demos live in `test/`. They step through the full pipeline (draw seed box → similarity map → affinity mask → CCA boxes) and save visualizations under `test/output/`.

| Folder / file | Purpose |
|---------------|---------|
| `test/images/` | Input images |
| `test/output/` | Saved heatmaps, masks, overlays |
| `test/dinov3test.py` | DINOv3 demo |
| `test/sam3test.py` | SAM3 demo |

Run from `feature_matching_scripts/test/`:

```bash
cd test
python dinov3test.py
python sam3test.py
```

With `DEBUG = True` (default in the test scripts), OpenCV windows appear at each step. Change `IMG_PATH` in the script to try a different image.

---

## Pseudo labels (`dataset_burrow`)

### Step 1 — Draw seed boxes (`get_seed.py`)

One seed burrow box per image, used as the reference for similarity matching.

```bash
python get_seed.py
```

**Controls:** drag box · `r` reset · `s` save & next · `n` skip · `q` quit

**Saved to:** `dataset_burrow/labels_seed/`  
One `.txt` per image (YOLO: `class cx cy w h`, normalized). Images that already have a seed file are skipped.

To review seeds only:

```bash
python show_seed.py
```

### Step 2 — Generate pseudo labels

Both scripts use the same pipeline settings (`IMAGE_SIZE=512`, `PATCH_SIZE=16`, `MASK_PERCENTILE=95`, `MIN_AREA=200`). Each writes **DINO** and **SAM3** labels, always keeps the seed box, and drops CCA boxes that overlap the seed region. At the end they print a metrics table vs `labels_gt/` and save a JSON report.

#### Local prototype (`get_pseudos_local.py`)

Uses the **seed-center patch on that frame** as the cosine-similarity query.

```bash
python get_pseudos_local.py
```

**Outputs:**
- `dataset_burrow/labels_dino_local/`
- `dataset_burrow/labels_sam_local/`
- `dataset_burrow/eval_local.json`

#### Global prototype (`get_pseudos_global.py`)

**Pass 1:** collect seed-center features from all seeded images and average them into one normalized prototype (per backbone).  
**Pass 2:** run cosine similarity against that global prototype on every image.

```bash
python get_pseudos_global.py
```

**Outputs:**
- `dataset_burrow/labels_dino_global/`
- `dataset_burrow/labels_sam_global/`
- `dataset_burrow/eval_global.json`

### Step 3 — Visualize pseudo labels (`show_pseudos.py`)

Top panel = DINOv3, bottom = SAM3. Green = pseudo boxes, red = seed.

```bash
python show_pseudos.py              # local labels (default)
python show_pseudos.py --local
python show_pseudos.py --global
```

**Controls:** `n` / space next · `p` previous · `q` quit
