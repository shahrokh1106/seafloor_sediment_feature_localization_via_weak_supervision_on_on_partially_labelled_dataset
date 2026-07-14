# Annotation Acceleration Setup

Shared code lives in `feature_matching_scripts/`:
- `utils.py` — box→similarity→affinity pipeline
- `dinov3/` — local Meta DINOv3 clone + weights

Demos live in `feature_matching_scripts/test/`:
- `images/` — input images
- `output/` — saved visualizations
- `dinov3test.py` / `sam3test.py`

Run demos from the `test/` folder.

---

# DINOv3 Setup (local Meta clone)

`test/dinov3test.py` extracts dense features with Meta DINOv3 via `torch.hub.load`
(`get_intermediate_layers`), then runs the shared similarity / affinity mask flow.

Repository: https://github.com/facebookresearch/dinov3

```bash
git clone https://github.com/facebookresearch/dinov3.git
# follow the official install steps, then place pretrained weights where the script expects:
#   feature_matching_scripts/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
```

1. Clone into `feature_matching_scripts/dinov3`
2. Install per the DINOv3 README and download the ViT-L/16 weights
3. From `feature_matching_scripts/test/`: `python dinov3test.py`

---

# SAM3 Setup (Hugging Face transformers)

`test/sam3test.py` uses the same box→similarity→affinity pipeline as `dinov3test.py`,
but pulls dense backbone features from Hugging Face SAM3 (`get_vision_features`).

```bash
pip install "transformers>=5.0.0"
hf auth login
```

1. Request access: https://huggingface.co/facebook/sam3
2. From `feature_matching_scripts/test/`: `python sam3test.py`
