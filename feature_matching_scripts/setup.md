# Annotation Acceleration Setup (DINOv3)

The annotation acceleration workflow in this `feature_matching_scripts/` folder is built on top of the official **DINOv3** implementation from Meta AI.

Repository:
- [https://github.com/facebookresearch/dinov3](https://github.com/facebookresearch/dinov3)

## Important Dependency

Before running scripts in this folder (for example `dinov3test.py`), you should clone and install the official DINOv3 repository according to its setup instructions.

## Recommended Steps

1. Clone DINOv3:
   ```bash
   git clone https://github.com/facebookresearch/dinov3.git
   ```
2. Follow the installation instructions in the official repository (environment, dependencies, model weights, etc.).
3. Ensure the local path expected by this project is available (the scripts currently reference `dinov3` as a local repository directory).
4. Run this project's DINO-based scripts only after DINOv3 is installed correctly.

## Notes

- This project uses DINOv3 feature extraction as the backbone for annotation acceleration utilities.
- If setup is incomplete, model loading via `torch.hub.load(...)` in the local scripts may fail.
