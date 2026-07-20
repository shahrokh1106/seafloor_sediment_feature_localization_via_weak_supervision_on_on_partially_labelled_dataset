"""
Show dataset_burrow pseudo boxes: DINOv3 (top) and SAM3 (bottom) in one window.
Green = pseudo boxes, red = seed from labels_seed.

Controls:
  n / space - next
  p         - previous
  q         - quit

Run from feature_matching_scripts/:
  python show_pseudos.py              # local labels (default)
  python show_pseudos.py --local
  python show_pseudos.py --global
"""

import argparse

from utils import *
from pathlib import Path

from pseudo_common import OUT_DINO_GLOBAL, OUT_DINO_LOCAL, OUT_SAM_GLOBAL, OUT_SAM_LOCAL

DATASET_DIR = Path("dataset_burrow")
IMAGES_DIR = DATASET_DIR / "images"
SEED_DIR = DATASET_DIR / "labels_seed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
COLOR_PSEUDO = (0, 255, 0)  # green
COLOR_SEED = (0, 0, 255)    # red


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    w = bw * img_w
    h = bh * img_h
    x1 = int(cx * img_w - w / 2)
    y1 = int(cy * img_h - h / 2)
    x2 = int(cx * img_w + w / 2)
    y2 = int(cy * img_h + h / 2)
    return x1, y1, x2, y2


def load_boxes(label_path, img_w, img_h):
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            boxes.append(yolo_to_xyxy(float(cx), float(cy), float(bw), float(bh), img_w, img_h))
    return boxes


def draw_panel(image_bgr, pseudo_boxes, seed_boxes, title):
    vis = image_bgr.copy()
    for x1, y1, x2, y2 in pseudo_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_PSEUDO, 2)
    for x1, y1, x2, y2 in seed_boxes:
        cv2.rectangle(vis, (x1, y1), (x2, y2), COLOR_SEED, 2)
    cv2.putText(vis, f"{title} ({len(pseudo_boxes)})", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_PSEUDO, 2, cv2.LINE_AA)
    return vis


def list_images():
    return sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize pseudo labels")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--local", action="store_true", help="show local labels (default)")
    mode_group.add_argument("--global", dest="use_global", action="store_true", help="show global labels")
    args = parser.parse_args()

    mode = "global" if args.use_global else "local"
    dino_dir = OUT_DINO_GLOBAL if mode == "global" else OUT_DINO_LOCAL
    sam_dir = OUT_SAM_GLOBAL if mode == "global" else OUT_SAM_LOCAL

    images = list_images()
    pairs = [
        p for p in images
        if (dino_dir / f"{p.stem}.txt").exists() or (sam_dir / f"{p.stem}.txt").exists()
    ]
    if not pairs:
        print(f"No {mode} labels found in {dino_dir} or {sam_dir}")
        raise SystemExit(1)

    print(f"Showing {len(pairs)} images ({mode} prototype)")
    print("Top = DINOv3 | Bottom = SAM3  (green=pseudo, red=seed)")
    print("Controls: n/space=next | p=prev | q=quit")

    idx = 0
    while True:
        img_path = pairs[idx]
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[{idx + 1}/{len(pairs)}] could not read: {img_path.name}")
            idx = (idx + 1) % len(pairs)
            continue

        h, w = image_bgr.shape[:2]
        seed_boxes = load_boxes(SEED_DIR / f"{img_path.stem}.txt", w, h)
        dino_boxes = load_boxes(dino_dir / f"{img_path.stem}.txt", w, h)
        sam_boxes = load_boxes(sam_dir / f"{img_path.stem}.txt", w, h)

        top = draw_panel(image_bgr, dino_boxes, seed_boxes, "DINOv3")
        bot = draw_panel(image_bgr, sam_boxes, seed_boxes, "SAM3")
        canvas = np.vstack([top, bot])

        print(f"[{idx + 1}/{len(pairs)}] {img_path.name}  dino={len(dino_boxes)}  sam={len(sam_boxes)}")
        cv2.imshow("pseudos (dino | sam)", canvas)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            idx = (idx - 1) % len(pairs)
        else:
            idx = (idx + 1) % len(pairs)

    cv2.destroyAllWindows()
    print("Done.")
