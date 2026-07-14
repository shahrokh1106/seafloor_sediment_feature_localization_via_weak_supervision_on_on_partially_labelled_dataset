"""
Show dataset_burrow images with their seed boxes.

Controls:
  n / space - next
  p         - previous
  q         - quit

Run from feature_matching_scripts/:
  python show_seed.py
"""

from utils import *
from pathlib import Path

DATASET_DIR = Path("dataset_burrow")
IMAGES_DIR = DATASET_DIR / "images"
SEED_DIR = DATASET_DIR / "labels_seed"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BOX_COLOR = (0, 0, 255)


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    """YOLO normalized cx cy w h → absolute xyxy pixels."""
    w = bw * img_w
    h = bh * img_h
    x1 = int(cx * img_w - w / 2)
    y1 = int(cy * img_h - h / 2)
    x2 = int(cx * img_w + w / 2)
    y2 = int(cy * img_h + h / 2)
    return x1, y1, x2, y2


def load_seed_boxes(seed_path, img_w, img_h):
    boxes = []
    if not seed_path.exists():
        return boxes
    with open(seed_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            boxes.append(yolo_to_xyxy(float(cx), float(cy), float(bw), float(bh), img_w, img_h))
    return boxes


def list_images():
    return sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


if __name__ == "__main__":
    images = list_images()
    seeded = [p for p in images if (SEED_DIR / f"{p.stem}.txt").exists()]
    if not seeded:
        print(f"No seed labels found in {SEED_DIR.resolve()}")
        raise SystemExit(1)

    print(f"Showing {len(seeded)} seeded images (of {len(images)} total)")
    print("Controls: n/space=next | p=prev | q=quit")

    idx = 0
    while True:
        img_path = seeded[idx]
        seed_path = SEED_DIR / f"{img_path.stem}.txt"
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[{idx + 1}/{len(seeded)}] could not read: {img_path.name}")
            idx = (idx + 1) % len(seeded)
            continue

        h, w = image_bgr.shape[:2]
        vis = image_bgr.copy()
        for x1, y1, x2, y2 in load_seed_boxes(seed_path, w, h):
            cv2.rectangle(vis, (x1, y1), (x2, y2), BOX_COLOR, 2)

        print(f"[{idx + 1}/{len(seeded)}] {img_path.name}")
        cv2.imshow("seed", vis)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            break
        if key == ord("p"):
            idx = (idx - 1) % len(seeded)
        else:
            # n, space, or any other key → next
            idx = (idx + 1) % len(seeded)

    cv2.destroyAllWindows()
    print("Done.")
