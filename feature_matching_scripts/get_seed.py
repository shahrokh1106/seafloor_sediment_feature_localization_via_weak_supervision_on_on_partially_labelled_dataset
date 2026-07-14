"""
Draw one seed bounding box per image for dataset_burrow.

Controls:
  drag  - draw box
  r     - reset box
  s     - save seed and go to next image
  n     - skip image (no save)
  q     - quit

Run from feature_matching_scripts/:
  python get_seed.py
"""

from utils import *
from pathlib import Path

DATASET_DIR = Path("dataset_burrow")
IMAGES_DIR = DATASET_DIR / "images"
SEED_DIR = DATASET_DIR / "labels_seed"
CLASS_ID = 0  # binary detection
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    """Convert absolute xyxy (original pixels) to YOLO normalized cx cy w h."""
    x1, x2 = sorted((max(0, min(img_w - 1, x1)), max(0, min(img_w - 1, x2))))
    y1, y2 = sorted((max(0, min(img_h - 1, y1)), max(0, min(img_h - 1, y2))))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = bw / img_w
    h = bh / img_h
    return cx, cy, w, h


def draw_seed_box(image_bgr):
    """
    Draw one box on the image at original size.
    Returns:
      ("save", [x1,y1,x2,y2]) | ("skip", None) | ("quit", None)
    """
    BOX_COLOR = (0, 0, 255)
    drawing = False
    points = []
    img_orig = image_bgr.copy()
    img = img_orig.copy()

    def on_mouse(event, x, y, flags, params):
        nonlocal drawing, points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = img_orig.copy()
            cv2.rectangle(img, points[0], (x, y), BOX_COLOR, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(points) == 1:
                points.append((x, y))
            else:
                points[1] = (x, y)
            img = img_orig.copy()
            cv2.rectangle(img, points[0], points[1], BOX_COLOR, 2)

    cv2.namedWindow("seed", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("seed", on_mouse)
    action = "quit"
    while True:
        cv2.imshow("seed", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            img = img_orig.copy()
            points = []
            drawing = False
        elif key == ord("s"):
            if len(points) == 2:
                action = "save"
                break
            print("  draw a box before saving (s)")
        elif key == ord("n"):
            action = "skip"
            break
        elif key == ord("q"):
            action = "quit"
            break
    cv2.destroyAllWindows()

    if action == "save":
        (x1, y1), (x2, y2) = points
        return action, [int(x1), int(y1), int(x2), int(y2)]
    return action, None


def list_images():
    return sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


if __name__ == "__main__":
    SEED_DIR.mkdir(parents=True, exist_ok=True)
    images = list_images()
    if not images:
        print(f"No images found in {IMAGES_DIR.resolve()}")
        raise SystemExit(1)

    print(f"Found {len(images)} images")
    print(f"Seeds will be saved to: {SEED_DIR.resolve()}")
    print("Controls: drag box | r=reset | s=save next | n=skip | q=quit")

    for i, img_path in enumerate(images, 1):
        seed_path = SEED_DIR / f"{img_path.stem}.txt"
        if seed_path.exists():
            print(f"[{i}/{len(images)}] skip (seed exists): {img_path.name}")
            continue

        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[{i}/{len(images)}] could not read: {img_path.name}")
            continue

        h, w = image_bgr.shape[:2]
        print(f"[{i}/{len(images)}] {img_path.name}  ({w}x{h})")

        action, box = draw_seed_box(image_bgr)
        if action == "quit":
            print("  quit.")
            break
        if action == "skip":
            print("  skipped.")
            continue

        x1, y1, x2, y2 = box
        cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
        with open(seed_path, "w") as f:
            f.write(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        print(f"  saved: {seed_path.name}  (xyxy={x1},{y1},{x2},{y2})")

    print("Done.")
