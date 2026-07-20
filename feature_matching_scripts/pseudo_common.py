"""
Shared helpers for local- and global-prototype pseudo-label generation.
"""

from __future__ import annotations

import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou
from transformers import Sam3Model, Sam3Processor

from utils import (
    TF,
    get_affinity_mask,
    get_bounding_boxes_from_mask,
    get_sim_map_box,
    resize_transform,
)

# --- shared pipeline settings (keep in sync across local/global scripts) ---
SEED = 42
PATCH_SIZE = 16
IMAGE_SIZE = 512
MASK_PERCENTILE = 95
MIN_AREA = 200
SEED_DUP_IOU = 0.3
SEED_COVER = 0.5
DEBUG = False
SHOWSCALE = 100
CLASS_ID = 0
IOU_THRESHOLDS = (0.50, 0.75) # for evaluation

ROOT = Path(__file__).resolve().parent
REPO_DIR = str(ROOT / "dinov3")
WEIGHTS = str(ROOT / "dinov3" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
MODEL_ID = "facebook/sam3"

DATASET_DIR = Path("dataset_burrow")
IMAGES_DIR = DATASET_DIR / "images"
SEED_DIR = DATASET_DIR / "labels_seed"
GT_DIR = DATASET_DIR / "labels_gt"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

OUT_DINO_LOCAL = DATASET_DIR / "labels_dino_local"
OUT_SAM_LOCAL = DATASET_DIR / "labels_sam_local"
OUT_DINO_GLOBAL = DATASET_DIR / "labels_dino_global"
OUT_SAM_GLOBAL = DATASET_DIR / "labels_sam_global"
EVAL_JSON_LOCAL = DATASET_DIR / "eval_local.json"
EVAL_JSON_GLOBAL = DATASET_DIR / "eval_global.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_str = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(SEED)


def pipeline_settings(mode: str) -> dict:
    return {
        "mode": mode,
        "IMAGE_SIZE": IMAGE_SIZE,
        "PATCH_SIZE": PATCH_SIZE,
        "MASK_PERCENTILE": MASK_PERCENTILE,
        "MIN_AREA": MIN_AREA,
        "SEED_DUP_IOU": SEED_DUP_IOU,
        "SEED_COVER": SEED_COVER,
        "CLASS_ID": CLASS_ID,
        "IOU_THRESHOLDS": list(IOU_THRESHOLDS),
    }


def yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h):
    w, h = bw * img_w, bh * img_h
    return [cx * img_w - w / 2, cy * img_h - h / 2, cx * img_w + w / 2, cy * img_h + h / 2]


def xyxy_to_yolo(x1, y1, x2, y2, img_w, img_h):
    x1, x2 = sorted((max(0.0, min(img_w - 1.0, x1)), max(0.0, min(img_w - 1.0, x2))))
    y1, y2 = sorted((max(0.0, min(img_h - 1.0, y1)), max(0.0, min(img_h - 1.0, y2))))
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    return cx, cy, bw / img_w, bh / img_h


def _inter_area(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    return iw * ih


def box_iou_xyxy(a, b):
    inter = _inter_area(a, b)
    if inter <= 0:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def conflicts_with_seed(pseudo, seed, iou_thr=SEED_DUP_IOU, cover_thr=SEED_COVER):
    sx1, sy1, sx2, sy2 = seed
    px1, py1, px2, py2 = pseudo
    scx, scy = 0.5 * (sx1 + sx2), 0.5 * (sy1 + sy2)
    if px1 <= scx <= px2 and py1 <= scy <= py2:
        return True
    seed_area = max(0.0, sx2 - sx1) * max(0.0, sy2 - sy1)
    if seed_area <= 0:
        return False
    inter = _inter_area(pseudo, seed)
    if inter / seed_area >= cover_thr:
        return True
    return box_iou_xyxy(pseudo, seed) >= iou_thr


def load_seed_xyxy(seed_path, img_w, img_h):
    with open(seed_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            return yolo_to_xyxy(float(cx), float(cy), float(bw), float(bh), img_w, img_h)
    return None


def load_yolo_xyxy(path: Path) -> torch.Tensor:
    boxes = []
    if path.exists():
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                _, cx, cy, w, h = map(float, parts[:5])
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    if not boxes:
        return torch.zeros((0, 4), dtype=torch.float32)
    return torch.tensor(boxes, dtype=torch.float32)


def scale_box(box, sx, sy):
    x1, y1, x2, y2 = box
    return [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]


def list_seeded_images():
    images = sorted(
        p for p in IMAGES_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )
    return [p for p in images if (SEED_DIR / f"{p.stem}.txt").exists()]


def extract_dino_grid(model, image_pil):
    image_resized = resize_transform(image_pil, IMAGE_SIZE, PATCH_SIZE)
    image_resized_norm = TF.normalize(
        image_resized, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE_str, dtype=torch.float32):
            feats = model.get_intermediate_layers(
                image_resized_norm.unsqueeze(0).to(DEVICE),
                n=range(24), reshape=True, norm=True,
            )
            x = feats[-1].squeeze().detach().cpu()
            x = x.view(x.shape[0], -1).permute(1, 0)
    h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
    return x.view(h_patches, w_patches, x.shape[1]).permute(2, 0, 1).unsqueeze(0)


def extract_sam_grid(model, processor, image_pil):
    inputs = processor(images=image_pil, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)
    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE_str, dtype=torch.float32):
            vision_out = model.get_vision_features(pixel_values=pixel_values)
            x = vision_out.last_hidden_state.squeeze(0).detach().cpu()
            side = int(x.shape[0] ** 0.5)
            x = x.view(side, side, x.shape[1]).permute(2, 0, 1).unsqueeze(0)
            h_patches = IMAGE_SIZE // PATCH_SIZE
            w_patches = IMAGE_SIZE // PATCH_SIZE
            x = F.interpolate(x, size=(h_patches, w_patches), mode="bilinear", align_corners=False)
    return x


def seed_center_feature(x_grid, seed_model_xyxy):
    feat = x_grid[0].detach().float()
    _, Hp, Wp = feat.shape
    x1, y1, x2, y2 = seed_model_xyxy
    cx_p = max(0, min(Wp - 1, int(np.floor(0.5 * (x1 + x2) / PATCH_SIZE))))
    cy_p = max(0, min(Hp - 1, int(np.floor(0.5 * (y1 + y2) / PATCH_SIZE))))
    return feat[:, cy_p, cx_p].clone()


def sim_map_from_proto(x_grid, proto):
    feat = x_grid[0].detach().float()
    C, Hp, Wp = feat.shape
    proto_n = F.normalize(proto.float(), dim=0, eps=1e-8)
    flat = F.normalize(feat.reshape(C, -1), dim=0, eps=1e-8)
    sim = torch.matmul(proto_n, flat).reshape(Hp, Wp)
    sim_local = sim.detach().cpu().numpy().astype(np.float32)
    sim_up = cv2.resize(sim_local, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    return np.clip(sim_up, 0.0, 1.0).astype(np.float32)


def cca_boxes_from_sim(sim_map, x_grid):
    _, mask = get_affinity_mask(
        sim_map, x_grid, DEVICE, SHOWSCALE, DEBUG, cv2.COLORMAP_INFERNO,
        mask_percentile=MASK_PERCENTILE,
    )
    boxes_model, _, _ = get_bounding_boxes_from_mask(mask, min_area=MIN_AREA)
    return boxes_model


def assemble_yolo_lines(boxes_model, seed_xyxy_orig, orig_w, orig_h):
    """Seed first + pseudos that do not conflict with the seed region."""
    sx1, sy1, sx2, sy2 = seed_xyxy_orig
    scx, scy, sbw, sbh = xyxy_to_yolo(sx1, sy1, sx2, sy2, orig_w, orig_h)
    lines = [f"{CLASS_ID} {scx:.6f} {scy:.6f} {sbw:.6f} {sbh:.6f}"]

    sx, sy = orig_w / IMAGE_SIZE, orig_h / IMAGE_SIZE
    n_dropped = 0
    for (x1, y1, x2, y2) in boxes_model:
        ox1, oy1, ox2, oy2 = x1 * sx, y1 * sy, x2 * sx, y2 * sy
        if conflicts_with_seed([ox1, oy1, ox2, oy2], seed_xyxy_orig):
            n_dropped += 1
            continue
        cx, cy, bw, bh = xyxy_to_yolo(ox1, oy1, ox2, oy2, orig_w, orig_h)
        lines.append(f"{CLASS_ID} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    return lines, len(lines), n_dropped


def run_local_pipeline(img_path, x_grid, seed_xyxy_orig, orig_w, orig_h):
    seed_model = scale_box(seed_xyxy_orig, IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h)
    _, sim_map, _ = get_sim_map_box(
        str(img_path), x_grid, IMAGE_SIZE, PATCH_SIZE, SHOWSCALE, DEBUG,
        cmap=cv2.COLORMAP_INFERNO, box_org=seed_model,
    )
    boxes_model = cca_boxes_from_sim(sim_map, x_grid)
    return assemble_yolo_lines(boxes_model, seed_xyxy_orig, orig_w, orig_h)


def run_global_pipeline(x_grid, proto, seed_xyxy_orig, orig_w, orig_h):
    sim_map = sim_map_from_proto(x_grid, proto)
    boxes_model = cca_boxes_from_sim(sim_map, x_grid)
    return assemble_yolo_lines(boxes_model, seed_xyxy_orig, orig_w, orig_h)


def save_yolo(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")


def to_tm_dicts(pred_boxes, gt_boxes):
    n_p, n_g = pred_boxes.shape[0], gt_boxes.shape[0]
    pred = {
        "boxes": pred_boxes,
        "scores": torch.ones(n_p, dtype=torch.float32),
        "labels": torch.full((n_p,), CLASS_ID, dtype=torch.int64),
    }
    target = {
        "boxes": gt_boxes,
        "labels": torch.full((n_g,), CLASS_ID, dtype=torch.int64),
    }
    return pred, target


def greedy_match_counts(pred_boxes, gt_boxes, thr):
    if pred_boxes.numel() == 0 or gt_boxes.numel() == 0:
        return 0, []
    ious = box_iou(pred_boxes, gt_boxes)
    flat = ious.reshape(-1)
    order = torch.argsort(flat, descending=True)
    ng = ious.shape[1]
    used_p, used_g = set(), set()
    matched = []
    for idx in order.tolist():
        iou = float(flat[idx])
        if iou < thr:
            break
        i, j = divmod(idx, ng)
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matched.append(iou)
    return len(matched), matched


def evaluate_set(pred_dir: Path, gt_dir: Path, stems):
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=False)
    metric.warn_on_many_detections = False

    n_gt = n_pred = 0
    tp = {t: 0 for t in IOU_THRESHOLDS}
    best_gt, best_pred = [], []
    matched_ious_50 = []

    for stem in stems:
        gt = load_yolo_xyxy(gt_dir / f"{stem}.txt")
        pred = load_yolo_xyxy(pred_dir / f"{stem}.txt")
        n_gt += gt.shape[0]
        n_pred += pred.shape[0]

        p_dict, t_dict = to_tm_dicts(pred, gt)
        metric.update([p_dict], [t_dict])

        if gt.shape[0] > 0 and pred.shape[0] > 0:
            ious = box_iou(pred, gt)
            best_gt.extend(ious.max(dim=0).values.tolist())
            best_pred.extend(ious.max(dim=1).values.tolist())
        elif gt.shape[0] > 0:
            best_gt.extend([0.0] * gt.shape[0])
        elif pred.shape[0] > 0:
            best_pred.extend([0.0] * pred.shape[0])

        for thr in IOU_THRESHOLDS:
            tps, matched = greedy_match_counts(pred, gt, thr)
            tp[thr] += tps
            if thr == 0.50:
                matched_ious_50.extend(matched)

    map_out = metric.compute()

    def _f(x):
        v = float(x.detach().cpu()) if torch.is_tensor(x) else float(x)
        return float("nan") if v < 0 else v

    results = {
        "images": len(stems),
        "gt_boxes": n_gt,
        "pred_boxes": n_pred,
        "avg_gt_per_img": n_gt / max(1, len(stems)),
        "avg_pred_per_img": n_pred / max(1, len(stems)),
        "map": _f(map_out["map"]),
        "map_50": _f(map_out["map_50"]),
        "map_75": _f(map_out["map_75"]),
        "mar_100": _f(map_out["mar_100"]),
        "mean_best_iou_gt": float(torch.tensor(best_gt).mean()) if best_gt else 0.0,
        "mean_best_iou_pred": float(torch.tensor(best_pred).mean()) if best_pred else 0.0,
        "mean_matched_iou_50": float(torch.tensor(matched_ious_50).mean()) if matched_ious_50 else 0.0,
        "by_thr": {},
    }
    for thr in IOU_THRESHOLDS:
        tps = tp[thr]
        prec = tps / n_pred if n_pred > 0 else 0.0
        rec = tps / n_gt if n_gt > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        results["by_thr"][f"{thr:.2f}"] = {
            "tp": tps,
            "fp": n_pred - tps,
            "fn": n_gt - tps,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }
    return results


def print_comparison(r_dino, r_sam, title=""):
    if title:
        print(f"\n=== {title} ===")

    def _fmt(v):
        return f"{v:10.4f}" if v == v else f"{'nan':>10}"

    print(f"\n  {'Metric':<28} {'DINO':>10} {'SAM3':>10}")
    print("-" * 52)
    print(f"  {'Images':<28} {r_dino['images']:10d} {r_sam['images']:10d}")
    print(f"  {'GT boxes (total)':<28} {r_dino['gt_boxes']:10d} {r_sam['gt_boxes']:10d}")
    print(f"  {'Detected boxes (total)':<28} {r_dino['pred_boxes']:10d} {r_sam['pred_boxes']:10d}")
    print(f"  {'Avg boxes / img':<28} {_fmt(r_dino['avg_pred_per_img'])} {_fmt(r_sam['avg_pred_per_img'])}")
    print(f"  {'Avg GT / img':<28} {_fmt(r_dino['avg_gt_per_img'])} {_fmt(r_sam['avg_gt_per_img'])}")
    for key, label in [
        ("map", "mAP"),
        ("map_50", "mAP50"),
        ("map_75", "mAP75"),
        ("mar_100", "mAR@100"),
        ("mean_best_iou_gt", "Mean best IoU / GT"),
        ("mean_best_iou_pred", "Mean best IoU / pred"),
    ]:
        print(f"  {label:<28} {_fmt(r_dino[key])} {_fmt(r_sam[key])}")
    for thr in IOU_THRESHOLDS:
        key = f"{thr:.2f}"
        md, ms = r_dino["by_thr"][key], r_sam["by_thr"][key]
        print(f"  {'TP@' + key:<28} {md['tp']:10d} {ms['tp']:10d}")
        print(f"  {'FP@' + key:<28} {md['fp']:10d} {ms['fp']:10d}")
        print(f"  {'FN@' + key:<28} {md['fn']:10d} {ms['fn']:10d}")
        print(f"  {'F1@' + key:<28} {md['f1']:10.4f} {ms['f1']:10.4f}")
        print(f"  {'Precision@' + key:<28} {md['precision']:10.4f} {ms['precision']:10.4f}")
        print(f"  {'Recall@' + key:<28} {md['recall']:10.4f} {ms['recall']:10.4f}")


def save_eval_json(path: Path, mode: str, r_dino: dict, r_sam: dict):
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "settings": pipeline_settings(mode),
        "labels_dino": str((OUT_DINO_LOCAL if mode == "local" else OUT_DINO_GLOBAL).resolve()),
        "labels_sam": str((OUT_SAM_LOCAL if mode == "local" else OUT_SAM_GLOBAL).resolve()),
        "dino": r_dino,
        "sam": r_sam,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved evaluation → {path.resolve()}")


def evaluate_and_report(mode: str, out_dino: Path, out_sam: Path, eval_json: Path):
    gt_stems = {p.stem for p in GT_DIR.glob("*.txt")}
    dino_stems = {p.stem for p in out_dino.glob("*.txt")}
    sam_stems = {p.stem for p in out_sam.glob("*.txt")}
    stems = sorted(gt_stems & dino_stems & sam_stems)
    if not stems:
        print(f"No overlapping labels with GT under {DATASET_DIR}")
        raise SystemExit(1)

    r_dino = evaluate_set(out_dino, GT_DIR, stems)
    r_sam = evaluate_set(out_sam, GT_DIR, stems)
    print_comparison(r_dino, r_sam, title=f"{mode.capitalize()} prototype metrics ({len(stems)} images)")
    save_eval_json(eval_json, mode, r_dino, r_sam)
    return r_dino, r_sam


def print_settings(mode: str):
    s = pipeline_settings(mode)
    print(f"Found seeded images under {SEED_DIR.resolve()}")
    print(
        f"Mode={mode} | IMAGE_SIZE={s['IMAGE_SIZE']}, PATCH_SIZE={s['PATCH_SIZE']}, "
        f"MASK_PERCENTILE={s['MASK_PERCENTILE']}, MIN_AREA={s['MIN_AREA']}, "
        f"SEED_DUP_IOU={s['SEED_DUP_IOU']}, SEED_COVER={s['SEED_COVER']}"
    )


def load_dino():
    model = torch.hub.load(REPO_DIR, "dinov3_vitl16", source="local", weights=WEIGHTS)
    model.to(DEVICE).eval()
    return model


def load_sam():
    model = Sam3Model.from_pretrained(MODEL_ID).to(DEVICE)
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model.eval()
    return model, processor


def clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
