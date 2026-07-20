"""
Global mean-prototype pseudo labels for dataset_burrow.

  1) Collect seed-center features from all seeded images (per backbone)
  2) Mean → global prototype
  3) Cosine vs proto on every image → affinity → CCA
  4) write YOLO = seed + non-conflicting pseudos
  5) evaluate vs labels_gt and save eval_global.json

Run from feature_matching_scripts/:
  python get_pseudos_global.py
"""

from pseudo_common import *


def generate_global_labels(name, extract_fn, seeded, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== [{name}] Pass 1: build mean seed prototype ===")
    seed_feats = []
    entries = []

    for i, img_path in enumerate(seeded, 1):
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        seed = load_seed_xyxy(SEED_DIR / f"{img_path.stem}.txt", orig_w, orig_h)
        if seed is None:
            print(f"[{name} {i}/{len(seeded)}] empty seed: {img_path.name}")
            continue
        seed_model = scale_box(seed, IMAGE_SIZE / orig_w, IMAGE_SIZE / orig_h)
        x_grid = extract_fn(image)
        seed_feats.append(seed_center_feature(x_grid, seed_model))
        entries.append((img_path, orig_w, orig_h, seed, x_grid))
        if i % 50 == 0 or i == len(seeded):
            print(f"  [{name}] collected {len(seed_feats)}/{i} seed features...")

    if not seed_feats:
        print(f"[{name}] No seed features collected.")
        return

    proto = F.normalize(torch.stack(seed_feats, dim=0).mean(dim=0), dim=0, eps=1e-8)
    print(
        f"  [{name}] Prototype: C={proto.shape[0]}, ||p||={float(proto.norm()):.4f}, "
        f"from {len(seed_feats)} seeds"
    )

    print(f"\n=== [{name}] Pass 2: global-proto cosine + affinity + CCA ===")
    for i, (img_path, orig_w, orig_h, seed_xyxy, x_grid) in enumerate(entries, 1):
        out_path = out_dir / f"{img_path.stem}.txt"
        lines, n, n_drop = run_global_pipeline(x_grid, proto, seed_xyxy, orig_w, orig_h)
        save_yolo(out_path, lines)
        if i % 50 == 0 or i == len(entries):
            print(
                f"[{name} {i}/{len(entries)}] {img_path.name}: "
                f"{n} boxes (seed+pseudos, dropped {n_drop} seed-overlap) → {out_path.name}"
            )


if __name__ == "__main__":
    set_seed()
    seeded = list_seeded_images()
    if not seeded:
        print(f"No seeded images in {SEED_DIR.resolve()}")
        raise SystemExit(1)

    print(f"Found {len(seeded)} seeded images")
    print_settings("global")

    print("\nLoading DINOv3...")
    dino = load_dino()
    generate_global_labels("dino", lambda img: extract_dino_grid(dino, img), seeded, OUT_DINO_GLOBAL)
    del dino
    clear_cuda()

    print("\nLoading SAM3...")
    sam, processor = load_sam()
    generate_global_labels(
        "sam", lambda img: extract_sam_grid(sam, processor, img), seeded, OUT_SAM_GLOBAL
    )
    del sam, processor
    clear_cuda()

    evaluate_and_report("global", OUT_DINO_GLOBAL, OUT_SAM_GLOBAL, EVAL_JSON_GLOBAL)
    print("\nDone.")
    print(f"  DINO labels → {OUT_DINO_GLOBAL.resolve()}")
    print(f"  SAM3 labels → {OUT_SAM_GLOBAL.resolve()}")
