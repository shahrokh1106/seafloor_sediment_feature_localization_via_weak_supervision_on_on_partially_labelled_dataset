"""
Local seed-prototype pseudo labels for dataset_burrow.

For each seeded image:
  1) extract dense features (DINOv3 or SAM3)
  2) cosine similarity from that frame's seed-center patch → affinity → CCA
  3) write YOLO = seed + non-conflicting pseudos
  4) evaluate vs labels_gt and save eval_local.json

Run from feature_matching_scripts/:
  python get_pseudos_local.py
"""

from pseudo_common import *


def generate_local_labels(name, extract_fn, seeded, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== [{name}] local seed prototype ===")
    for i, img_path in enumerate(seeded, 1):
        out_path = out_dir / f"{img_path.stem}.txt"
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size
        seed = load_seed_xyxy(SEED_DIR / f"{img_path.stem}.txt", orig_w, orig_h)
        if seed is None:
            print(f"[{name} {i}/{len(seeded)}] empty seed: {img_path.name}")
            continue

        x_grid = extract_fn(image)
        lines, n, n_drop = run_local_pipeline(img_path, x_grid, seed, orig_w, orig_h)
        save_yolo(out_path, lines)
        if i % 50 == 0 or i == len(seeded):
            print(
                f"[{name} {i}/{len(seeded)}] {img_path.name}: "
                f"{n} boxes (seed+pseudos, dropped {n_drop} seed-overlap) → {out_path.name}"
            )


if __name__ == "__main__":
    set_seed()
    seeded = list_seeded_images()
    if not seeded:
        print(f"No seeded images in {SEED_DIR.resolve()}")
        raise SystemExit(1)

    print(f"Found {len(seeded)} seeded images")
    print_settings("local")

    print("\nLoading DINOv3...")
    dino = load_dino()
    generate_local_labels("dino", lambda img: extract_dino_grid(dino, img), seeded, OUT_DINO_LOCAL)
    del dino
    clear_cuda()

    print("\nLoading SAM3...")
    sam, processor = load_sam()
    generate_local_labels(
        "sam", lambda img: extract_sam_grid(sam, processor, img), seeded, OUT_SAM_LOCAL
    )
    del sam, processor
    clear_cuda()

    evaluate_and_report("local", OUT_DINO_LOCAL, OUT_SAM_LOCAL, EVAL_JSON_LOCAL)
    print("\nDone.")
    print(f"  DINO labels → {OUT_DINO_LOCAL.resolve()}")
    print(f"  SAM3 labels → {OUT_SAM_LOCAL.resolve()}")
