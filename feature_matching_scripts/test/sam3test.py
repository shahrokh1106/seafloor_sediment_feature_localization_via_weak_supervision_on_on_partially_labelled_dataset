import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils import *
from transformers import Sam3Model, Sam3Processor

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

DATASET_PATH = os.path.join("test","output")
os.makedirs(DATASET_PATH, exist_ok=True)
PATCH_SIZE = 16
IMAGE_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_str = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "facebook/sam3"
IMG_PATH = os.path.join("test","images", "b.jpg")
DEBUG = True
OUTPUT = {}
OUTPUTSIZE = IMAGE_SIZE // 2
SHOWSCALE = 100
MASK_PERCENTILE = 95
OUTPUT_PATH = os.path.join(DATASET_PATH, "sam3outputs")
os.makedirs(OUTPUT_PATH, exist_ok=True)

if __name__ == "__main__":
    model = Sam3Model.from_pretrained(MODEL_ID).to(DEVICE)
    processor = Sam3Processor.from_pretrained(MODEL_ID)
    model.eval()

    image = Image.open(IMG_PATH).convert("RGB")
    # SAM3 processor resizes to its native input (1008); features are then
    # reshaped/resized onto the same HxW patch grid utils expects.
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(DEVICE)

    with torch.inference_mode():
        with torch.autocast(device_type=DEVICE_str, dtype=torch.float32):
            vision_out = model.get_vision_features(pixel_values=pixel_values)
            # last_hidden_state: [1, 72*72, 1024] on 1008x1008 input
            x = vision_out.last_hidden_state.squeeze(0).detach().cpu()  # [N, C]
            side = int(x.shape[0] ** 0.5)
            x = x.view(side, side, x.shape[1]).permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
            h_patches = IMAGE_SIZE // PATCH_SIZE
            w_patches = IMAGE_SIZE // PATCH_SIZE
            x_grid = F.interpolate(x, size=(h_patches, w_patches), mode="bilinear", align_corners=False)

    xgrid_show = x_grid[0].detach().cpu().numpy()
    xgrid_show = np.mean(xgrid_show, axis=0)
    xgrid_show = cv2.normalize(xgrid_show, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    xgrid_show = cv2.resize(xgrid_show, (IMAGE_SIZE, IMAGE_SIZE))
    xgrid_show = cv2.applyColorMap(xgrid_show, cv2.COLORMAP_INFERNO)
    OUTPUT.update({"mean_features": xgrid_show})

    #################### MASK GENERATION USING BOX #####################
    # SAM3 backbone features
    sim_map_box_rgb, sim_map_box, box = get_sim_map_box(IMG_PATH, x_grid, IMAGE_SIZE, PATCH_SIZE, SHOWSCALE, DEBUG, cmap=cv2.COLORMAP_INFERNO)
    overlay_heatmap_box = get_overlay_heatmap(sim_map_box_rgb, cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)), DEBUG, SHOWSCALE)
    OUTPUT.update({"sim_map_box_rgb": sim_map_box_rgb})
    OUTPUT.update({"overlay_heatmap_box": overlay_heatmap_box})
    mask_box_rgb, mask_box = get_affinity_mask(
        sim_map_box, x_grid, DEVICE, SHOWSCALE, DEBUG, cv2.COLORMAP_INFERNO,
        mask_percentile=MASK_PERCENTILE,
    )
    overlay_mask_box = get_overlay_heatmap(mask_box_rgb, cv2.resize(cv2.imread(IMG_PATH), (IMAGE_SIZE, IMAGE_SIZE)), DEBUG, SHOWSCALE)
    OUTPUT.update({"mask_box_rgb": mask_box_rgb})
    OUTPUT.update({"mask_box_binary": (mask_box).astype(np.uint8)})
    OUTPUT.update({"overlay_mask_box": overlay_mask_box})

    boxes, num_boxes, component_stats = get_bounding_boxes_from_mask(mask_box, min_area=100)
    print(f"Found {num_boxes} bounding boxes from mask")

    image_bgr_full = cv2.imread(IMG_PATH)
    image_bgr_full = cv2.resize(image_bgr_full, (IMAGE_SIZE, IMAGE_SIZE))
    image_with_boxes = visualize_bounding_boxes(image_bgr_full, boxes, color=(0, 255, 0), thickness=3)
    OUTPUT.update({"image_with_boxes": image_with_boxes})

    for key in OUTPUT.keys():
        if key != "feat_rgb" and key != "mean_features" and key != "image_with_boxes":
            OUTPUT[key] = cv2.resize(OUTPUT[key], (OUTPUTSIZE, OUTPUTSIZE))

    image_bgr = cv2.imread(IMG_PATH)
    OUTPUT.update({"image": cv2.resize(image_bgr, (OUTPUTSIZE, OUTPUTSIZE))})

    if "image_with_boxes" in OUTPUT:
        OUTPUT["image_with_boxes"] = cv2.resize(OUTPUT["image_with_boxes"], (OUTPUTSIZE, OUTPUTSIZE))

    OUTPUT_PATH_ = os.path.join(OUTPUT_PATH, str(IMAGE_SIZE), os.path.basename(IMG_PATH[:-4]))
    print(OUTPUT_PATH_)
    os.makedirs(OUTPUT_PATH_, exist_ok=True)
    for key in OUTPUT.keys():
        cv2.imwrite(os.path.join(OUTPUT_PATH_, key + ".png"), OUTPUT[key])

    print("ALL GOOD")
