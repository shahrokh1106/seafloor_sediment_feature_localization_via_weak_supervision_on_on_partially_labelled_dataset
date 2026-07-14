from utils import *
import yaml
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from collections import defaultdict
from sklearn.decomposition import PCA
import pandas as pd


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(SEED)

PATCH_SIZE = 16
IMAGE_SIZE = 1024
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE_str = "cuda" if torch.cuda.is_available() else "cpu"
REPO_DIR = "dinov3"

# Dataset paths
DATASET_PATH = "../detector_dataset_simple"
VAL_TXT = os.path.join(DATASET_PATH, "val.txt")
DATA_YAML = os.path.join(DATASET_PATH, "data.yaml")
OUTPUT_DIR = "dinov3_validation_set"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_yolo_annotations(label_path, img_width, img_height):
    """
    Load YOLO format annotations and convert to pixel coordinates.

    Args:
        label_path: Path to YOLO label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        boxes: List of [class_id, x1, y1, x2, y2] in pixel coordinates
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2

                boxes.append([class_id, int(x1), int(y1), int(x2), int(y2)])
            except (ValueError, IndexError):
                continue

    return boxes

def extract_box_features_from_grid(x_grid, box, img_width, img_height):
    """
    Extract DINOv3 features for the entire bounding box by averaging features over all patches within the box.

    Args:
        x_grid: Feature tensor [1, C, H, W] at patch resolution
        box: [class_id, x1, y1, x2, y2] in pixel coordinates of original image
        img_width: Original image width
        img_height: Original image height

    Returns:
        box_feature: Feature vector [C] averaged over all patches in the box
    """
    _, x1, y1, x2, y2 = box

    # Get feature grid dimensions
    h_patches, w_patches = x_grid.shape[2], x_grid.shape[3]

    # Scale coordinates from original image to IMAGE_SIZE
    scale_x = IMAGE_SIZE / img_width
    scale_y = IMAGE_SIZE / img_height

    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    # Convert to patch coordinates
    x1_patch = max(0, min(w_patches - 1, int(x1_scaled / PATCH_SIZE)))
    y1_patch = max(0, min(h_patches - 1, int(y1_scaled / PATCH_SIZE)))
    x2_patch = max(0, min(w_patches - 1, int(np.ceil(x2_scaled / PATCH_SIZE))))
    y2_patch = max(0, min(h_patches - 1, int(np.ceil(y2_scaled / PATCH_SIZE))))

    # Extract features for all patches within the box
    feats = x_grid[0].detach().cpu().numpy()  # [C, H, W]
    box_patches = feats[:, y1_patch:y2_patch+1, x1_patch:x2_patch+1]  # [C, H_box, W_box]

    # Average over spatial dimensions to get a single feature vector
    box_feature = box_patches.mean(axis=(1, 2))  # [C]

    return box_feature

def compute_class_feature_metrics(features_dict):
    """
    Compute per-class centroids and intra-class cosine distances from centroid.
    """
    metrics = {
        'intra_class_distances': {},
        'class_centroids': {},
    }

    for class_id, feats_list in features_dict.items():
        if len(feats_list) == 0:
            continue
        feats_array = np.array(feats_list)
        centroid = feats_array.mean(axis=0)
        metrics['class_centroids'][class_id] = centroid

        feats_norm = feats_array / (np.linalg.norm(feats_array, axis=1, keepdims=True) + 1e-8)
        centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-8)

        distances = 1 - (feats_norm @ centroid_norm)
        metrics['intra_class_distances'][class_id] = {
            'mean': distances.mean(),
            'std': distances.std(),
            'min': distances.min(),
            'max': distances.max(),
        }

    return metrics


def visualize_intra_class_distances(metrics, class_names, output_dir):
    """
    Visualize intra-class distances as a separate figure.
    Colors represent distance magnitude (darker = higher variability).
    Classes are sorted by distance (highest at top).
    """
    intra_data = []
    for class_id, dist_info in metrics['intra_class_distances'].items():
        intra_data.append({
            'Class': class_names[class_id],
            'Mean Distance': dist_info['mean'],
            'Std Distance': dist_info['std']
        })

    if not intra_data:
        return

    df_intra = pd.DataFrame(intra_data)
    df_intra = df_intra.sort_values('Mean Distance', ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    bars = ax.barh(df_intra['Class'], df_intra['Mean Distance'],
                   xerr=df_intra['Std Distance'], capsize=5)
    ax.set_xlabel('Mean Cosine Distance from Centroid', fontsize=22)
    ax.set_title('Intra-Class Feature Variability\n(Lower = more consistent features)', fontsize=24)
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, 1)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)

    max_dist = df_intra['Mean Distance'].max()
    for i, bar in enumerate(bars):
        dist = df_intra['Mean Distance'].iloc[i]
        normalized_dist = dist / max_dist if max_dist > 0 else 0
        bar.set_color(plt.cm.viridis(normalized_dist))

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                               norm=Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Distance Magnitude', pad=0.02)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label('Distance Magnitude', fontsize=22)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'intra_class_distances.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved intra-class distances visualization to {output_path}")

def visualize_class_centroids_pca(metrics, class_names, output_dir):
    """
    Visualize class centroids in 2D using PCA.
    Each class is represented by its centroid in the reduced feature space.
    """
    centroids = []
    centroid_class_ids = []

    for class_id in sorted(metrics['class_centroids'].keys()):
        centroids.append(metrics['class_centroids'][class_id])
        centroid_class_ids.append(class_id)

    if len(centroids) < 2:
        print("Not enough classes for PCA visualization")
        return

    centroids_array = np.array(centroids)
    centroids_norm = centroids_array / (np.linalg.norm(centroids_array, axis=1, keepdims=True) + 1e-8)

    pca = PCA(n_components=2, random_state=SEED)
    centroids_2d = pca.fit_transform(centroids_norm)

    fig, ax = plt.subplots(figsize=(12, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, len(class_names)))

    for i, class_id in enumerate(centroid_class_ids):
        ax.scatter(centroids_2d[i, 0], centroids_2d[i, 1],
                  c=[colors[class_id]], label=class_names[class_id],
                  s=200, alpha=0.8, edgecolors='black', linewidths=2)

    explained_var = pca.explained_variance_ratio_
    total_var = explained_var.sum()

    for i, class_id in enumerate(centroid_class_ids):
        ax.annotate(class_names[class_id],
                   (centroids_2d[i, 0], centroids_2d[i, 1]),
                   xytext=(0, -15), textcoords='offset points',
                   fontsize=16, alpha=0.8, fontweight='bold',
                   ha='center', va='top')

    ax.set_xlabel(f'PC1 ({explained_var[0]*100:.1f}% variance)', fontsize=22)
    ax.set_ylabel(f'PC2 ({explained_var[1]*100:.1f}% variance)', fontsize=22)
    ax.set_title(f'Class Centroids in 2D Feature Space (PCA)\nTotal Variance Explained: {total_var*100:.1f}%', fontsize=24)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=20)

    x_min, x_max = centroids_2d[:, 0].min(), centroids_2d[:, 0].max()
    y_min, y_max = centroids_2d[:, 1].min(), centroids_2d[:, 1].max()
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.15 * y_range, y_max + 0.05 * y_range)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'class_centroids_pca.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PCA visualization of class centroids to {output_path}")


if __name__ == "__main__":
    print("Loading dataset configuration...")
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)

    class_names = [data_config['names'][str(i)] for i in range(data_config['nc'])]
    print(f"Classes: {class_names}")

    print("Loading validation images...")
    val_images = []
    with open(VAL_TXT, 'r') as f:
        for line in f:
            img_path = line.strip()
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(DATASET_PATH), img_path)
            val_images.append(img_path)

    print(f"Found {len(val_images)} validation images")

    print("Loading DINOv3 model...")
    model = torch.hub.load(REPO_DIR, 'dinov3_vitl16', source='local', weights="dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth")
    model.to(DEVICE)
    model.eval()

    print("\nExtracting DINOv3 features from annotated boxes...")
    print("Strategy: Extract full image features once, then extract box features from that")

    features_dict = defaultdict(list)

    processed = 0
    for img_path in val_images:
        if not os.path.exists(img_path):
            continue

        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        label_path = img_path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt')
        boxes = load_yolo_annotations(label_path, img_width, img_height)

        if len(boxes) == 0:
            continue

        image_resized = resize_transform(image, IMAGE_SIZE, PATCH_SIZE)
        image_resized_norm = TF.normalize(image_resized, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

        with torch.inference_mode():
            with torch.autocast(device_type=DEVICE_str, dtype=torch.float32):
                feats = model.get_intermediate_layers(image_resized_norm.unsqueeze(0).to(DEVICE),
                                                     n=range(24), reshape=True, norm=True)
                x = feats[-1].squeeze().detach().cpu()
                x = x.view(x.shape[0], -1)
                x = x.permute(1, 0)

        h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]
        x_grid = x.view(h_patches, w_patches, x.shape[1]).permute(2, 0, 1).unsqueeze(0)

        for box in boxes:
            box_feature = extract_box_features_from_grid(x_grid, box, img_width, img_height)
            if box_feature is not None:
                features_dict[box[0]].append(box_feature)

        processed += 1
        if processed % 10 == 0:
            print(f"  Processed {processed}/{len(val_images)} images...")

    print(f"\nExtracted features from {sum(len(v) for v in features_dict.values())} bounding boxes")
    print(f"Distribution by class:")
    for class_id in sorted(features_dict.keys()):
        print(f"  {class_names[class_id]:20s}: {len(features_dict[class_id])} boxes")

    print("\nComputing feature quality metrics...")
    metrics = compute_class_feature_metrics(features_dict)

    print("\nCreating visualizations...")
    visualize_intra_class_distances(metrics, class_names, OUTPUT_DIR)
    visualize_class_centroids_pca(metrics, class_names, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print("=" * 70)
