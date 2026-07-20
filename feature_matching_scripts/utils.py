import os
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as F
import cv2
import random
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

import sys

SEED = 42
random.seed(SEED)                   
np.random.seed(SEED)               
torch.manual_seed(SEED)            
torch.cuda.manual_seed(SEED)       
torch.cuda.manual_seed_all(SEED)   
torch.backends.cudnn.deterministic = True   
torch.backends.cudnn.benchmark = False      
os.environ['PYTHONHASHSEED'] = str(SEED)   


class TorchPCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
    def fit(self, X):
        self.mean_ = X.mean(dim=0)
        unbiased = X - self.mean_.unsqueeze(0)
        U, S, V = torch.pca_lowrank(unbiased, q=self.n_components, center=False, niter=4)
        self.components_ = V.T
        self.singular_values_ = S
        return self
    def transform(self, X):
        t0 = X - self.mean_.unsqueeze(0)
        projected = t0 @ self.components_.T
        return projected

def pca(image_feats_list, dim=3, fit_pca=None, use_torch_pca=True, max_samples=None):
    device = image_feats_list[0].device
    def flatten(tensor, target_size=None):
        if target_size is not None and fit_pca is None:
            tensor = F.interpolate(tensor, (target_size, target_size), mode="bilinear")
        B, C, H, W = tensor.shape
        return tensor.permute(1, 0, 2, 3).reshape(C, B * H * W).permute(1, 0).detach().cpu()
    if len(image_feats_list) > 1 and fit_pca is None:
        target_size = image_feats_list[0].shape[2]
    else:
        target_size = None

    flattened_feats = []
    for feats in image_feats_list:
        flattened_feats.append(flatten(feats, target_size))
    x = torch.cat(flattened_feats, dim=0)
    # Subsample the data if max_samples is set and the number of samples exceeds max_samples
    if max_samples is not None and x.shape[0] > max_samples:
        indices = torch.randperm(x.shape[0])[:max_samples]
        x = x[indices]

    if fit_pca is None:
        if use_torch_pca:
            fit_pca = TorchPCA(n_components=dim).fit(x)
        else:
            fit_pca = PCA(n_components=dim).fit(x)
    reduced_feats = []
    for feats in image_feats_list:
        x_red = fit_pca.transform(flatten(feats))
        if isinstance(x_red, np.ndarray):
            x_red = torch.from_numpy(x_red)
        x_red -= x_red.min(dim=0, keepdim=True).values
        x_red /= x_red.max(dim=0, keepdim=True).values
        B, C, H, W = feats.shape
        reduced_feats.append(x_red.reshape(B, H, W, dim).permute(0, 3, 1, 2).to(device))
    return reduced_feats, fit_pca


def resize_transform(mask_image: Image,image_size,patch_size):
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    # w_patches = int((w * image_size) / (h * patch_size))
    w_patches =  int(image_size / patch_size)
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))

def get_rgb_feature_map(x_grid,input_size,show_scale, debug):
    [feats_pca, _], _ = pca([x_grid, x_grid])
    feat = feats_pca[0]
    feat_np = feat.permute(1, 2, 0).detach().cpu().numpy()
    feat_norm = cv2.normalize(feat_np, None, 0, 255,norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    feat_norm = cv2.resize(feat_norm, (input_size,input_size))
    w = int(feat_norm.shape[1] * show_scale / 100)
    h = int(feat_norm.shape[0] * show_scale / 100)
    feat_norm_show = cv2.resize(feat_norm, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("Feature map", feat_norm_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return feat_norm


def get_roi(image,show_scale = 300):
    BOX_COLOR = (0, 0, 255)  # red (BGR)
    drawing = False

    def draw_rectangle(event, x, y, flags, params):
        nonlocal drawing
        global rectangle_points, img, img_orig
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            rectangle_points = [(x, y)]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            img = img_orig.copy()
            cv2.rectangle(img, rectangle_points[0], (x, y), BOX_COLOR, 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(rectangle_points) == 1:
                rectangle_points.append((x, y))
            else:
                rectangle_points[1] = (x, y)
            img = img_orig.copy()
            cv2.rectangle(img, rectangle_points[0], rectangle_points[1], BOX_COLOR, 2)

    global img,img_orig,rectangle_points
    rectangle_points = []
    
    w = int(image.shape[1] * show_scale / 100)
    h = int(image.shape[0] * show_scale / 100)
    image = cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)
    img = image.copy()
    img_orig = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_rectangle)
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): # If 'r' is pressed, reset the cropping region
            img = img_orig.copy()
            rectangle_points = []
            drawing = False
        elif key == ord("s"): # If 's' is pressed, break from the loop and do the cropping
            break
        elif key == ord("q"):
            break    
    cv2.destroyAllWindows()
    if len(rectangle_points) == 2:
        roi =  [(int(rectangle_points[0][0]*100/show_scale),int(rectangle_points[0][1]*100/show_scale)),(int(rectangle_points[1][0]*100/show_scale),int(rectangle_points[1][1]*100/show_scale))]
        return [roi[0][0],roi[0][1], roi[1][0],roi[1][1]]
    else:
        raise Exception("No roi has selected")
    


def get_sim_map_box(path, x, inputsize, patchsize, show_scale, debug, cmap=cv2.COLORMAP_INFERNO, box_org=None):
    """
    Cosine-similarity map from a seed box on dense features x [1,C,Hp,Wp].

    DINOv3-style matching: single seed prototype from the patch that contains
    the geometric center of the drawn box (pixel center → patch index).
    Avoids averaging background and avoids centering on the snapped patch
    rectangle (floor/ceil corners), which can shift the query patch.

    box_org: optional [x1,y1,x2,y2] in *inputsize* pixel coords.
             If None, opens interactive ROI (demo path).

    Returns:
        mean_vis: colormap visualization at inputsize
        sim_map:  normalized cosine map at inputsize (for affinity/mask)
        box_org:  seed box used [x1,y1,x2,y2]
    """
    # box_org = [288, 333, 307, 361]
    if box_org is None:
        image_bgr = cv2.imread(path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
        box_org = get_roi(image_bgr, show_scale=show_scale)
        print(f"Selected box: {box_org}")
    else:
        x1, y1, x2, y2 = box_org
        x1, x2 = sorted((int(round(x1)), int(round(x2))))
        y1, y2 = sorted((int(round(y1)), int(round(y2))))
        x1 = max(0, min(inputsize - 1, x1))
        x2 = max(0, min(inputsize - 1, x2))
        y1 = max(0, min(inputsize - 1, y1))
        y2 = max(0, min(inputsize - 1, y2))
        box_org = [x1, y1, x2, y2]
        


    feat = x[0].detach().float().clone()  # [C, Hp, Wp]
    C, Hp, Wp = feat.shape

    # Pixel-box center → patch that contains that point
    cx = 0.5 * (box_org[0] + box_org[2])
    cy = 0.5 * (box_org[1] + box_org[3])
    cx_p = int(np.floor(cx / patchsize))
    cy_p = int(np.floor(cy / patchsize))
    cx_p = max(0, min(Wp - 1, cx_p))
    cy_p = max(0, min(Hp - 1, cy_p))

    # --- dense cosine: that single patch as prototype ---
    proto = F.normalize(feat[:, cy_p, cx_p], dim=0, eps=1e-8)  # [C]
    flat = F.normalize(feat.reshape(C, -1), dim=0, eps=1e-8)  # [C, HW]
    sim_map = torch.matmul(proto, flat).reshape(Hp, Wp)
    sim_map = sim_map.detach().cpu().numpy().astype(np.float32)
    sim_local = sim_map.copy()

    # Percentile normalize the contrast map (shared by viz + affinity)
    flat_l = sim_local.reshape(-1)
    p5 = float(np.percentile(flat_l, 5))
    p95 = float(np.percentile(flat_l, 95))
    if p95 - p5 < 1e-8:
        # fallback if contrast map is nearly flat
        p5 = float(sim_map.min())
        p95 = float(sim_map.max())
        sim_norm = (sim_map - p5) / (p95 - p5 + 1e-8)
    else:
        sim_norm = (sim_local - p5) / (p95 - p5 + 1e-8)
    sim_norm = np.clip(sim_norm, 0.0, 1.0).astype(np.float32)
    sim_norm  = sim_local.copy()

    mean_vis = (sim_norm * 255).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap)
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize), interpolation=cv2.INTER_LINEAR)
    sim_for_mask_up = cv2.resize(sim_norm, (inputsize, inputsize), interpolation=cv2.INTER_LINEAR)
    sim_for_mask_up = np.clip(sim_for_mask_up, 0.0, 1.0).astype(np.float32)

    if debug:
        w = int(mean_vis.shape[1] * show_scale / 100)
        h = int(mean_vis.shape[0] * show_scale / 100)
        mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("mean_vis_show", mean_vis_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return mean_vis, sim_for_mask_up, box_org

def random_walk_refine(Y0, affinity, alpha=0.5, num_iter=10):
    """
    Affinity-normalized random walk refinement.

    Args:
        Y0: [B, C, H, W] initial soft predictions (e.g. bg/fg probs)
        affinity: [B, H, W, 8] directional affinities
        alpha: propagation strength (0 < alpha < 1)
        num_iter: number of propagation iterations
    Returns:
        Refined soft predictions [B, C, H, W]
    """
    DIRECTIONS = [
        (-1,  0),  # up
        ( 1,  0),  # down
        ( 0, -1),  # left
        ( 0,  1),  # right
        (-1, -1),  # up-left
        (-1,  1),  # up-right
        ( 1, -1),  # down-left
        ( 1,  1),  # down-right
    ]
    B, C, H, W = Y0.shape
    # Normalize affinities over neighbors so each pixel is a convex combination
    affinity_sum = affinity.sum(dim=-1, keepdim=True).clamp_min(1e-8)  # [B, H, W, 1]
    affinity_norm = affinity / affinity_sum

    Y = Y0.clone()
    for _ in range(num_iter):
        Y_new = torch.zeros_like(Y)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            affinity_weight = affinity_norm[..., i].unsqueeze(1)  # [B, 1, H, W]
            # Pull label from neighbor in direction (dx, dy); replicate pads avoid wrap-around
            shifted = F.pad(Y, (1, 1, 1, 1), mode='replicate')
            shifted = shifted[:, :, 1 + dy:H + 1 + dy, 1 + dx:W + 1 + dx]
            Y_new += affinity_weight * shifted
        Y = alpha * Y_new + (1 - alpha) * Y0
        # Keep valid probabilities across channels
        Y = Y.clamp(0, 1)
        Y = Y / Y.sum(dim=1, keepdim=True).clamp_min(1e-8)

    return Y


def compute_affinity_from_features(feat_map):
    """
    Compute 8-directional affinity map from feature embeddings.
    feat_map: Tensor of shape (B, D, H, W)
    Returns: Tensor of shape (B, H, W, 8)
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    
    feat_pad = F.pad(feat_map, (1, 1, 1, 1), mode='replicate')
    affinities = []
    for dx, dy in directions:
        # Neighbor at (h+dx, w+dy) relative to unpadded location
        shifted = feat_pad[:, :, 1 + dx:1 + dx + feat_map.shape[2], 1 + dy:1 + dy + feat_map.shape[3]]
        dist = torch.norm(feat_map - shifted, dim=1, keepdim=True)  # L2 over D
        affinity = torch.exp(-dist * 10)
        affinities.append(affinity)  # (B, 1, H, W)
    return torch.cat(affinities, dim=1).permute(0, 2, 3, 1)


def get_affinity_mask(sim_map, image_embedding_high, device, show_scale, debug, cmap, mask_percentile=70):
    """
    Generate localized mask using affinity-based random walk (paper §S1.3).

    Strategy (as claimed):
    1. Interpret the similarity map S as a soft two-channel mask (fg=S, bg=1-S)
    2. Propagate along feature affinity with a normalized random walk
    3. Detrend refined FG (subtract blur halo) then percentile-threshold the residual

    Args:
        mask_percentile: percentile of detrended FG used as the binary threshold (default 70)
    """
    sim_map_np = np.asarray(sim_map, dtype=np.float32).copy()
    sim_map_np = np.clip(sim_map_np, 0.0, 1.0)
    if debug:
        sim_map_np_vis = (np.clip(sim_map_np, 0, 1) * 255).astype(np.uint8)
        sim_map_np_vis = cv2.applyColorMap(sim_map_np_vis, cmap)
        w = int(sim_map_np_vis.shape[1] * show_scale / 100)
        h = int(sim_map_np_vis.shape[0] * show_scale / 100)
        sim_map_np_vis = cv2.resize(sim_map_np_vis, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("sim_map_np_vis", sim_map_np_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    # Step 1: soft mask from full similarity — fg = S, bg = 1 - S
    foreground_prob = sim_map_np
    background_prob = 1.0 - foreground_prob

    sim_map_tensor = torch.tensor(
        np.stack([background_prob, foreground_prob], axis=0),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)  # (1, 2, H, W)


    # Expect channel-first features [B, C, H, W].
    # If channel-last [B, H, W, C] is passed, convert once.
    feat = image_embedding_high
    if feat.dim() != 4:
        raise ValueError(f"Expected 4D feature tensor, got shape {tuple(feat.shape)}")
    if feat.shape[1] < feat.shape[-1] and feat.shape[-1] > 64:
        # Likely [B, H, W, C]
        feat = feat.permute(0, 3, 1, 2).contiguous()
    feat = feat.to(device)

    # Step 2: Affinity at feature resolution, then upsample to sim_map size
    affinity_map = compute_affinity_from_features(feat)
    affinity_map = F.interpolate(
        affinity_map.permute(0, 3, 1, 2),
        size=(sim_map_tensor.shape[2], sim_map_tensor.shape[3]),
        mode='bilinear',
        align_corners=False,
    ).permute(0, 2, 3, 1)

    # Step 3: Affinity-based random walk propagation
    refined_mask_soft = random_walk_refine(sim_map_tensor, affinity_map, alpha=0.3, num_iter=8)

    # Step 4: Foreground probability after refinement
    refined_mask_prob = refined_mask_soft[0, 1].detach().cpu().numpy()

    if debug:
        soft_vis = (np.clip(refined_mask_prob, 0, 1) * 255).astype(np.uint8)
        soft_vis = cv2.applyColorMap(soft_vis, cmap)
        w = int(soft_vis.shape[1] * show_scale / 100)
        h = int(soft_vis.shape[0] * show_scale / 100)
        soft_vis = cv2.resize(soft_vis, (w, h), interpolation=cv2.INTER_AREA)
        cv2.imshow("refined soft foreground", soft_vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    
    # Step 5: adaptive detrend → threshold
    # Seed is the global peak and dominates a smooth halo; subtract that low-frequency
    # trend so secondary peaks compete, then percentile-threshold the residual.
    H, W = refined_mask_prob.shape
    k = max(31, (min(H, W) // 8) | 1)  # odd; ~seed-halo scale

    trend = cv2.GaussianBlur(refined_mask_prob, (k, k), sigmaX=0)
    detrended = np.maximum(refined_mask_prob - trend, 0.0).astype(np.float32)

    if float(detrended.max()) < 1e-8:
        # Detrend removed all signal — threshold the raw refined map instead.
        score = refined_mask_prob
        final_threshold = float(np.percentile(score, mask_percentile))
    else:
        score = detrended
        final_threshold = float(np.percentile(score, mask_percentile))
        if final_threshold <= 1e-8:
            # Sparse residual: most pixels are 0, so the global percentile collapses to 0.
            # Threshold only over positive peaks; use a slightly lower percentile so we
            # do not wipe secondary detections when mask_percentile is very high.
            pos = score[score > 0]
            if pos.size > 0:
                pos_pct = max(50, mask_percentile - 30)
                final_threshold = float(np.percentile(pos, pos_pct))
            else:
                score = refined_mask_prob
                final_threshold = float(np.percentile(score, mask_percentile))

    refined_mask = (score > final_threshold).astype(np.uint8) * 255
    kernel_small = np.ones((5, 5), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)

    if debug:
        det_vis = (np.clip(score / (score.max() + 1e-8), 0, 1) * 255).astype(np.uint8)
        det_vis = cv2.applyColorMap(det_vis, cmap)
        w = int(refined_mask.shape[1] * show_scale / 100)
        h = int(refined_mask.shape[0] * show_scale / 100)
        cv2.imshow("step5_detrended_score", cv2.resize(det_vis, (w, h), interpolation=cv2.INTER_AREA))
        cv2.imshow("refined_mask_vis", cv2.resize(refined_mask, (w, h), interpolation=cv2.INTER_AREA))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    mask_rgb = cv2.applyColorMap(
        cv2.normalize(refined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cmap,
    )
    return mask_rgb, refined_mask


def get_overlay_heatmap(sim_map,img_org,debug,show_scale):
    heatmap = cv2.applyColorMap(np.uint8(sim_map), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_org.shape[1], img_org.shape[0]))
    overlay = cv2.addWeighted(img_org, 0.5, heatmap, 0.5, 0)
    if debug:
        w = int(overlay.shape[1] * show_scale / 100)
        h = int(overlay.shape[0] * show_scale / 100)
        overlay_vis = cv2.resize(overlay, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow("overlay_vis", overlay_vis); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return overlay


def get_bounding_boxes_from_mask(mask, min_area=100):
    """
    connected component analysis with 8-connectivity,
    filter small components, extract axis-aligned bounding boxes.
    """
    # Ensure mask is binary (0 and 255)
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Connected component analysis (8-connectivity)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    boxes = []
    component_stats = []
    
    # Skip background (label 0); filter small components; axis-aligned boxes
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue
        
        x1 = stats[label_id, cv2.CC_STAT_LEFT]
        y1 = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]
        x2 = x1 + width
        y2 = y1 + height
        
        boxes.append((x1, y1, x2, y2))
        component_stats.append({
            'area': area,
            'box': (x1, y1, x2, y2),
            'centroid': (int(centroids[label_id, 0]), int(centroids[label_id, 1]))
        })
    
    return boxes, len(boxes), component_stats


def visualize_bounding_boxes(image, boxes, color=(0, 255, 0), thickness=2, label_text=None):
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image (BGR format)
        boxes: List of bounding boxes [(x1, y1, x2, y2), ...]
        color: Box color in BGR format (default: green)
        thickness: Box line thickness (default: 2)
        label_text: Optional list of labels for each box
    
    Returns:
        image_with_boxes: Image with bounding boxes drawn
    """
    image_with_boxes = image.copy()
    
    for idx, (x1, y1, x2, y2) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, thickness)
        
        # Draw label if provided
        if label_text is not None and idx < len(label_text):
            label = str(label_text[idx])
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Draw background rectangle for text
            cv2.rectangle(image_with_boxes, (x1, y1 - text_height - 10), 
                         (x1 + text_width, y1), color, -1)
            # Draw text
            cv2.putText(image_with_boxes, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return image_with_boxes
