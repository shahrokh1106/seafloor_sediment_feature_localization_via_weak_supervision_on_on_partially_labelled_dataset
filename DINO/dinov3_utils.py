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

def get_points(image, show_scale=300):
    global drawing, prev_pt
    drawing = False  
    prev_pt = None
    def draw_freehand(event, x, y, flags, params):
        global drawing, prev_pt, points, img
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            prev_pt = (x, y)
            points.append(prev_pt)
            cv2.circle(img, prev_pt, 4, (0, 0, 255), -1)
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            cur_pt = (x, y)
            cv2.line(img, prev_pt, cur_pt, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            num_steps = max(abs(cur_pt[0]-prev_pt[0]), abs(cur_pt[1]-prev_pt[1])) + 1
            for t in range(num_steps):
                xi = int(prev_pt[0] + (cur_pt[0]-prev_pt[0]) * t / num_steps)
                yi = int(prev_pt[1] + (cur_pt[1]-prev_pt[1]) * t / num_steps)
                points.append((xi, yi))
            prev_pt = cur_pt
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            prev_pt = None
    global img, img_orig, points
    points = []
    w = int(image.shape[1] * show_scale / 100)
    h = int(image.shape[0] * show_scale / 100)
    image_small = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    img = image_small.copy()
    img_orig = img.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", draw_freehand)
    while True:
        cv2.imshow("image", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"): 
            img = img_orig.copy()
            points = []
            prev_pt = None
        elif key == ord("s"):  # save/return points
            break
        elif key == ord("q"):
            break
    cv2.destroyAllWindows()
    if len(points) != 0:
        rescaled_points = []
        for (px, py) in points:
            rx = int(px * 100 / show_scale)
            ry = int(py * 100 / show_scale)
            rescaled_points.append((rx, ry))
        return rescaled_points
    else:
        raise Exception("No roi has selected")

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
    def draw_rectangle(event, x, y, flags, params):
        global rectangle_points, img, img_orig
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle_points = [(x, y)]
        elif event == cv2.EVENT_LBUTTONUP:
            rectangle_points.append((x, y))
            cv2.rectangle(img, rectangle_points[0], rectangle_points[1], (0, 255, 0), 2)
            cv2.imshow("image", img)
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
        elif key == ord("s"): # If 's' is pressed, break from the loop and do the cropping
            break
        elif key == ord("q"):
            break    
    cv2.destroyAllWindows()
    if len(rectangle_points)!=0:
        roi =  [(int(rectangle_points[0][0]*100/show_scale),int(rectangle_points[0][1]*100/show_scale)),(int(rectangle_points[1][0]*100/show_scale),int(rectangle_points[1][1]*100/show_scale))]
        return [roi[0][0],roi[0][1], roi[1][0],roi[1][1]]
    else:
        raise Exception("No roi has selected")
    


def get_sim_map_box(path,x, inputsize,patchsize,show_scale,debug, cmap=cv2.COLORMAP_INFERNO):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
    box_org = get_roi(image_bgr,show_scale = show_scale)
    box = [b//patchsize for b in box_org]
    x1,y1,x2,y2 = box
    feats = x[0].detach().cpu().numpy()  
    box_features = feats[:,y1:y2, x1:x2]
    box_features = box_features.mean(axis=(1, 2)) 
    C, H, W = feats.shape
    feats = feats.reshape(C,H*W)
    #L2
    feats /= (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    box_features /= (np.linalg.norm(box_features) + 1e-8)
    sim = box_features.T@feats
    sim_map = sim.reshape(H,W)
    
    # Normalize similarity map to [0,1] for mask generation (keep full range)
    sim_map_normalized = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    
    # For visualization: use percentile-based normalization to reduce impact of selected box
    # This makes other similar regions more visible while keeping the selected box visible
    sim_flat = sim_map.flatten()
    # Use 95th percentile as max for visualization (excludes extreme outliers like the selected box)
    p95 = np.percentile(sim_flat, 95)
    p5 = np.percentile(sim_flat, 5)
    
    # Create visualization map with percentile-based normalization
    sim_vis = np.clip(sim_map, p5, p95)
    sim_vis = (sim_vis - p5) / (p95 - p5 + 1e-8)
    
    # Apply gamma correction (power scaling) to enhance contrast for medium similarity values
    # This makes similar regions more distinct while preserving the selected box visibility
    gamma = 0.7  # Values < 1 enhance contrast for lower values
    sim_vis = np.power(sim_vis, gamma)
    
    # Convert to uint8 for colormap
    mean_vis = (sim_vis * 255).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap) 
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize))
    
    w = int(mean_vis.shape[1] * show_scale / 100)
    h = int(mean_vis.shape[0] * show_scale / 100)
    mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("DINO FEATURES", mean_vis_show); cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return normalized similarity map (full range) for mask generation
    return mean_vis, cv2.resize(sim_map_normalized, (inputsize,inputsize)), box_org

def get_sim_map_points(path,x, inputsize,patchsize,show_scale,debug, cmap=cv2.COLORMAP_INFERNO):
    image_bgr = cv2.imread(path)
    image_bgr = cv2.resize(image_bgr, (inputsize, inputsize))
    points_org = get_points (image_bgr ,show_scale = show_scale)
    points = [(point[0]//patchsize,point[1]//patchsize) for point in points_org]
    feats = x[0].detach().cpu().numpy()  
    point_features = []
    for point in points:
        point_features.append(feats[:,point[1],point[0]])
    point_features = np.vstack(point_features)
    point_features = point_features.mean(0)
    C, H, W = feats.shape
    feats = feats.reshape(C,H*W)

    #L2
    feats /= (np.linalg.norm(feats, axis=0, keepdims=True) + 1e-8)
    point_features /= (np.linalg.norm(point_features) + 1e-8)

    sim = point_features.T@feats
    sim_map = sim.reshape(H,W)
    
    # Normalize similarity map to [0,1] for mask generation (keep full range)
    sim_map_normalized = (sim_map - sim_map.min()) / (sim_map.max() - sim_map.min() + 1e-8)
    
    # For visualization: use percentile-based normalization to reduce impact of selected points
    # This makes other similar regions more visible while keeping selected points visible
    sim_flat = sim_map.flatten()
    # Use 95th percentile as max for visualization (excludes extreme outliers like selected points)
    p95 = np.percentile(sim_flat, 95)
    p5 = np.percentile(sim_flat, 5)
    
    # Create visualization map with percentile-based normalization
    sim_vis = np.clip(sim_map, p5, p95)
    sim_vis = (sim_vis - p5) / (p95 - p5 + 1e-8)
    
    # Apply gamma correction (power scaling) to enhance contrast for medium similarity values
    # This makes similar regions more distinct while preserving selected points visibility
    gamma = 0.7  # Values < 1 enhance contrast for lower values
    sim_vis = np.power(sim_vis, gamma)
    
    # Convert to uint8 for colormap
    mean_vis = (sim_vis * 255).astype(np.uint8)
    mean_vis = cv2.applyColorMap(mean_vis, cmap) 
    mean_vis = cv2.resize(mean_vis, (inputsize, inputsize))
    
    w = int(mean_vis.shape[1] * show_scale / 100)
    h = int(mean_vis.shape[0] * show_scale / 100)
    mean_vis_show = cv2.resize(mean_vis, (w, h), interpolation = cv2.INTER_AREA)
    if debug:
        cv2.imshow("DINO FEATURES", mean_vis_show); cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Return normalized similarity map (full range) for mask generation
    return mean_vis, cv2.resize(sim_map_normalized, (inputsize,inputsize)), points_org
    
def random_walk_refine(Y0, affinity, alpha=0.5, num_iter=10):
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
    """
    Fast GPU-based random walk refinement.
    Args:
        Y0: [B, C, H, W] initial soft predictions (e.g., softmax outputs)
        affinity: [B, H, W, 8] affinity map
        alpha: propagation strength (0 < alpha < 1)
        num_iter: number of propagation iterations
    Returns:
        Refined soft predictions [B, C, H, W]
    """
    B, C, H, W = Y0.shape
    Y = Y0.clone()
    for _ in range(num_iter):
        Y_new = torch.zeros_like(Y)
        for i, (dx, dy) in enumerate(DIRECTIONS):
            affinity_weight = affinity[..., i]  # [B, H, W]
            affinity_weight = affinity_weight.unsqueeze(1)  # [B, 1, H, W]
            # Shift prediction
            shifted = F.pad(Y, (1, 1, 1, 1), mode='replicate')  # pad to handle borders
            shifted = shifted[:, :, 1+dy:H+1+dy, 1+dx:W+1+dx]  # shifted prediction
            Y_new += affinity_weight * shifted
        Y = alpha * Y_new + (1 - alpha) * Y0
    
    return Y

def compute_affinity_from_features(feat_map):
    """
    Compute 8-directional affinity map from feature embeddings.
    feat_map: Tensor of shape (B, D, H, W)
    Returns: Tensor of shape (B, H, W, 8)
    """
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1)]
    affinities = []
    for dx, dy in directions:
        shifted = torch.roll(feat_map, shifts=(dx, dy), dims=(2, 3))  # shift H, W
        dist = torch.norm(feat_map - shifted, dim=1, keepdim=True)    # L2 over D

        affinity = torch.exp(-dist*10)  # similarity from distance
        affinities.append(affinity)  # (B, 1, H, W)
    # Stack and return as (B, H, W, 8)
    return torch.cat(affinities, dim=1).permute(0, 2, 3, 1)


def get_affinity_mask(sim_map,image_embedding_high,device,show_sclae,debug,cmap, seed_percentile=90):
    """
    Generate localized mask using affinity-based random walk.
    
    Strategy:
    1. Start with VERY strict seed mask (only top similarity regions, e.g., 90th percentile = top 10%)
    2. Use conservative affinity-based random walk to propagate ONLY to highly similar regions
    3. Apply strict thresholding on refined probabilities to keep mask localized
    
    Args:
        sim_map: Similarity map [H, W] normalized to [0, 1]
        image_embedding_high: High-res feature embeddings
        device: Device for computation
        show_sclae: Display scale
        debug: Debug mode
        cmap: Colormap for visualization
        seed_percentile: Percentile threshold for seed regions (default: 90 = top 10%)
    
    Returns:
        mask_rgb: Colored mask visualization
        mask_binary: Binary mask [H, W]
    """
    sim_map_np = sim_map.copy()
    
    # Step 1: Create VERY strict seed mask from only the highest similarity regions
    # Increased from 85th to 90th percentile = only top 10% similarity as seeds (more conservative)
    seed_threshold = np.percentile(sim_map_np.flatten(), seed_percentile)
    
    # Create soft seed predictions: high foreground probability only for very similar regions
    # Use smooth transition to avoid hard boundaries but keep it localized
    foreground_prob = np.clip((sim_map_np - seed_threshold) / (1.0 - seed_threshold + 1e-8), 0, 1)
    # Increased power scaling from 1.5 to 2.0 to make seed regions even more distinct (more conservative)
    foreground_prob = np.power(foreground_prob, 2.0)  # Higher power = more conservative seeds
    
    background_prob = 1.0 - foreground_prob
    
    # Convert to tensor for random walk
    sim_map_tensor = torch.tensor(np.stack([background_prob, foreground_prob], axis=0), dtype=torch.float32, device=device)
    sim_map_tensor = sim_map_tensor.unsqueeze(0)  # (1, 2, H, W)
    
    image_embedding_high = image_embedding_high.permute(0,3,1,2)
    
    # Step 2: Compute affinity map from features - ensures propagation only along similar features
    affinity_map = compute_affinity_from_features(image_embedding_high).to(device)
    affinity_map = F.interpolate(affinity_map.permute(0, 3, 1, 2), size=(sim_map_tensor.shape[2], sim_map_tensor.shape[3]), mode='bilinear', align_corners=False) 
    affinity_map = affinity_map.permute(0, 2, 3, 1)
    
    # Step 3: Apply CONSERVATIVE random walk refinement - less aggressive propagation
    # Reduced alpha from 0.4 to 0.3: less propagation (more conservative)
    # Reduced iterations from 8 to 6: less spreading (more localized)
    refined_mask_soft = random_walk_refine(sim_map_tensor, affinity_map, alpha=0.3, num_iter=6)
    
    # Step 4: Get refined foreground probabilities (after random walk propagation)
    refined_mask_prob = refined_mask_soft[0, 1, :, :].detach().cpu().numpy()  # Foreground probability
    
    # Step 5: Apply STRICT thresholding on refined probabilities to keep mask localized
    # Use higher percentile threshold (60th instead of 50th) for stricter final mask
    try:
        refined_mask_uint8 = (refined_mask_prob * 255).astype(np.uint8)
        threshold_value, _ = cv2.threshold(refined_mask_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        final_threshold = threshold_value / 255.0
        # If Otsu threshold is too low, use percentile-based threshold instead
        if final_threshold < 0.5:
            final_threshold = np.percentile(refined_mask_prob.flatten(), 60)  # 60th percentile = stricter
    except:
        # Fallback: Use 60th percentile of refined probabilities (stricter than median)
        final_threshold = np.percentile(refined_mask_prob.flatten(), 60)
    
    # Create binary mask from refined probabilities with strict threshold
    refined_mask = (refined_mask_prob > final_threshold).astype(np.uint8) * 255
    
    # Optional: Small morphological operations to clean up while preserving regions
    kernel_small = np.ones((3, 3), np.uint8)
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_OPEN, kernel_small, iterations=1)  # Remove small noise
    refined_mask = cv2.morphologyEx(refined_mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)  # Fill small gaps
    
    if debug:
        w = int(refined_mask.shape[1] * show_sclae / 100)
        h = int(refined_mask.shape[0] * show_sclae / 100)
        refined_mask_vis = cv2.resize(refined_mask, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow("DINO FEATURES", refined_mask_vis); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return cv2.applyColorMap(cv2.normalize(refined_mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8), cmap) ,refined_mask


def get_overlay_heatmap(sim_map,img_org,debug,show_scale):
    heatmap = cv2.applyColorMap(np.uint8(sim_map), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (img_org.shape[1], img_org.shape[0]))
    overlay = cv2.addWeighted(img_org, 0.5, heatmap, 0.5, 0)
    if debug:
        w = int(overlay.shape[1] * show_scale / 100)
        h = int(overlay.shape[0] * show_scale / 100)
        overlay_vis = cv2.resize(overlay, (w, h), interpolation = cv2.INTER_AREA)
        cv2.imshow("DINO FEATURES", overlay_vis); cv2.waitKey(0)
        cv2.destroyAllWindows()
    return overlay


def get_bounding_boxes_from_mask(mask, min_area=100):
    """
    Perform connected component analysis on binary mask and return bounding boxes.
    
    Args:
        mask: Binary mask [H, W] with values 0 (background) and 255 (foreground)
        min_area: Minimum area (in pixels) for a component to be included (default: 100)
    
    Returns:
        boxes: List of bounding boxes in format [(x1, y1, x2, y2), ...]
        num_components: Number of connected components found
        component_stats: List of dicts with stats for each component {'area': int, 'box': tuple, 'centroid': tuple}
    """
    # Ensure mask is binary (0 and 255)
    if mask.max() > 1:
        mask_binary = (mask > 127).astype(np.uint8)
    else:
        mask_binary = (mask > 0.5).astype(np.uint8) * 255
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_binary, connectivity=8)
    
    boxes = []
    component_stats = []
    
    # Skip background component (label 0)
    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        
        # Filter by minimum area
        if area < min_area:
            continue
        
        # Get bounding box coordinates
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
