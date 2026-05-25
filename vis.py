import cv2
import numpy as np
import torch
import torch.nn.functional as F
import wandb
import os
from utils import extract_instances, six_d_to_matrix

def depth_to_color(depth_map, d_min=None, d_max=None):
    if d_min is None: d_min = depth_map.min()
    if d_max is None: d_max = depth_map.max()
    if d_max > d_min:
        d_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(depth_map)
    d_uint8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)

def flow_to_color(flow_np):
    flow_np = flow_np.astype(np.float32)
    # Subtract median to remove global camera motion and highlight relative parallax/object motion
    flow_np[..., 0] -= np.median(flow_np[..., 0])
    flow_np[..., 1] -= np.median(flow_np[..., 1])
    
    h, w = flow_np.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    mag_max = np.max(mag)
    if mag_max > 1e-3:
        hsv[..., 2] = (mag / mag_max * 255).astype(np.uint8)
    else:
        hsv[..., 2] = 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def save_visualization(video_t, target_t, pred_t, step, warped_img=None, output_dir="vis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img_tensor = video_t[0].permute(1, 2, 0).cpu().numpy()
    base_bgr = cv2.cvtColor((img_tensor * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = base_bgr.shape[:2]
    
    # ---------------------------
    # Left: Prediction (Box & Mask)
    # ---------------------------
    pred_canvas = base_bgr.copy()
    with torch.no_grad():
        instances = extract_instances(pred_t, score_thresh=0.3, nms_thresh=0.5)
    inst = instances[0]
    if inst is not None and len(inst["scores"]) > 0:
        for k in range(len(inst["scores"])):
            # classes: 0 -> Active (Red), 1 -> Passive (Blue)
            # fallback to Red if classification is missing
            cls_idx = inst["classes"][k].item() if inst["classes"] is not None else 0
            color = (0, 0, 255) if cls_idx == 0 else (255, 0, 0)
            
            if inst["masks"] is not None:
                m = inst["masks"][k].cpu().numpy()
                pred_canvas[m] = pred_canvas[m] * 0.5 + np.array(color) * 0.5
            
            b = inst["boxes"][k].cpu().numpy()
            cv2.rectangle(pred_canvas, (int(b[0]*W), int(b[1]*H)), (int(b[2]*W), int(b[3]*H)), color, 2)
            # Add text label for class
            label = "Active" if cls_idx == 0 else "Passive"
            cv2.putText(pred_canvas, label, (int(b[0]*W), max(10, int(b[1]*H) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
    cv2.putText(pred_canvas, "Prediction", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # ---------------------------
    # Middle: Ground Truth (Box & Mask)
    # ---------------------------
    gt_canvas = base_bgr.copy()
    if "seg_raw" in target_t and "is_dynamic" in target_t:
        seg = target_t["seg_raw"][0].cpu().numpy()
        is_dyn = target_t["is_dynamic"][0].cpu().numpy()
        
        max_uid = int(np.max(seg))
        for uid in range(1, max_uid + 1):
            m = (seg == uid)
            if np.any(m):
                # is_dynamic mapping: True -> Active (Red), False -> Passive (Blue)
                # is_dyn index is uid - 1
                if (uid - 1) < len(is_dyn) and is_dyn[uid - 1]:
                    color = (0, 0, 255) # Red (BGR)
                    label = "Active"
                else:
                    color = (255, 0, 0) # Blue (BGR)
                    label = "Passive"
                
                gt_canvas[m] = gt_canvas[m] * 0.5 + np.array(color) * 0.5
                
                # Draw bounding box
                y_idx, x_idx = np.where(m)
                ymin, ymax = y_idx.min(), y_idx.max()
                xmin, xmax = x_idx.min(), x_idx.max()
                cv2.rectangle(gt_canvas, (xmin, ymin), (xmax, ymax), color, 2)
                cv2.putText(gt_canvas, label, (xmin, max(10, ymin - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
    elif "bboxes_dense" in target_t and "obj_dense" in target_t:
        # Fallback if seg_raw is missing
        obj_t = target_t["obj_dense"][0, 0].cpu().numpy()
        boxes_t = target_t["bboxes_dense"][0].cpu().numpy()
        y_idx, x_idx = np.where(obj_t > 0.5)
        for y, x in zip(y_idx, x_idx):
            b = boxes_t[:, y, x]
            # b is [l, t, r, b] in stride units. grid size is 8.0
            grid_x = x * 8.0 + 4.0
            grid_y = y * 8.0 + 4.0
            xmin = int(grid_x - b[0] * 8.0)
            ymin = int(grid_y - b[1] * 8.0)
            xmax = int(grid_x + b[2] * 8.0)
            ymax = int(grid_y + b[3] * 8.0)
            cv2.rectangle(gt_canvas, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            
    cv2.putText(gt_canvas, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # ---------------------------
    # Right: 6-Grid
    # ---------------------------
    half_h, half_w = H // 2, W // 2
    
    # 1. Anomaly Heatmap
    anom_map = pred_t["anomaly_map"][0].cpu().detach().numpy()
    anom_max = max(float(np.max(anom_map)), 0.001)
    anom_norm = np.clip(anom_map / anom_max, 0, 1)
    anom_img = cv2.applyColorMap((anom_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    anom_img = cv2.resize(anom_img, (half_w, half_h))
    cv2.putText(anom_img, "Anomaly", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 2. Pred Flow (moved to 2nd position)
    pred_flow = pred_t.get("flow")
    if pred_flow is not None:
        pred_flow_np = pred_flow[0].cpu().detach().numpy()
        pred_flow_img = cv2.resize(flow_to_color(pred_flow_np.transpose(1, 2, 0)), (half_w, half_h))
    else:
        pred_flow_img = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    cv2.putText(pred_flow_img, "Pred Flow", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 3. GT Depth
    gt_depth_np = target_t["depth"][0].cpu().numpy()
    pred_depth_np = pred_t["depth"][0].cpu().detach().numpy()
    d_min = min(gt_depth_np.min(), pred_depth_np.min())
    d_max = max(gt_depth_np.max(), pred_depth_np.max())
    gt_depth_img = cv2.resize(depth_to_color(gt_depth_np, d_min, d_max), (half_w, half_h))
    cv2.putText(gt_depth_img, "GT Depth", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 4. Pred Depth
    pred_depth_img = cv2.resize(depth_to_color(pred_depth_np, d_min, d_max), (half_w, half_h))
    cv2.putText(pred_depth_img, "Pred Depth", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 5. GT Flow
    gt_flow_np = target_t.get("flow_target", torch.zeros((1, 2, H, W)))[0].cpu().numpy()
    gt_flow_img = cv2.resize(flow_to_color(gt_flow_np.transpose(1, 2, 0)), (half_w, half_h))
    cv2.putText(gt_flow_img, "GT Flow", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 6. Warped Image (Photometric Error)
    if warped_img is not None:
        warp_np = warped_img[0].permute(1, 2, 0).cpu().detach().numpy()
        warp_bgr = cv2.cvtColor((np.clip(warp_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        warp_img = cv2.resize(warp_bgr, (half_w, half_h))
    else:
        warp_img = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    cv2.putText(warp_img, "Warped (Photo Error)", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Assemble 6-Grid
    row1 = np.hstack([anom_img, pred_flow_img])
    row2 = np.hstack([gt_depth_img, pred_depth_img])
    row3 = np.hstack([gt_flow_img, warp_img])
    grid = np.vstack([row1, row2, row3])
    
    # Scale grid to match height H
    grid = cv2.resize(grid, (int(grid.shape[1] * H / grid.shape[0]), H))
    
    final_img = np.hstack([pred_canvas, gt_canvas, grid])
    filepath = os.path.join(output_dir, f"vis_step_{step:05d}.jpg")
    cv2.imwrite(filepath, final_img)
    return filepath

# =====================================================================
# 6. 核心物理监督 Loss 与主循环
# =====================================================================
def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    K_inv = torch.inverse(K)
    return K, K_inv

def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    device = depth.device
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.flatten().expand(B, -1)
    y = y.flatten().expand(B, -1)
    ones = torch.ones_like(x)
    pixels = torch.stack([x, y, ones], dim=1) 
    
    points_3d = torch.bmm(K_inv.expand(B, 3, 3), pixels.float()) 
    points_3d = points_3d * depth.view(B, 1, H*W)
    
    t = pose[:, :3].unsqueeze(2)
    R = six_d_to_matrix(pose[:, 3:])
    
    points_3d_next = torch.bmm(R, points_3d) + t
    
    pixels_next = torch.bmm(K.expand(B, 3, 3), points_3d_next)
    z_next_raw = pixels_next[:, 2:3, :]
    z_next_safe = torch.clamp(z_next_raw, min=0.01).float()
    x_next = (pixels_next[:, 0:1, :].float() / z_next_safe).to(pixels_next.dtype)
    y_next = (pixels_next[:, 1:2, :].float() / z_next_safe).to(pixels_next.dtype)
    
    x_norm = 2.0 * x_next / (W - 1) - 1.0
    y_norm = 2.0 * y_next / (H - 1) - 1.0
    
    grid = torch.cat([x_norm, y_norm], dim=1).view(B, 2, H, W).permute(0, 2, 3, 1)
    grid = torch.clamp(grid, -2.0, 2.0)
    
    warped_img = F.grid_sample(img_next, grid, mode='bilinear', padding_mode='border', align_corners=True)
    warped_img = torch.nan_to_num(warped_img, 0.0)
    
    valid_mask = ((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0)).view(B, 1, H, W).float()
    safe_depth_mask = ((depth > 0.01) & (z_next_raw.view(B, 1, H, W) > 0.01)).float()
    valid_mask = valid_mask * safe_depth_mask
    
    return warped_img, valid_mask

def edge_aware_smoothness_loss(depth, img):
    mean_depth = depth.mean(dim=[2, 3], keepdim=True).float()
    norm_depth = (depth.float() / torch.clamp(mean_depth, min=1e-4)).to(depth.dtype)
    grad_depth_x = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
    grad_depth_y = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    return grad_depth_x.mean() + grad_depth_y.mean()

def dice_loss(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    targets = targets.flatten()
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1.0 - dice

