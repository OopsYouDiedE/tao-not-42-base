import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision



# =====================================================================

def flow_to_color(flow_np):
    flow_np = flow_np.astype(np.float32) - np.median(flow_np, axis=(0, 1))
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-5) * 255, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_visualization(video_t, target_t, pred_t, step, warped_img=None, output_dir="vis_outputs", draw_track=None):
    os.makedirs(output_dir, exist_ok=True)
    img_tensor = video_t[0].permute(1, 2, 0).cpu().numpy()
    base_bgr = cv2.cvtColor(
        (img_tensor * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = base_bgr.shape[:2]

    def add_title(img, text, pos=(10, 30)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2)
        return img

    # --- Prediction ---
    pred_canvas = base_bgr.copy()
    with torch.no_grad():
        insts = extract_instances(pred_t, score_thresh=0.3, nms_thresh=0.5)
        inst = insts[0] if insts else None

    if inst and len(inst["scores"]) > 0:
        masks_iter = inst["masks"] if inst["masks"] is not None else [
            None] * len(inst["scores"])
        for c, m, b in zip(inst["classes"], masks_iter, inst["boxes"]):
            cls_val = c.item() if c is not None else 1
            # 统一为红色，因为我们现在进行的是类不可知 (class-agnostic) 的物体发现
            color = (0, 0, 255)

            if m is not None:
                m_np = m.cpu().numpy()
                pred_canvas[m_np] = pred_canvas[m_np] * \
                    0.5 + np.array(color) * 0.5

            b_np = b.cpu().numpy() * [W, H, W, H]
            cv2.rectangle(pred_canvas, (int(b_np[0]), int(
                b_np[1])), (int(b_np[2]), int(b_np[3])), color, 2)
                
    if draw_track is None:
        try:
            from utils.losses import get_loss_weights
            w = get_loss_weights(step)
            draw_track = (w.get("track", 0.0) > 0.0)
        except Exception:
            draw_track = True

    # 依据当前的追踪任务损失权重自适应决定是否绘制追踪框，杜绝未激活阶段的随机噪声干扰，拒绝绝对数值硬编码
    if "track_boxes" in pred_t and "track_alive" in pred_t and draw_track:
        tb = pred_t["track_boxes"].detach()
        ta = pred_t["track_alive"].detach()

        if tb.dim() >= 3:
            Q = tb.shape[-2]
            t_boxes = tb.reshape(-1, Q, 4)[0]
            t_alive = ta.reshape(-1, Q, 1)[0].sigmoid()

            for i in range(Q):
                if t_alive[i, 0] > 0.5:
                    b_np = t_boxes[i].cpu().numpy()
                    cx, cy, bw, bh = b_np
                    x1, y1 = (cx - bw/2) * W, (cy - bh/2) * H
                    x2, y2 = (cx + bw/2) * W, (cy + bh/2) * H
                    cv2.rectangle(pred_canvas, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 3)
                    cv2.putText(pred_canvas, f"ID:{i}", (int(x1), max(10, int(y1)-5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
    add_title(pred_canvas, "Prediction")

    # --- Ground Truth ---
    gt_canvas = base_bgr.copy()
    if target_t.get("seg_raw") is not None:
        seg = target_t["seg_raw"][0].cpu().numpy()

        for uid in range(1, int(np.max(seg)) + 1):
            m = seg == uid
            if np.any(m):
                # 统一为蓝色，因为我们现在进行的是类不可知 (class-agnostic) 的物体发现与物理追踪，不再区分动静态颜色的框
                color = (255, 0, 0)
                gt_canvas[m] = gt_canvas[m] * 0.5 + np.array(color) * 0.5

                y_idx, x_idx = np.where(m)
                cv2.rectangle(gt_canvas, (x_idx.min(), y_idx.min()),
                              (x_idx.max(), y_idx.max()), color, 2)

    elif "bboxes_dense" in target_t and "obj_dense" in target_t:
        obj_t = target_t["obj_dense"][0, 0].cpu().numpy()
        boxes_t = target_t["bboxes_dense"][0].cpu().numpy()
        for y, x in zip(*np.where(obj_t > 0.5)):
            b = boxes_t[:, y, x] * 8.0
            gx, gy = x * 8.0 + 4.0, y * 8.0 + 4.0
            # 统一为蓝色以保持一致性
            cv2.rectangle(gt_canvas, (int(
                gx - b[0]), int(gy - b[1])), (int(gx + b[2]), int(gy + b[3])), (255, 0, 0), 2)

    add_title(gt_canvas, "Ground Truth")

    # --- 6-Grid Output ---
    hw, hh = W // 2, H // 2

    def prep_cell(img, title):
        img_res = cv2.resize(img, (hw, hh))
        cv2.putText(img_res, title, (5, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img_res

    anom = pred_t["anomaly_map"][0].cpu().detach().numpy().squeeze()
    anom_norm = np.clip(anom / max(anom.max(), 1e-3), 0, 1)
    anom_img = cv2.applyColorMap(
        (anom_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)

    p_flow = pred_t.get("flow")
    p_flow_img = flow_to_color(p_flow[0].cpu().detach().numpy().transpose(
        1, 2, 0)) if p_flow is not None else np.zeros((H, W, 3), np.uint8)

    g_dep = target_t["depth"][0].cpu().numpy()
    p_dep = pred_t["depth"][0].cpu().detach().numpy()
    d_min, d_max = min(g_dep.min(), p_dep.min()), max(g_dep.max(), p_dep.max())

    flow_tgt = target_t.get("flow_target")
    g_flow_np = flow_tgt[0].cpu().numpy().transpose(
        1, 2, 0) if flow_tgt is not None else np.zeros((H, W, 2), np.float32)

    if warped_img is None:
        warp_img_bgr = np.zeros((H, W, 3), np.uint8)
    else:
        warp_img_rgb = np.clip(warped_img[0].permute(
            1, 2, 0).cpu().detach().numpy(), 0, 1) * 255
        warp_img_bgr = cv2.cvtColor(
            warp_img_rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)

    grid = np.vstack([
        np.hstack([prep_cell(anom_img, "Anomaly"), prep_cell(
            warp_img_bgr, "Warped (Photo Error)")]),
        np.hstack([prep_cell(depth_to_color(g_dep, d_min, d_max), "GT Depth"), prep_cell(
            depth_to_color(p_dep, d_min, d_max), "Pred Depth")]),
        np.hstack([prep_cell(flow_to_color(g_flow_np), "GT Flow"),
                  prep_cell(p_flow_img, "Pred Flow")]),
    ])

    grid_resized = cv2.resize(
        grid, (int(grid.shape[1] * H / grid.shape[0]), H))
    final_img = np.hstack([pred_canvas, gt_canvas, grid_resized])

    filepath = os.path.join(output_dir, f"vis_step_{step:05d}.jpg")
    cv2.imwrite(filepath, final_img)
    return filepath


def depth_to_color(depth_map, d_min=None, d_max=None):
    d_min = d_min if d_min is not None else depth_map.min()
    d_max = d_max if d_max is not None else depth_map.max()
    d_norm = (depth_map - d_min) / \
        (d_max - d_min) if d_max > d_min else np.zeros_like(depth_map)
    return cv2.applyColorMap((np.clip(d_norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)



def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20, with_nms=True):
    obj_list = preds.get("objectness", [])
    box_list = preds.get("boxes", [])

    if not isinstance(obj_list, list):
        obj_list, box_list = [obj_list], [box_list]
        cls_list = [preds.get("classification")
                    ] if "classification" in preds else []
        coef_list = [preds.get("mask_coefficients")
                     ] if "mask_coefficients" in preds else []
    else:
        cls_list = preds.get("classification", [])
        coef_list = preds.get("mask_coefficients", [])

    B = obj_list[0].shape[0] if obj_list else 0
    device = obj_list[0].device if obj_list else torch.device("cuda")
    H_img, W_img = (obj_list[0].shape[2] * 8,
                    obj_list[0].shape[3] * 8) if obj_list else (0, 0)
    results = []

    for b in range(B):
        all_scores, all_boxes, all_masks_info, all_classes = [], [], [], []
        for i, (obj, box) in enumerate(zip(obj_list, box_list)):
            stride = 8 * (2 ** i)
            if box is None:
                continue

            obj_score = torch.sigmoid(obj[b, 0])
            # 由于实际训练为类无关的物体发现 (class-agnostic object discovery)，分类头未训练，
            # 乘法融合会导致预测置信度被未训练的分类概率严重抑制，因此直接使用 obj_score 作为置信度。
            final_score = obj_score

            valid = final_score > score_thresh
            if not valid.any():
                continue

            sel_scores = final_score[valid]
            decoded_boxes = box[b][:, valid].T
            cy, cx = valid.nonzero(as_tuple=True)

            grid_x_norm = (cx.float() * stride + stride / 2.0) / W_img
            grid_y_norm = (cy.float() * stride + stride / 2.0) / H_img

            pl_norm = decoded_boxes[:, 0] * stride / W_img
            pt_norm = decoded_boxes[:, 1] * stride / H_img
            pr_norm = decoded_boxes[:, 2] * stride / W_img
            pb_norm = decoded_boxes[:, 3] * stride / H_img

            decoded_boxes_norm = torch.stack([
                torch.clamp(grid_x_norm - pl_norm, 0.0, 1.0),
                torch.clamp(grid_y_norm - pt_norm, 0.0, 1.0),
                torch.clamp(grid_x_norm + pr_norm, 0.0, 1.0),
                torch.clamp(grid_y_norm + pb_norm, 0.0, 1.0)
            ], dim=-1)

            all_scores.append(sel_scores)
            all_boxes.append(decoded_boxes_norm)

            if cls_list and i < len(cls_list) and cls_list[i] is not None:
                all_classes.append(torch.argmax(
                    cls_list[i][b, :, cy, cx].T, dim=-1))
            else:
                all_classes.append(torch.zeros_like(
                    sel_scores, dtype=torch.long))

            if coef_list and i < len(coef_list) and coef_list[i] is not None:
                all_masks_info.append(coef_list[i][b, :, cy, cx].T)

        if not all_scores:
            results.append(None)
            continue

        all_scores = torch.cat(all_scores, dim=0)
        all_boxes = torch.cat(all_boxes, dim=0)
        all_classes = torch.cat(all_classes, dim=0)

        if with_nms:
            boxes_scaled = all_boxes * torch.tensor([W_img, H_img, W_img, H_img], device=device)
            keep = torchvision.ops.nms(boxes_scaled, all_scores, nms_thresh)[:max_det]
        else:
            # 端到端检测，无需 NMS，直接取 Top-K
            num_keep = min(max_det, all_scores.shape[0])
            _, keep = torch.topk(all_scores, num_keep)

        protos = preds.get("mask_prototypes")
        protos = protos[0] if isinstance(protos, list) else protos
        masks_bool = None

        if protos is not None and all_masks_info:
            all_masks_info = torch.cat(all_masks_info, dim=0)
            masks = F.interpolate(
                torch.einsum(
                    "kp,phw->khw", all_masks_info[keep], protos[b]).unsqueeze(0),
                size=(H_img, W_img), mode="bilinear", align_corners=False
            )[0]

            x1, y1, x2, y2 = (
                all_boxes[keep] * torch.tensor([W_img, H_img, W_img, H_img], device=device)).unbind(-1)
            rows = torch.arange(H_img, device=device).view(1, H_img, 1)
            cols = torch.arange(W_img, device=device).view(1, 1, W_img)

            masks_bool = (masks > 0) & (cols >= x1.view(-1, 1, 1)) & (cols < x2.view(-1,
                                                                                     1, 1)) & (rows >= y1.view(-1, 1, 1)) & (rows < y2.view(-1, 1, 1))

        results.append({"scores": all_scores[keep], "boxes": all_boxes[keep],
                       "masks": masks_bool, "classes": all_classes[keep]})

    return results


