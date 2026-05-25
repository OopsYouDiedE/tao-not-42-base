import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision


def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    w2 = w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    matrix = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * (xy - zw),
            2 * (xz + yw),
            2 * (xy + zw),
            w2 - x2 + y2 - z2,
            2 * (yz - xw),
            2 * (xz - yw),
            2 * (yz + xw),
            w2 - x2 - y2 + z2,
        ],
        dim=-1,
    ).view(*q.shape[:-1], 3, 3)
    return matrix


def matrix_to_6d(matrix):
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def six_d_to_matrix(d6):
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    y = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
    y = F.normalize(y, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack([x, y, z], dim=-1)


# =====================================================================
# 1. 物理特征网络架构
# =====================================================================
def decode_dfl_boxes(pred_dist, reg_max=16):
    if isinstance(pred_dist, list):
        return [decode_dfl_boxes(x, reg_max) for x in pred_dist]
    # pred_dist: (B, 4*reg_max, H, W)
    B, C, H, W = pred_dist.shape
    prob = F.softmax(pred_dist.view(B, 4, reg_max, H, W), dim=2)
    weights = torch.arange(reg_max, dtype=torch.float32, device=pred_dist.device)
    distances = (prob * weights.view(1, 1, reg_max, 1, 1)).sum(dim=2)  # (B, 4, H, W)
    return distances


import urllib.request  # 确保文件顶部有引入，或者直接在函数里引入


def load_yolo_backbone_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 权重文件 {checkpoint_path} 不存在，正在尝试自动从 GitHub 下载...")
        try:
            # Ultralytics 的官方权重下载链接 (YOLO11 的权重存放在 v8.3.0 release 中)
            url = f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{checkpoint_path}"
            urllib.request.urlretrieve(url, checkpoint_path)
            print(f"✅ 自动下载成功: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            print(f"👉 请手动下载 {checkpoint_path} 并放置在项目根目录下。")
            return

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            state_dict = ckpt["model"].state_dict()
        elif hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            state_dict = ckpt
    except Exception as e:
        print(f"⚠️ 加载权重失败: {e}")
        return

    target_state = model.state_dict()
    updates = {}

    for src_key, src_val in state_dict.items():
        # Ultralytics pt 文件的 key 通常是 model.model.0.conv.weight
        # 我们 MyYOLOE 的 key 是 segmenter.model.0.conv.weight
        if src_key.startswith("model.model."):
            tgt_key = src_key.replace("model.model.", "segmenter.model.")
        elif src_key.startswith("model."):
            tgt_key = src_key.replace("model.", "segmenter.model.")
        else:
            tgt_key = src_key

        if tgt_key in target_state and target_state[tgt_key].shape == src_val.shape:
            updates[tgt_key] = src_val

    target_state.update(updates)
    model.load_state_dict(target_state)
    print(
        f"✅ 成功加载 YOLO 预训练权重: {checkpoint_path} (匹配了 {len(updates)} 个张量)"
    )


def freeze_backbone(model):
    print("❄️ 正在冻结 YOLOE 分割模块 (保持其强大的 Zero-shot 基础能力)...")
    for name, param in model.segmenter.named_parameters():
        param.requires_grad = False


# =====================================================================
# 3. 经验回放池 (Replay Buffer) 与 GPU 数据流水线
# =====================================================================
def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20):
    if isinstance(preds["objectness"], list):
        # Simplify extraction by using the highest resolution scale (P3)
        preds = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}

    B = preds["objectness"].shape[0]
    H_feat, W_feat = preds["objectness"].shape[2:]
    device = preds["objectness"].device

    results = []
    H_img, W_img = H_feat * 8, W_feat * 8

    for b in range(B):
        boxes = preds.get("boxes")
        if boxes is None:
            results.append(None)
            continue

        decoded_boxes = boxes[b]
        obj = preds["objectness"][b, 0]

        scores = torch.sigmoid(obj)
        valid = scores > score_thresh
        if not valid.any():
            results.append(None)
            continue

        sel_scores = scores[valid]
        decoded_boxes = decoded_boxes[:, valid].T

        indices = valid.nonzero()
        cy = indices[:, 0].float()
        cx = indices[:, 1].float()

        grid_x_norm = (cx * 8.0 + 4.0) / W_img
        grid_y_norm = (cy * 8.0 + 4.0) / H_img

        pl_norm = decoded_boxes[:, 0] * 8.0 / W_img
        pt_norm = decoded_boxes[:, 1] * 8.0 / H_img
        pr_norm = decoded_boxes[:, 2] * 8.0 / W_img
        pb_norm = decoded_boxes[:, 3] * 8.0 / H_img

        x1 = torch.clamp(grid_x_norm - pl_norm, 0.0, 1.0)
        y1 = torch.clamp(grid_y_norm - pt_norm, 0.0, 1.0)
        x2 = torch.clamp(grid_x_norm + pr_norm, 0.0, 1.0)
        y2 = torch.clamp(grid_y_norm + pb_norm, 0.0, 1.0)

        decoded_boxes_norm = torch.stack([x1, y1, x2, y2], dim=-1)
        pixel_boxes = decoded_boxes_norm * torch.tensor(
            [W_img, H_img, W_img, H_img], device=device
        )

        keep = torchvision.ops.nms(pixel_boxes, sel_scores, nms_thresh)[:max_det]

        coeffs = preds.get("mask_coefficients")
        protos = preds.get("mask_prototypes")

        if coeffs is not None and protos is not None:
            sel_coeffs = coeffs[b, :, valid.nonzero()[:, 0], valid.nonzero()[:, 1]].T
            kept_coeffs = sel_coeffs[keep]
            masks = torch.einsum("kp,phw->khw", kept_coeffs, protos[b])
            masks = F.interpolate(
                masks.unsqueeze(0),
                size=(H_img, W_img),
                mode="bilinear",
                align_corners=False,
            )[0]

            # Box crop for masks
            boxes_pixel = pixel_boxes[keep]
            N_masks = masks.shape[0]
            rows = torch.arange(H_img, device=device).view(1, H_img, 1)
            cols = torch.arange(W_img, device=device).view(1, 1, W_img)
            x1, y1, x2, y2 = boxes_pixel.unbind(-1)
            mask_crop = (
                (cols >= x1.view(N_masks, 1, 1))
                & (cols < x2.view(N_masks, 1, 1))
                & (rows >= y1.view(N_masks, 1, 1))
                & (rows < y2.view(N_masks, 1, 1))
            )
            masks = masks * mask_crop.float() - 10.0 * (~mask_crop).float()

            masks_bool = (masks > 0) & mask_crop
        else:
            masks_bool = None

        classes = None
        if "classification" in preds:
            cls_logits = preds["classification"][
                b, :, valid.nonzero()[:, 0], valid.nonzero()[:, 1]
            ].T
            # class index with max probability
            classes = torch.argmax(cls_logits, dim=-1)[keep]

        results.append(
            {
                "scores": sel_scores[keep],
                "boxes": decoded_boxes_norm[keep],
                "masks": masks_bool,
                "classes": classes,
            }
        )
    return results


def compute_instance_loss(preds, targets, step=0):
    if isinstance(preds["objectness"], list):
        loss_obj_total = 0.0
        loss_box_total = 0.0
        loss_mask_total = 0.0
        loss_cls_total = 0.0
        num_scales = len(preds["objectness"])

        for i in range(num_scales):
            p_i = {}
            for k, v in preds.items():
                if isinstance(v, list) and len(v) == num_scales:
                    p_i[k] = v[i]
                else:
                    p_i[k] = v
            t_i = {}
            for k, v in targets.items():
                if isinstance(v, list) and len(v) == num_scales:
                    t_i[k] = v[i]
                else:
                    t_i[k] = v
            
            o, b, m, c = compute_instance_loss_single_scale(p_i, t_i, step)
            loss_obj_total += o
            loss_box_total += b
            loss_mask_total += m
            loss_cls_total += c
            
        return loss_obj_total, loss_box_total, loss_mask_total, loss_cls_total

    return compute_instance_loss_single_scale(preds, targets, step)


def compute_instance_loss_single_scale(preds, targets, step=0):
    device = preds["objectness"].device

    loss_obj = focal_loss(preds["objectness"], targets["obj_dense"])
    if "dense_objectness" in preds:
        loss_obj = (
            loss_obj + focal_loss(preds["dense_objectness"], targets["obj_dense"])
        ) * 0.5

    pos_mask = targets["obj_dense"][:, 0] > 0.5


    loss_box = torch.tensor(0.0, device=device)
    loss_mask = torch.tensor(0.0, device=device)
    w = get_loss_weights(step)

    if w["box"] > 0:
        pred_boxes = preds["boxes"].permute(0, 2, 3, 1)
        gt_boxes = targets["bboxes_dense"].permute(0, 2, 3, 1)

        loss_giou = giou_loss_with_l1_warmup(pred_boxes, gt_boxes, step=step)
        pred_dist = preds["box_dist"].permute(0, 2, 3, 1)
        loss_dfl = dfl_loss(pred_dist, gt_boxes, reg_max=32)

        loss_box_dense = loss_giou * 1.5 + loss_dfl * 0.5
        loss_box = (loss_box_dense * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

        if "dense_box_dist" in preds:
            pred_boxes_dense = decode_dfl_boxes(
                preds["dense_box_dist"], reg_max=32
            ).permute(0, 2, 3, 1)
            loss_giou_dense2 = giou_loss_with_l1_warmup(
                pred_boxes_dense, gt_boxes, step=step
            )
            pred_dist_dense = preds["dense_box_dist"].permute(0, 2, 3, 1)
            loss_dfl_dense = dfl_loss(pred_dist_dense, gt_boxes, reg_max=32)
            
            loss_box_dense2 = loss_giou_dense2 * 1.5 + loss_dfl_dense * 0.5
            loss_box = (loss_box + (loss_box_dense2 * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)) * 0.5

    if w["mask"] > 0:
        loss_mask = compute_per_instance_mask_loss(
            preds, targets, pos_mask, key="mask_coefficients"
        )
        if "dense_mask_coefficients" in preds:
            loss_mask_dense = compute_per_instance_mask_loss(
                preds, targets, pos_mask, key="dense_mask_coefficients"
            )
            loss_mask = (loss_mask + loss_mask_dense) * 0.5

    loss_cls = torch.tensor(0.0, device=device)
    if (
        w.get("cls", 0) > 0
        and "dense_classification" in preds
        and "cls_dense" in targets
    ):
        pred_cls_dense = preds["dense_classification"].permute(0, 2, 3, 1)
        pred_cls_o2o = preds["classification"].permute(0, 2, 3, 1)

        gt_cls = targets["cls_dense"][:, 0].long()

        loss_cls_dense = F.cross_entropy(pred_cls_dense.flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)
        loss_cls_o2o = F.cross_entropy(pred_cls_o2o.flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)
        
        loss_cls = ((loss_cls_dense + loss_cls_o2o) * 0.5 * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

    return loss_obj, loss_box, loss_mask, loss_cls


def compute_per_instance_mask_loss(preds, targets, pos_mask, key="mask_coefficients"):
    B, _, H_feat, W_feat = preds["objectness"].shape
    device = preds["objectness"].device
    seg_raw = targets["seg_raw"]
    H, W = seg_raw.shape[1], seg_raw.shape[2]

    stride = H // H_feat
    y_grid, x_grid = torch.meshgrid(torch.arange(H_feat, device=device), torch.arange(W_feat, device=device), indexing="ij")
    center_y = torch.clamp(y_grid * stride + stride // 2, 0, H - 1).unsqueeze(0).expand(B, -1, -1)
    center_x = torch.clamp(x_grid * stride + stride // 2, 0, W - 1).unsqueeze(0).expand(B, -1, -1)
    
    inst_ids = torch.gather(
        seg_raw.view(B, H * W), 
        1, 
        (center_y * W + center_x).view(B, H_feat * W_feat)
    ).view(B, H_feat, W_feat).long()

    coeffs = preds[key] # [B, 32, H_feat, W_feat]
    protos = preds["mask_prototypes"] # [B, 32, H_proto, W_proto]
    
    pred_logits = torch.einsum("bchw,bcHW->bhwHW", coeffs, protos)
    
    seg_batch = targets["seg_small"].unsqueeze(1).unsqueeze(2) # [B, 1, 1, H_proto, W_proto]
    gt_masks = (seg_batch == inst_ids.view(B, H_feat, W_feat, 1, 1)).float()

    if gt_masks.shape[-2:] != pred_logits.shape[-2:]:
        gt_masks = F.interpolate(
            gt_masks.flatten(0, 2).unsqueeze(1),
            size=pred_logits.shape[-2:],
            mode="nearest",
        ).squeeze(1).view_as(pred_logits)

    preds_sig = torch.sigmoid(pred_logits)
    intersection = (preds_sig * gt_masks).sum(dim=(3, 4))
    union = preds_sig.sum(dim=(3, 4)) + gt_masks.sum(dim=(3, 4))

    pos_count = gt_masks.sum(dim=(3, 4)).clamp(min=1.0)
    smooth = pos_count * 0.01
    loss_dice = 1.0 - (2.0 * intersection + smooth) / (union + smooth)

    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction="none")
    p_t = torch.exp(-bce)
    loss_bce = (0.25 * (1 - p_t) ** 2 * bce).mean(dim=(3, 4))

    loss_mask_dense = loss_dice * 2.0 + loss_bce * 1.0
    
    valid_mask = (inst_ids > 0).float() * pos_mask.float()
    loss_mask = (loss_mask_dense * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)
    return loss_mask


def setup_finetune_mode(model):
    for param in model.segmenter.parameters():
        param.requires_grad = False
    for param in model.depth_decoder.parameters():
        param.requires_grad = True
    for param in model.pose_head.parameters():
        param.requires_grad = True
    for param in model.conv_gru.parameters():
        param.requires_grad = True
    for param in model.feature_predictor.parameters():
        param.requires_grad = True
    for param in model.state_update_gate_head.parameters():
        param.requires_grad = True

    if hasattr(model.segmenter, "model"):
        model.segmenter.model[-1].obj_proj.requires_grad_(True)
        model.segmenter.model[-1].one2one_obj_proj.requires_grad_(True)
        if hasattr(model.segmenter.model[-1], "class_prompts"):
            model.segmenter.model[-1].class_prompts.requires_grad_(True)

    for param in model.flow_head.parameters():
        param.requires_grad = True


# =====================================================================
# 5. 论文级可视化
# =====================================================================
def depth_to_color(depth_map, d_min=None, d_max=None):
    if d_min is None:
        d_min = depth_map.min()
    if d_max is None:
        d_max = depth_map.max()
    if d_max > d_min:
        d_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(depth_map)
    d_uint8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)


def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(preds_logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    loss = alpha * (1 - p_t) ** gamma * bce
    return loss.mean()


def dfl_loss(pred_dist, target_distances, reg_max=16):
    target_left = target_distances.long()
    target_right = target_left + 1
    weight_left = target_right.float() - target_distances
    weight_right = 1.0 - weight_left

    target_left = torch.clamp(target_left, 0, reg_max - 1)
    target_right = torch.clamp(target_right, 0, reg_max - 1)

    pred_dist = pred_dist.view(-1, 4, reg_max)
    loss_left = (
        F.cross_entropy(
            pred_dist.view(-1, reg_max), target_left.view(-1), reduction="none"
        ).view(-1, 4)
        * weight_left
    )
    loss_right = (
        F.cross_entropy(
            pred_dist.view(-1, reg_max), target_right.view(-1), reduction="none"
        ).view(-1, 4)
        * weight_right
    )

    return (loss_left + loss_right).mean(dim=-1)


def giou_loss(preds, targets):
    pl, pt, pr, pb = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    tl, tt, tr, tb = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    inter_w = torch.min(pl, tl) + torch.min(pr, tr)
    inter_h = torch.min(pt, tt) + torch.min(pb, tb)
    inter_area = inter_w * inter_h

    p_area = (pl + pr) * (pt + pb)
    t_area = (tl + tr) * (tt + tb)

    union_area = p_area + t_area - inter_area + 1e-6
    iou = inter_area / union_area

    enclose_w = torch.max(pl, tl) + torch.max(pr, tr)
    enclose_h = torch.max(pt, tt) + torch.max(pb, tb)
    enclose_area = enclose_w * enclose_h + 1e-6

    giou = iou - (enclose_area - union_area) / enclose_area
    return 1.0 - giou


def giou_loss_with_l1_warmup(preds, targets, step, warmup_steps=500):
    l1 = F.smooth_l1_loss(preds, targets, beta=1.0, reduction="none").mean(dim=-1)
    if step < warmup_steps:
        return l1
    giou = giou_loss(preds, targets)
    alpha = min((step - warmup_steps) / 1000.0, 1.0)
    return l1 * (1 - alpha) + giou * alpha


def ssim_loss(x, y):
    C1 = 0.01**2
    C2 = 0.03**2
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")
    y_pad = F.pad(y, (1, 1, 1, 1), mode="reflect")
    mu_x = F.avg_pool2d(x_pad, 3, 1)
    mu_y = F.avg_pool2d(y_pad, 3, 1)
    sigma_x = F.avg_pool2d(x_pad**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(y_pad**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x_pad * y_pad, 3, 1) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def get_loss_weights(step):
    def ramp(start, end, val):
        if step < start:
            return 0.0
        if step > end:
            return val
        return val * (step - start) / (end - start)

    return {
        "obj": 1.0,
        "box": 1.5,
        "mask": 1.0,
        "depth": 3.0 if step < 3000 else 1.5,
        "photo": ramp(1000, 3000, 1.0),
        "ego": ramp(100, 600, 3.0),
        "flow": ramp(300, 1000, 1.0),
        "cls": ramp(1000, 1001, 1.0),
        "anom": ramp(4000, 6000, 1.0),
        "smooth": 0.05,
        "gate": 0.05,
    }


LOSS_EMA = {}


def get_ema_loss(name, current_val, alpha=0.95):
    global LOSS_EMA
    with torch.no_grad():
        val = current_val.detach()
        if name not in LOSS_EMA:
            LOSS_EMA[name] = torch.tensor(1.0, device=val.device)
            
        update_mask = (val > 0.0).float()
        LOSS_EMA[name] = LOSS_EMA[name] * (1.0 - update_mask * (1.0 - alpha)) + val * update_mask * (1.0 - alpha)
        
        ema_val = torch.clamp(LOSS_EMA[name], min=1e-4)
        return torch.where(val == 0.0, torch.tensor(1.0, device=val.device), ema_val)


def compute_physics_loss(
    preds,
    targets,
    img_t=None,
    img_next=None,
    mode="supervised",
    teacher_forcing_ego=None,
    step=0,
):
    device = preds["depth"].device
    B, H, W = preds["depth"].shape
    w = get_loss_weights(step)

    loss_obj, loss_box, loss_mask, loss_cls = compute_instance_loss(
        preds, targets, step=step
    )

    loss_ego = torch.tensor(0.0, device=device)
    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        c_mat_t = quaternion_to_matrix(targets["cam_quat_t"])
        c_mat_n = quaternion_to_matrix(targets["cam_quat_next"])
        R_n_inv = c_mat_n.transpose(1, 2)
        R_delta = torch.bmm(R_n_inv, c_mat_t)
        T_delta = torch.bmm(
            R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)
        ).squeeze(-1)
        gt_ego = torch.cat([T_delta, matrix_to_6d(R_delta)], dim=1)
        loss_ego = F.smooth_l1_loss(preds["ego_pose"], gt_ego)

    # Depth loss with sky exclusion
    loss_depth = torch.tensor(0.0, device=device)
    if mode == "supervised":
        raw_loss_depth = F.smooth_l1_loss(
            preds["log_depth"], targets["log_depth"], reduction="none"
        )
        valid_depth_mask = (~targets["sky_mask"]).float()
        loss_depth = (
            raw_loss_depth * valid_depth_mask
        ).sum() / valid_depth_mask.sum().clamp(min=1)
        
        # Depth Edge-Gradient Loss for sharpness
        grad_pred_x = preds["depth"][:, :, 1:] - preds["depth"][:, :, :-1]
        grad_gt_x = targets["depth"][:, :, 1:] - targets["depth"][:, :, :-1]
        grad_pred_y = preds["depth"][:, 1:, :] - preds["depth"][:, :-1, :]
        grad_gt_y = targets["depth"][:, 1:, :] - targets["depth"][:, :-1, :]
        
        mask_x = valid_depth_mask[:, :, 1:] * valid_depth_mask[:, :, :-1]
        mask_y = valid_depth_mask[:, 1:, :] * valid_depth_mask[:, :-1, :]
        
        loss_grad_x = F.smooth_l1_loss(grad_pred_x * mask_x, grad_gt_x * mask_x, reduction="sum")
        loss_grad_y = F.smooth_l1_loss(grad_pred_y * mask_y, grad_gt_y * mask_y, reduction="sum")
        loss_depth += 0.5 * (loss_grad_x + loss_grad_y) / valid_depth_mask.sum().clamp(min=1)

    loss_flow = torch.tensor(0.0, device=device)
    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        raw_loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"], reduction="none")
        if "has_next" in targets:
            valid_flow_mask = targets["has_next"].view(-1, 1, 1, 1).float()
            # Normalize by valid mask sum and the other dimensions (C, H, W)
            loss_flow = (raw_loss_flow * valid_flow_mask).sum() / (valid_flow_mask.sum().clamp(min=1) * raw_loss_flow.shape[1] * raw_loss_flow.shape[2] * raw_loss_flow.shape[3])
        else:
            loss_flow = raw_loss_flow.mean()

    warped_img = None
    loss_photo = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)

    if img_t is not None:
        loss_smooth = edge_aware_smoothness_loss(preds["depth"].unsqueeze(1), img_t)

    if img_t is not None and img_next is not None:
        K, K_inv = generate_intrinsics(H, W, device)
        warped_img, valid_warp_mask = inverse_warp(
            img_next, preds["depth"].unsqueeze(1), preds["ego_pose"], K, K_inv
        )
        
        if w["photo"] > 0:
            # Photo loss uses L1 + SSIM
            def photo_loss_fn(pred, tgt):
                l1 = F.l1_loss(pred, tgt, reduction="none").mean(dim=1, keepdim=True)
                ssim = ssim_loss(pred, tgt).mean(dim=1, keepdim=True)
                return 0.15 * l1 + 0.85 * ssim

            warp_loss = photo_loss_fn(warped_img, img_t)
            identity_loss = photo_loss_fn(img_next, img_t)

            auto_mask = (warp_loss < identity_loss).float()
            sky_mask_1 = targets["sky_mask"].float().unsqueeze(1)

            mask = valid_warp_mask * (1 - sky_mask_1) * auto_mask
            if "has_next" in targets:
                has_next_mask = targets["has_next"].view(-1, 1, 1, 1).float()
                mask = mask * has_next_mask
                
            loss_photo = (warp_loss * mask).sum() / mask.sum().clamp(min=1)

    loss_anom = preds["feature_error"].mean()
    loss_gate = preds["state_update_gate"].abs().mean() * 0.01

    norm_obj = loss_obj / get_ema_loss("Obj", loss_obj)
    norm_box = loss_box / get_ema_loss("Box", loss_box)
    norm_mask = loss_mask / get_ema_loss("Mask", loss_mask)
    norm_depth = loss_depth / get_ema_loss("Dep", loss_depth)
    norm_photo = loss_photo / get_ema_loss("Pht", loss_photo)
    norm_ego = loss_ego / get_ema_loss("Ego", loss_ego)
    norm_flow = loss_flow / get_ema_loss("Flw", loss_flow)
    norm_anom = loss_anom / get_ema_loss("Ano", loss_anom)
    norm_cls = loss_cls / get_ema_loss("Cls", loss_cls)

    total_loss = (
        w.get("obj", 1.0) * norm_obj
        + w.get("box", 0.0) * norm_box
        + w.get("mask", 0.0) * norm_mask
        + w.get("depth", 1.0) * norm_depth
        + w.get("photo", 0.0) * norm_photo
        + w.get("ego", 1.0) * norm_ego
        + w.get("flow", 0.0) * norm_flow
        + w.get("anom", 0.0) * norm_anom
        + w.get("cls", 0.0) * norm_cls
        + w.get("smooth", 0.05) * loss_smooth
        + w.get("gate", 0.05) * loss_gate
    )

    loss_dict = {
        "Obj": loss_obj.detach(),
        "Box": loss_box.detach(),
        "Mask": loss_mask.detach(),
        "Depth": loss_depth.detach(),
        "Photo": loss_photo.detach(),
        "Ego": loss_ego.detach(),
        "Flow": loss_flow.detach(),
        "Anom": loss_anom.detach(),
        "Gate": loss_gate.detach(),
        "Cls": loss_cls.detach(),
        "Tot": total_loss.detach(),
    }

    return total_loss, loss_dict, warped_img


def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32
    )
    K_inv = torch.inverse(K)
    return K, K_inv


def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(
        torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
    )
    x = x.flatten().expand(B, -1)
    y = y.flatten().expand(B, -1)
    ones = torch.ones_like(x)
    pixels = torch.stack([x, y, ones], dim=1)

    points_3d = torch.bmm(K_inv.expand(B, 3, 3), pixels.float())
    points_3d = points_3d * depth.view(B, 1, H * W)

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

    warped_img = F.grid_sample(
        img_next, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    warped_img = torch.nan_to_num(warped_img, 0.0)

    valid_mask = (
        ((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0))
        .view(B, 1, H, W)
        .float()
    )
    safe_depth_mask = ((depth > 0.01) & (z_next_raw.view(B, 1, H, W) > 0.01)).float()
    valid_mask = valid_mask * safe_depth_mask

    return warped_img, valid_mask


def edge_aware_smoothness_loss(depth, img):
    mean_depth = depth.mean(dim=[2, 3], keepdim=True).float()
    norm_depth = (depth.float() / torch.clamp(mean_depth, min=1e-4)).to(depth.dtype)
    grad_depth_x = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
    grad_depth_y = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    grad_img_x = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True
    )
    grad_img_y = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True
    )
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    return grad_depth_x.mean() + grad_depth_y.mean()


def dice_loss(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    targets = targets.flatten()
    intersection = (preds * targets).sum()
    dice = (2.0 * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1.0 - dice
