import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

from utils.geometry import *

@torch.no_grad()
def flow_epe_px(pred_flow_norm, gt_flow_norm, valid_mask=None, img_size=256):
    """
    pred_flow_norm / gt_flow_norm: [B, 2, H, W]
    当前代码约定：flow_norm = flow_px * 2 / img_size
    """
    pred_px = pred_flow_norm * (img_size / 2.0)
    gt_px = gt_flow_norm * (img_size / 2.0)
    epe = torch.linalg.vector_norm(pred_px - gt_px, dim=1)

    if valid_mask is not None:
        valid = valid_mask.float().expand_as(epe)
        return (epe * valid).sum() / valid.sum().clamp(min=1.0)

    return epe.mean()


@torch.no_grad()
def depth_metrics(pred_depth, gt_depth, valid_mask):
    pred = pred_depth.clamp(min=1e-4)
    gt = gt_depth.clamp(min=1e-4)
    valid = valid_mask.bool()

    pred = pred[valid]
    gt = gt[valid]

    if pred.numel() == 0:
        z = torch.tensor(0.0, device=pred_depth.device)
        return {"AbsRel": z, "RMSElog": z, "Delta1": z}

    abs_rel = (pred - gt).abs().div(gt).mean()
    rmse_log = torch.sqrt(((torch.log(pred) - torch.log(gt)) ** 2).mean())

    ratio = torch.maximum(pred / gt, gt / pred)
    delta1 = (ratio < 1.25).float().mean()

    return {
        "AbsRel": abs_rel,
        "RMSElog": rmse_log,
        "Delta1": delta1,
    }

# =====================================================================

def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(
        preds_logits, targets, reduction="none")
    return (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()


def dfl_loss(pred_dist, target_distances, reg_max=16):
    if pred_dist.shape[-1] == 4:
        return torch.zeros(pred_dist.shape[:-1], device=pred_dist.device, dtype=pred_dist.dtype)

    tl = torch.clamp(target_distances.long(), 0, reg_max - 1)
    tr = torch.clamp(target_distances.long() + 1, 0, reg_max - 1)
    wl = tr.float() - target_distances
    wr = 1.0 - wl

    pred_dist = pred_dist.reshape(-1, 4, reg_max)
    loss_left = F.cross_entropy(
        pred_dist.reshape(-1, reg_max), tl.reshape(-1), reduction="none").reshape(wl.shape)
    loss_right = F.cross_entropy(
        pred_dist.reshape(-1, reg_max), tr.reshape(-1), reduction="none").reshape(wr.shape)

    return (loss_left * wl + loss_right * wr).mean(dim=-1)


def giou_loss(preds, targets):
    pl, pt, pr, pb = preds[..., :4].unbind(-1)
    tl, tt, tr, tb = targets[..., :4].unbind(-1)

    inter_area = (torch.min(pl, tl) + torch.min(pr, tr)) * \
        (torch.min(pt, tt) + torch.min(pb, tb))
    union_area = (pl + pr) * (pt + pb) + (tl + tr) * \
        (tt + tb) - inter_area + 1e-6

    convex_w = torch.max(pl, tl) + torch.max(pr, tr)
    convex_h = torch.max(pt, tt) + torch.max(pb, tb)
    convex_area = convex_w * convex_h + 1e-6

    iou = inter_area / union_area
    giou = iou - (convex_area - union_area) / convex_area
    return 1.0 - giou


def ssim_loss(x, y):
    pad_x = F.pad(x, (1, 1, 1, 1), mode="reflect")
    pad_y = F.pad(y, (1, 1, 1, 1), mode="reflect")

    mu_x = F.avg_pool2d(pad_x, 3, 1)
    mu_y = F.avg_pool2d(pad_y, 3, 1)

    sigma_x = F.avg_pool2d(pad_x**2, 3, 1) - mu_x**2
    sigma_y = F.avg_pool2d(pad_y**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(pad_x * pad_y, 3, 1) - mu_x * mu_y

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
        ((mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2))
    return torch.clamp((1 - ssim_map) / 2, 0, 1)


def edge_aware_smoothness_loss(depth, img):
    norm_depth = (depth.float(
    ) / torch.clamp(depth.mean(dim=[2, 3], keepdim=True).float(), min=1e-4)).to(depth.dtype)

    depth_dx = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
    img_dx = torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)

    depth_dy = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    img_dy = torch.mean(
        torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)

    return (depth_dx * torch.exp(-img_dx)).mean() + (depth_dy * torch.exp(-img_dy)).mean()


def get_loss_weights(step):
    # 阶段 1：先让检测、分割、深度、ego 对齐
    if step < 2000:
        return {
            "obj": 1.0,
            "box": 1.5,
            "mask": 1.0,
            "depth": 3.0,
            "photo": 0.0,
            "ego": 2.0,
            "flow": 0.0,
            "cls": 0.0,
            "attr": 0.0,
            "anom": 0.0,
            "smooth": 0.02,
            "gate": 0.0,
            "track": 0.0,
        }

    # 阶段 2：加入监督光流，但仍不加 tracking
    if step < 5000:
        return {
            "obj": 1.0,
            "box": 1.5,
            "mask": 1.0,
            "depth": 3.0,
            "photo": 0.0,
            "ego": 2.0,
            "flow": 2.0,
            "cls": 0.0,
            "attr": 0.2,
            "anom": 0.0,
            "smooth": 0.02,
            "gate": 0.0,
            "track": 0.0,
        }

    # 阶段 3：加入 tracking
    return {
        "obj": 1.0,
        "box": 1.5,
        "mask": 1.0,
        "depth": 3.0,
        "photo": 0.0,
        "ego": 2.0,
        "flow": 2.0,
        "cls": 0.0,
        "attr": 0.5,
        "anom": 0.2,
        "smooth": 0.02,
        "gate": 0.0,
        "track": 1.0,
    }


LOSS_EMA = {}


def get_ema_loss(name, current_val, alpha=0.95):
    global LOSS_EMA
    with torch.no_grad():
        val = current_val.detach()
        if name not in LOSS_EMA:
            LOSS_EMA[name] = val.clone() if val > 0.0 else torch.tensor(
                1.0, device=val.device)
        if val > 0.0:
            LOSS_EMA[name] = LOSS_EMA[name] * alpha + val * (1.0 - alpha)
        return torch.clamp(LOSS_EMA[name], min=1e-4) if val > 0.0 else torch.tensor(1.0, device=val.device)

# =====================================================================
# 端到端追踪损失函数
# =====================================================================


def compute_track_loss(preds, targets, step):
    if "track_boxes" not in preds:
        device = next(iter(preds.values())
                      ).device if preds else torch.device("cpu")
        return torch.tensor(0., device=device)

    track_boxes = preds["track_boxes"]
    track_alive = preds["track_alive"]
    B, T, N, _ = track_boxes.shape
    device = track_boxes.device

    track_gt_boxes = targets.get("track_gt_boxes")
    track_gt_valid = targets.get("track_gt_valid")

    if track_gt_boxes is None or track_gt_valid is None:
        return torch.tensor(0., device=device)

    # track_gt_boxes initially has shape [B, T, MAX_INSTANCES, 4]
    # track_gt_valid initially has shape [B, T, MAX_INSTANCES]
    # But if targets was passed through _extract_target_chunk, it was flattened to [B * T, ...]
    # Let's reconstruct the [B, T, ...] shape
    if track_gt_boxes.dim() == 3:
        track_gt_boxes = track_gt_boxes.view(B, T, -1, 4)
    if track_gt_valid.dim() == 2:
        track_gt_valid = track_gt_valid.view(B, T, -1)

    loss_box = torch.tensor(0., device=device)
    loss_alive = torch.tensor(0., device=device)
    n_matched_total = 0

    # Batch compute cost matrix for all B and T on GPU
    flat_pred_boxes = track_boxes.flatten(0, 1)      # [B*T, N, 4]
    flat_gt_boxes = track_gt_boxes.flatten(0, 1)      # [B*T, M, 4]
    cost_matrix_all = torch.cdist(flat_pred_boxes.detach(), flat_gt_boxes, p=1) # [B*T, N, M]

    # Transfer cost matrix and valid masks to CPU in exactly ONE synchronization!
    cost_matrix_cpu = cost_matrix_all.cpu().numpy()
    flat_gt_valid_cpu = track_gt_valid.flatten(0, 1).cpu().numpy()

    b_list, t_list, q_list, g_list = [], [], [], []
    assignments = {}  # (b, gt_idx) -> query_id

    for t in range(T):
        alive_t = track_alive[:, t, :, 0]
        alive_target = torch.zeros(B, N, device=device)

        for b in range(B):
            idx = b * T + t
            valid_ids = np.where(flat_gt_valid_cpu[idx])[0].tolist()

            if len(valid_ids) == 0:
                continue

            used_queries = {
                q for (bb, _gid), q in assignments.items()
                if bb == b
            }

            # 1. 已绑定且当前可见的 GT，继续监督同一个 query
            new_gt_ids = []
            for gt_idx in valid_ids:
                key = (b, int(gt_idx))
                if key in assignments:
                    qi = assignments[key]
                    if qi < N:
                        alive_target[b, qi] = 1.0
                        b_list.append(b)
                        t_list.append(t)
                        q_list.append(qi)
                        g_list.append(gt_idx)
                else:
                    new_gt_ids.append(gt_idx)

            # 2. 新出现 GT 才使用 Hungarian 分配空闲 query
            if len(new_gt_ids) > 0:
                free_queries = [q for q in range(N) if q not in used_queries]
                if len(free_queries) == 0:
                    continue

                cost = cost_matrix_cpu[idx][np.array(free_queries)][:, np.array(new_gt_ids)]

                if _lsa is not None:
                    q_local, g_local = _lsa(cost)
                else:
                    q_local, g_local = [], []
                    used_local = set()
                    for gi in range(len(new_gt_ids)):
                        best = None
                        best_cost = 1e18
                        for qi_local in range(len(free_queries)):
                            if qi_local in used_local:
                                continue
                            c = cost[qi_local, gi]
                            if c < best_cost:
                                best = qi_local
                                best_cost = c
                        if best is not None:
                            used_local.add(best)
                            q_local.append(best)
                            g_local.append(gi)

                for qli, gli in zip(q_local, g_local):
                    qi = int(free_queries[qli])
                    gt_idx = int(new_gt_ids[gli])
                    assignments[(b, gt_idx)] = qi

                    alive_target[b, qi] = 1.0
                    b_list.append(b)
                    t_list.append(t)
                    q_list.append(qi)
                    g_list.append(gt_idx)

        loss_alive = loss_alive + F.binary_cross_entropy_with_logits(
            alive_t, alive_target
        )

    if len(b_list) > 0:
        b_idx = torch.tensor(b_list, dtype=torch.long, device=device)
        t_idx = torch.tensor(t_list, dtype=torch.long, device=device)
        q_idx = torch.tensor(q_list, dtype=torch.long, device=device)
        g_idx = torch.tensor(g_list, dtype=torch.long, device=device)

        pred_boxes_matched = track_boxes[b_idx, t_idx, q_idx]
        gt_boxes_matched = track_gt_boxes[b_idx, t_idx, g_idx]
        loss_box = F.smooth_l1_loss(pred_boxes_matched, gt_boxes_matched, beta=0.1, reduction="sum")
        n_matched_total = len(b_list)
    else:
        n_matched_total = 1

    n_matched_total = max(n_matched_total, 1)
    loss_box = loss_box / n_matched_total
    loss_alive = loss_alive / T

    return 1.5 * loss_box + 0.5 * loss_alive

def compute_instance_loss(preds, targets, step):
    B = preds["objectness"][0].shape[0]
    device = preds["objectness"][0].device
    num_scales = len(preds["objectness"])

    loss_obj = torch.tensor(0.0, device=device)
    loss_box = torch.tensor(0.0, device=device)
    loss_mask = torch.tensor(0.0, device=device)
    loss_cls = torch.tensor(0.0, device=device)

    w = get_loss_weights(step)

    for i in range(num_scales):
        p_obj, t_obj = preds["objectness"][i], targets["obj_dense"][i]

        loss_obj += focal_loss(p_obj, t_obj)
        if "dense_objectness" in preds:
            loss_obj += focal_loss(preds["dense_objectness"][i], t_obj) * 0.5

        pos_mask = t_obj[:, 0] > 0.5

        if w["box"] > 0:
            pb = preds["boxes"][i].permute(0, 2, 3, 1)
            tb = targets["bboxes_dense"][i].permute(0, 2, 3, 1)
            pdist = preds["box_dist"][i].permute(0, 2, 3, 1)

            l1_w = min(1.0, max(0.0, (step - 500) / 1000.0))
            if step >= 500:
                giou = F.smooth_l1_loss(pb, tb, beta=1.0, reduction="none").mean(
                    dim=-1) * (1 - l1_w) + giou_loss(pb, tb) * l1_w
            else:
                giou = F.smooth_l1_loss(
                    pb, tb, beta=1.0, reduction="none").mean(dim=-1)

            box_l = (giou * 1.5 + dfl_loss(pdist, tb, 32)
                     * 0.5) * pos_mask.float()
            loss_box += box_l.sum() / pos_mask.float().sum().clamp(min=1.0)

        if w["mask"] > 0:
            pos_mask_b = pos_mask.bool()
            if pos_mask_b.any():
                b_idx, y_idx, x_idx = torch.where(pos_mask_b)
                mc = preds["mask_coefficients"][i].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
                
                protos = preds["mask_prototypes"]
                protos_n = protos[b_idx]
                
                pred_logits = torch.einsum("nc,nchw->nhw", mc, protos_n)
                
                H, W = targets["seg_raw"].shape[1], targets["seg_raw"].shape[2]
                stride = 8 * (2 ** i)
                
                gy = (y_idx.float() * stride + stride / 2.0).long().clamp(0, H - 1)
                gx = (x_idx.float() * stride + stride / 2.0).long().clamp(0, W - 1)
                
                inst_ids = targets["seg_raw"][b_idx, gy, gx]
                gt_masks_full = (targets["seg_small"][b_idx] == inst_ids.view(-1, 1, 1)).float()
                
                if gt_masks_full.shape[-2:] != pred_logits.shape[-2:]:
                    gt_masks_full = F.interpolate(gt_masks_full.unsqueeze(1), size=pred_logits.shape[-2:], mode="nearest").squeeze(1)
                
                tb = targets["bboxes_dense"][i].permute(0, 2, 3, 1)[b_idx, y_idx, x_idx]
                
                mask_stride = H / pred_logits.shape[-2]
                gy_m = (y_idx.float() * stride + stride / 2.0) / mask_stride
                gx_m = (x_idx.float() * stride + stride / 2.0) / mask_stride
                
                pl_m = tb[:, 0] * stride / mask_stride
                pt_m = tb[:, 1] * stride / mask_stride
                pr_m = tb[:, 2] * stride / mask_stride
                pb_m = tb[:, 3] * stride / mask_stride
                
                x1 = (gx_m - pl_m).clamp(0, pred_logits.shape[-1] - 1)
                y1 = (gy_m - pt_m).clamp(0, pred_logits.shape[-2] - 1)
                x2 = (gx_m + pr_m).clamp(0, pred_logits.shape[-1] - 1)
                y2 = (gy_m + pb_m).clamp(0, pred_logits.shape[-2] - 1)
                
                rows = torch.arange(pred_logits.shape[-2], device=device).view(1, -1, 1)
                cols = torch.arange(pred_logits.shape[-1], device=device).view(1, 1, -1)
                
                box_mask = (cols >= x1.view(-1, 1, 1)) & (cols <= x2.view(-1, 1, 1)) & \
                           (rows >= y1.view(-1, 1, 1)) & (rows <= y2.view(-1, 1, 1))
                
                pred_logits_crop = pred_logits.masked_fill(~box_mask, -10.0)
                gt_masks_crop = gt_masks_full * box_mask.float()
                
                intersection = (torch.sigmoid(pred_logits_crop) * gt_masks_crop).sum(dim=(1, 2))
                union = torch.sigmoid(pred_logits_crop).sum(dim=(1, 2)) + gt_masks_crop.sum(dim=(1, 2))
                
                bce = F.binary_cross_entropy_with_logits(pred_logits_crop, gt_masks_crop, reduction="none")
                focal_bce = (0.25 * (1 - torch.exp(-bce)) ** 2 * bce).mean(dim=(1, 2))
                
                dice_loss = 1.0 - (2.0 * intersection + gt_masks_crop.sum(dim=(1, 2)).clamp(min=1.0) * 0.01) / \
                                  (union + gt_masks_crop.sum(dim=(1, 2)).clamp(min=1.0) * 0.01)
                
                valid_mask_inst = (inst_ids > 0).float()
                loss_mask += ((dice_loss * 2.0 + focal_bce) * valid_mask_inst).sum() / valid_mask_inst.sum().clamp(min=1.0)

        # [FIX] 如果 get_loss_weights 中的 cls 为 0，此处将被跳过，保护分类器字典不被错误标签淹没。
        if w.get("cls", 0) > 0 and "dense_classification" in preds and "cls_dense" in targets:
            gt_cls = targets["cls_dense"][i][:, 0].long()
            dense_cls_loss = F.cross_entropy(preds["dense_classification"][i].permute(
                0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)
            main_cls_loss = F.cross_entropy(preds["classification"][i].permute(
                0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)

            loss_cls += ((dense_cls_loss + main_cls_loss) * 0.5 *
                         pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

    return loss_obj, loss_box, loss_mask, loss_cls

def compute_attribute_loss(preds, targets):
    if "attributes" not in preds:
        return torch.tensor(0.0, device=next(iter(preds.values())).device)

    loss = torch.tensor(0.0, device=preds["attributes"][0].device)
    n_terms = 0

    for i, pred_attr in enumerate(preds["attributes"]):
        obj = targets["obj_dense"][i]
        pos = obj[:, 0] > 0.5

        if pos.sum() == 0:
            continue

        init_dyn = targets["initial_dynamic_dense"][i][:, 0]
        cur_mov = targets["current_moving_dense"][i][:, 0]
        init_valid = targets.get("initial_dynamic_valid_dense", None)
        cur_valid = targets.get("current_moving_valid_dense", None)

        if init_valid is not None:
            init_pos = pos & (init_valid[i][:, 0] > 0.5)
        else:
            init_pos = pos

        if init_pos.any():
            loss_init = F.binary_cross_entropy_with_logits(
                pred_attr[:, 0], init_dyn.float(), reduction="none"
            )
            loss = loss + (loss_init * init_pos.float()).sum() / init_pos.float().sum().clamp(min=1.0)
            n_terms += 1

        if cur_valid is not None:
            cur_pos = pos & (cur_valid[i][:, 0] > 0.5)
        else:
            cur_pos = pos

        if cur_pos.any():
            loss_cur = F.binary_cross_entropy_with_logits(
                pred_attr[:, 1], cur_mov.float(), reduction="none"
            )
            loss = loss + (loss_cur * cur_pos.float()).sum() / cur_pos.float().sum().clamp(min=1.0)
            n_terms += 1

    return loss / max(n_terms, 1)


def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", step=0):
    device = preds["depth"].device
    H, W = preds["depth"].shape[-2:]
    w = get_loss_weights(step)

    loss_obj, loss_box, loss_mask, loss_cls = compute_instance_loss(
        preds, targets, step)
    loss_ego, loss_depth, loss_flow, loss_photo, loss_smooth = [
        torch.tensor(0.0, device=device) for _ in range(5)]
    loss_track = torch.tensor(0.0, device=device)
    loss_attr = torch.tensor(0.0, device=device)

    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        R_n_inv = quaternion_to_matrix(
            targets["cam_quat_next"]).transpose(1, 2)
        trans_diff = torch.bmm(
            R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)).squeeze(-1)
        rot_diff = matrix_to_6d(
            torch.bmm(R_n_inv, quaternion_to_matrix(targets["cam_quat_t"])))
        gt_pose = torch.cat([trans_diff, rot_diff], dim=1)

        loss_ego = F.smooth_l1_loss(preds["ego_pose"], gt_pose)

        v_d_mask = (~targets["sky_mask"]).float()
        l_depth_base = (F.smooth_l1_loss(
            preds["log_depth"], targets["log_depth"], reduction="none") * v_d_mask).sum() / v_d_mask.sum().clamp(min=1)

        pd_dx = preds["depth"][:, :, 1:] - preds["depth"][:, :, :-1]
        td_dx = targets["depth"][:, :, 1:] - targets["depth"][:, :, :-1]
        mask_dx = v_d_mask[:, :, 1:] * v_d_mask[:, :, :-1]
        l_depth_dx = F.smooth_l1_loss(
            pd_dx * mask_dx, td_dx * mask_dx, reduction="sum")

        pd_dy = preds["depth"][:, 1:, :] - preds["depth"][:, :-1, :]
        td_dy = targets["depth"][:, 1:, :] - targets["depth"][:, :-1, :]
        mask_dy = v_d_mask[:, 1:, :] * v_d_mask[:, :-1, :]
        l_depth_dy = F.smooth_l1_loss(
            pd_dy * mask_dy, td_dy * mask_dy, reduction="sum")

        loss_depth = l_depth_base + 0.5 * \
            (l_depth_dx + l_depth_dy) / v_d_mask.sum().clamp(min=1)

    ret_flow_epe = torch.tensor(0.0, device=device)
    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        if "has_next" in targets:
            has_n = targets["has_next"].view(-1, 1, 1, 1).float()
            l_flow_raw = F.smooth_l1_loss(
                preds["flow"], targets["flow_target"], reduction="none") * has_n
            loss_flow = l_flow_raw.sum() / (has_n.sum().clamp(min=1) *
                                             preds["flow"].shape[1] * H * W)
        else:
            loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"])

    if preds.get("flow") is not None and "flow_target" in targets:
        ret_flow_epe = flow_epe_px(
            preds["flow"].detach(),
            targets["flow_target"].detach(),
            valid_mask=targets.get("has_next", None).view(-1, 1, 1)
            if "has_next" in targets else None,
            img_size=W,
        )

    depth_abs_rel = torch.tensor(0.0, device=device)
    depth_rmse_log = torch.tensor(0.0, device=device)
    depth_delta1 = torch.tensor(0.0, device=device)
    if mode == "supervised" and "depth" in targets and "sky_mask" in targets:
        with torch.no_grad():
            d_metrics = depth_metrics(
                preds["depth"].detach(),
                targets["depth"].detach(),
                ~targets["sky_mask"].detach(),
            )
            depth_abs_rel = d_metrics["AbsRel"]
            depth_rmse_log = d_metrics["RMSElog"]
            depth_delta1 = d_metrics["Delta1"]

    warped_img = None
    if img_t is not None:
        loss_smooth = edge_aware_smoothness_loss(
            preds["depth"].unsqueeze(1), img_t)
        if img_next is not None:
            K, K_inv = generate_intrinsics(
                H,
                W,
                device,
                focal_length=targets.get("camera_focal_length", None),
                sensor_width=targets.get("camera_sensor_width", None),
                dtype=preds["depth"].dtype,
            )
            warped_img, v_w_mask = inverse_warp(
                img_next,
                preds["depth"].unsqueeze(1),
                preds["ego_pose"],
                K,
                K_inv,
                depth_is_distance=True,
            )

            if w["photo"] > 0:
                def p_loss(p, t):
                    return 0.15 * F.l1_loss(p, t, reduction="none").mean(dim=1, keepdim=True) + 0.85 * ssim_loss(p, t).mean(dim=1, keepdim=True)

                w_loss = p_loss(warped_img, img_t)
                has_n_factor = targets["has_next"].view(
                    -1, 1, 1, 1).float() if "has_next" in targets else 1.0
                m = v_w_mask * (1 - targets["sky_mask"].float().unsqueeze(1)) * (
                    w_loss < p_loss(img_next, img_t)).float() * has_n_factor

                loss_photo = (w_loss * m).sum() / m.sum().clamp(min=1)

    loss_anom = preds["feature_error"].mean()
    loss_gate = preds["state_update_gate"].abs().mean() * 0.01

    if w.get("track", 0) > 0 and "track_boxes" in preds:
        loss_track = compute_track_loss(preds, targets, step)

    if w.get("attr", 0) > 0 and "attributes" in preds:
        loss_attr = compute_attribute_loss(preds, targets)

    loss_components = {
        "Obj": loss_obj, "Box": loss_box, "Mask": loss_mask,
        "Depth": loss_depth, "Photo": loss_photo, "Ego": loss_ego,
        "Flow": loss_flow, "Anom": loss_anom, "Cls": loss_cls, "Attr": loss_attr
    }

    tot = sum(w.get(k.lower(), 0) *
              (l / get_ema_loss(k[:3], l)) for k, l in loss_components.items())
    tot += w.get("smooth", 0.05) * loss_smooth + w.get("gate",
                                                       0.05) * loss_gate + w.get("track", 0) * loss_track

    ret_dict = {k: v.detach() for k, v in loss_components.items() if w.get(k.lower(), 0) > 0}
    if w.get("gate", 0) > 0:
        ret_dict["Gate"] = loss_gate.detach()
    if w.get("track", 0) > 0:
        ret_dict["Track"] = loss_track.detach()

    ret_dict["FlowEPEpx"] = ret_flow_epe.detach()
    ret_dict["DepthAbsRel"] = depth_abs_rel.detach()
    ret_dict["DepthRMSElog"] = depth_rmse_log.detach()
    ret_dict["DepthDelta1"] = depth_delta1.detach()
    ret_dict["Tot"] = tot.detach()

    return tot, ret_dict, warped_img

# =====================================================================
