import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

from utils.geometry import *

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
    return {
        "obj": 1.0, "box": 1.5, "mask": 1.0,
        "depth": 3.0,
        "photo": 0.0,
        "ego": 3.0,
        "flow": 2.0,
        "cls": 0.0,  # <--- [FIX] 将分类损失彻底切断，保住 4585 维语义特征空间不坍缩为二分类
        "anom": 1.0,
        "smooth": 0.05,
        "gate": 0.05,
        "track": 1.0
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

    seg_BT = targets.get("seg_raw")
    if seg_BT is None:
        return torch.tensor(0., device=device)

    H_img, W_img = seg_BT.shape[-2:]
    seg = seg_BT.view(B, T, H_img, W_img)

    loss_box = torch.tensor(0., device=device)
    loss_alive = torch.tensor(0., device=device)
    loss_consist = torch.tensor(0., device=device)
    n_matched_total = 0

    prev_assignments = {}

    for t in range(T):
        boxes_t = track_boxes[:, t]
        alive_t = track_alive[:, t, :, 0]
        alive_target = torch.zeros(B, N, device=device)
        cur_assignments = {}

        for b in range(B):
            seg_bt = seg[b, t]
            inst_ids = [int(i) for i in seg_bt.unique().tolist() if i > 0]
            if not inst_ids:
                continue

            gt_boxes_list, valid_ids = [], []
            for iid in inst_ids:
                m = seg_bt == iid
                if not m.any():
                    continue
                ys, xs = m.nonzero(as_tuple=True)

                y1, y2 = ys.float().min() / H_img, ys.float().max() / H_img
                x1, x2 = xs.float().min() / W_img, xs.float().max() / W_img

                cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
                bw = (x2 - x1).clamp(min=1.0 / W_img)
                bh = (y2 - y1).clamp(min=1.0 / H_img)

                gt_boxes_list.append(torch.stack([cx, cy, bw, bh]))
                valid_ids.append(iid)

            if not gt_boxes_list:
                continue

            gt_boxes = torch.stack(gt_boxes_list)

            with torch.no_grad():
                cost = torch.cdist(
                    boxes_t[b].detach(), gt_boxes, p=1).cpu().numpy()

            if _lsa is not None:
                q_inds, g_inds = _lsa(cost)
            else:
                q_inds, g_inds = [], []
                used_q = set()
                for gi in range(min(len(valid_ids), N)):
                    qi = int(
                        np.argmin([cost[q, gi] if q not in used_q else 1e9 for q in range(N)]))
                    q_inds.append(qi)
                    g_inds.append(gi)
                    used_q.add(qi)

            for qi, gi in zip(q_inds, g_inds):
                iid = valid_ids[gi]
                cur_assignments[(b, iid)] = int(qi)
                alive_target[b, qi] = 1.0

                loss_box = loss_box + \
                    F.smooth_l1_loss(boxes_t[b, qi], gt_boxes[gi], beta=0.1)
                n_matched_total += 1

                if (b, iid) in prev_assignments:
                    prev_qi = prev_assignments[(b, iid)]
                    if prev_qi != int(qi) and prev_qi < N:
                        loss_consist = loss_consist + F.binary_cross_entropy_with_logits(
                            track_alive[b, t, prev_qi, 0:1], torch.ones(
                                1, device=device)
                        )

        loss_alive = loss_alive + \
            F.binary_cross_entropy_with_logits(alive_t, alive_target)
        prev_assignments = cur_assignments

    n_matched_total = max(n_matched_total, 1)
    loss_box = loss_box / n_matched_total
    loss_alive = loss_alive / T
    loss_consist = loss_consist / max(T * B, 1)

    return 1.5 * loss_box + 0.5 * loss_alive + 0.3 * loss_consist


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
            H_feat, W_feat = p_obj.shape[2], p_obj.shape[3]
            H, W = targets["seg_raw"].shape[1], targets["seg_raw"].shape[2]

            y_g, x_g = torch.meshgrid(torch.arange(H_feat, device=device), torch.arange(
                W_feat, device=device), indexing="ij")
            y_idx = torch.clamp(y_g * (H // H_feat) + (H // H_feat) //
                                2, 0, H - 1).unsqueeze(0).expand(B, -1, -1)
            x_idx = torch.clamp(x_g * (H // H_feat) + (H // H_feat) //
                                2, 0, W - 1).unsqueeze(0).expand(B, -1, -1)
            flat_idx = (y_idx * W + x_idx).reshape(B, H_feat * W_feat)

            inst_ids = torch.gather(targets["seg_raw"].reshape(
                B, H * W), 1, flat_idx).reshape(B, H_feat, W_feat).long()
            pred_logits = torch.einsum(
                "bchw,bcHW->bhwHW", preds["mask_coefficients"][i], preds["mask_prototypes"])
            gt_masks = (targets["seg_small"].unsqueeze(1).unsqueeze(
                2) == inst_ids.view(B, H_feat, W_feat, 1, 1)).float()

            if gt_masks.shape[-2:] != pred_logits.shape[-2:]:
                gt_masks = F.interpolate(gt_masks.flatten(0, 2).unsqueeze(
                    1), size=pred_logits.shape[-2:], mode="nearest").squeeze(1).view_as(pred_logits)

            intersection = (torch.sigmoid(pred_logits)
                            * gt_masks).sum(dim=(3, 4))
            union = torch.sigmoid(pred_logits).sum(
                dim=(3, 4)) + gt_masks.sum(dim=(3, 4))

            bce = F.binary_cross_entropy_with_logits(
                pred_logits, gt_masks, reduction="none")

            dice_loss = 1.0 - (2.0 * intersection + gt_masks.sum(dim=(3, 4)).clamp(
                min=1.0) * 0.01) / (union + gt_masks.sum(dim=(3, 4)).clamp(min=1.0) * 0.01)
            focal_bce = (0.25 * (1 - torch.exp(-bce))
                         ** 2 * bce).mean(dim=(3, 4))

            valid_mask_inst = (inst_ids > 0).float() * pos_mask.float()
            loss_mask += ((dice_loss * 2.0 + focal_bce) *
                          valid_mask_inst).sum() / valid_mask_inst.sum().clamp(min=1.0)

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


def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", step=0):
    device = preds["depth"].device
    H, W = preds["depth"].shape[-2:]
    w = get_loss_weights(step)

    loss_obj, loss_box, loss_mask, loss_cls = compute_instance_loss(
        preds, targets, step)
    loss_ego, loss_depth, loss_flow, loss_photo, loss_smooth = [
        torch.tensor(0.0, device=device) for _ in range(5)]
    loss_track = torch.tensor(0.0, device=device)

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

    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        if "has_next" in targets:
            has_n = targets["has_next"].view(-1, 1, 1, 1).float()
            l_flow_raw = F.smooth_l1_loss(
                preds["flow"], targets["flow_target"], reduction="none") * has_n
            loss_flow = l_flow_raw.sum() / (has_n.sum().clamp(min=1) *
                                            preds["flow"].shape[1] * H * W)
        else:
            loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"])

    if img_t is not None:
        loss_smooth = edge_aware_smoothness_loss(
            preds["depth"].unsqueeze(1), img_t)
        if img_next is not None:
            K, K_inv = generate_intrinsics(H, W, device)
            warped_img, v_w_mask = inverse_warp(
                img_next, preds["depth"].unsqueeze(1), preds["ego_pose"], K, K_inv)

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

    loss_components = {
        "Obj": loss_obj, "Box": loss_box, "Mask": loss_mask,
        "Depth": loss_depth, "Photo": loss_photo, "Ego": loss_ego,
        "Flow": loss_flow, "Anom": loss_anom, "Cls": loss_cls
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
    ret_dict["Tot"] = tot.detach()

    return tot, ret_dict, warped_img

# =====================================================================
