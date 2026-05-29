import numpy as np
import torch
import torch.nn.functional as F

def process_batch_on_gpu(batch, device, target_size=256):
    def to_gpu(k, dtype=None):
        val = batch[k]
        if isinstance(val, (list, tuple)):
            stacked_cpu = torch.stack(val)
        else:
            stacked_cpu = val
        stacked_gpu = stacked_cpu.to(device, non_blocking=True)
        return stacked_gpu.to(dtype) if dtype else stacked_gpu

    video = to_gpu("video")
    depth_raw = to_gpu("depth")
    seg_raw = to_gpu("segmentation")
    flow_raw = to_gpu("forward_flow")
    cam_pos = to_gpu("cam_pos")
    cam_quat = to_gpu("cam_quat")
    B, T = video.shape[:2]

    if depth_raw.dtype == torch.uint16 or depth_raw.dtype == torch.int16:
        depth_range = to_gpu("depth_range")
        min_v = depth_range[:, 0].view(B, 1, 1, 1)
        max_v = depth_range[:, 1].view(B, 1, 1, 1)
        depth_raw = depth_raw.float() / 65535.0 * (max_v - min_v) + min_v
    else:
        depth_raw = depth_raw.float()

    if flow_raw.dtype == torch.uint16 or flow_raw.dtype == torch.int16:
        flow_range = to_gpu("forward_flow_range")
        min_f = flow_range[:, 0].view(B, 1, 1, 1, 1)
        max_f = flow_range[:, 1].view(B, 1, 1, 1, 1)
        flow_raw = flow_raw.float() / 65535.0 * (max_f - min_f) + min_f
    else:
        flow_raw = flow_raw.float()

    def pad_instances(key):
        """填充每个实例的字段，同时保留批次维度。

        TFDS 有时会忽略单个样本上的可选实例元数据。
        旧的实现跳过了 ``None`` 条目，这可能会在无意中缩小 B 并随后使实例 ID 与视频帧错位。
        """
        values = batch.get(key)
        if isinstance(values, torch.Tensor):
            return values.to(device, non_blocking=True)
        if not values:
            return None

        present = [x for x in values if x is not None]
        if not present:
            return None

        max_len = max(len(x) for x in present)
        exemplar = present[0]
        padded = []
        for x in values:
            if x is None:
                fill_shape = (max_len, *exemplar.shape[1:])
                padded.append(torch.zeros(fill_shape, dtype=exemplar.dtype))
                continue

            pad_dims = []
            for _ in range(x.dim() - 1):
                pad_dims.extend([0, 0])
            pad_dims.extend([0, max_len - len(x)])
            padded.append(F.pad(x, tuple(pad_dims)))

        return torch.stack(padded).to(device, non_blocking=True)

    def instance_presence(key):
        values = batch.get(key)
        if isinstance(values, torch.Tensor):
            return torch.ones(values.shape[0], device=device, dtype=torch.bool)
        if not values:
            return None
        return torch.tensor([x is not None for x in values], device=device, dtype=torch.bool)

    is_dyn_out = pad_instances("is_dynamic")
    is_dyn_present = instance_presence("is_dynamic")
    velocities_out = pad_instances("velocities")
    velocities_present = instance_presence("velocities")
    angular_velocities_out = pad_instances("angular_velocities")
    angular_velocities_present = instance_presence("angular_velocities")
    visibility_out = pad_instances("visibility")
    visibility_present = instance_presence("visibility")

    depth_m = torch.clamp(depth_raw, 0.01, 100.0)
    depth_m[depth_raw == 0] = 100.0
    video_p = video.permute(0, 1, 4, 2, 3).float() / 255.0

    if video_p.shape[-1] != target_size:
        video_p = F.interpolate(video_p.flatten(0, 1), size=(target_size, target_size),
                                mode="bilinear", align_corners=False).view(B, T, 3, target_size, target_size)
        seg = F.interpolate(seg_raw.float().flatten(0, 1).unsqueeze(1), size=(
            target_size, target_size), mode="nearest").view(B, T, target_size, target_size).long()
        depth_m = F.interpolate(depth_m.flatten(0, 1).unsqueeze(1), size=(
            target_size, target_size), mode="bilinear", align_corners=False).squeeze(1).view(B, T, target_size, target_size)
        sky_mask = F.interpolate((depth_raw == 0).float().flatten(0, 1).unsqueeze(1), size=(
            target_size, target_size), mode="nearest").squeeze(1).view(B, T, target_size, target_size).bool()
    else:
        seg = seg_raw.long()
        sky_mask = (depth_raw == 0)

    # MOVi forward_flow[..., 0] = delta_row = dy
    # MOVi forward_flow[..., 1] = delta_column = dx
    # 模型内部统一使用 flow_xy[..., 0] = dx, flow_xy[..., 1] = dy
    src_h, src_w = flow_raw.shape[2], flow_raw.shape[3]

    flow_xy = flow_raw[..., [1, 0]].contiguous()

    # 如果以后训练分辨率不是原始分辨率，光流像素位移也要同步缩放
    if src_w != target_size:
        flow_xy[..., 0] = flow_xy[..., 0] * (float(target_size) / float(src_w))
    if src_h != target_size:
        flow_xy[..., 1] = flow_xy[..., 1] * (float(target_size) / float(src_h))

    flow_norm = torch.clamp(
        flow_xy * 2.0 / float(target_size), -1.5, 1.5
    ).permute(0, 1, 4, 2, 3)

    if flow_norm.shape[-1] != target_size:
        flow_norm = F.interpolate(
            flow_norm.flatten(0, 1),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, 2, target_size, target_size)

    bboxes_dense, obj_dense, cls_dense = [], [], []
    initial_dynamic_dense = []
    initial_dynamic_valid_dense = []
    current_moving_dense = []
    current_moving_valid_dense = []
    MAX_INSTANCES = 32

    # 1. 准备展平的一维网格坐标
    y_coords = torch.arange(target_size, dtype=torch.float32, device=device).view(target_size, 1).expand(target_size, target_size).flatten().view(1, 1, -1).expand(B, T, -1)
    x_coords = torch.arange(target_size, dtype=torch.float32, device=device).view(1, target_size).expand(target_size, target_size).flatten().view(1, 1, -1).expand(B, T, -1)
    flat_seg = seg.view(B, T, -1).long()
    valid_seg_for_scatter = (flat_seg >= 0) & (flat_seg <= MAX_INSTANCES)
    scatter_seg = torch.where(valid_seg_for_scatter, flat_seg, torch.zeros_like(flat_seg))

    # 2. 利用 scatter_reduce_ 直接求出边界坐标
    ymin_target = torch.full((B, T, MAX_INSTANCES + 1), float(target_size), dtype=torch.float32, device=device)
    ymin_target.scatter_reduce_(dim=2, index=scatter_seg, src=y_coords, reduce="amin", include_self=False)
    ymin = ymin_target[:, :, 1:].permute(2, 0, 1)

    ymax_target = torch.full((B, T, MAX_INSTANCES + 1), -1.0, dtype=torch.float32, device=device)
    ymax_target.scatter_reduce_(dim=2, index=scatter_seg, src=y_coords, reduce="amax", include_self=False)
    ymax = ymax_target[:, :, 1:].permute(2, 0, 1)

    xmin_target = torch.full((B, T, MAX_INSTANCES + 1), float(target_size), dtype=torch.float32, device=device)
    xmin_target.scatter_reduce_(dim=2, index=scatter_seg, src=x_coords, reduce="amin", include_self=False)
    xmin = xmin_target[:, :, 1:].permute(2, 0, 1)

    xmax_target = torch.full((B, T, MAX_INSTANCES + 1), -1.0, dtype=torch.float32, device=device)
    xmax_target.scatter_reduce_(dim=2, index=scatter_seg, src=x_coords, reduce="amax", include_self=False)
    xmax = xmax_target[:, :, 1:].permute(2, 0, 1)

    # 3. 利用 scatter_add_ 直接统计实例真实面积
    true_area_target = torch.zeros((B, T, MAX_INSTANCES + 1), dtype=torch.int32, device=device)
    ones = torch.where(valid_seg_for_scatter, torch.ones_like(flat_seg, dtype=torch.int32), torch.zeros_like(flat_seg, dtype=torch.int32))
    true_area_target.scatter_add_(dim=2, index=scatter_seg, src=ones)
    true_area = true_area_target[:, :, 1:].permute(2, 0, 1)

    valid_bt = (true_area > 0)
    box_area = torch.clamp((xmax - xmin) * (ymax - ymin), min=1)

    for stride in [8, 16, 32]:
        H_f, W_f = target_size // stride, target_size // stride
        b_d = torch.zeros(B, T, 4, H_f, W_f, device=device)
        o_d = torch.zeros(B, T, 1, H_f, W_f, device=device)
        c_d = torch.full((B, T, 1, H_f, W_f), fill_value=-100, dtype=torch.long, device=device)
        dyn_d = torch.zeros(B, T, 1, H_f, W_f, device=device)
        dyn_valid_d = torch.zeros(B, T, 1, H_f, W_f, device=device)
        cur_mov_d = torch.zeros(B, T, 1, H_f, W_f, device=device)
        cur_mov_valid_d = torch.zeros(B, T, 1, H_f, W_f, device=device)

        if stride == 8:
            s_mask = (box_area < 32**2)
        elif stride == 16:
            s_mask = (box_area >= 32**2) & (box_area < 96**2)
        else:
            s_mask = (box_area >= 96**2)

        n_idx, b_idx, t_idx = torch.where((true_area >= 10) & (
            box_area <= 4 * true_area) & valid_bt & s_mask)

        if len(n_idx) > 0:
            areas = box_area[n_idx, b_idx, t_idx]
            sort_idx = torch.argsort(areas, descending=True)
            n_idx, b_idx, t_idx = n_idx[sort_idx], b_idx[sort_idx], t_idx[sort_idx]

            cy = torch.clamp(
                ((ymin[n_idx, b_idx, t_idx] + ymax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, H_f - 1)
            cx = torch.clamp(
                ((xmin[n_idx, b_idx, t_idx] + xmax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, W_f - 1)

            o_d[b_idx, t_idx, 0, cy, cx] = 1.0

            dyn_val = torch.zeros_like(n_idx, dtype=torch.float32, device=device)
            if is_dyn_out is not None and is_dyn_out.numel() > 0:
                valid_dyn = n_idx.long() < is_dyn_out.shape[1]
                if is_dyn_present is not None:
                    valid_dyn = valid_dyn & is_dyn_present[b_idx]
                if valid_dyn.any():
                    dyn_val[valid_dyn] = is_dyn_out[b_idx[valid_dyn], n_idx[valid_dyn].long()].float()
                dyn_d[b_idx, t_idx, 0, cy, cx] = dyn_val
                dyn_valid_d[b_idx[valid_dyn], t_idx[valid_dyn], 0, cy[valid_dyn], cx[valid_dyn]] = 1.0

            cur_mov_val = torch.zeros_like(n_idx, dtype=torch.float32, device=device)
            cur_mov_valid_val = torch.zeros_like(n_idx, dtype=torch.float32, device=device)
            moving_flag = torch.zeros_like(n_idx, dtype=torch.bool, device=device)
            motion_defined = torch.zeros_like(n_idx, dtype=torch.bool, device=device)

            if velocities_out is not None and velocities_out.numel() > 0 and velocities_out.dim() >= 4:
                valid_vel = (n_idx.long() < velocities_out.shape[1]) & (t_idx.long() < velocities_out.shape[2])
                if velocities_present is not None:
                    valid_vel = valid_vel & velocities_present[b_idx]
                if valid_vel.any():
                    v = velocities_out[b_idx[valid_vel], n_idx[valid_vel].long(), t_idx[valid_vel]]
                    moving_flag[valid_vel] |= torch.linalg.vector_norm(v, dim=-1) > 0.03
                    motion_defined[valid_vel] = True

            if angular_velocities_out is not None and angular_velocities_out.numel() > 0 and angular_velocities_out.dim() >= 4:
                valid_ang = (n_idx.long() < angular_velocities_out.shape[1]) & (t_idx.long() < angular_velocities_out.shape[2])
                if angular_velocities_present is not None:
                    valid_ang = valid_ang & angular_velocities_present[b_idx]
                if valid_ang.any():
                    av = angular_velocities_out[b_idx[valid_ang], n_idx[valid_ang].long(), t_idx[valid_ang]]
                    moving_flag[valid_ang] |= torch.linalg.vector_norm(av, dim=-1) > 0.03
                    motion_defined[valid_ang] = True

            visible_flag = torch.ones_like(n_idx, dtype=torch.bool, device=device)
            if visibility_out is not None and visibility_out.numel() > 0 and visibility_out.dim() >= 3:
                valid_vis = (n_idx.long() < visibility_out.shape[1]) & (t_idx.long() < visibility_out.shape[2])
                if visibility_present is not None:
                    valid_vis = valid_vis & visibility_present[b_idx]
                if valid_vis.any():
                    vis = visibility_out[b_idx[valid_vis], n_idx[valid_vis].long(), t_idx[valid_vis]].float()
                    # MOVi 可见性通常是可见像素计数。
                    # 低阈值可避免对几乎隐藏的对象监督运动状态，
                    # 而分割正例仍提供边界框/掩码。
                    visible_flag[valid_vis] = vis > 10.0

            if motion_defined.any():
                cur_mov_val[motion_defined] = (moving_flag[motion_defined] & visible_flag[motion_defined]).float()
                cur_mov_valid_val[motion_defined] = 1.0
                cur_mov_d[b_idx, t_idx, 0, cy, cx] = cur_mov_val
                cur_mov_valid_d[b_idx, t_idx, 0, cy, cx] = cur_mov_valid_val

            gx, gy = cx.float() * stride + stride / 2.0, cy.float() * stride + stride / 2.0

            x_min_f = xmin[n_idx, b_idx, t_idx].float()
            y_min_f = ymin[n_idx, b_idx, t_idx].float()
            x_max_f = xmax[n_idx, b_idx, t_idx].float()
            y_max_f = ymax[n_idx, b_idx, t_idx].float()

            b_d[b_idx, t_idx, :, cy, cx] = torch.stack([
                torch.clamp((gx - x_min_f) / stride, min=1e-4),
                torch.clamp((gy - y_min_f) / stride, min=1e-4),
                torch.clamp((x_max_f - gx) / stride, min=1e-4),
                torch.clamp((y_max_f - gy) / stride, min=1e-4),
            ], dim=-1)

        bboxes_dense.append(b_d)
        obj_dense.append(o_d)
        cls_dense.append(c_d)
        initial_dynamic_dense.append(dyn_d)
        initial_dynamic_valid_dense.append(dyn_valid_d)
        current_moving_dense.append(cur_mov_d)
        current_moving_valid_dense.append(cur_mov_valid_d)

    seg_small = F.interpolate(seg.float().flatten(0, 1).unsqueeze(1), size=(
        target_size // 8, target_size // 8), mode="nearest").squeeze(1).view(B, T, target_size // 8, target_size // 8)

    # 预计算 Ground Truth Tracking 边界框以避免 CPU-GPU 同步瓶颈
    ymin_f = ymin.float() / target_size
    ymax_f = ymax.float() / target_size
    xmin_f = xmin.float() / target_size
    xmax_f = xmax.float() / target_size

    cx = (xmin_f + xmax_f) * 0.5
    cy = (ymin_f + ymax_f) * 0.5
    bw = (xmax_f - xmin_f).clamp(min=1.0 / target_size)
    bh = (ymax_f - ymin_f).clamp(min=1.0 / target_size)

    # 形状: [MAX_INSTANCES, B, T, 4] -> 置换为 [B, T, MAX_INSTANCES, 4]
    track_gt_boxes = torch.stack([cx, cy, bw, bh], dim=-1).permute(1, 2, 0, 3)
    # 形状: [MAX_INSTANCES, B, T] -> 置换为 [B, T, MAX_INSTANCES]
    track_gt_valid = (true_area > 0).permute(1, 2, 0)

    if "camera_focal_length" in batch and batch["camera_focal_length"] is not None and len(batch["camera_focal_length"]) > 0:
        try:
            camera_focal_length = to_gpu("camera_focal_length", torch.float32)
        except Exception:
            camera_focal_length = torch.tensor([35.0] * B, device=device, dtype=torch.float32)
    else:
        camera_focal_length = torch.tensor([35.0] * B, device=device, dtype=torch.float32)

    if "camera_sensor_width" in batch and batch["camera_sensor_width"] is not None and len(batch["camera_sensor_width"]) > 0:
        try:
            camera_sensor_width = to_gpu("camera_sensor_width", torch.float32)
        except Exception:
            camera_sensor_width = torch.tensor([32.0] * B, device=device, dtype=torch.float32)
    else:
        camera_sensor_width = torch.tensor([32.0] * B, device=device, dtype=torch.float32)

    return {
        "video": video_p, "seg_raw": seg, "depth": depth_m, "log_depth": torch.log(depth_m),
        "flow": flow_norm, "cam_pos": cam_pos, "cam_quat": cam_quat, "is_dynamic": is_dyn_out, "sky_mask": sky_mask,
        "seg_small": seg_small, "bboxes_dense": bboxes_dense, "obj_dense": obj_dense, "cls_dense": cls_dense,
        "initial_dynamic_dense": initial_dynamic_dense,
        "initial_dynamic_valid_dense": initial_dynamic_valid_dense,
        "current_moving_dense": current_moving_dense,
        "current_moving_valid_dense": current_moving_valid_dense,
        "track_gt_boxes": track_gt_boxes, "track_gt_valid": track_gt_valid,
        "camera_focal_length": camera_focal_length,
        "camera_sensor_width": camera_sensor_width,
    }
