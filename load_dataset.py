import sys

import torch

try:
    import google.colab

    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import tensorflow as tf
    import tensorflow_datasets as tfds
import random
import threading
from collections import deque

import numpy as np
import torch.nn.functional as F


class AsyncDataBuffer:
    def __init__(
        self, split="train", max_buffer_size=64, batch_size=16, max_samples=None
    ):
        self.split = split
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock)

        print("\n" + "=" * 60)
        print(f"🚀 [异步管线] 正在启动后台独立 I/O 数据流缓冲池...")
        print(f"   >> 最大数据缓冲池: {max_buffer_size} 个序列 (滚动窗口)")
        print(f"   >> 动态批次抽样: 每次随机抽取 {batch_size} 条 (拒绝空转)")
        print("=" * 60 + "\n")

        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()

    def _fetch_loop(self):
        read_config = tfds.ReadConfig(
            interleave_cycle_length=16,
            num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE,
        )
        ds = tfds.load(
            "movi_e",
            data_dir="gs://kubric-public/tfds",
            split=self.split,
            read_config=read_config,
        )
        ds = ds.repeat()

        def process_video_frames(x):
            out = {
                "video": x["video"],
                "segmentations": x["segmentations"],
                "depth": x["depth"],
                "forward_flow": x["forward_flow"],
                "cam_pos": x["camera"]["positions"],
                "cam_quat": x["camera"]["quaternions"],
            }
            if "instances" in x and "is_dynamic" in x["instances"]:
                out["is_dynamic"] = x["instances"]["is_dynamic"]
            return out

        ds = ds.map(process_video_frames, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)

        for item in tfds.as_numpy(ds):
            pinned_item = {
                "video": torch.from_numpy(item["video"]).pin_memory(),
                "segmentation": torch.from_numpy(
                    item["segmentations"][..., 0]
                ).pin_memory(),
                "depth": torch.from_numpy(item["depth"][..., 0]).pin_memory(),
                "cam_pos": torch.from_numpy(item["cam_pos"]).pin_memory(),
                "cam_quat": torch.from_numpy(item["cam_quat"]).pin_memory(),
            }
            if "is_dynamic" in item:
                pinned_item["is_dynamic"] = torch.from_numpy(
                    item["is_dynamic"]
                ).pin_memory()

            # Decode forward_flow from uint16 (Fallback logic works perfectly for Kubric MOVi-E)
            flow_np = item["forward_flow"].astype(np.float32)
            if "metadata" in item and "forward_flow_range" in item["metadata"]:
                minv, maxv = item["metadata"]["forward_flow_range"]
                flow_np = flow_np / 65535.0 * (maxv - minv) + minv
            else:
                flow_np = (flow_np - 32768.0) / 64.0
            pinned_item["forward_flow"] = torch.from_numpy(flow_np).pin_memory()

            with self.lock:
                self.buffer.append(pinned_item)
                self.has_data.notify_all()

    def get_batch(self):
        with self.lock:
            while len(self.buffer) < self.batch_size:
                if not self.thread.is_alive():
                    raise RuntimeError(
                        "❌ 后台数据流线程异常崩溃，请检查网络或 TFDS 配置！"
                    )
                self.has_data.wait(timeout=5.0)
            batch_list = random.sample(self.buffer, self.batch_size)

        return {
            "video": [item["video"] for item in batch_list],
            "segmentation": [item["segmentation"] for item in batch_list],
            "depth": [item["depth"] for item in batch_list],
            "forward_flow": [item["forward_flow"] for item in batch_list],
            "cam_pos": [item["cam_pos"] for item in batch_list],
            "cam_quat": [item["cam_quat"] for item in batch_list],
            "is_dynamic": [item.get("is_dynamic") for item in batch_list],
        }


def process_batch_on_gpu(batch, device, target_size=256):
    video_raw = torch.stack([x.to(device, non_blocking=True) for x in batch["video"]])
    depth_raw_uint16 = torch.stack(
        [x.to(device, non_blocking=True) for x in batch["depth"]]
    ).float()
    seg_raw = torch.stack(
        [x.to(device, non_blocking=True) for x in batch["segmentation"]]
    )
    flow_raw = torch.stack(
        [x.to(device, non_blocking=True) for x in batch["forward_flow"]]
    ).float()
    cam_pos = torch.stack([x.to(device, non_blocking=True) for x in batch["cam_pos"]])
    cam_quat = torch.stack([x.to(device, non_blocking=True) for x in batch["cam_quat"]])

    B, T = video_raw.shape[:2]

    is_dyn_out = None
    if "is_dynamic" in batch and batch["is_dynamic"][0] is not None:
        dyn_list = [x.to(device, non_blocking=True) for x in batch["is_dynamic"]]
        max_len = max(len(x) for x in dyn_list)
        padded_dyn = [F.pad(x, (0, max_len - len(x))) for x in dyn_list]
        is_dyn_out = torch.stack(padded_dyn)

    depth_raw_m = depth_raw_uint16 / 1000.0
    sky_mask_raw = (depth_raw_uint16 == 0)
    depth_raw_m[sky_mask_raw] = 100.0
    depth_raw_m = torch.clamp(depth_raw_m, 0.01, 100.0)

    video = video_raw.permute(0, 1, 4, 2, 3).float() / 255.0

    if video.shape[-1] != target_size:
        video = F.interpolate(
            video.flatten(0, 1),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        ).view(B, T, 3, target_size, target_size)
        seg = F.interpolate(
            seg_raw.float().flatten(0, 1).unsqueeze(1),
            size=(target_size, target_size),
            mode="nearest",
        ).view(B, T, target_size, target_size)
        depth_m = (
            F.interpolate(
                depth_raw_m.float().flatten(0, 1).unsqueeze(1),
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
            .squeeze(1)
            .view(B, T, target_size, target_size)
        )
        sky_mask = F.interpolate(
            sky_mask_raw.float().flatten(0, 1).unsqueeze(1),
            size=(target_size, target_size),
            mode="nearest"
        ).squeeze(1).view(B, T, target_size, target_size).bool()
    else:
        seg = seg_raw.float()
        depth_m = depth_raw_m
        sky_mask = sky_mask_raw
    H, W = target_size, target_size
    seg_long = seg.long()

    depth_m_clamped = torch.clamp(depth_m, 0.01, 100.0)
    log_depth_target = torch.log(depth_m_clamped)

    flow_norm = torch.clamp(flow_raw * 2.0 / target_size, -1.5, 1.5)
    if flow_norm.shape[2] != target_size:
        flow_norm = F.interpolate(
            flow_norm.flatten(0, 1).permute(0, 3, 1, 2),
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )
        flow_norm = flow_norm.view(B, T, 2, target_size, target_size)
    else:
        flow_norm = flow_norm.permute(0, 1, 4, 2, 3)

    active_mask = seg_long > 0
    active_mask_float = active_mask.float()
    
    seg_small = (
        F.interpolate(
            seg.flatten(0, 1).unsqueeze(1), size=(H // 8, W // 8), mode="nearest"
        )
        .squeeze(1)
        .view(B, T, H // 8, W // 8)
    )

    # Build multi-scale targets for P3 (stride 8), P4 (stride 16), P5 (stride 32)
    bboxes_dense = []
    obj_dense = []
    cls_dense = []

    for stride in [8, 16, 32]:
        H_feat, W_feat = H // stride, W // stride
        
        b_d = torch.zeros(B, T, 4, H_feat, W_feat, device=device)
        o_d = torch.zeros(B, T, 1, H_feat, W_feat, device=device)
        c_d = torch.zeros(B, T, 1, H_feat, W_feat, device=device)
        
        y_grid = torch.arange(H, device=device, dtype=torch.int16).view(1, 1, 1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=torch.int16).view(1, 1, 1, 1, W)

        max_uid = int(seg_long.max().item())
        if max_uid > 0:
            uids = torch.arange(1, max_uid + 1, device=device, dtype=torch.int16).view(
                -1, 1, 1, 1, 1
            )
            masks = seg_long.to(torch.int16).unsqueeze(0) == uids
            valid_bt = masks.any(dim=-1).any(dim=-1)

            val_H = torch.tensor(H, dtype=torch.int16, device=device)
            val_W = torch.tensor(W, dtype=torch.int16, device=device)
            val_neg1 = torch.tensor(-1, dtype=torch.int16, device=device)

            ymin = torch.where(masks, y_grid, val_H).amin(dim=(3, 4))
            ymax = torch.where(masks, y_grid, val_neg1).amax(dim=(3, 4))

            xmin = torch.where(masks, x_grid, val_W).amin(dim=(3, 4))
            xmax = torch.where(masks, x_grid, val_neg1).amax(dim=(3, 4))

            true_area = masks.sum(dim=(3, 4), dtype=torch.int32)
            box_area = torch.clamp((xmax - xmin) * (ymax - ymin), min=1)

            # Valid area conditions based on FPN stride
            if stride == 8:
                stride_mask = box_area < (32 ** 2)
            elif stride == 16:
                stride_mask = (box_area >= (32 ** 2)) & (box_area < (96 ** 2))
            else:
                stride_mask = box_area >= (96 ** 2)

            valid_mask = (true_area >= 10) & (box_area <= 4 * true_area) & valid_bt & stride_mask

            n_idx, b_idx, t_idx = torch.where(valid_mask)

            if len(n_idx) > 0:
                areas = box_area[n_idx, b_idx, t_idx]
                sort_idx = torch.argsort(areas, descending=True)
                
                n_idx = n_idx[sort_idx]
                b_idx = b_idx[sort_idx]
                t_idx = t_idx[sort_idx]
                
                cy = torch.clamp(((ymin[n_idx, b_idx, t_idx] + ymax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, H_feat - 1)
                cx = torch.clamp(((xmin[n_idx, b_idx, t_idx] + xmax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, W_feat - 1)
                
                o_d[b_idx, t_idx, 0, cy, cx] = 1.0

                if is_dyn_out is not None:
                    is_dyn_batch = is_dyn_out[b_idx]
                    is_dyn_val = is_dyn_batch[torch.arange(len(n_idx), device=device), n_idx.long()]
                    c_d[b_idx, t_idx, 0, cy, cx] = is_dyn_val.float()
                else:
                    c_d[b_idx, t_idx, 0, cy, cx] = 1.0

                grid_x = cx.float() * stride + (stride / 2.0)
                grid_y = cy.float() * stride + (stride / 2.0)

                valid_boxes = torch.stack(
                    [
                        torch.clamp((grid_x - xmin[n_idx, b_idx, t_idx].float()) / float(stride), min=1e-4),
                        torch.clamp((grid_y - ymin[n_idx, b_idx, t_idx].float()) / float(stride), min=1e-4),
                        torch.clamp((xmax[n_idx, b_idx, t_idx].float() - grid_x) / float(stride), min=1e-4),
                        torch.clamp((ymax[n_idx, b_idx, t_idx].float() - grid_y) / float(stride), min=1e-4),
                    ],
                    dim=-1,
                )
                b_d[b_idx, t_idx, :, cy, cx] = valid_boxes
                
        bboxes_dense.append(b_d)
        obj_dense.append(o_d)
        cls_dense.append(c_d)

    h_val = torch.arange(H, device=device).view(1, 1, H, 1)
    w_val = torch.arange(W, device=device).view(1, 1, 1, W)
    ys = torch.where(active_mask, h_val, torch.full_like(h_val, H))
    xs = torch.where(active_mask, w_val, torch.full_like(w_val, W))
    ymin_g = torch.clamp(ys.amin(dim=(2, 3)).float(), 0.0, float(H))
    xmin_g = torch.clamp(xs.amin(dim=(2, 3)).float(), 0.0, float(W))
    ys_max = torch.where(active_mask, h_val, torch.full_like(h_val, -1))
    xs_max = torch.where(active_mask, w_val, torch.full_like(w_val, -1))
    ymax_g = torch.clamp(ys_max.amax(dim=(2, 3)).float(), 0.0, float(H))
    xmax_g = torch.clamp(xs_max.amax(dim=(2, 3)).float(), 0.0, float(W))

    bboxes_global = torch.stack(
        [xmin_g / W, ymin_g / H, xmax_g / W, ymax_g / H], dim=-1
    )
    empty = ~active_mask.view(B, T, -1).any(dim=-1)
    bboxes_global[empty] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)

    return {
        "video": video,
        "seg_raw": seg_long,
        "seg_small": seg_small,
        "depth": depth_m_clamped,
        "log_depth": log_depth_target,
        "flow": flow_norm,
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
        "bboxes_dense": bboxes_dense,
        "obj_dense": obj_dense,
        "cls_dense": cls_dense,
        "bboxes_global": bboxes_global,
        "is_dynamic": is_dyn_out,
        "sky_mask": sky_mask,
    }


import queue


class CUDAPrefetcher:
    """Overlaps GPU data processing with training to maximize GPU utilization."""

    def __init__(self, buffer, device, target_size=256, max_prefetch=4):
        self.buffer = buffer
        self.device = device
        self.target_size = target_size
        self.queue = queue.Queue(maxsize=max_prefetch)
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        while True:
            try:
                batch = self.buffer.get_batch()
                if self.stream is not None:
                    with torch.cuda.stream(self.stream):
                        batch_gpu = process_batch_on_gpu(
                            batch, self.device, self.target_size
                        )
                else:
                    batch_gpu = process_batch_on_gpu(
                        batch, self.device, self.target_size
                    )
                self.queue.put(batch_gpu)
            except Exception as e:
                print(f"Prefetcher worker error: {e}")
                import time

                time.sleep(1)

    def next(self):
        batch_gpu = self.queue.get()
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            for k, v in batch_gpu.items():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
        return batch_gpu
