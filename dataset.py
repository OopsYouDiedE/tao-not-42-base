import time
import queue
import random
import threading
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

try:
    import tensorflow as tf
    import tensorflow_datasets as tfds
except ImportError:
    tf = None
    tfds = None


def decode_uint16_range(encoded, value_range):
    encoded = encoded.astype(np.float32)
    minv, maxv = np.asarray(value_range, dtype=np.float32)
    return encoded / 65535.0 * (maxv - minv) + minv

# =====================================================================

# 5. 数据流加载 (Data Loader & Pipeline)
# =====================================================================


class AsyncDataBuffer:
    def __init__(self, split="train", max_buffer_size=64, batch_size=16):
        self.split = split
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock)
        threading.Thread(target=self._fetch_loop, daemon=True).start()

    def _fetch_loop(self):
        if tfds is None or tf is None:
            while True:
                item = {
                    "video": torch.randint(0, 256, (12, 256, 256, 3), dtype=torch.uint8),
                    "segmentation": torch.randint(0, 3, (12, 256, 256), dtype=torch.int32),
                    "depth": torch.rand(12, 256, 256, dtype=torch.float32) * 15.0 + 3.0,
                    "forward_flow": torch.zeros(12, 256, 256, 2, dtype=torch.float32),
                    "cam_pos": torch.zeros(12, 3, dtype=torch.float32),
                    "cam_quat": torch.tensor([1., 0., 0., 0.], dtype=torch.float32).expand(12, 4).clone(),
                    "is_dynamic": torch.zeros(5, dtype=torch.bool)
                }
                with self.lock:
                    self.buffer.append(item)
                    self.has_data.notify_all()
                time.sleep(0.5)
            return

        ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", split=self.split,
                       read_config=tfds.ReadConfig(interleave_cycle_length=16)).repeat()

        def map_fn(x):
            return {
                "video": x["video"], "segmentations": x["segmentations"], "depth": x["depth"],
                "forward_flow": x["forward_flow"], "cam_pos": x["camera"]["positions"],
                "cam_quat": x["camera"]["quaternions"],
                "depth_range": x["metadata"]["depth_range"],
                "forward_flow_range": x["metadata"]["forward_flow_range"],
                "camera_focal_length": x["camera"]["focal_length"],
                "camera_sensor_width": x["camera"]["sensor_width"],
                **({"is_dynamic": x["instances"]["is_dynamic"]} if "instances" in x and "is_dynamic" in x["instances"] else {})
            }

        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)

        for item in tfds.as_numpy(ds):
            p_item = {k: torch.from_numpy(item[k_i]).pin_memory() for k, k_i in [
                ("video", "video"), ("cam_pos", "cam_pos"), ("cam_quat", "cam_quat")]}
            p_item["segmentation"] = torch.from_numpy(
                item["segmentations"][..., 0]).pin_memory()
            p_item["depth"] = torch.from_numpy(decode_uint16_range(
                item["depth"][..., 0], item["depth_range"])).pin_memory()
            p_item["depth_range"] = torch.from_numpy(
                item["depth_range"]).pin_memory()
            p_item["forward_flow_range"] = torch.from_numpy(
                item["forward_flow_range"]).pin_memory()
            p_item["camera_focal_length"] = torch.as_tensor(
                item["camera_focal_length"]).pin_memory()
            p_item["camera_sensor_width"] = torch.as_tensor(
                item["camera_sensor_width"]).pin_memory()

            if "is_dynamic" in item:
                p_item["is_dynamic"] = torch.from_numpy(
                    item["is_dynamic"]).pin_memory()

            f_np = decode_uint16_range(
                item["forward_flow"], item["forward_flow_range"])
            p_item["forward_flow"] = torch.from_numpy(f_np).pin_memory()

            with self.lock:
                self.buffer.append(p_item)
                self.has_data.notify_all()

    def get_batch(self):
        with self.lock:
            while len(self.buffer) < self.batch_size:
                self.has_data.wait(timeout=5.0)
                if len(self.buffer) == 0 and not IN_COLAB:
                    return None
            batch = random.sample(self.buffer, self.batch_size)

        keys = [
            "video", "segmentation", "depth", "forward_flow", "cam_pos",
            "cam_quat", "is_dynamic", "depth_range", "forward_flow_range",
            "camera_focal_length", "camera_sensor_width",
        ]
        return {k: [i.get(k) for i in batch] for k in keys}


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
    depth_raw = to_gpu("depth", torch.float32)
    seg_raw = to_gpu("segmentation")
    flow_raw = to_gpu("forward_flow", torch.float32)
    cam_pos = to_gpu("cam_pos")
    cam_quat = to_gpu("cam_quat")
    B, T = video.shape[:2]

    is_dyn_out = None
    if batch.get("is_dynamic") and batch["is_dynamic"][0] is not None:
        max_dyn_len = max(
            [len(d) for d in batch["is_dynamic"] if d is not None], default=0)
        is_dyn_out = torch.stack(
            [F.pad(x, (0, max_dyn_len - len(x))) for x in batch["is_dynamic"]]
        ).to(device, non_blocking=True)

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

    flow_norm = torch.clamp(
        flow_raw * 2.0 / target_size, -1.5, 1.5).permute(0, 1, 4, 2, 3)
    if flow_norm.shape[-1] != target_size:
        flow_norm = F.interpolate(flow_norm.flatten(0, 1), size=(
            target_size, target_size), mode="bilinear", align_corners=False).view(B, T, 2, target_size, target_size)

    bboxes_dense, obj_dense, cls_dense = [], [], []
    MAX_INSTANCES = 24
    uids = torch.arange(1, MAX_INSTANCES + 1, device=device,
                        dtype=torch.int16).view(-1, 1, 1, 1, 1)
    masks = (seg.to(torch.int16).unsqueeze(0) == uids)
    valid_bt = masks.any(dim=-1).any(dim=-1)

    y_grid = torch.arange(target_size, device=device,
                          dtype=torch.int16).view(1, 1, 1, target_size, 1)
    x_grid = torch.arange(target_size, device=device,
                          dtype=torch.int16).view(1, 1, 1, 1, target_size)

    ymin = torch.where(masks, y_grid, torch.tensor(
        target_size, dtype=torch.int16, device=device)).amin(dim=(3, 4))
    ymax = torch.where(masks, y_grid, torch.tensor(-1,
                       dtype=torch.int16, device=device)).amax(dim=(3, 4))
    xmin = torch.where(masks, x_grid, torch.tensor(
        target_size, dtype=torch.int16, device=device)).amin(dim=(3, 4))
    xmax = torch.where(masks, x_grid, torch.tensor(-1,
                       dtype=torch.int16, device=device)).amax(dim=(3, 4))

    true_area = masks.sum(dim=(3, 4), dtype=torch.int32)
    box_area = torch.clamp((xmax - xmin) * (ymax - ymin), min=1)

    for stride in [8, 16, 32]:
        H_f, W_f = target_size // stride, target_size // stride
        b_d = torch.zeros(B, T, 4, H_f, W_f, device=device)
        o_d = torch.zeros(B, T, 1, H_f, W_f, device=device)
        c_d = torch.zeros(B, T, 1, H_f, W_f, device=device)

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
            if is_dyn_out is not None:
                c_d[b_idx, t_idx, 0, cy, cx] = is_dyn_out[b_idx, n_idx.long()
                                                          ].float()
            else:
                c_d[b_idx, t_idx, 0, cy, cx] = 1.0

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

    # shape: [MAX_INSTANCES, B, T, 4] -> permute to [B, T, MAX_INSTANCES, 4]
    track_gt_boxes = torch.stack([cx, cy, bw, bh], dim=-1).permute(1, 2, 0, 3)
    # shape: [MAX_INSTANCES, B, T] -> permute to [B, T, MAX_INSTANCES]
    track_gt_valid = (true_area > 0).permute(1, 2, 0)

    return {
        "video": video_p, "seg_raw": seg, "depth": depth_m, "log_depth": torch.log(depth_m),
        "flow": flow_norm, "cam_pos": cam_pos, "cam_quat": cam_quat, "is_dynamic": is_dyn_out, "sky_mask": sky_mask,
        "seg_small": seg_small, "bboxes_dense": bboxes_dense, "obj_dense": obj_dense, "cls_dense": cls_dense,
        "track_gt_boxes": track_gt_boxes, "track_gt_valid": track_gt_valid,
    }

class CUDAPrefetcher:
    def __init__(self, buffer, device, target_size=256):
        self.buffer = buffer
        self.device = device
        self.target_size = target_size
        self.queue = queue.Queue(maxsize=4)
        self.stream = torch.cuda.Stream(
            device=device) if device.type == "cuda" else None
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            batch = self.buffer.get_batch()
            if batch is None:
                time.sleep(1)
                continue
            try:
                if self.stream:
                    with torch.cuda.stream(self.stream):
                        batch_gpu = process_batch_on_gpu(
                            batch, self.device, self.target_size)
                else:
                    batch_gpu = process_batch_on_gpu(
                        batch, self.device, self.target_size)
                self.queue.put(batch_gpu)
            except Exception as e:
                print(f"Prefetcher err: {e}")
                time.sleep(1)

    def next(self):
        batch = self.queue.get()
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            for v in batch.values():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
        return batch

# =====================================================================
