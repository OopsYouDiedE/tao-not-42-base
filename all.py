import os
import time
import queue
import random
import argparse
import threading
import contextlib
import urllib.request
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from mamba_ssm import Mamba
    import tensorflow as tf
    import tensorflow_datasets as tfds
else:
    Mamba = None
    tf = None
    tfds = None

try:
    import wandb
except ImportError:
    wandb = None

# =====================================================================
# 1. 基础工具与可视化函数
# =====================================================================


def flow_to_color(flow_np):
    flow_np = flow_np.astype(np.float32) - np.median(flow_np, axis=(0, 1))
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv = np.zeros((*flow_np.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = ang * 90 / np.pi
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag / (mag.max() + 1e-5) * 255, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def save_visualization(video_t, target_t, pred_t, step, warped_img=None, output_dir="vis_outputs"):
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
                
    if "track_boxes" in pred_t and "track_alive" in pred_t:
        t_boxes = pred_t["track_boxes"].view(-1, 16, 4)[0] if pred_t["track_boxes"].dim() >= 3 else None
        t_alive = pred_t["track_alive"].view(-1, 16, 1)[0].sigmoid() if pred_t["track_alive"].dim() >= 3 else None
        
        if t_boxes is not None and t_alive is not None:
            for i in range(16):
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
    if target_t.get("seg_raw") is not None and target_t.get("is_dynamic") is not None:
        seg = target_t["seg_raw"][0].cpu().numpy()
        is_dyn = target_t["is_dynamic"][0].cpu().numpy()

        for uid in range(1, int(np.max(seg)) + 1):
            m = seg == uid
            if np.any(m):
                is_dynamic_obj = (uid - 1 < len(is_dyn)) and is_dyn[uid - 1]
                color = (0, 0, 255) if is_dynamic_obj else (255, 0, 0)
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
            cv2.rectangle(gt_canvas, (int(
                gx - b[0]), int(gy - b[1])), (int(gx + b[2]), int(gy + b[3])), (0, 255, 0), 2)

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


def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2,
    ], dim=-1).view(*q.shape[:-1], 3, 3)


def matrix_to_6d(matrix):
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def six_d_to_matrix(d6):
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     device=device, dtype=torch.float32)
    return K, torch.inverse(K)


def depth_to_color(depth_map, d_min=None, d_max=None):
    d_min = d_min if d_min is not None else depth_map.min()
    d_max = d_max if d_max is not None else depth_map.max()
    d_norm = (depth_map - d_min) / \
        (d_max - d_min) if d_max > d_min else np.zeros_like(depth_map)
    return cv2.applyColorMap((np.clip(d_norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)


def decode_dfl_boxes(pred_dist, reg_max=16):
    if isinstance(pred_dist, list):
        return [decode_dfl_boxes(x, reg_max) for x in pred_dist]
    B, C, H, W = pred_dist.shape
    if C == 4:
        return pred_dist
    prob = F.softmax(pred_dist.view(B, 4, reg_max, H, W), dim=2)
    weights = torch.arange(reg_max, dtype=torch.float32,
                           device=pred_dist.device)
    return (prob * weights.view(1, 1, reg_max, 1, 1)).sum(dim=2)

# =====================================================================
# 2. 模型核心组件 (Blocks - 整合与去重)
# =====================================================================


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k,
                                          int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(
            k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else (
            act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(
            self.c, self.c, shortcut, g, k=(3, 3), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(1, 3), e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 1, shortcut, g) if c3k else Bottleneck(
                self.c, self.c, shortcut, g)
            for _ in range(n)
        )


class C3k2Attention(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList([
            nn.Sequential(
                Bottleneck(self.c, self.c, shortcut, g=1, e=1.0),
                PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
            )
        ])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        feat = y[-1]
        for layer in self.m[0]:
            feat = layer(feat)
        y.append(feat)
        return self.cv2(torch.cat(y, 1))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5

        self.qkv = Conv(dim, dim + self.key_dim * num_heads * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x).view(B, self.num_heads,
                               self.key_dim * 2 + self.head_dim, H * W)
        q, k, v = qkv.split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k * self.scale).softmax(dim=-1)
        out = (v @ attn.transpose(-2, -1)).view(B, C, H, W)
        return self.proj(out + self.pe(v.reshape(B, C, H, W)))


class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, num_heads=num_heads, attn_ratio=attn_ratio)
        self.ffn = nn.Sequential(
            Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        y = [self.cv1(x)]
        for _ in range(3):
            y.append(self.m(y[-1]))
        return self.cv2(torch.cat(y, 1))

# =====================================================================
# 3. 物理与时间模块 (Time & Physics Modules)
# =====================================================================


class SpatioTemporalMambaBlock(nn.Module):
    def __init__(self, channels, num_frequencies=16):
        super().__init__()
        self.channels = channels
        self.conv3d = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(channels)
        self.act = nn.SiLU(inplace=True)
        self.mamba = Mamba(d_model=channels, d_state=16,
                           d_conv=4, expand=2) if Mamba else None
        self.norm = nn.LayerNorm(channels)

        self.register_buffer("frequencies", torch.exp(
            torch.linspace(-5, 3, num_frequencies)))
        self.time_mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, 64), nn.SiLU(), nn.Linear(64, channels))
        self.gamma = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x, t):
        B, T, C, H, W = x.shape
        x3d = x.permute(0, 2, 1, 3, 4)
        x3d = self.act(self.bn3d(self.conv3d(x3d)))
        x3d = x3d.permute(0, 2, 1, 3, 4).contiguous()

        scaled_time = t.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        fourier_feats = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        time_embed = self.time_mlp(fourier_feats)

        x3d = x3d + time_embed.view(B, T, C, 1, 1)
        x_flat = x3d.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        mamba_out = self.mamba(x_flat) if self.mamba else x_flat
        x_flat = self.norm(x_flat + mamba_out)

        out = x_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return x + self.gamma * out


class UnifiedGeometryDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48, pose_dim=9):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_p3, ch_f2, 3))
        self.conv1 = Conv(ch_f2 * 2, ch_f2, 3)
        self.up2 = nn.Sequential(nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_f2, ch_f1, 3))
        self.conv2 = Conv(ch_f1 * 2, ch_f1, 3)

        self.depth_branch = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            Conv(ch_f1, ch_f1, 3),
            Conv(ch_f1, ch_f1 // 2, 3),
            nn.Conv2d(ch_f1 // 2, 1, 3, padding=1)
        )

        self.flow_up = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False)
        self.flow_conv = nn.Sequential(
            Conv(ch_f1 + pose_dim, ch_f1, 3),
            Conv(ch_f1, ch_f1 // 2, 3),
            nn.Conv2d(ch_f1 // 2, 2, 3, padding=1)
        )

    def forward(self, f1, f2, p3, ego_pose_feat=None, need_flow=True):
        x1 = self.conv1(torch.cat([self.up1(p3), f2], dim=1))
        x2 = self.conv2(torch.cat([self.up2(x1), f1], dim=1))

        depth_out = self.depth_branch(x2)

        flow_out = None
        if need_flow:
            flow_feat = self.flow_up(x2)
            if ego_pose_feat is not None:
                B, C, H, W = flow_feat.shape
                pose_map = ego_pose_feat.view(B, -1, 1, 1).expand(-1, -1, H, W)
                flow_feat = torch.cat([flow_feat, pose_map], dim=1)
            flow_out = self.flow_conv(flow_feat)

        return depth_out, flow_out


class EgoPoseHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64), nn.SiLU(), nn.Linear(64, 9))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        pose = self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
        rot_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=pose.device,
                              dtype=pose.dtype) + torch.tanh(pose[:, 3:]) * 0.5
        return torch.cat([torch.tanh(pose[:, :3]) * 5.0, rot_6d], dim=1)


class FeaturePredictorHead(nn.Module):
    def __init__(self, channels=256, action_dim=9):
        super().__init__()
        self.stem = Conv(channels + action_dim, channels, 1)
        self.net = nn.Sequential(Bottleneck(channels, channels), Bottleneck(
            channels, channels), Conv(channels, channels, 3))

    def forward(self, state, action):
        action_map = action.view(
            *action.shape, 1, 1).expand(-1, -1, state.shape[2], state.shape[3])
        return self.net(self.stem(torch.cat([state, action_map], dim=1)))

# =====================================================================
# 3b. 端到端追踪模块 (End-to-End Track Query Module)
# =====================================================================


class TrackQueryModule(nn.Module):
    def __init__(self, feat_channels=128, num_queries=16, num_heads=4, nc=80, nm=32):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.query_mamba = Mamba(d_model=feat_channels, d_state=16, d_conv=4,
                                 expand=2) if Mamba else nn.Linear(feat_channels, feat_channels)
        self.query_norm = nn.LayerNorm(feat_channels)

        self.cross_attn = nn.MultiheadAttention(
            feat_channels, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(feat_channels)

        self.box_head = nn.Sequential(
            nn.Linear(feat_channels, 64), nn.SiLU(), nn.Linear(64, 4), nn.Sigmoid())
        self.cls_head = nn.Linear(feat_channels, nc)
        self.mask_head = nn.Linear(feat_channels, nm)
        self.alive_head = nn.Linear(feat_channels, 1)
        nn.init.constant_(self.alive_head.bias, -4.0)

    def forward(self, st_p3):
        B, T, C, H, W = st_p3.shape
        N = self.num_queries
        queries = self.query_embed.weight.unsqueeze(
            0).expand(B, -1, -1).clone()

        query_seq = []
        for t in range(T):
            feat_flat = st_p3[:, t].flatten(2).permute(0, 2, 1)
            q_attn, _ = self.cross_attn(queries, feat_flat, feat_flat)
            queries = self.cross_attn_norm(queries + q_attn)
            query_seq.append(queries)

        q_seq = torch.stack(query_seq, dim=1)
        q_flat = q_seq.permute(0, 2, 1, 3).reshape(B * N, T, C)
        q_temp = self.query_mamba(q_flat)
        q_temp = self.query_norm(q_flat + q_temp)
        q_temp = q_temp.view(B, N, T, C).permute(0, 2, 1, 3)

        return {
            "track_boxes":   self.box_head(q_temp),
            "track_classes": self.cls_head(q_temp),
            "track_alive":   self.alive_head(q_temp),
            "track_masks":   self.mask_head(q_temp),
        }


class Proto(nn.Module):
    def __init__(self, c1, c_, c2):
        super().__init__()
        self.cv1 = Conv(c1, c_, 3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)
        self.cv2 = Conv(c_, c_, 3)
        self.cv3 = Conv(c_, c2, 1)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class Proto26(Proto):
    def __init__(self, ch, c_=256, c2=32, nc=80):
        super().__init__(c_, c_, c2)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], 1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, 3)
        self.semseg = nn.Sequential(
            Conv(ch[0], c_, 3), Conv(c_, c_, 3), nn.Conv2d(c_, nc, 1))

    def forward(self, x: list):
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            up_feat = F.interpolate(
                f(x[i + 1]), size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        return super().forward(self.feat_fuse(feat)), self.semseg(feat)


class DWConv(nn.Sequential):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True):
        super().__init__(
            Conv(c1, c1, k, s, p, g=c1, d=d, act=True),
            Conv(c1, c2, 1, 1, act=act)
        )


class LRPCLayer(nn.Module):
    def __init__(self, c_in, c_loc_in, nc=4585, is_linear=True):
        super().__init__()
        if is_linear:
            self.vocab = nn.Linear(c_in, nc)
        else:
            self.vocab = nn.Conv2d(c_in, nc, 1)
        self.pf = nn.Conv2d(c_in, 1, 1)
        self.loc = nn.Conv2d(c_loc_in, 4, 1)


class YOLOESegment26(nn.Module):
    def __init__(self, nc=80, nm=32, npr=256, embed=512, reg_max=1, ch=(), **kwargs):
        super().__init__()
        self.proto = Proto26(ch, npr, nm, nc)

        c2 = max((16, ch[0] // 4, reg_max * 4))
        self.cv2 = nn.ModuleList(nn.Sequential(
            Conv(x, c2, 3), Conv(c2, c2, 3)) for x in ch)
        self.one2one_cv2 = nn.ModuleList(nn.Sequential(
            Conv(x, c2, 3), Conv(c2, c2, 3)) for x in ch)

        c3 = max(ch[0], min(nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(
            DWConv(x, c3, 3), DWConv(c3, c3, 3)) for x in ch)
        self.one2one_cv3 = nn.ModuleList(nn.Sequential(
            DWConv(x, c3, 3), DWConv(c3, c3, 3)) for x in ch)

        c5 = max(ch[0] // 4, nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(
            c5, c5, 3), nn.Conv2d(c5, nm, 1)) for x in ch)
        self.one2one_cv5 = nn.ModuleList(nn.Sequential(
            Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, nm, 1)) for x in ch)

        self.lrpc = nn.ModuleList([
            LRPCLayer(c3, c2, nc=4585, is_linear=True),
            LRPCLayer(c3, c2, nc=4585, is_linear=True),
            LRPCLayer(c3, c2, nc=4585, is_linear=False)
        ])

    def forward(self, x):
        proto_out, semseg = self.proto(x)

        def process_branch(cv2_list, cv3_list, cv5_list):
            boxes, scores, mc, obj, cls = [], [], [], [], []
            for i, f in enumerate(x):
                feat_box = cv2_list[i](f)
                feat_cls = cv3_list[i](f)
                feat_mask = cv5_list[i](f)

                bbox = self.lrpc[i].loc(feat_box)
                boxes.append(bbox)

                gate_logits = self.lrpc[i].pf(feat_cls)
                obj.append(gate_logits)

                gate = torch.sigmoid(gate_logits)
                gated_cls = feat_cls * gate

                if isinstance(self.lrpc[i].vocab, nn.Linear):
                    cls_pred = self.lrpc[i].vocab(
                        gated_cls.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                else:
                    cls_pred = self.lrpc[i].vocab(gated_cls)

                scores.append(cls_pred)
                cls.append(cls_pred)
                mc.append(feat_mask)
            return obj, cls, boxes, mc

        obj, cls, boxes, mc = process_branch(self.cv2, self.cv3, self.cv5)
        obj_o2o, cls_o2o, boxes_o2o, mc_o2o = process_branch(
            self.one2one_cv2, self.one2one_cv3, self.one2one_cv5)

        return {
            "features": x,
            "objectness": obj, "classification": cls, "boxes": boxes, "mask_coefficients": mc,
            "o2o_objectness": obj_o2o, "o2o_classification": cls_o2o, "o2o_boxes": boxes_o2o, "o2o_mask_coefficients": mc_o2o,
            "mask_prototypes": proto_out, "semseg": semseg
        }

# =====================================================================
# 4. 主模型架构 (Vision Model)
# =====================================================================


class MyYOLOE(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            Conv(3, 32, 3, 2),  # 0
            Conv(32, 64, 3, 2),  # 1
            C3k2(64, 128, n=1, shortcut=True, c3k=False, e=0.25),  # 2
            Conv(128, 128, 3, 2),  # 3
            C3k2(128, 256, n=1, shortcut=True, c3k=False, e=0.25),  # 4
            Conv(256, 256, 3, 2),  # 5
            C3k2(256, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 6
            Conv(256, 512, 3, 2),  # 7
            C3k2(512, 512, n=1, shortcut=False, c3k=True, e=0.5),  # 8
            SPPF(512, 512, k=5),  # 9
            C2PSA(512, 512, n=1, e=0.5),  # 10
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 11
            Concat(1),  # 12
            C3k2(768, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 13
            nn.Upsample(scale_factor=2.0, mode='nearest'),  # 14
            Concat(1),  # 15
            C3k2(512, 128, n=1, shortcut=False, c3k=True, e=0.5),  # 16 (P3)
            Conv(128, 128, 3, 2),  # 17
            Concat(1),  # 18
            C3k2(384, 256, n=1, shortcut=False, c3k=True, e=0.5),  # 19 (P4)
            Conv(256, 256, 3, 2),  # 20
            Concat(1),  # 21
            C3k2Attention(768, 512, n=1, shortcut=False, e=0.5),  # 22 (P5)
            YOLOESegment26(nc=80, nm=32, npr=128, embed=80,
                           reg_max=1, ch=(128, 256, 512))  # 23
        ])

        self.routes = {12: [-1, 6], 15: [-1, 4],
                       18: [-1, 13], 21: [-1, 10], 23: [16, 19, 22]}

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if i == 23:
                break
            if i in self.routes:
                f = self.routes[i]
                x = m([x if j == -1 else y[j] for j in f]
                      if isinstance(f, list) else (y[f] if f != -1 else x))
            else:
                x = m(x)
            y.append(x)
        return y[0], y[1], y[16], y[19], y[22]


class TAONot42VisionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.segmenter = MyYOLOE()
        self.geom_decoder = UnifiedGeometryDecoder(128, 64, 32)
        self.st_block = SpatioTemporalMambaBlock(128)
        self.st_block_p4 = SpatioTemporalMambaBlock(256)
        self.st_block_p5 = SpatioTemporalMambaBlock(512)
        self.pose_head = EgoPoseHead(128)
        self.feature_predictor = FeaturePredictorHead(128)
        self.state_update_gate_head = nn.Sequential(
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
        self.track_module = TrackQueryModule(
            feat_channels=128, num_queries=16, num_heads=4, nc=4585, nm=32)

        self.f1_temporal = nn.Conv3d(32, 32, kernel_size=(
            3, 1, 1), padding=(1, 0, 0), groups=32)
        self.f2_temporal = nn.Conv3d(64, 64, kernel_size=(
            3, 1, 1), padding=(1, 0, 0), groups=64)

    def extract_features(self, peripheral):
        return self.segmenter(peripheral)

    def forward_physics(self, f1, f2, p3_fused, p4, p5, dt, step, get_loss_weights_fn=None, original_shape=None):
        B, T = f1.shape[:2]
        h, w = original_shape if original_shape else (
            f1.shape[3] * 2, f1.shape[4] * 2)

        t0 = torch.rand(B, 1, device=f1.device) * 1000.0
        t_abs = t0 + torch.cumsum(dt, dim=1)

        def update_st(block, p_feat):
            B_s, T_s, C_s, H_s, W_s = p_feat.shape
            pooled = F.avg_pool2d(p_feat.flatten(0, 1), 2, 2).view(
                B_s, T_s, C_s, H_s//2, W_s//2)
            st_out = block(pooled, t_abs)
            st_out_up = F.interpolate(st_out.flatten(0, 1), size=(
                H_s, W_s), mode="bilinear", align_corners=False).view(B_s, T_s, C_s, H_s, W_s)
            return st_out, p_feat + st_out_up

        next_st, spatiotemporal_p3 = update_st(self.st_block, p3_fused)
        next_st_p4, spatiotemporal_p4 = update_st(self.st_block_p4, p4)
        next_st_p5, spatiotemporal_p5 = update_st(self.st_block_p5, p5)

        preds = self.segmenter.model[-1]([
            spatiotemporal_p3.flatten(0, 1),
            spatiotemporal_p4.flatten(0, 1),
            spatiotemporal_p5.flatten(0, 1)
        ])

        lw = get_loss_weights_fn(step) if get_loss_weights_fn else {
            "flow": 1, "box": 1, "mask": 1, "anom": 1}

        ego_pose = self.pose_head(spatiotemporal_p3.flatten(0, 1))

        # f1: [B, T, 32, H, W] -> [B, 32, T, H, W] -> apply 3D conv -> back to [B, T, 32, H, W]
        f1_t = self.f1_temporal(f1.permute(
            0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        f2_t = self.f2_temporal(f2.permute(
            0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)

        depth_raw, flow_raw = self.geom_decoder(
            f1_t.flatten(0, 1), f2_t.flatten(0, 1), spatiotemporal_p3.flatten(0, 1), ego_pose_feat=ego_pose, need_flow=(lw["flow"] > 0)
        )

        depth_pred = torch.exp(torch.clamp(F.interpolate(
            depth_raw, size=(h, w), mode="bilinear", align_corners=False).squeeze(1), min=-4.6, max=4.6)).view(B*T, h, w)
        flow_pred = flow_raw * 1.5 if flow_raw is not None else None

        gate_logits = self.state_update_gate_head(
            spatiotemporal_p3.mean(dim=[3, 4]).flatten(0, 1))
        gate = torch.sigmoid(gate_logits).view(B*T)

        feat_err = torch.zeros(
            B, T, next_st.shape[-2], next_st.shape[-1], device=f1.device)
        if lw["anom"] > 0 and T > 1:
            prev_st = next_st[:, :-1].flatten(0, 1)
            prev_ego = ego_pose.view(B, T, 9)[:, :-1].flatten(0, 1)
            predicted_st = self.feature_predictor(
                prev_st, prev_ego).view(B, T-1, *next_st.shape[2:])
            feat_err[:, 1:] = F.smooth_l1_loss(
                predicted_st, next_st[:, 1:], reduction="none").mean(dim=2)

        track_out = self.track_module(spatiotemporal_p3)

        return {
            "objectness": preds["o2o_objectness"], "classification": preds["o2o_classification"],
            "box_dist": preds["o2o_boxes"] if lw["box"] > 0 else None,
            "boxes": decode_dfl_boxes(preds["o2o_boxes"], 32) if lw["box"] > 0 else None,
            "mask_coefficients": preds["o2o_mask_coefficients"] if lw["mask"] > 0 else None,
            "mask_prototypes": preds["mask_prototypes"] if lw["mask"] > 0 else None,
            "depth": depth_pred, "log_depth": torch.log(depth_pred), "ego_pose": ego_pose,
            "flow": flow_pred,
            "features": spatiotemporal_p3.flatten(0, 1), "anomaly_map": feat_err.flatten(0, 1),
            "feature_error": feat_err.mean(), "state_update_gate": gate,
            "dense_objectness": preds["objectness"], "dense_classification": preds["classification"],
            "dense_box_dist": preds["boxes"], "dense_mask_coefficients": preds["mask_coefficients"],
            "track_boxes":   track_out["track_boxes"],
            "track_classes": track_out["track_classes"],
            "track_alive":   track_out["track_alive"],
            "track_masks":   track_out["track_masks"],
        }

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
        if not IN_COLAB:
            while True:
                item = {
                    "video": torch.randint(0, 256, (12, 256, 256, 3), dtype=torch.uint8),
                    "segmentation": torch.randint(0, 3, (12, 256, 256), dtype=torch.int32),
                    "depth": torch.rand(12, 256, 256, dtype=torch.float32) * 1000,
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
                **({"is_dynamic": x["instances"]["is_dynamic"]} if "instances" in x and "is_dynamic" in x["instances"] else {})
            }

        ds = ds.map(map_fn, num_parallel_calls=tf.data.AUTOTUNE).prefetch(
            tf.data.AUTOTUNE)

        for item in tfds.as_numpy(ds):
            p_item = {k: torch.from_numpy(item[k_i]).pin_memory() for k, k_i in [(
                "video", "video"), ("cam_pos", "cam_pos"), ("cam_quat", "cam_quat")]}
            p_item.update({k: torch.from_numpy(item[k_i][..., 0]).pin_memory() for k, k_i in [
                          ("segmentation", "segmentations"), ("depth", "depth")]})

            if "is_dynamic" in item:
                p_item["is_dynamic"] = torch.from_numpy(
                    item["is_dynamic"]).pin_memory()

            f_np = item["forward_flow"].astype(np.float32)
            if "metadata" in item and "forward_flow_range" in item["metadata"]:
                minv, maxv = item["metadata"]["forward_flow_range"]
                f_np = f_np / 65535.0 * (maxv - minv) + minv
            else:
                f_np = (f_np - 32768.0) / 64.0

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

        return {k: [i.get(k) for i in batch] for k in ["video", "segmentation", "depth", "forward_flow", "cam_pos", "cam_quat", "is_dynamic"]}


def process_batch_on_gpu(batch, device, target_size=256):
    def to_gpu(k, dtype=None):
        stacked = torch.stack([x.to(device, non_blocking=True)
                              for x in batch[k]])
        return stacked.to(dtype) if dtype else stacked

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
            [F.pad(x.to(device), (0, max_dyn_len - len(x))) for x in batch["is_dynamic"]])

    depth_m = torch.clamp(depth_raw / 1000.0, 0.01, 100.0)
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

    return {
        "video": video_p, "seg_raw": seg, "depth": depth_m, "log_depth": torch.log(depth_m),
        "flow": flow_norm, "cam_pos": cam_pos, "cam_quat": cam_quat, "is_dynamic": is_dyn_out, "sky_mask": sky_mask,
        "seg_small": seg_small, "bboxes_dense": bboxes_dense, "obj_dense": obj_dense, "cls_dense": cls_dense,
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
# 6. Loss 计算工具
# =====================================================================


def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20):
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
    device = obj_list[0].device if obj_list else torch.device("cpu")
    H_img, W_img = (obj_list[0].shape[2] * 8,
                    obj_list[0].shape[3] * 8) if obj_list else (0, 0)
    results = []

    for b in range(B):
        all_scores, all_boxes, all_masks_info, all_classes = [], [], [], []
        for i, (obj, box) in enumerate(zip(obj_list, box_list)):
            stride = 8 * (2 ** i)
            if box is None:
                continue

            valid = torch.sigmoid(obj[b, 0]) > score_thresh
            if not valid.any():
                continue

            sel_scores = torch.sigmoid(obj[b, 0])[valid]
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

        keep = torchvision.ops.nms(all_boxes * torch.tensor(
            [W_img, H_img, W_img, H_img], device=device), all_scores, nms_thresh)[:max_det]

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


def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(
        W, device=depth.device), indexing="ij")

    pixels = torch.stack([x.flatten().expand(
        B, -1), y.flatten().expand(B, -1), torch.ones_like(x.flatten().expand(B, -1))], dim=1)

    pose_rot = six_d_to_matrix(pose[:, 3:])
    pose_trans = pose[:, :3].unsqueeze(2)

    # 3D points
    points_3d = torch.bmm(K_inv.expand(B, 3, 3),
                          pixels.float()) * depth.view(B, 1, H * W)
    # Transform to next frame
    points_next = torch.bmm(pose_rot, points_3d) + pose_trans
    # Project back to 2D
    pixels_next = torch.bmm(K.expand(B, 3, 3), points_next)

    depth_next = torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()
    x_n = 2.0 * (pixels_next[:, 0:1, :].float() / depth_next) / (W - 1) - 1.0
    y_n = 2.0 * (pixels_next[:, 1:2, :].float() / depth_next) / (H - 1) - 1.0

    grid = torch.cat([x_n, y_n], dim=1).view(B, 2, H, W).permute(0, 2, 3, 1)
    grid = torch.clamp(grid, -2.0, 2.0)

    warped = F.grid_sample(img_next, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    warped = torch.nan_to_num(warped, 0.0)

    valid_mask = ((x_n > -1.0) & (x_n < 1.0) & (y_n > -1.0)
                  & (y_n < 1.0)).view(B, 1, H, W).float()
    depth_mask = ((depth > 0.01) & (
        pixels_next[:, 2:3, :].view(B, 1, H, W) > 0.01)).float()

    return warped, valid_mask * depth_mask


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
# 7. TAO 训练器 (OOP Refactoring)
# =====================================================================


class TAOTrainer:
    def __init__(self, args, model, buffer, prefetcher):
        self.args = args
        self.device = torch.device(args.device)
        self.model = model.to(self.device)
        self.buffer = buffer
        self.prefetcher = prefetcher

        if self.args.yolo_weights:
            self._load_yolo_weights()
            if self.args.freeze:
                for param in self.model.segmenter.parameters():
                    param.requires_grad = False

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.scaler = torch.amp.GradScaler(
            self.device.type) if self.device.type == "cuda" else None
        self.global_step, self.start_time, self.best_loss, self.epochs_no_improve = 0, time.time(), float("inf"), 0
        self.mode = "supervised"

    def _load_yolo_weights(self):
        if not os.path.exists(self.args.yolo_weights):
            print(f"Downloading {self.args.yolo_weights} from Ultralytics...")
            urllib.request.urlretrieve(
                f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{self.args.yolo_weights}", self.args.yolo_weights)
            print("Download complete.")

        for name, module in self.model.named_modules():
            if module.__class__.__name__ == 'Conv':
                c1, c2 = module.conv.in_channels, module.conv.out_channels
                k, s = module.conv.kernel_size, module.conv.stride
                p, g, d = module.conv.padding, module.conv.groups, module.conv.dilation

                new_conv = nn.Conv2d(
                    c1, c2, k, s, p, groups=g, dilation=d, bias=True)
                new_conv.to(module.conv.weight.device)
                module.conv = new_conv
                module.bn = nn.Identity()
            elif module.__class__.__name__ == 'PSABlock':
                if hasattr(module, 'add_norm1'):
                    module.add_norm1 = nn.Identity()
                if hasattr(module, 'add_norm2'):
                    module.add_norm2 = nn.Identity()

        ckpt = torch.load(self.args.yolo_weights,
                          map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (
            ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

        tgt = self.model.state_dict()

        def map_key(k):
            return k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k

        loaded_keys = {k for k, v in sd.items() if map_key(
            k) in tgt and tgt[map_key(k)].shape == v.shape}
        print(f"[YOLO] Successfully loaded {len(loaded_keys)}/{len(sd)} keys")
        tgt.update({map_key(k): v for k, v in sd.items() if k in loaded_keys})
        self.model.load_state_dict(tgt)

    def _setup_finetune(self):
        for param in self.model.segmenter.parameters():
            param.requires_grad = False

        trainable_modules = [
            self.model.geom_decoder, self.model.pose_head, self.model.st_block,
            self.model.st_block_p4, self.model.st_block_p5,
            self.model.feature_predictor, self.model.state_update_gate_head
        ]
        for m in trainable_modules:
            for p in m.parameters():
                p.requires_grad = True

        if hasattr(self.model.segmenter.model[-1], "obj_proj"):
            self.model.segmenter.model[-1].obj_proj.requires_grad_(True)
        if hasattr(self.model.segmenter.model[-1], "one2one_obj_proj"):
            self.model.segmenter.model[-1].one2one_obj_proj.requires_grad_(
                True)
        if hasattr(self.model.segmenter.model[-1], "class_prompts"):
            self.model.segmenter.model[-1].class_prompts.requires_grad_(True)

        # [FIX] 强制冻结 LRPCLayer 的 vocab（类别词典权重）
        # 保护 4585 维语义特征空间不被坍缩
        if hasattr(self.model.segmenter.model[-1], "lrpc"):
            for layer in self.model.segmenter.model[-1].lrpc:
                layer.vocab.requires_grad_(False)

        self.optimizer = torch.optim.AdamW(filter(
            lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr * 0.1)

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            if self.args.finetune_after_epoch and epoch > self.args.finetune_after_epoch and self.mode == "supervised":
                self.mode = "self_supervised"
                self._setup_finetune()

            epoch_loss = self._train_epoch(epoch)
            print(
                f"\n✅ Epoch {epoch} End | Avg Loss: {epoch_loss:.4f} | Mode: {self.mode}")
            torch.save(self.model.state_dict(), self.args.checkpoint.replace(
                ".pth", f"_epoch_{epoch}.pth"))

            if epoch_loss < self.best_loss:
                self.best_loss, self.epochs_no_improve = epoch_loss, 0
                torch.save(self.model.state_dict(),
                           self.args.checkpoint.replace(".pth", "_best.pth"))
                print(f"🌟 Best Model saved (Loss: {self.best_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.args.early_stop_patience:
                    print(f"\n🛑 Early Stopping Triggered!")
                    break

    def _train_epoch(self, epoch):
        loss_sum = 0.0
        for _ in range(self.args.steps_per_epoch):
            batch = self.prefetcher.next()
            if batch is None:
                continue

            loss_sum += self._train_chunk(batch)

            if self.global_step == 500 and self.mode == "supervised" and hasattr(self.model.segmenter.model[-1], "class_prompts"):
                self.model.segmenter.model[-1].class_prompts.requires_grad = True

            if self.mode == "supervised" and self.global_step in [self.args.unfreeze_step_1, self.args.unfreeze_step_2]:
                target_range = range(
                    20, 23) if self.global_step == self.args.unfreeze_step_1 else range(16, 20)
                for n, p in self.model.segmenter.named_parameters():
                    if any(f"model.{i}." in n for i in target_range):
                        p.requires_grad = True

        return loss_sum / self.args.steps_per_epoch

    def _extract_target_chunk(self, batch, c_start, c_end, max_t):
        T = c_end - c_start
        B = batch["video"].shape[0]
        tgt = {}
        for k, v in batch.items():
            if k in ("video", "flow"):
                continue

            if k == "is_dynamic":
                tgt[k] = v.unsqueeze(
                    1).expand(-1, T, -1).flatten(0, 1) if v is not None else None
            elif isinstance(v, list):
                tgt[k] = [x[:, c_start:c_end].flatten(0, 1) for x in v]
            else:
                tgt[k] = v[:, c_start:c_end].flatten(
                    0, 1) if v is not None else None

        flow = batch.get("flow")
        if flow is not None:
            flow_tgt = torch.zeros_like(flow[:, c_start:c_end])
            for i, step in enumerate(range(c_start, c_end)):
                flow_tgt[:, i] = flow[:, step] if step + \
                    1 < max_t else torch.zeros_like(flow[:, 0])
            tgt["flow_target"] = flow_tgt.flatten(0, 1)

        tgt["cam_pos_t"] = batch["cam_pos"][:, c_start:c_end].flatten(0, 1)
        tgt["cam_quat_t"] = batch["cam_quat"][:, c_start:c_end].flatten(0, 1)

        cam_pos_next = torch.zeros_like(batch["cam_pos"][:, c_start:c_end])
        cam_quat_next = torch.zeros_like(batch["cam_quat"][:, c_start:c_end])
        has_next = torch.zeros(B, T, device=self.device, dtype=torch.bool)

        for i, step in enumerate(range(c_start, c_end)):
            next_idx = step + 1 if step + 1 < max_t else step
            cam_pos_next[:, i] = batch["cam_pos"][:, next_idx]
            cam_quat_next[:, i] = batch["cam_quat"][:, next_idx]
            has_next[:, i] = step + 1 < max_t

        tgt["cam_pos_next"] = cam_pos_next.flatten(0, 1)
        tgt["cam_quat_next"] = cam_quat_next.flatten(0, 1)
        tgt["has_next"] = has_next.flatten(0, 1)

        if "cls_dense" in tgt:
            for i, step in enumerate(range(c_start, c_end)):
                if self.global_step < 1000 or step < 2:
                    if isinstance(tgt["cls_dense"], list):
                        for x in tgt["cls_dense"]:
                            x.view(B, T, *x.shape[1:])[:, i] = -100
                    else:
                        tgt["cls_dense"].view(
                            B, T, *tgt["cls_dense"].shape[1:])[:, i] = -100

        return tgt

    def _train_chunk(self, batch):
        v_seq, t_max = batch["video"], batch["video"].shape[1]
        total_loss = 0.0

        loss_acc = {k: 0.0 for k in [
            "Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls"]}
        total_frames = 0

        for c_start in range(0, t_max, self.args.seq_len):
            c_end = min(c_start + self.args.seq_len, t_max)
            c_vids = v_seq[:, c_start:c_end]
            T_chunk = c_end - c_start
            total_frames += T_chunk
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                allow_backbone_grad = self.mode == "supervised" and self.global_step >= self.args.unfreeze_step_1
                with contextlib.nullcontext() if allow_backbone_grad else torch.no_grad():
                    extracted = self.model.extract_features(
                        c_vids.reshape(-1, *c_vids.shape[2:]))
                    feats = [f.view(v_seq.shape[0], T_chunk, *f.shape[1:])
                             for f in extracted]

                dt = torch.full(
                    (v_seq.shape[0], T_chunk), 1.0 / 24.0, device=self.device)
                preds = self.model.forward_physics(
                    *feats, dt, self.global_step, get_loss_weights, c_vids.shape[-2:])
                tgts = self._extract_target_chunk(batch, c_start, c_end, t_max)

                img_next = torch.zeros_like(c_vids)
                for i, step in enumerate(range(c_start, c_end)):
                    img_next[:, i] = v_seq[:, min(step+1, t_max-1)]

                loss, l_dict, w_img = compute_physics_loss(preds, tgts, c_vids.flatten(
                    0, 1), img_next.flatten(0, 1), self.mode, self.global_step)

            if self.scaler:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()

            total_loss += loss.item()
            for k in loss_acc:
                loss_acc[k] += l_dict[k] * T_chunk

            if (self.global_step + 1) % self.args.vis_interval == 0:
                def slice_second_frame(v):
                    if v is None:
                        return None
                    if isinstance(v, list):
                        res = []
                        for x in v:
                            if x.dim() == 0:
                                res.append(x)
                            elif x.shape[0] == v_seq.shape[0] * T_chunk:
                                res.append(
                                    x[(v_seq.shape[0] - 1) * T_chunk + 1: (v_seq.shape[0] - 1) * T_chunk + 2])
                            else:
                                res.append(x[-v_seq.shape[0]:])
                        return res
                    if v.dim() == 0:
                        return v
                    if v.shape[0] == v_seq.shape[0] * T_chunk:
                        return v[(v_seq.shape[0] - 1) * T_chunk + 1: (v_seq.shape[0] - 1) * T_chunk + 2]
                    return v[-v_seq.shape[0]:]

                fp = save_visualization(
                    c_vids[-1:, 1],
                    {k: slice_second_frame(v) for k, v in tgts.items()},
                    {k: slice_second_frame(v) for k, v in preds.items()},
                    self.global_step + 1,
                    slice_second_frame(w_img) if w_img is not None else None
                )
                if wandb and fp:
                    wandb.log({"Vis": wandb.Image(fp)}, step=self.global_step)

            self.global_step += 1
            if self.global_step % 10 == 0:
                print(f"[{time.time()-self.start_time:.1f}s] S{self.global_step} | Tot:{loss.item():.4f} | " + " ".join(
                    [f"{k}:{loss_acc[k]/total_frames:.2f}" for k in ["Obj", "Box", "Mask", "Depth", "Ego", "Flow", "Anom"]]))
                if wandb:
                    log_dict = {
                        f"Loss/{k}": loss_acc[k]/total_frames for k in loss_acc}
                    log_dict.update(
                        {"Loss/Total": loss.item(), "Step": self.global_step})
                    wandb.log(log_dict, step=self.global_step)

        return total_loss


# =====================================================================
# Main 参数配置与执行入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_buffer_size", type=int, default=64)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--unfreeze_step_1", type=int, default=1000)
    parser.add_argument("--unfreeze_step_2", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str,
                        default="tao_not_42_weights.pth")
    parser.add_argument("--yolo_weights", type=str,
                        default="yoloe-26s-seg-pf.pt")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--finetune_after_epoch", type=int, default=0)
    args = parser.parse_args()

    if args.use_wandb and wandb:
        wandb.init(project="tao_not_42", config=vars(args))
    elif not args.use_wandb:
        wandb = None

    try:
        data_buffer = AsyncDataBuffer(
            max_buffer_size=args.max_buffer_size, batch_size=args.batch_size)
        prefetcher = CUDAPrefetcher(
            data_buffer, torch.device(args.device), args.img_size)
        model = TAONot42VisionModel()

        trainer = TAOTrainer(args, model, data_buffer, prefetcher)
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")
