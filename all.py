import os
import time
import queue
import random
import argparse
import threading
import contextlib
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# =====================================================================
# 0. 环境与可选依赖配置
# =====================================================================
try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    import tensorflow as tf
    import tensorflow_datasets as tfds

try:
    import wandb
except ImportError:
    wandb = None

def flow_to_color(flow_np):
    # 向量化减去中位数以突出相对运动
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
    base_bgr = cv2.cvtColor((img_tensor * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = base_bgr.shape[:2]

    def add_title(img, text, pos=(10, 30), scale=0.8, thickness=2):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness)
        return img

    # --- Left: Prediction (Box & Mask) ---
    pred_canvas = base_bgr.copy()
    with torch.no_grad():
        inst = extract_instances(pred_t, score_thresh=0.3, nms_thresh=0.5)[0]
    
    if inst and len(inst["scores"]) > 0:
        masks_iter = inst["masks"] if inst["masks"] is not None else [None] * len(inst["scores"])
        for c, m, b in zip(inst["classes"], masks_iter, inst["boxes"]):
            color = (0, 0, 255) if (c.item() if c is not None else 1) == 1 else (255, 0, 0)
            if m is not None:
                m_np = m.cpu().numpy()
                pred_canvas[m_np] = pred_canvas[m_np] * 0.5 + np.array(color) * 0.5
            b_np = b.cpu().numpy() * [W, H, W, H]
            cv2.rectangle(pred_canvas, (int(b_np[0]), int(b_np[1])), (int(b_np[2]), int(b_np[3])), color, 2)
    add_title(pred_canvas, "Prediction")

    # --- Middle: Ground Truth (Box & Mask) ---
    gt_canvas = base_bgr.copy()
    if "seg_raw" in target_t and "is_dynamic" in target_t:
        seg, is_dyn = target_t["seg_raw"][0].cpu().numpy(), target_t["is_dynamic"][0].cpu().numpy()
        for uid in range(1, int(np.max(seg)) + 1):
            m = seg == uid
            if np.any(m):
                color = (0, 0, 255) if (uid - 1 < len(is_dyn) and is_dyn[uid - 1]) else (255, 0, 0)
                gt_canvas[m] = gt_canvas[m] * 0.5 + np.array(color) * 0.5
                y_idx, x_idx = np.where(m)
                cv2.rectangle(gt_canvas, (x_idx.min(), y_idx.min()), (x_idx.max(), y_idx.max()), color, 2)
    elif "bboxes_dense" in target_t and "obj_dense" in target_t:
        obj_t, boxes_t = target_t["obj_dense"][0, 0].cpu().numpy(), target_t["bboxes_dense"][0].cpu().numpy()
        for y, x in zip(*np.where(obj_t > 0.5)):
            b, gx, gy = boxes_t[:, y, x] * 8.0, x * 8.0 + 4.0, y * 8.0 + 4.0
            cv2.rectangle(gt_canvas, (int(gx - b[0]), int(gy - b[1])), (int(gx + b[2]), int(gy + b[3])), (0, 255, 0), 2)
    add_title(gt_canvas, "Ground Truth")

    # --- Right: 6-Grid Physics Output ---
    hw, hh = W // 2, H // 2
    def prep_cell(img, title):
        return add_title(cv2.resize(img, (hw, hh)), title, pos=(5, 20), scale=0.5, thickness=1)

    anom = pred_t["anomaly_map"][0].cpu().detach().numpy().squeeze()
    anom_img = cv2.applyColorMap((np.clip(anom / max(anom.max(), 1e-3), 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    
    p_flow = pred_t.get("flow")
    p_flow_img = flow_to_color(p_flow[0].cpu().detach().numpy().transpose(1, 2, 0)) if p_flow is not None else np.zeros((H, W, 3), np.uint8)
    
    g_dep, p_dep = target_t["depth"][0].cpu().numpy(), pred_t["depth"][0].cpu().detach().numpy()
    d_min, d_max = min(g_dep.min(), p_dep.min()), max(g_dep.max(), p_dep.max())
    
    g_flow_np = target_t.get("flow_target", torch.zeros((1, 2, H, W)))[0].cpu().numpy().transpose(1, 2, 0)
    warp_img = np.zeros((H, W, 3), np.uint8) if warped_img is None else cv2.cvtColor((np.clip(warped_img[0].permute(1, 2, 0).cpu().detach().numpy(), 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

    grid = np.vstack([
        np.hstack([prep_cell(anom_img, "Anomaly"), prep_cell(p_flow_img, "Pred Flow")]),
        np.hstack([prep_cell(depth_to_color(g_dep, d_min, d_max), "GT Depth"), prep_cell(depth_to_color(p_dep, d_min, d_max), "Pred Depth")]),
        np.hstack([prep_cell(flow_to_color(g_flow_np), "GT Flow"), prep_cell(warp_img, "Warped (Photo Error)")]),
    ])

    final_img = np.hstack([pred_canvas, gt_canvas, cv2.resize(grid, (int(grid.shape[1] * H / grid.shape[0]), H))])
    filepath = os.path.join(output_dir, f"vis_step_{step:05d}.jpg")
    cv2.imwrite(filepath, final_img)
    return filepath

# =====================================================================
# 1. 基础工具与几何函数
# =====================================================================
def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    w2 = w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    matrix = torch.stack(
        [
            w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw),
            2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw),
            2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2,
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


def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor(
        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32
    )
    K_inv = torch.inverse(K)
    return K, K_inv


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


def decode_dfl_boxes(pred_dist, reg_max=16):
    if isinstance(pred_dist, list):
        return [decode_dfl_boxes(x, reg_max) for x in pred_dist]
    # pred_dist: (B, 4*reg_max, H, W)
    B, C, H, W = pred_dist.shape
    prob = F.softmax(pred_dist.view(B, 4, reg_max, H, W), dim=2)
    weights = torch.arange(reg_max, dtype=torch.float32, device=pred_dist.device)
    distances = (prob * weights.view(1, 1, reg_max, 1, 1)).sum(dim=2)  # (B, 4, H, W)
    return distances

# =====================================================================
# 2. 模型核心组件 (Blocks)
# =====================================================================
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class YOLOConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, shortcut=True, n=3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n = n
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, attn_ratio: float = 0.5):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(
            B, self.num_heads, self.key_dim * 2 + self.head_dim, N
        ).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(
            v.reshape(B, C, H, W)
        )
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    def __init__(
        self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True
    ):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, self.c * 2, 1, 1)
        )
        self.cv2 = nn.Sequential(
            Conv(self.c * 2, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, c1, 1, 1)
        )
        self.m = nn.Sequential(
            *(
                PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64)
                for _ in range(n)
            )
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class Bottleneck(nn.Module):
    def __init__(
        self, c1: int, c2: int, shortcut: bool = True,
        g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5,
    ):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv(c1, self.c, 1, 1)
        self.cv3 = Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(
            *(
                Bottleneck(self.c, self.c, shortcut, g, k=(k, k), e=1.0)
                for _ in range(n)
            )
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k2(nn.Module):
    def __init__(
        self, c1, c2, n=1, c3k=False, e=0.5, e2=1.0, g=1, shortcut=True, attn=False
    ):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        if attn:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g, e2)
                if c3k
                else Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e2)
                for _ in range(n - 1)
            )
            self.m.append(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64))
        else:
            self.m = nn.ModuleList(
                C3k(self.c, self.c, 2, shortcut, g, e2)
                if c3k
                else Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e2)
                for _ in range(n)
            )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

# =====================================================================
# 3. 物理与时间模块 (Time & Physics Modules)
# =====================================================================
class TimeAwareConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_frequencies=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_frequencies = num_frequencies

        # 基频设置为 16s (Base Period T = 16s, Base Frequency = 2*pi / 16 = pi / 8)
        # 固定倍率为 2.0
        base_period = 16.0
        base_omega = (2.0 * torch.pi) / base_period
        self.register_buffer(
            "frequencies", 2.0 ** torch.arange(num_frequencies) * base_omega
        )

        time_embed_dim = num_frequencies * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 64), nn.SiLU(), nn.Linear(64, hidden_channels * 2)
        )

        gate_channels = input_channels + hidden_channels
        self.update_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)

    def forward(self, x, dt, state=None):
        if state is None:
            state = x.new_zeros(
                x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]
            )
        elif state.shape[-2:] != x.shape[-2:]:
            state = F.interpolate(
                state, size=x.shape[-2:], mode="bilinear", align_corners=False
            )

        scaled_time = dt.view(-1, 1) * self.frequencies.view(1, -1)
        time_emb = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)

        time_params = self.time_mlp(time_emb)
        gamma, beta = time_params.chunk(2, dim=-1)
        gamma = gamma.view(-1, self.hidden_channels, 1, 1)
        beta = beta.view(-1, self.hidden_channels, 1, 1)

        modulated_state = state * (gamma + 1.0) + beta

        gates_in = torch.cat([x, modulated_state], dim=1)
        update = torch.sigmoid(self.update_gate(gates_in))
        reset = torch.sigmoid(self.reset_gate(gates_in))
        candidate = torch.tanh(
            self.candidate(torch.cat([x, reset * modulated_state], dim=1))
        )

        return (1.0 - update) * modulated_state + update * candidate


class FlowDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_p3, ch_f2, kernel_size=3),
        )
        self.conv1 = YOLOConv(ch_f2 * 2, ch_f2, kernel_size=3)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_f2, ch_f1, kernel_size=3),
        )
        self.conv2 = YOLOConv(ch_f1 * 2, ch_f1, kernel_size=3)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_f1, ch_f1, kernel_size=3),
        )
        self.head = nn.Sequential(
            YOLOConv(ch_f1, ch_f1 // 2, kernel_size=3),
            nn.Conv2d(ch_f1 // 2, 2, kernel_size=3, padding=1)
        )

    def forward(self, f1, f2, p3):
        x = self.up1(p3)
        x = torch.cat([x, f2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        return self.head(x)


class EgoPoseHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        c3 = 64
        self.fc = nn.Sequential(nn.Linear(in_channels, c3), nn.SiLU(), nn.Linear(c3, 9))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        pose = self.fc(pooled)
        t = torch.tanh(pose[:, :3]) * 5.0
        # 零初始化保证初始输出为0，加上理想单位正交基
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=pose.device)
        rot_6d = identity_6d + torch.tanh(pose[:, 3:]) * 0.5
        return torch.cat([t, rot_6d], dim=1)


class FeaturePredictorHead(nn.Module):
    def __init__(self, channels=256, action_dim=9):
        super().__init__()
        self.stem = YOLOConv(channels + action_dim, channels, kernel_size=1)
        self.net = nn.Sequential(
            Bottleneck(channels, channels, shortcut=True),
            Bottleneck(channels, channels, shortcut=True),
            YOLOConv(channels, channels, kernel_size=3),
        )

    def forward(self, state, action):
        action_map = action.view(action.shape[0], action.shape[1], 1, 1).expand(
            -1, -1, state.shape[2], state.shape[3]
        )
        x = torch.cat([state, action_map], dim=1)
        return self.net(self.stem(x))


class DepthDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_p3, ch_f2, kernel_size=3),
        )
        self.conv1 = YOLOConv(ch_f2 * 2, ch_f2, kernel_size=3)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_f2, ch_f1, kernel_size=3),
        )
        self.conv2 = YOLOConv(ch_f1 * 2, ch_f1, kernel_size=3)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False),
            YOLOConv(ch_f1, ch_f1, kernel_size=3),
        )
        self.depth_out = nn.Sequential(
            YOLOConv(ch_f1, ch_f1 // 2, kernel_size=3),
            nn.Conv2d(ch_f1 // 2, 1, kernel_size=3, padding=1),
        )

    def forward(self, f1, f2, p3):
        x = self.up1(p3)
        x = torch.cat([x, f2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        return self.depth_out(x)


class Proto26(nn.Module):
    def __init__(self, ch=(), c_=256, c2=32, nc=80):
        super().__init__()
        self.cv1 = Conv(c_, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2, k=1)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)

    def forward(self, x):
        feat = x[0]
        for i, m in enumerate(self.feat_refine):
            up_feat = m(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode="nearest")
            feat = feat + up_feat
        p = self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(feat)))))
        return p


class YOLOESegment26(nn.Module):
    def __init__(self, nc=80, nm=32, npr=256, embed=512, reg_max=1, ch=()):
        super().__init__()
        self.nm = nm
        self.npr = npr
        self.nc = nc
        self.reg_max = reg_max
        self.proto = Proto26(ch, npr, nm, nc)

        c5 = max(ch[0] // 4, nm)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, nm, 1)) for x in ch)
        self.one2one_cv5 = nn.ModuleList(nn.Sequential(Conv(x, c5, 3), Conv(c5, c5, 3), nn.Conv2d(c5, nm, 1)) for x in ch)

        c2 = max(ch[0] // 4, 16)
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch)
        self.one2one_cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch)

        c3 = max(ch[0], min(nc, 100))
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)
        self.one2one_cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, embed, 1)) for x in ch)

        self.obj_proj = nn.ModuleList(nn.Conv2d(embed, 1, 1) for _ in ch)
        self.one2one_obj_proj = nn.ModuleList(nn.Conv2d(embed, 1, 1) for _ in ch)

        # Open-Vocabulary Semantic Prompts: Active (Class 0) and Passive (Class 1)
        # Replacing the traditional classification conv with explicit prompt embeddings
        self.class_prompts = nn.Parameter(torch.randn(2, embed))
        self.class_prompts.requires_grad_(False)

    def forward(self, x):
        proto_out = self.proto(x)
        
        boxes = []
        scores = []
        mc = []
        boxes_o2o = []
        scores_o2o = []
        mc_o2o = []
        obj_foreground = []
        obj_foreground_o2o = []
        cls_scores = []
        cls_scores_o2o = []

        norm_prompts = F.normalize(self.class_prompts, p=2, dim=1)

        for i in range(len(x)):
            boxes.append(self.cv2[i](x[i]))
            scores.append(self.cv3[i](x[i]))
            mc.append(self.cv5[i](x[i]))

            boxes_o2o.append(self.one2one_cv2[i](x[i]))
            scores_o2o.append(self.one2one_cv3[i](x[i]))
            mc_o2o.append(self.one2one_cv5[i](x[i]))

            obj_foreground.append(self.obj_proj[i](scores[i]))
            obj_foreground_o2o.append(self.one2one_obj_proj[i](scores_o2o[i]))

            norm_scores = F.normalize(scores[i], p=2, dim=1)
            cls_scores.append(torch.einsum("b c h w, k c -> b k h w", norm_scores, norm_prompts) * 10.0)

            norm_scores_o2o = F.normalize(scores_o2o[i], p=2, dim=1)
            cls_scores_o2o.append(torch.einsum("b c h w, k c -> b k h w", norm_scores_o2o, norm_prompts) * 10.0)

        # features is just the list of spatial features
        return {
            "features": x,
            "objectness": obj_foreground,
            "classification": cls_scores,
            "boxes": boxes,
            "mask_coefficients": mc,
            "o2o_objectness": obj_foreground_o2o,
            "o2o_classification": cls_scores_o2o,
            "o2o_boxes": boxes_o2o,
            "o2o_mask_coefficients": mc_o2o,
            "mask_prototypes": proto_out,
        }

# =====================================================================
# 4. 主模型架构 (Vision Model)
# =====================================================================
class MyYOLOE(nn.Module):
    def __init__(self):
        super().__init__()

        def c(dim):
            return int(dim * 0.5)

        def n(depth):
            return max(round(depth * 0.5), 1)

        self.model = nn.Sequential(
            Conv(3, c(64), 3, 2),  # 0 (f1)
            Conv(c(64), c(128), 3, 2),  # 1 (f2)
            C3k2(c(128), c(256), n=n(2), c3k=False, e=0.25),  # 2
            Conv(c(256), c(256), 3, 2),  # 3
            C3k2(c(256), c(512), n=n(2), c3k=False, e=0.25),  # 4
            Conv(c(512), c(512), 3, 2),  # 5
            C3k2(c(512), c(512), n=n(2), c3k=True),  # 6
            Conv(c(512), c(1024), 3, 2),  # 7
            C3k2(c(1024), c(1024), n=n(2), c3k=True),  # 8
            SPPF(c(1024), c(1024), k=5, n=3, shortcut=True),  # 9
            C2PSA(c(1024), c(1024), n=n(2), e=0.5),  # 10
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 11
            Concat(dimension=1),  # 12. Takes [11, 6]
            C3k2(c(1024) + c(512), c(512), n=n(2), c3k=True),  # 13
            nn.Upsample(scale_factor=2.0, mode="nearest"),  # 14
            Concat(dimension=1),  # 15. Takes [14, 4]
            C3k2(c(512) + c(512), c(256), n=n(2), c3k=True),  # 16 (P3)
            Conv(c(256), c(256), 3, 2),  # 17
            Concat(dimension=1),  # 18. Takes [17, 13]
            C3k2(c(256) + c(512), c(512), n=n(2), c3k=True),  # 19 (P4)
            Conv(c(512), c(512), 3, 2),  # 20
            Concat(dimension=1),  # 21. Takes [20, 10]
            C3k2(
                c(512) + c(1024), c(1024), n=n(2), c3k=True, e=0.5, attn=True
            ),  # 22 (P5)
            YOLOESegment26(
                nc=80, nm=32, npr=256, embed=512, reg_max=32, ch=(128, 256, 512)
            ),  # 23
        )

        for m in self.model:
            m.f = -1

        self.model[12].f = [-1, 6]
        self.model[15].f = [-1, 4]
        self.model[18].f = [-1, 13]
        self.model[21].f = [-1, 10]
        self.model[23].f = [16, 19, 22]

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if i == 23:
                break
            x = m([y[j] for j in m.f] if isinstance(m.f, list) else x)
            y.append(x)

        return y[0], y[1], y[16], y[19], y[22]


class TAONot42VisionModel(nn.Module):
    def __init__(self, base_channels=48, hidden_channels=768):
        super().__init__()
        self.segmenter = MyYOLOE()
        self.depth_decoder = DepthDecoder(128, 64, 32)
        self.conv_gru = TimeAwareConvGRUCell(128, 128)
        self.conv_gru_p4 = TimeAwareConvGRUCell(256, 256)
        self.conv_gru_p5 = TimeAwareConvGRUCell(512, 512)
        self.pose_head = EgoPoseHead(128)
        self.flow_head = FlowDecoder(128, 64, 32)
        self.feature_predictor = FeaturePredictorHead(128)
        self.state_update_gate_head = nn.Sequential(
            nn.Linear(128 + 1, 64), nn.SiLU(), nn.Linear(64, 1)
        )

    def extract_features(self, peripheral):
        return self.segmenter(peripheral)

    def forward_physics(
        self, f1, f2, p3_fused, p4, p5, dt, step,
        state=None, get_loss_weights_fn=None, original_shape=None,
    ):
        b = f1.shape[0]
        h, w = original_shape if original_shape else (f1.shape[2] * 2, f1.shape[3] * 2)
        state = state or {}

        # 辅助函数：简化 GRU 时空融合逻辑
        def update_gru(gru_cell, p_feat, gru_state):
            p_down = F.avg_pool2d(p_feat, kernel_size=2, stride=2)
            next_state = gru_cell(p_down, dt, gru_state)
            up_state = F.interpolate(next_state, size=p_feat.shape[-2:], mode="bilinear", align_corners=False)
            return next_state, p_feat + up_state

        next_gru_state, spatiotemporal_p3 = update_gru(self.conv_gru, p3_fused, state.get("gru"))
        next_gru_state_p4, spatiotemporal_p4 = update_gru(self.conv_gru_p4, p4, state.get("gru_p4"))
        next_gru_state_p5, spatiotemporal_p5 = update_gru(self.conv_gru_p5, p5, state.get("gru_p5"))

        # Step 3: 直接调用检测头 (YOLOESegment26)
        import torch.utils.checkpoint as checkpoint

        def run_yolo_head(p3, p4, p5):
            return self.segmenter.model[-1]([p3, p4, p5])

        preds = checkpoint.checkpoint(
            run_yolo_head, spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5, use_reentrant=False
        )

        depth_logits = self.depth_decoder(f1, f2, spatiotemporal_p3)
        depth_logits = F.interpolate(depth_logits, size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        log_depth_pred = depth_logits
        depth_pred = torch.exp(torch.clamp(log_depth_pred, min=-4.6, max=4.6))

        ego_pose = self.pose_head(spatiotemporal_p3)
        lw = get_loss_weights_fn(step) if get_loss_weights_fn else {"flow": 1, "box": 1, "mask": 1, "anom": 1}

        pred_flow = self.flow_head(f1, f2, spatiotemporal_p3) * 1.5 if lw["flow"] > 0 else None

        gate_in = torch.cat([spatiotemporal_p3.mean(dim=[2, 3]), dt.view(-1, 1)], dim=-1)
        gate = torch.sigmoid(self.state_update_gate_head(gate_in)).view(-1, 1, 1, 1)

        def mix_state(old_st, next_st):
            return old_st * (1.0 - gate) + next_st * gate if old_st is not None else next_st

        final_gru_state = mix_state(state.get("gru"), next_gru_state)
        final_gru_state_p4 = mix_state(state.get("gru_p4"), next_gru_state_p4)
        final_gru_state_p5 = mix_state(state.get("gru_p5"), next_gru_state_p5)

        prev_ego_pose = state.get("prev_ego", torch.zeros_like(ego_pose))
        if state.get("gru") is not None and lw["anom"] > 0:
            pred_current_feature = self.feature_predictor(state.get("gru"), prev_ego_pose)
            feature_error_map = F.smooth_l1_loss(pred_current_feature, final_gru_state.detach(), reduction="none").mean(dim=1)
        else:
            feature_error_map = torch.zeros(b, next_gru_state.shape[2], next_gru_state.shape[3], device=f1.device)

        return {
            "objectness": preds["o2o_objectness"],
            "classification": preds["o2o_classification"],
            "box_dist": preds["o2o_boxes"] if lw["box"] > 0 else None,
            "boxes": decode_dfl_boxes(preds["o2o_boxes"], reg_max=32) if lw["box"] > 0 else None,
            "mask_coefficients": preds["o2o_mask_coefficients"] if lw["mask"] > 0 else None,
            "mask_prototypes": preds["mask_prototypes"] if lw["mask"] > 0 else None,
            "depth": depth_pred,
            "log_depth": log_depth_pred,
            "ego_pose": ego_pose,
            "flow": pred_flow,
            "features": spatiotemporal_p3,
            "anomaly_map": feature_error_map,
            "feature_error": feature_error_map.mean(),
            "state_update_gate": gate.view(b),
            "next_state": {"gru": final_gru_state, "gru_p4": final_gru_state_p4, "gru_p5": final_gru_state_p5, "prev_ego": ego_pose},
            "dense_objectness": preds["objectness"],
            "dense_classification": preds["classification"],
            "dense_box_dist": preds["boxes"],
            "dense_mask_coefficients": preds["mask_coefficients"],
        }

    def forward(self, peripheral, dt, step, state=None, get_loss_weights_fn=None):
        b, _, h, w = peripheral.shape
        f1, f2, p3_fused, p4, p5 = self.extract_features(peripheral)
        return self.forward_physics(
            f1, f2, p3_fused, p4, p5, dt, step, state, get_loss_weights_fn, original_shape=(h, w)
        )

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

        print("\n" + "=" * 60)
        print(f"🚀 [异步管线] 正在启动后台独立 I/O 数据流缓冲池...")
        print(f"   >> 最大数据缓冲池: {max_buffer_size} 个序列 (滚动窗口)")
        print(f"   >> 动态批次抽样: 每次随机抽取 {batch_size} 条 (拒绝空转)")
        print("=" * 60 + "\n")

        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()

    def _fetch_loop(self):
        if not IN_COLAB:
            print("❌ TFDS requires colab env, Data Buffer is mock running.")
            return

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
                if not self.thread.is_alive() and IN_COLAB:
                    raise RuntimeError(
                        "❌ 后台数据流线程异常崩溃，请检查网络或 TFDS 配置！"
                    )
                # Mock if not in colab
                if not IN_COLAB:
                    return None
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
            ).squeeze(1).view(B, T, target_size, target_size)
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
    
    seg_small = (
        F.interpolate(
            seg.flatten(0, 1).unsqueeze(1), size=(H // 8, W // 8), mode="nearest"
        ).squeeze(1).view(B, T, H // 8, W // 8)
    )

    # Build multi-scale targets for P3 (stride 8), P4 (stride 16), P5 (stride 32)
    bboxes_dense = []
    obj_dense = []
    cls_dense = []

    max_uid = int(seg_long.max().item())
    valid_bt = None
    ymin = ymax = xmin = xmax = true_area = box_area = None
    if max_uid > 0:
        uids = torch.arange(1, max_uid + 1, device=device, dtype=torch.int16).view(
            -1, 1, 1, 1, 1
        )
        masks = seg_long.to(torch.int16).unsqueeze(0) == uids
        valid_bt = masks.any(dim=-1).any(dim=-1)

        val_H = torch.tensor(H, dtype=torch.int16, device=device)
        val_W = torch.tensor(W, dtype=torch.int16, device=device)
        val_neg1 = torch.tensor(-1, dtype=torch.int16, device=device)

        y_grid = torch.arange(H, device=device, dtype=torch.int16).view(1, 1, 1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=torch.int16).view(1, 1, 1, 1, W)

        ymin = torch.where(masks, y_grid, val_H).amin(dim=(3, 4))
        ymax = torch.where(masks, y_grid, val_neg1).amax(dim=(3, 4))

        xmin = torch.where(masks, x_grid, val_W).amin(dim=(3, 4))
        xmax = torch.where(masks, x_grid, val_neg1).amax(dim=(3, 4))

        true_area = masks.sum(dim=(3, 4), dtype=torch.int32)
        box_area = torch.clamp((xmax - xmin) * (ymax - ymin), min=1)

    for stride in [8, 16, 32]:
        H_feat, W_feat = H // stride, W // stride
        
        b_d = torch.zeros(B, T, 4, H_feat, W_feat, device=device)
        o_d = torch.zeros(B, T, 1, H_feat, W_feat, device=device)
        c_d = torch.zeros(B, T, 1, H_feat, W_feat, device=device)

        if max_uid > 0:
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
                if batch is None:
                    time.sleep(1)
                    continue
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
                time.sleep(1)

    def next(self):
        batch_gpu = self.queue.get()
        if self.stream is not None:
            torch.cuda.current_stream().wait_stream(self.stream)
            for k, v in batch_gpu.items():
                if isinstance(v, torch.Tensor):
                    v.record_stream(torch.cuda.current_stream())
        return batch_gpu

# =====================================================================
# 6. Loss 计算与提取工具
# =====================================================================
def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20):
    if isinstance(preds["objectness"], list):
        preds = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}

    B = preds["objectness"].shape[0]
    H_feat, W_feat = preds["objectness"].shape[2:]
    device = preds["objectness"].device
    H_img, W_img = H_feat * 8, W_feat * 8
    results = []

    for b in range(B):
        boxes = preds.get("boxes")
        if boxes is None:
            results.append(None)
            continue

        obj = preds["objectness"][b, 0]
        scores = torch.sigmoid(obj)
        valid = scores > score_thresh
        if not valid.any():
            results.append(None)
            continue

        sel_scores = scores[valid]
        decoded_boxes = boxes[b][:, valid].T
        cy, cx = valid.nonzero(as_tuple=True)

        grid_x_norm = (cx.float() * 8.0 + 4.0) / W_img
        grid_y_norm = (cy.float() * 8.0 + 4.0) / H_img

        pl_norm, pt_norm, pr_norm, pb_norm = (decoded_boxes[:, i] * 8.0 / d for i, d in enumerate([W_img, H_img, W_img, H_img]))

        decoded_boxes_norm = torch.stack([
            torch.clamp(grid_x_norm - pl_norm, 0.0, 1.0),
            torch.clamp(grid_y_norm - pt_norm, 0.0, 1.0),
            torch.clamp(grid_x_norm + pr_norm, 0.0, 1.0),
            torch.clamp(grid_y_norm + pb_norm, 0.0, 1.0)
        ], dim=-1)
        
        pixel_boxes = decoded_boxes_norm * torch.tensor([W_img, H_img, W_img, H_img], device=device)
        keep = torchvision.ops.nms(pixel_boxes, sel_scores, nms_thresh)[:max_det]

        coeffs, protos = preds.get("mask_coefficients"), preds.get("mask_prototypes")
        if coeffs is not None and protos is not None:
            kept_coeffs = coeffs[b, :, cy, cx].T[keep]
            masks = torch.einsum("kp,phw->khw", kept_coeffs, protos[b])
            masks = F.interpolate(masks.unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False)[0]

            boxes_pixel = pixel_boxes[keep]
            N_masks = masks.shape[0]
            rows = torch.arange(H_img, device=device).view(1, H_img, 1)
            cols = torch.arange(W_img, device=device).view(1, 1, W_img)
            x1, y1, x2, y2 = boxes_pixel.unbind(-1)
            
            mask_crop = ((cols >= x1.view(N_masks, 1, 1)) & (cols < x2.view(N_masks, 1, 1)) & 
                         (rows >= y1.view(N_masks, 1, 1)) & (rows < y2.view(N_masks, 1, 1)))
            masks_bool = (masks > 0) & mask_crop
        else:
            masks_bool = None

        classes = torch.argmax(preds["classification"][b, :, cy, cx].T, dim=-1)[keep] if "classification" in preds else None
        
        results.append({
            "scores": sel_scores[keep],
            "boxes": decoded_boxes_norm[keep],
            "masks": masks_bool,
            "classes": classes,
        })
    return results


def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(preds_logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    return (alpha * (1 - p_t) ** gamma * bce).mean()


def dfl_loss(pred_dist, target_distances, reg_max=16):
    target_left = torch.clamp(target_distances.long(), 0, reg_max - 1)
    target_right = torch.clamp(target_left + 1, 0, reg_max - 1)
    weight_left = target_right.float() - target_distances
    weight_right = 1.0 - weight_left

    pred_dist = pred_dist.reshape(-1, 4, reg_max)
    loss_left = F.cross_entropy(pred_dist.reshape(-1, reg_max), target_left.reshape(-1), reduction="none").reshape(weight_left.shape) * weight_left
    loss_right = F.cross_entropy(pred_dist.reshape(-1, reg_max), target_right.reshape(-1), reduction="none").reshape(weight_right.shape) * weight_right

    return (loss_left + loss_right).mean(dim=-1)


def giou_loss(preds, targets):
    pl, pt, pr, pb = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    tl, tt, tr, tb = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]

    inter_area = (torch.min(pl, tl) + torch.min(pr, tr)) * (torch.min(pt, tt) + torch.min(pb, tb))
    union_area = (pl + pr) * (pt + pb) + (tl + tr) * (tt + tb) - inter_area + 1e-6
    enclose_area = (torch.max(pl, tl) + torch.max(pr, tr)) * (torch.max(pt, tt) + torch.max(pb, tb)) + 1e-6

    return 1.0 - (inter_area / union_area - (enclose_area - union_area) / enclose_area)


def giou_loss_with_l1_warmup(preds, targets, step, warmup_steps=500):
    l1 = F.smooth_l1_loss(preds, targets, beta=1.0, reduction="none").mean(dim=-1)
    if step < warmup_steps:
        return l1
    return l1 * (1 - min((step - warmup_steps) / 1000.0, 1.0)) + giou_loss(preds, targets) * min((step - warmup_steps) / 1000.0, 1.0)


def ssim_loss(x, y):
    C1, C2 = 0.01**2, 0.03**2
    x_pad, y_pad = F.pad(x, (1, 1, 1, 1), mode="reflect"), F.pad(y, (1, 1, 1, 1), mode="reflect")
    mu_x, mu_y = F.avg_pool2d(x_pad, 3, 1), F.avg_pool2d(y_pad, 3, 1)
    sigma_x, sigma_y = F.avg_pool2d(x_pad**2, 3, 1) - mu_x**2, F.avg_pool2d(y_pad**2, 3, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(x_pad * y_pad, 3, 1) - mu_x * mu_y
    SSIM_n, SSIM_d = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2), (mu_x**2 + mu_y**2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def edge_aware_smoothness_loss(depth, img):
    norm_depth = (depth.float() / torch.clamp(depth.mean(dim=[2, 3], keepdim=True).float(), min=1e-4)).to(depth.dtype)
    grad_depth_x, grad_depth_y = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:]), torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    grad_img_x, grad_img_y = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True), torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    return (grad_depth_x * torch.exp(-grad_img_x)).mean() + (grad_depth_y * torch.exp(-grad_img_y)).mean()


def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    device = depth.device

    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
    pixels = torch.stack([x.flatten().expand(B, -1), y.flatten().expand(B, -1), torch.ones_like(x.flatten().expand(B, -1))], dim=1)

    points_3d = torch.bmm(K_inv.expand(B, 3, 3), pixels.float()) * depth.view(B, 1, H * W)
    points_3d_next = torch.bmm(six_d_to_matrix(pose[:, 3:]), points_3d) + pose[:, :3].unsqueeze(2)

    pixels_next = torch.bmm(K.expand(B, 3, 3), points_3d_next)
    z_next_safe = torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()
    x_norm = 2.0 * (pixels_next[:, 0:1, :].float() / z_next_safe) / (W - 1) - 1.0
    y_norm = 2.0 * (pixels_next[:, 1:2, :].float() / z_next_safe) / (H - 1) - 1.0

    grid = torch.clamp(torch.cat([x_norm, y_norm], dim=1).view(B, 2, H, W).permute(0, 2, 3, 1), -2.0, 2.0)
    warped_img = torch.nan_to_num(F.grid_sample(img_next, grid, mode="bilinear", padding_mode="border", align_corners=True), 0.0)

    valid_mask = (((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0)).view(B, 1, H, W).float() *
                  ((depth > 0.01) & (pixels_next[:, 2:3, :].view(B, 1, H, W) > 0.01)).float())
    return warped_img, valid_mask


def get_loss_weights(step):
    ramp = lambda s, e, v: 0.0 if step < s else (v if step > e else v * (step - s) / (e - s))
    return {
        "obj": 1.0, "box": 1.5, "mask": 1.0,
        "depth": 3.0 if step < 3000 else 1.5,
        "photo": ramp(1000, 3000, 1.0),
        "ego": ramp(100, 600, 3.0),
        "flow": ramp(300, 1000, 1.0),
        "cls": ramp(1000, 1001, 1.0),
        "anom": ramp(4000, 6000, 1.0),
        "smooth": 0.05, "gate": 0.05,
    }


LOSS_EMA = {}
def get_ema_loss(name, current_val, alpha=0.95):
    global LOSS_EMA
    with torch.no_grad():
        val = current_val.detach()
        if name not in LOSS_EMA:
            LOSS_EMA[name] = torch.tensor(1.0, device=val.device)
        if val > 0.0:
            LOSS_EMA[name] = LOSS_EMA[name] * alpha + val * (1.0 - alpha)
        return torch.clamp(LOSS_EMA[name], min=1e-4) if val > 0.0 else torch.tensor(1.0, device=val.device)


def compute_per_instance_mask_loss(preds, targets, pos_mask, key="mask_coefficients"):
    B, _, H_feat, W_feat = preds["objectness"].shape
    H, W = targets["seg_raw"].shape[1:3]
    stride = H // H_feat

    y_grid, x_grid = torch.meshgrid(torch.arange(H_feat, device=preds["objectness"].device), torch.arange(W_feat, device=preds["objectness"].device), indexing="ij")
    center_y, center_x = torch.clamp(y_grid * stride + stride // 2, 0, H - 1).unsqueeze(0).expand(B, -1, -1), torch.clamp(x_grid * stride + stride // 2, 0, W - 1).unsqueeze(0).expand(B, -1, -1)
    
    inst_ids = torch.gather(targets["seg_raw"].reshape(B, H * W), 1, (center_y * W + center_x).reshape(B, H_feat * W_feat)).reshape(B, H_feat, W_feat).long()
    pred_logits = torch.einsum("bchw,bcHW->bhwHW", preds[key], preds["mask_prototypes"])
    
    gt_masks = (targets["seg_small"].unsqueeze(1).unsqueeze(2) == inst_ids.view(B, H_feat, W_feat, 1, 1)).float()
    if gt_masks.shape[-2:] != pred_logits.shape[-2:]:
        gt_masks = F.interpolate(gt_masks.flatten(0, 2).unsqueeze(1), size=pred_logits.shape[-2:], mode="nearest").squeeze(1).view_as(pred_logits)

    preds_sig = torch.sigmoid(pred_logits)
    intersection, union = (preds_sig * gt_masks).sum(dim=(3, 4)), preds_sig.sum(dim=(3, 4)) + gt_masks.sum(dim=(3, 4))
    smooth = gt_masks.sum(dim=(3, 4)).clamp(min=1.0) * 0.01

    bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction="none")
    loss_bce = (0.25 * (1 - torch.exp(-bce)) ** 2 * bce).mean(dim=(3, 4))
    loss_mask_dense = (1.0 - (2.0 * intersection + smooth) / (union + smooth)) * 2.0 + loss_bce * 1.0
    
    valid_mask = (inst_ids > 0).float() * pos_mask.float()
    return (loss_mask_dense * valid_mask).sum() / valid_mask.sum().clamp(min=1.0)


def compute_instance_loss_single_scale(preds, targets, step=0):
    device, pos_mask = preds["objectness"].device, targets["obj_dense"][:, 0] > 0.5
    w = get_loss_weights(step)

    loss_obj = focal_loss(preds["objectness"], targets["obj_dense"]) + (focal_loss(preds["dense_objectness"], targets["obj_dense"]) * 0.5 if "dense_objectness" in preds else 0.0)

    loss_box = loss_mask = loss_cls = torch.tensor(0.0, device=device)

    if w["box"] > 0:
        pred_boxes, gt_boxes = preds["boxes"].permute(0, 2, 3, 1), targets["bboxes_dense"].permute(0, 2, 3, 1)
        loss_box_dense = giou_loss_with_l1_warmup(pred_boxes, gt_boxes, step) * 1.5 + dfl_loss(preds["box_dist"].permute(0, 2, 3, 1), gt_boxes, 32) * 0.5
        loss_box = (loss_box_dense * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

        if "dense_box_dist" in preds:
            pred_boxes_dense = decode_dfl_boxes(preds["dense_box_dist"], 32).permute(0, 2, 3, 1)
            loss_box_dense2 = giou_loss_with_l1_warmup(pred_boxes_dense, gt_boxes, step) * 1.5 + dfl_loss(preds["dense_box_dist"].permute(0, 2, 3, 1), gt_boxes, 32) * 0.5
            loss_box = (loss_box + (loss_box_dense2 * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)) * 0.5

    if w["mask"] > 0:
        loss_mask = compute_per_instance_mask_loss(preds, targets, pos_mask, "mask_coefficients")
        if "dense_mask_coefficients" in preds:
            loss_mask = (loss_mask + compute_per_instance_mask_loss(preds, targets, pos_mask, "dense_mask_coefficients")) * 0.5

    if w.get("cls", 0) > 0 and "dense_classification" in preds and "cls_dense" in targets:
        gt_cls = targets["cls_dense"][:, 0].long()
        loss_cls_dense = F.cross_entropy(preds["dense_classification"].permute(0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)
        loss_cls_o2o = F.cross_entropy(preds["classification"].permute(0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)
        loss_cls = ((loss_cls_dense + loss_cls_o2o) * 0.5 * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

    return loss_obj, loss_box, loss_mask, loss_cls


def compute_instance_loss(preds, targets, step=0):
    if not isinstance(preds["objectness"], list):
        return compute_instance_loss_single_scale(preds, targets, step)

    losses = [0.0, 0.0, 0.0, 0.0]
    num_scales = len(preds["objectness"])
    for i in range(num_scales):
        p_i = {k: (v[i] if isinstance(v, list) and len(v) == num_scales else v) for k, v in preds.items()}
        t_i = {k: (v[i] if isinstance(v, list) and len(v) == num_scales else v) for k, v in targets.items()}
        for j, l in enumerate(compute_instance_loss_single_scale(p_i, t_i, step)):
            losses[j] += l
    return tuple(losses)


def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", step=0):
    device, B, H, W = preds["depth"].device, *preds["depth"].shape
    w = get_loss_weights(step)

    loss_obj, loss_box, loss_mask, loss_cls = compute_instance_loss(preds, targets, step)
    loss_ego = loss_depth = loss_flow = loss_photo = loss_smooth = torch.tensor(0.0, device=device)
    warped_img = None

    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        c_mat_t, c_mat_n = quaternion_to_matrix(targets["cam_quat_t"]), quaternion_to_matrix(targets["cam_quat_next"])
        R_n_inv = c_mat_n.transpose(1, 2)
        gt_ego = torch.cat([torch.bmm(R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)).squeeze(-1), matrix_to_6d(torch.bmm(R_n_inv, c_mat_t))], dim=1)
        loss_ego = F.smooth_l1_loss(preds["ego_pose"], gt_ego)

        valid_depth_mask = (~targets["sky_mask"]).float()
        loss_depth = (F.smooth_l1_loss(preds["log_depth"], targets["log_depth"], reduction="none") * valid_depth_mask).sum() / valid_depth_mask.sum().clamp(min=1)
        
        grad_pred_x, grad_gt_x = preds["depth"][:, :, 1:] - preds["depth"][:, :, :-1], targets["depth"][:, :, 1:] - targets["depth"][:, :, :-1]
        grad_pred_y, grad_gt_y = preds["depth"][:, 1:, :] - preds["depth"][:, :-1, :], targets["depth"][:, 1:, :] - targets["depth"][:, :-1, :]
        mask_x, mask_y = valid_depth_mask[:, :, 1:] * valid_depth_mask[:, :, :-1], valid_depth_mask[:, 1:, :] * valid_depth_mask[:, :-1, :]
        
        loss_depth += 0.5 * (F.smooth_l1_loss(grad_pred_x * mask_x, grad_gt_x * mask_x, reduction="sum") + F.smooth_l1_loss(grad_pred_y * mask_y, grad_gt_y * mask_y, reduction="sum")) / valid_depth_mask.sum().clamp(min=1)

    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        raw_loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"], reduction="none")
        if "has_next" in targets:
            valid_flow_mask = targets["has_next"].view(-1, 1, 1, 1).float()
            loss_flow = (raw_loss_flow * valid_flow_mask).sum() / (valid_flow_mask.sum().clamp(min=1) * raw_loss_flow.shape[1] * raw_loss_flow.shape[2] * raw_loss_flow.shape[3])
        else:
            loss_flow = raw_loss_flow.mean()

    if img_t is not None:
        loss_smooth = edge_aware_smoothness_loss(preds["depth"].unsqueeze(1), img_t)

    if img_t is not None and img_next is not None:
        K, K_inv = generate_intrinsics(H, W, device)
        warped_img, valid_warp_mask = inverse_warp(img_next, preds["depth"].unsqueeze(1), preds["ego_pose"], K, K_inv)
        
        if w["photo"] > 0:
            photo_loss_fn = lambda p, t: 0.15 * F.l1_loss(p, t, reduction="none").mean(dim=1, keepdim=True) + 0.85 * ssim_loss(p, t).mean(dim=1, keepdim=True)
            warp_loss, identity_loss = photo_loss_fn(warped_img, img_t), photo_loss_fn(img_next, img_t)
            mask = valid_warp_mask * (1 - targets["sky_mask"].float().unsqueeze(1)) * (warp_loss < identity_loss).float() * (targets["has_next"].view(-1, 1, 1, 1).float() if "has_next" in targets else 1.0)
            loss_photo = (warp_loss * mask).sum() / mask.sum().clamp(min=1)

    loss_anom, loss_gate = preds["feature_error"].mean(), preds["state_update_gate"].abs().mean() * 0.01

    total_loss = (w.get("obj", 1.0) * (loss_obj / get_ema_loss("Obj", loss_obj)) + w.get("box", 0.0) * (loss_box / get_ema_loss("Box", loss_box)) +
                  w.get("mask", 0.0) * (loss_mask / get_ema_loss("Mask", loss_mask)) + w.get("depth", 1.0) * (loss_depth / get_ema_loss("Dep", loss_depth)) +
                  w.get("photo", 0.0) * (loss_photo / get_ema_loss("Pht", loss_photo)) + w.get("ego", 1.0) * (loss_ego / get_ema_loss("Ego", loss_ego)) +
                  w.get("flow", 0.0) * (loss_flow / get_ema_loss("Flw", loss_flow)) + w.get("anom", 0.0) * (loss_anom / get_ema_loss("Ano", loss_anom)) +
                  w.get("cls", 0.0) * (loss_cls / get_ema_loss("Cls", loss_cls)) + w.get("smooth", 0.05) * loss_smooth + w.get("gate", 0.05) * loss_gate)

    return total_loss, {k: v.detach() for k, v in zip(["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls", "Tot"], [loss_obj, loss_box, loss_mask, loss_depth, loss_photo, loss_ego, loss_flow, loss_anom, loss_gate, loss_cls, total_loss])}, warped_img

# =====================================================================
# 7. 权重加载与预处理
# =====================================================================
def load_yolo_backbone_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 权重文件 {checkpoint_path} 不存在，正在尝试自动从 GitHub 下载...")
        try:
            urllib.request.urlretrieve(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{checkpoint_path}", checkpoint_path)
            print(f"✅ 自动下载成功: {checkpoint_path}")
        except Exception as e:
            print(f"❌ 下载失败: {e}\n👉 请手动下载 {checkpoint_path} 并放置在项目根目录下。")
            return

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
    except Exception as e:
        print(f"⚠️ 加载权重失败: {e}")
        return

    target_state = model.state_dict()
    updates = {k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k: v for k, v in state_dict.items() if (k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k) in target_state and target_state[(k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k)].shape == v.shape}
    target_state.update(updates)
    model.load_state_dict(target_state)
    print(f"✅ 成功加载 YOLO 预训练权重: {checkpoint_path} (匹配了 {len(updates)} 个张量)")


def freeze_backbone(model):
    print("❄️ 正在冻结 YOLOE 分割模块 (保持其强大的 Zero-shot 基础能力)...")
    for param in model.segmenter.parameters():
        param.requires_grad = False


def setup_finetune_mode(model):
    freeze_backbone(model)
    for m in [model.depth_decoder, model.pose_head, model.conv_gru, model.conv_gru_p4, model.conv_gru_p5, model.feature_predictor, model.state_update_gate_head, model.flow_head]:
        for param in m.parameters():
            param.requires_grad = True

    if hasattr(model.segmenter, "model"):
        model.segmenter.model[-1].obj_proj.requires_grad_(True)
        model.segmenter.model[-1].one2one_obj_proj.requires_grad_(True)
        if hasattr(model.segmenter.model[-1], "class_prompts"):
            model.segmenter.model[-1].class_prompts.requires_grad_(True)

# =====================================================================
# 8. 主训练循环
# =====================================================================
def train_model(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    model = TAONot42VisionModel(base_channels=48, hidden_channels=768).to(device)
    if getattr(args, "compile_model", False) and hasattr(torch, "compile"):
        try:
            model.segmenter = torch.compile(model.segmenter, mode="reduce-overhead")
            model.depth_decoder = torch.compile(model.depth_decoder, mode="reduce-overhead")
            print("🚀 torch.compile 成功开启！")
        except Exception as e:
            print(f"⚠️ torch.compile 开启失败 (将使用正常模式): {e}")

    if args.yolo_weights:
        load_yolo_backbone_weights(model, args.yolo_weights)
        if args.freeze:
            freeze_backbone(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

    buffer = AsyncDataBuffer(split="train", max_buffer_size=args.max_buffer_size, batch_size=args.batch_size)
    prefetcher = CUDAPrefetcher(buffer, device, target_size=args.img_size)

    if args.use_wandb and wandb is not None:
        try:
            from google.colab import userdata
            if key := userdata.get("WANDB_API_KEY"): wandb.login(key=key)
        except Exception:
            pass
        wandb.init(project="tao-not-42", config=vars(args))

    model.train()
    mode = "supervised"
    print(f"\n🚀 开始 TAO-NOT-42 V12 训练 (Device: {device}, Mode: {mode})")

    global_step, start_time, best_loss, epochs_without_improvement = 0, time.time(), float("inf"), 0

    for epoch in range(1, args.epochs + 1):
        if args.finetune_after_epoch and epoch > args.finetune_after_epoch and mode == "supervised":
            mode = "self_supervised"
            setup_finetune_mode(model)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1)
            if scaler is not None: scaler = torch.amp.GradScaler(device.type)

        print(f"\n{'=' * 40}\n🌟 Epoch {epoch}/{args.epochs} [Mode: {mode}]\n{'=' * 40}")
        epoch_loss_sum = 0.0

        for _ in range(args.steps_per_epoch):
            batch = prefetcher.next()
            if batch is None: continue

            videos = batch["video"]
            b, t, c, h, w = videos.shape
            state = None

            for chunk_start in range(0, t, args.seq_len):
                chunk_end = min(chunk_start + args.seq_len, t)
                optimizer.zero_grad(set_to_none=True)
                loss_dict_acc = {k: 0.0 for k in ["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls"]}
                chunk_preds, chunk_targets, chunk_x_t, chunk_x_next = [], [], [], []

                chunk_steps = chunk_end - chunk_start
                chunk_videos = videos[:, chunk_start:chunk_end]
                
                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    with contextlib.nullcontext() if (mode == "supervised" and global_step >= args.unfreeze_step_1) else torch.no_grad():
                        f1_seq, f2_seq, p3_seq, p4_seq, p5_seq = [feat.view(b, chunk_steps, *feat.shape[1:]) for feat in model.extract_features(chunk_videos.reshape(b * chunk_steps, c, h, w))]

                for i_step, step in enumerate(range(chunk_start, chunk_end)):
                    x_t, x_next = chunk_videos[:, i_step], videos[:, step + 1] if step + 1 < t else chunk_videos[:, i_step]
                    dt_t = torch.full((b,), 1.0 / 24.0 if step > 0 else 0.0, device=device)

                    target_t = {k: (v if k == "is_dynamic" else ([x[:, step] for x in v] if isinstance(v, list) else (v[:, step] if v is not None else None))) for k, v in batch.items() if k not in ("video", "flow")}
                    target_t.update({
                        "flow_target": batch["flow"][:, step] if step + 1 < t else torch.zeros_like(batch["flow"][:, 0]),
                        "cam_pos_next": batch["cam_pos"][:, step + 1 if step + 1 < t else step],
                        "cam_quat_next": batch["cam_quat"][:, step + 1 if step + 1 < t else step],
                        "cam_pos_t": batch["cam_pos"][:, step], "cam_quat_t": batch["cam_quat"][:, step],
                        "has_next": torch.full((b,), step + 1 < t, device=device, dtype=torch.bool)
                    })
                    
                    if "cls_dense" in target_t and (global_step < 1000 or step < 2):
                        target_t["cls_dense"] = [torch.full_like(x, -100) for x in target_t["cls_dense"]] if isinstance(target_t["cls_dense"], list) else torch.full_like(target_t["cls_dense"], -100)

                    with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                        out = model.forward_physics(f1_seq[:, i_step], f2_seq[:, i_step], p3_seq[:, i_step], p4_seq[:, i_step], p5_seq[:, i_step], dt_t, global_step, state, get_loss_weights, (h, w))
                        state = out.pop("next_state")

                    chunk_preds.append(out)
                    chunk_targets.append(target_t)
                    chunk_x_t.append(x_t)
                    chunk_x_next.append(x_next)

                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    def batch_dicts(dict_list):
                        res = {}
                        for k, val in dict_list[0].items():
                            if val is None: res[k] = None
                            elif isinstance(val, list): res[k] = [torch.cat([d[k][i] for d in dict_list], dim=0) for i in range(len(val))]
                            elif val.dim() == 0: res[k] = torch.stack([d[k] for d in dict_list])
                            else: res[k] = torch.cat([d[k] for d in dict_list], dim=0)
                        return res

                    total_seq_loss, loss_dict, warped_img = compute_physics_loss(batch_dicts(chunk_preds), batch_dicts(chunk_targets), torch.cat(chunk_x_t, dim=0), torch.cat(chunk_x_next, dim=0), mode=mode, step=global_step)

                for k in loss_dict_acc: loss_dict_acc[k] += loss_dict[k] * chunk_steps

                if (global_step + 1) % args.vis_interval == 0:
                    if filepath := save_visualization(chunk_x_t[-1], chunk_targets[-1], chunk_preds[-1], global_step + 1, warped_img[-b:] if warped_img is not None else None):
                        if args.use_wandb and wandb is not None: wandb.log({"Visualization": wandb.Image(filepath)}, step=global_step)

                if scaler is not None:
                    scaler.scale(total_seq_loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    total_seq_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

                state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}

                if global_step == 500 and mode == "supervised":
                    print(f"\n🔓 [Step 500] 顿悟时刻：解冻分类 Prompt，开启运动定性！")
                    if hasattr(model.segmenter.model[-1], "class_prompts"): model.segmenter.model[-1].class_prompts.requires_grad = True

                if global_step == args.unfreeze_step_1 and mode == "supervised":
                    print(f"\n🔓 [Step {args.unfreeze_step_1}] 解冻 P5 高层语义 (Stage 5)...")
                    for name, param in model.segmenter.named_parameters():
                        if any(f"model.{i}." in name for i in range(20, 23)): param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\n🔓 [Step {args.unfreeze_step_2}] 解冻 P4 中层特征 (Stage 4)...")
                    for name, param in model.segmenter.named_parameters():
                        if any(f"model.{i}." in name for i in range(16, 20)): param.requires_grad = True

                global_step += 1
                epoch_loss_sum += total_seq_loss.item()

                if global_step % 10 == 0:
                    cs, get_val = chunk_steps, lambda k: (loss_dict_acc[k].item() if isinstance(loss_dict_acc[k], torch.Tensor) else loss_dict_acc[k]) / cs
                    print(f"[{time.time() - start_time:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{total_seq_loss.item():.4f} | "
                          f"Obj:{get_val('Obj'):.2f} Bx:{get_val('Box'):.2f} Msk:{get_val('Mask'):.2f} Dep:{get_val('Depth'):.2f} "
                          f"Pht:{get_val('Photo'):.2f} Ego:{get_val('Ego'):.2f} Flw:{get_val('Flow'):.2f} Ano:{get_val('Anom'):.2f} Cls:{get_val('Cls'):.2f}")

                    if args.use_wandb and wandb is not None:
                        wandb.log({**{f"Loss/{k}": get_val(k) for k in loss_dict_acc}, "Loss/Total": total_seq_loss.item(), "System/Step": global_step, "System/Epoch": epoch, "System/Mode": 0 if mode == "supervised" else 1, "System/Buffer_Size": len(buffer.buffer)}, step=global_step)

        avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
        print(f"\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
        torch.save(model.state_dict(), epoch_ckpt_path := args.checkpoint.replace(".pth", f"_epoch_{epoch}.pth"))
        print(f"💾 已保存: {epoch_ckpt_path}")

        if avg_epoch_loss < best_loss:
            best_loss, epochs_without_improvement = avg_epoch_loss, 0
            torch.save(model.state_dict(), args.checkpoint.replace(".pth", "_best.pth"))
            print(f"🌟 最佳模型 (Loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ 未优化 ({epochs_without_improvement}/{args.early_stop_patience})")

        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n🛑 早停触发！")
            break

    print(f"🎉 训练完成。最佳: {args.checkpoint.replace('.pth', '_best.pth')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TAO-NOT-42 V12 Training")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_buffer_size", type=int, default=64, help="异步流数据缓冲池大小")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--compile_model", action="store_true", default=False)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    parser.add_argument("--unfreeze_step_1", type=int, default=200)
    parser.add_argument("--unfreeze_step_2", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint", type=str, default="tao_not_42_weights.pth")
    parser.add_argument("--yolo_weights", type=str, default="yolo11s-seg.pt")
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--finetune_after_epoch", type=int, default=0, help="在第几个Epoch后开启自监督微调 (填0表示不开启)")
    args = parser.parse_args()

    try:
        train_model(args)
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")