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
    base_bgr = cv2.cvtColor((img_tensor * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = base_bgr.shape[:2]

    def add_title(img, text, pos=(10, 30)):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return img

    # --- Prediction ---
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

    # --- Ground Truth ---
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

    # --- 6-Grid Output ---
    hw, hh = W // 2, H // 2
    def prep_cell(img, title):
        img_res = cv2.resize(img, (hw, hh))
        cv2.putText(img_res, title, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return img_res

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

def autopad(k, p=None, d=1):
    if d > 1: k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None: p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2,
    ], dim=-1).view(*q.shape[:-1], 3, 3)

def matrix_to_6d(matrix): return matrix[..., :2].reshape(*matrix.shape[:-2], 6)

def six_d_to_matrix(d6):
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)

def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    return K, torch.inverse(K)

def depth_to_color(depth_map, d_min=None, d_max=None):
    d_min = d_min if d_min is not None else depth_map.min()
    d_max = d_max if d_max is not None else depth_map.max()
    d_norm = (depth_map - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(depth_map)
    return cv2.applyColorMap((np.clip(d_norm, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)

def decode_dfl_boxes(pred_dist, reg_max=16):
    if isinstance(pred_dist, list): return [decode_dfl_boxes(x, reg_max) for x in pred_dist]
    B, C, H, W = pred_dist.shape
    prob = F.softmax(pred_dist.view(B, 4, reg_max, H, W), dim=2)
    weights = torch.arange(reg_max, dtype=torch.float32, device=pred_dist.device)
    return (prob * weights.view(1, 1, reg_max, 1, 1)).sum(dim=2)

def concat_dicts(dict_list):
    """Utility to batch a list of dicts along dim 0."""
    res = {}
    for k, val in dict_list[0].items():
        if val is None: res[k] = None
        elif isinstance(val, list): res[k] = [torch.cat([d[k][i] for d in dict_list], dim=0) for i in range(len(val))]
        elif val.dim() == 0: res[k] = torch.stack([d[k] for d in dict_list])
        else: res[k] = torch.cat([d[k] for d in dict_list], dim=0)
    return res

# =====================================================================
# 2. 模型核心组件 (Blocks)
# =====================================================================
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x): return self.act(self.bn(self.conv(x)))

class YOLOConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x): return self.net(x)

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x): return torch.cat(x, self.d)

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5, shortcut=True, n=3):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (n + 1), c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.n, self.add = n, shortcut and c1 == c2

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(self.n))
        y = self.cv2(torch.cat(y, 1))
        return y + x if self.add else y

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        self.qkv = Conv(dim, dim + self.key_dim * num_heads * 2, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, H * W).split([self.key_dim, self.key_dim, self.head_dim], dim=2)
        attn = (q.transpose(-2, -1) @ k * self.scale).softmax(dim=-1)
        return self.proj((v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W)))

class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.add else self.attn(x)
        return x + self.ffn(x) if self.add else self.ffn(x)

class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = nn.Sequential(Conv(c1, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, self.c * 2, 1, 1))
        self.cv2 = nn.Sequential(Conv(self.c * 2, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, c1, 1, 1))
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        return self.cv2(torch.cat((a, self.m(b)), 1))

class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, c_, k[0], 1), Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3k(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1, self.cv2, self.cv3 = Conv(c1, self.c, 1, 1), Conv(c1, self.c, 1, 1), Conv(2 * self.c, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))

    def forward(self, x): return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, e2=1.0, g=1, shortcut=True, attn=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1, self.cv2 = Conv(c1, 2 * self.c, 1, 1), Conv((2 + n) * self.c, c2, 1, 1)
        self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, e2) if c3k else Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e2) for _ in range(n - (1 if attn else 0)))
        if attn: self.m.append(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m: y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

# =====================================================================
# 3. 物理与时间模块 (Time & Physics Modules)
# =====================================================================
class TimeAwareConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_frequencies=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.register_buffer("frequencies", 2.0 ** torch.arange(num_frequencies) * ((2.0 * torch.pi) / 16.0))
        self.time_mlp = nn.Sequential(nn.Linear(num_frequencies * 2, 64), nn.SiLU(), nn.Linear(64, hidden_channels * 2))
        gate_channels = input_channels + hidden_channels
        self.update_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)

    def forward(self, x, dt, state=None):
        state = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3]) if state is None else F.interpolate(state, size=x.shape[-2:], mode="bilinear", align_corners=False)
        scaled_time = dt.view(-1, 1) * self.frequencies.view(1, -1)
        gamma, beta = self.time_mlp(torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)).chunk(2, dim=-1)
        modulated_state = state * (gamma.view(-1, self.hidden_channels, 1, 1) + 1.0) + beta.view(-1, self.hidden_channels, 1, 1)

        gates_in = torch.cat([x, modulated_state], dim=1)
        update, reset = torch.sigmoid(self.update_gate(gates_in)), torch.sigmoid(self.reset_gate(gates_in))
        return (1.0 - update) * modulated_state + update * torch.tanh(self.candidate(torch.cat([x, reset * modulated_state], dim=1)))

class FlowDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_p3, ch_f2, 3))
        self.conv1 = YOLOConv(ch_f2 * 2, ch_f2, 3)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_f2, ch_f1, 3))
        self.conv2 = YOLOConv(ch_f1 * 2, ch_f1, 3)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_f1, ch_f1, 3))
        self.head = nn.Sequential(YOLOConv(ch_f1, ch_f1 // 2, 3), nn.Conv2d(ch_f1 // 2, 2, 3, padding=1))

    def forward(self, f1, f2, p3):
        return self.head(self.up3(self.conv2(torch.cat([self.up2(self.conv1(torch.cat([self.up1(p3), f2], dim=1))), f1], dim=1))))

class EgoPoseHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(in_channels, 64), nn.SiLU(), nn.Linear(64, 9))
        nn.init.zeros_(self.fc[-1].weight); nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        pose = self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
        rot_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=pose.device) + torch.tanh(pose[:, 3:]) * 0.5
        return torch.cat([torch.tanh(pose[:, :3]) * 5.0, rot_6d], dim=1)

class FeaturePredictorHead(nn.Module):
    def __init__(self, channels=256, action_dim=9):
        super().__init__()
        self.stem = YOLOConv(channels + action_dim, channels, 1)
        self.net = nn.Sequential(Bottleneck(channels, channels), Bottleneck(channels, channels), YOLOConv(channels, channels, 3))

    def forward(self, state, action):
        return self.net(self.stem(torch.cat([state, action.view(*action.shape, 1, 1).expand(-1, -1, state.shape[2], state.shape[3])], dim=1)))

class DepthDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_p3, ch_f2, 3))
        self.conv1 = YOLOConv(ch_f2 * 2, ch_f2, 3)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_f2, ch_f1, 3))
        self.conv2 = YOLOConv(ch_f1 * 2, ch_f1, 3)
        self.up3 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), YOLOConv(ch_f1, ch_f1, 3))
        self.depth_out = nn.Sequential(YOLOConv(ch_f1, ch_f1 // 2, 3), nn.Conv2d(ch_f1 // 2, 1, 3, padding=1))

    def forward(self, f1, f2, p3):
        return self.depth_out(self.up3(self.conv2(torch.cat([self.up2(self.conv1(torch.cat([self.up1(p3), f2], dim=1))), f1], dim=1))))

class YOLOESegment26(nn.Module):
    def __init__(self, nc=80, nm=32, npr=256, embed=512, reg_max=1, ch=()):
        super().__init__()
        self.proto = nn.Sequential(Conv(npr, npr, 3), nn.Upsample(scale_factor=2, mode="nearest"), Conv(npr, npr, 3), Conv(npr, nm, 1))
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], 1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], npr, 3)

        c2, c3, c5 = max(ch[0] // 4, 16), max(ch[0], min(nc, 100)), max(ch[0] // 4, nm)
        def build_heads(c_in, c_out): return nn.ModuleList(nn.Sequential(Conv(x, c_in, 3), Conv(c_in, c_in, 3), nn.Conv2d(c_in, c_out, 1)) for x in ch)
        self.cv2, self.one2one_cv2 = build_heads(c2, 4 * reg_max), build_heads(c2, 4 * reg_max)
        self.cv3, self.one2one_cv3 = build_heads(c3, embed), build_heads(c3, embed)
        self.cv5, self.one2one_cv5 = build_heads(c5, nm), build_heads(c5, nm)
        self.obj_proj, self.one2one_obj_proj = nn.ModuleList(nn.Conv2d(embed, 1, 1) for _ in ch), nn.ModuleList(nn.Conv2d(embed, 1, 1) for _ in ch)
        self.class_prompts = nn.Parameter(torch.randn(2, embed), requires_grad=False)

    def forward(self, x):
        feat = x[0]
        for i, m in enumerate(self.feat_refine): feat = feat + F.interpolate(m(x[i + 1]), size=feat.shape[2:], mode="nearest")
        proto_out = self.proto(self.feat_fuse(feat))
        
        boxes, scores, mc, boxes_o2o, scores_o2o, mc_o2o, obj, obj_o2o, cls, cls_o2o = ([] for _ in range(10))
        norm_prompts = F.normalize(self.class_prompts, p=2, dim=1)

        for i, f in enumerate(x):
            boxes.append(self.cv2[i](f)); scores.append(self.cv3[i](f)); mc.append(self.cv5[i](f))
            boxes_o2o.append(self.one2one_cv2[i](f)); scores_o2o.append(self.one2one_cv3[i](f)); mc_o2o.append(self.one2one_cv5[i](f))
            obj.append(self.obj_proj[i](scores[i])); obj_o2o.append(self.one2one_obj_proj[i](scores_o2o[i]))
            cls.append(torch.einsum("bchw,kc->bkhw", F.normalize(scores[i], p=2, dim=1), norm_prompts) * 10.0)
            cls_o2o.append(torch.einsum("bchw,kc->bkhw", F.normalize(scores_o2o[i], p=2, dim=1), norm_prompts) * 10.0)

        return {"features": x, "objectness": obj, "classification": cls, "boxes": boxes, "mask_coefficients": mc,
                "o2o_objectness": obj_o2o, "o2o_classification": cls_o2o, "o2o_boxes": boxes_o2o, "o2o_mask_coefficients": mc_o2o, "mask_prototypes": proto_out}

# =====================================================================
# 4. 主模型架构 (Vision Model)
# =====================================================================
class MyYOLOE(nn.Module):
    def __init__(self):
        super().__init__()
        c, n = lambda d: int(d * 0.5), lambda d: max(round(d * 0.5), 1)
        self.model = nn.Sequential(
            Conv(3, c(64), 3, 2), Conv(c(64), c(128), 3, 2), C3k2(c(128), c(256), n=n(2), c3k=False, e=0.25),
            Conv(c(256), c(256), 3, 2), C3k2(c(256), c(512), n=n(2), c3k=False, e=0.25), Conv(c(512), c(512), 3, 2),
            C3k2(c(512), c(512), n=n(2), c3k=True), Conv(c(512), c(1024), 3, 2), C3k2(c(1024), c(1024), n=n(2), c3k=True),
            SPPF(c(1024), c(1024), k=5, n=3), C2PSA(c(1024), c(1024), n=n(2), e=0.5), nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(1), C3k2(c(1024) + c(512), c(512), n=n(2), c3k=True), nn.Upsample(scale_factor=2.0, mode="nearest"),
            Concat(1), C3k2(c(512) + c(512), c(256), n=n(2), c3k=True), Conv(c(256), c(256), 3, 2),
            Concat(1), C3k2(c(256) + c(512), c(512), n=n(2), c3k=True), Conv(c(512), c(512), 3, 2),
            Concat(1), C3k2(c(512) + c(1024), c(1024), n=n(2), c3k=True, e=0.5, attn=True),
            YOLOESegment26(nc=80, nm=32, npr=256, embed=512, reg_max=32, ch=(128, 256, 512)),
        )
        for m in self.model: m.f = -1
        self.model[12].f, self.model[15].f, self.model[18].f, self.model[21].f, self.model[23].f = [-1, 6], [-1, 4], [-1, 13], [-1, 10], [16, 19, 22]

    def forward(self, x):
        y = []
        for i, m in enumerate(self.model):
            if i == 23: break
            x = m([y[j] for j in m.f] if isinstance(m.f, list) else x); y.append(x)
        return y[0], y[1], y[16], y[19], y[22]

class TAONot42VisionModel(nn.Module):
    def __init__(self, base_channels=48, hidden_channels=768):
        super().__init__()
        self.segmenter = MyYOLOE()
        self.depth_decoder, self.flow_head = DepthDecoder(128, 64, 32), FlowDecoder(128, 64, 32)
        self.conv_gru, self.conv_gru_p4, self.conv_gru_p5 = TimeAwareConvGRUCell(128, 128), TimeAwareConvGRUCell(256, 256), TimeAwareConvGRUCell(512, 512)
        self.pose_head, self.feature_predictor = EgoPoseHead(128), FeaturePredictorHead(128)
        self.state_update_gate_head = nn.Sequential(nn.Linear(129, 64), nn.SiLU(), nn.Linear(64, 1))

    def extract_features(self, peripheral): return self.segmenter(peripheral)

    def forward_physics(self, f1, f2, p3_fused, p4, p5, dt, step, state=None, get_loss_weights_fn=None, original_shape=None):
        b, state = f1.shape[0], state or {}
        h, w = original_shape if original_shape else (f1.shape[2] * 2, f1.shape[3] * 2)

        def update_gru(gru_cell, p_feat, gru_state):
            next_state = gru_cell(F.avg_pool2d(p_feat, 2, 2), dt, gru_state)
            return next_state, p_feat + F.interpolate(next_state, size=p_feat.shape[-2:], mode="bilinear", align_corners=False)

        next_gru, spatiotemporal_p3 = update_gru(self.conv_gru, p3_fused, state.get("gru"))
        next_gru_p4, spatiotemporal_p4 = update_gru(self.conv_gru_p4, p4, state.get("gru_p4"))
        next_gru_p5, spatiotemporal_p5 = update_gru(self.conv_gru_p5, p5, state.get("gru_p5"))

        preds = torch.utils.checkpoint.checkpoint(lambda p3, p4, p5: self.segmenter.model[-1]([p3, p4, p5]), spatiotemporal_p3, spatiotemporal_p4, spatiotemporal_p5, use_reentrant=False)

        depth_pred = torch.exp(torch.clamp(F.interpolate(self.depth_decoder(f1, f2, spatiotemporal_p3), size=(h, w), mode="bilinear", align_corners=False).squeeze(1), min=-4.6, max=4.6))
        ego_pose = self.pose_head(spatiotemporal_p3)
        lw = get_loss_weights_fn(step) if get_loss_weights_fn else {"flow": 1, "box": 1, "mask": 1, "anom": 1}
        
        gate = torch.sigmoid(self.state_update_gate_head(torch.cat([spatiotemporal_p3.mean(dim=[2, 3]), dt.view(-1, 1)], dim=-1))).view(-1, 1, 1, 1)
        mix_st = lambda o, n: o * (1.0 - gate) + n * gate if o is not None else n

        final_gru, final_gru_p4, final_gru_p5 = mix_st(state.get("gru"), next_gru), mix_st(state.get("gru_p4"), next_gru_p4), mix_st(state.get("gru_p5"), next_gru_p5)

        feat_err = F.smooth_l1_loss(self.feature_predictor(state.get("gru"), state.get("prev_ego", torch.zeros_like(ego_pose))), final_gru.detach(), reduction="none").mean(dim=1) if state.get("gru") is not None and lw["anom"] > 0 else torch.zeros(b, next_gru.shape[2], next_gru.shape[3], device=f1.device)

        return {
            "objectness": preds["o2o_objectness"], "classification": preds["o2o_classification"],
            "box_dist": preds["o2o_boxes"] if lw["box"] > 0 else None, "boxes": decode_dfl_boxes(preds["o2o_boxes"], 32) if lw["box"] > 0 else None,
            "mask_coefficients": preds["o2o_mask_coefficients"] if lw["mask"] > 0 else None, "mask_prototypes": preds["mask_prototypes"] if lw["mask"] > 0 else None,
            "depth": depth_pred, "log_depth": torch.log(depth_pred), "ego_pose": ego_pose,
            "flow": self.flow_head(f1, f2, spatiotemporal_p3) * 1.5 if lw["flow"] > 0 else None,
            "features": spatiotemporal_p3, "anomaly_map": feat_err, "feature_error": feat_err.mean(), "state_update_gate": gate.view(b),
            "next_state": {"gru": final_gru, "gru_p4": final_gru_p4, "gru_p5": final_gru_p5, "prev_ego": ego_pose},
            "dense_objectness": preds["objectness"], "dense_classification": preds["classification"], "dense_box_dist": preds["boxes"], "dense_mask_coefficients": preds["mask_coefficients"],
        }

# =====================================================================
# 5. 数据流加载 (Data Loader & Pipeline)
# =====================================================================
class AsyncDataBuffer:
    def __init__(self, split="train", max_buffer_size=64, batch_size=16):
        self.split, self.max_buffer_size, self.batch_size = split, max_buffer_size, batch_size
        self.buffer, self.lock = deque(maxlen=max_buffer_size), threading.Lock()
        self.has_data = threading.Condition(self.lock)
        threading.Thread(target=self._fetch_loop, daemon=True).start()

    def _fetch_loop(self):
        if not IN_COLAB: return
        ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", split=self.split, read_config=tfds.ReadConfig(interleave_cycle_length=16)).repeat()
        ds = ds.map(lambda x: {"video": x["video"], "segmentations": x["segmentations"], "depth": x["depth"], "forward_flow": x["forward_flow"], "cam_pos": x["camera"]["positions"], "cam_quat": x["camera"]["quaternions"], **({"is_dynamic": x["instances"]["is_dynamic"]} if "instances" in x and "is_dynamic" in x["instances"] else {})}, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

        for item in tfds.as_numpy(ds):
            p_item = {k: torch.from_numpy(item[k_i]).pin_memory() for k, k_i in [("video", "video"), ("cam_pos", "cam_pos"), ("cam_quat", "cam_quat")]}
            p_item.update({k: torch.from_numpy(item[k_i][..., 0]).pin_memory() for k, k_i in [("segmentation", "segmentations"), ("depth", "depth")]})
            if "is_dynamic" in item: p_item["is_dynamic"] = torch.from_numpy(item["is_dynamic"]).pin_memory()
            
            f_np = item["forward_flow"].astype(np.float32)
            if "metadata" in item and "forward_flow_range" in item["metadata"]:
                minv, maxv = item["metadata"]["forward_flow_range"]
                f_np = f_np / 65535.0 * (maxv - minv) + minv
            else: f_np = (f_np - 32768.0) / 64.0
            p_item["forward_flow"] = torch.from_numpy(f_np).pin_memory()

            with self.lock:
                self.buffer.append(p_item)
                self.has_data.notify_all()

    def get_batch(self):
        with self.lock:
            while len(self.buffer) < self.batch_size:
                if not IN_COLAB: return None
                self.has_data.wait(timeout=5.0)
            batch = random.sample(self.buffer, self.batch_size)
        return {k: [i.get(k) for i in batch] for k in ["video", "segmentation", "depth", "forward_flow", "cam_pos", "cam_quat", "is_dynamic"]}

def process_batch_on_gpu(batch, device, target_size=256):
    to_gpu = lambda k, dtype=None: torch.stack([x.to(device, non_blocking=True) for x in batch[k]]).to(dtype) if dtype else torch.stack([x.to(device, non_blocking=True) for x in batch[k]])
    video, depth_raw, seg_raw, flow_raw = to_gpu("video"), to_gpu("depth", torch.float32), to_gpu("segmentation"), to_gpu("forward_flow", torch.float32)
    cam_pos, cam_quat = to_gpu("cam_pos"), to_gpu("cam_quat")
    B, T = video.shape[:2]

    is_dyn_out = torch.stack([F.pad(x.to(device), (0, max(len(d) for d in batch["is_dynamic"]) - len(x))) for x in batch["is_dynamic"]]) if batch.get("is_dynamic") and batch["is_dynamic"][0] is not None else None

    depth_m = torch.clamp(depth_raw / 1000.0, 0.01, 100.0)
    depth_m[depth_raw == 0] = 100.0  # Sky mask applied later
    video_p = video.permute(0, 1, 4, 2, 3).float() / 255.0

    if video_p.shape[-1] != target_size:
        video_p = F.interpolate(video_p.flatten(0, 1), size=(target_size, target_size), mode="bilinear", align_corners=False).view(B, T, 3, target_size, target_size)
        seg = F.interpolate(seg_raw.float().flatten(0, 1).unsqueeze(1), size=(target_size, target_size), mode="nearest").view(B, T, target_size, target_size).long()
        depth_m = F.interpolate(depth_m.flatten(0, 1).unsqueeze(1), size=(target_size, target_size), mode="bilinear", align_corners=False).squeeze(1).view(B, T, target_size, target_size)
        sky_mask = F.interpolate((depth_raw == 0).float().flatten(0, 1).unsqueeze(1), size=(target_size, target_size), mode="nearest").squeeze(1).view(B, T, target_size, target_size).bool()
    else:
        seg, sky_mask = seg_raw.long(), depth_raw == 0

    flow_norm = torch.clamp(flow_raw * 2.0 / target_size, -1.5, 1.5).permute(0, 1, 4, 2, 3)
    if flow_norm.shape[-1] != target_size: flow_norm = F.interpolate(flow_norm.flatten(0, 1), size=(target_size, target_size), mode="bilinear", align_corners=False).view(B, T, 2, target_size, target_size)

    bboxes_dense, obj_dense, cls_dense = [], [], []
    
    # 消除 GPU-CPU 同步瓶颈：预设最大实例数，纯矩阵计算
    MAX_INSTANCES = 24 
    uids = torch.arange(1, MAX_INSTANCES + 1, device=device, dtype=torch.int16).view(-1, 1, 1, 1, 1)
    masks = seg.to(torch.int16).unsqueeze(0) == uids
    valid_bt = masks.any(dim=-1).any(dim=-1)

    y_grid = torch.arange(target_size, device=device, dtype=torch.int16).view(1, 1, 1, target_size, 1)
    x_grid = torch.arange(target_size, device=device, dtype=torch.int16).view(1, 1, 1, 1, target_size)

    ymin, ymax = torch.where(masks, y_grid, torch.tensor(target_size, dtype=torch.int16, device=device)).amin(dim=(3, 4)), torch.where(masks, y_grid, torch.tensor(-1, dtype=torch.int16, device=device)).amax(dim=(3, 4))
    xmin, xmax = torch.where(masks, x_grid, torch.tensor(target_size, dtype=torch.int16, device=device)).amin(dim=(3, 4)), torch.where(masks, x_grid, torch.tensor(-1, dtype=torch.int16, device=device)).amax(dim=(3, 4))
    true_area, box_area = masks.sum(dim=(3, 4), dtype=torch.int32), torch.clamp((xmax - xmin) * (ymax - ymin), min=1)

    for stride in [8, 16, 32]:
        H_f, W_f = target_size // stride, target_size // stride
        b_d, o_d, c_d = torch.zeros(B, T, 4, H_f, W_f, device=device), torch.zeros(B, T, 1, H_f, W_f, device=device), torch.zeros(B, T, 1, H_f, W_f, device=device)
        
        s_mask = (box_area < 32**2) if stride == 8 else ((box_area >= 32**2) & (box_area < 96**2) if stride == 16 else (box_area >= 96**2))
        n_idx, b_idx, t_idx = torch.where((true_area >= 10) & (box_area <= 4 * true_area) & valid_bt & s_mask)

        if len(n_idx) > 0:
            areas = box_area[n_idx, b_idx, t_idx]
            sort_idx = torch.argsort(areas, descending=True)
            n_idx, b_idx, t_idx = n_idx[sort_idx], b_idx[sort_idx], t_idx[sort_idx]
            
            cy, cx = torch.clamp(((ymin[n_idx, b_idx, t_idx] + ymax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, H_f - 1), torch.clamp(((xmin[n_idx, b_idx, t_idx] + xmax[n_idx, b_idx, t_idx]) / 2 / stride).long(), 0, W_f - 1)
            o_d[b_idx, t_idx, 0, cy, cx] = 1.0
            c_d[b_idx, t_idx, 0, cy, cx] = is_dyn_out[b_idx, n_idx.long()].float() if is_dyn_out is not None else 1.0

            gx, gy = cx.float() * stride + stride / 2.0, cy.float() * stride + stride / 2.0
            b_d[b_idx, t_idx, :, cy, cx] = torch.stack([
                torch.clamp((gx - xmin[n_idx, b_idx, t_idx].float()) / stride, min=1e-4),
                torch.clamp((gy - ymin[n_idx, b_idx, t_idx].float()) / stride, min=1e-4),
                torch.clamp((xmax[n_idx, b_idx, t_idx].float() - gx) / stride, min=1e-4),
                torch.clamp((ymax[n_idx, b_idx, t_idx].float() - gy) / stride, min=1e-4),
            ], dim=-1)
            
        bboxes_dense.append(b_d); obj_dense.append(o_d); cls_dense.append(c_d)

    return {
        "video": video_p, "seg_raw": seg, "depth": depth_m, "log_depth": torch.log(depth_m),
        "flow": flow_norm, "cam_pos": cam_pos, "cam_quat": cam_quat, "is_dynamic": is_dyn_out, "sky_mask": sky_mask,
        "seg_small": F.interpolate(seg.float().flatten(0, 1).unsqueeze(1), size=(target_size // 8, target_size // 8), mode="nearest").squeeze(1).view(B, T, target_size // 8, target_size // 8),
        "bboxes_dense": bboxes_dense, "obj_dense": obj_dense, "cls_dense": cls_dense,
    }

class CUDAPrefetcher:
    def __init__(self, buffer, device, target_size=256):
        self.buffer, self.device, self.target_size = buffer, device, target_size
        self.queue, self.stream = queue.Queue(maxsize=4), torch.cuda.Stream(device=device) if device.type == "cuda" else None
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            if (batch := self.buffer.get_batch()) is None: time.sleep(1); continue
            try:
                if self.stream:
                    with torch.cuda.stream(self.stream): batch_gpu = process_batch_on_gpu(batch, self.device, self.target_size)
                else: batch_gpu = process_batch_on_gpu(batch, self.device, self.target_size)
                self.queue.put(batch_gpu)
            except Exception as e: print(f"Prefetcher err: {e}"); time.sleep(1)

    def next(self):
        batch = self.queue.get()
        if self.stream:
            torch.cuda.current_stream().wait_stream(self.stream)
            for v in batch.values():
                if isinstance(v, torch.Tensor): v.record_stream(torch.cuda.current_stream())
        return batch

# =====================================================================
# 6. Loss 计算工具
# =====================================================================
def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20):
    preds = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
    B, device, H_img, W_img = preds["objectness"].shape[0], preds["objectness"].device, preds["objectness"].shape[2] * 8, preds["objectness"].shape[3] * 8
    results = []

    for b in range(B):
        if (boxes := preds.get("boxes")) is None or not (valid := torch.sigmoid(preds["objectness"][b, 0]) > score_thresh).any():
            results.append(None); continue

        sel_scores, decoded_boxes, (cy, cx) = torch.sigmoid(preds["objectness"][b, 0])[valid], boxes[b][:, valid].T, valid.nonzero(as_tuple=True)
        grid_x_norm, grid_y_norm = (cx.float() * 8.0 + 4.0) / W_img, (cy.float() * 8.0 + 4.0) / H_img
        pl_norm, pt_norm, pr_norm, pb_norm = (decoded_boxes[:, i] * 8.0 / d for i, d in enumerate([W_img, H_img, W_img, H_img]))

        decoded_boxes_norm = torch.stack([torch.clamp(grid_x_norm - pl_norm, 0.0, 1.0), torch.clamp(grid_y_norm - pt_norm, 0.0, 1.0), torch.clamp(grid_x_norm + pr_norm, 0.0, 1.0), torch.clamp(grid_y_norm + pb_norm, 0.0, 1.0)], dim=-1)
        keep = torchvision.ops.nms(decoded_boxes_norm * torch.tensor([W_img, H_img, W_img, H_img], device=device), sel_scores, nms_thresh)[:max_det]

        coeffs, protos = preds.get("mask_coefficients"), preds.get("mask_prototypes")
        if coeffs is not None and protos is not None:
            masks = F.interpolate(torch.einsum("kp,phw->khw", coeffs[b, :, cy, cx].T[keep], protos[b]).unsqueeze(0), size=(H_img, W_img), mode="bilinear", align_corners=False)[0]
            x1, y1, x2, y2 = (decoded_boxes_norm[keep] * torch.tensor([W_img, H_img, W_img, H_img], device=device)).unbind(-1)
            rows, cols = torch.arange(H_img, device=device).view(1, H_img, 1), torch.arange(W_img, device=device).view(1, 1, W_img)
            masks_bool = (masks > 0) & ((cols >= x1.view(-1, 1, 1)) & (cols < x2.view(-1, 1, 1)) & (rows >= y1.view(-1, 1, 1)) & (rows < y2.view(-1, 1, 1)))
        else: masks_bool = None

        results.append({"scores": sel_scores[keep], "boxes": decoded_boxes_norm[keep], "masks": masks_bool, "classes": torch.argmax(preds["classification"][b, :, cy, cx].T, dim=-1)[keep] if "classification" in preds else None})
    return results

def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(preds_logits, targets, reduction="none")
    return (alpha * (1 - torch.exp(-bce)) ** gamma * bce).mean()

def dfl_loss(pred_dist, target_distances, reg_max=16):
    tl, tr = torch.clamp(target_distances.long(), 0, reg_max - 1), torch.clamp(target_distances.long() + 1, 0, reg_max - 1)
    wl, wr = tr.float() - target_distances, 1.0 - (tr.float() - target_distances)
    pred_dist = pred_dist.reshape(-1, 4, reg_max)
    return (F.cross_entropy(pred_dist.reshape(-1, reg_max), tl.reshape(-1), reduction="none").reshape(wl.shape) * wl + 
            F.cross_entropy(pred_dist.reshape(-1, reg_max), tr.reshape(-1), reduction="none").reshape(wr.shape) * wr).mean(dim=-1)

def giou_loss(preds, targets):
    pl, pt, pr, pb = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
    tl, tt, tr, tb = targets[:, 0], targets[:, 1], targets[:, 2], targets[:, 3]
    inter_area = (torch.min(pl, tl) + torch.min(pr, tr)) * (torch.min(pt, tt) + torch.min(pb, tb))
    union_area = (pl + pr) * (pt + pb) + (tl + tr) * (tt + tb) - inter_area + 1e-6
    return 1.0 - (inter_area / union_area - ((torch.max(pl, tl) + torch.max(pr, tr)) * (torch.max(pt, tt) + torch.max(pb, tb)) + 1e-6 - union_area) / ((torch.max(pl, tl) + torch.max(pr, tr)) * (torch.max(pt, tt) + torch.max(pb, tb)) + 1e-6))

def ssim_loss(x, y):
    mu_x, mu_y = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect"), 3, 1), F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="reflect"), 3, 1)
    sigma_x, sigma_y = F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect")**2, 3, 1) - mu_x**2, F.avg_pool2d(F.pad(y, (1, 1, 1, 1), mode="reflect")**2, 3, 1) - mu_y**2
    return torch.clamp((1 - (2 * mu_x * mu_y + 0.01**2) * (2 * (F.avg_pool2d(F.pad(x, (1, 1, 1, 1), mode="reflect") * F.pad(y, (1, 1, 1, 1), mode="reflect"), 3, 1) - mu_x * mu_y) + 0.03**2) / ((mu_x**2 + mu_y**2 + 0.01**2) * (sigma_x + sigma_y + 0.03**2))) / 2, 0, 1)

def edge_aware_smoothness_loss(depth, img):
    norm_depth = (depth.float() / torch.clamp(depth.mean(dim=[2, 3], keepdim=True).float(), min=1e-4)).to(depth.dtype)
    return ((torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:]) * torch.exp(-torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True))).mean() + 
            (torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :]) * torch.exp(-torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True))).mean())

def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(W, device=depth.device), indexing="ij")
    pixels = torch.stack([x.flatten().expand(B, -1), y.flatten().expand(B, -1), torch.ones_like(x.flatten().expand(B, -1))], dim=1)
    pixels_next = torch.bmm(K.expand(B, 3, 3), torch.bmm(six_d_to_matrix(pose[:, 3:]), torch.bmm(K_inv.expand(B, 3, 3), pixels.float()) * depth.view(B, 1, H * W)) + pose[:, :3].unsqueeze(2))
    
    x_n, y_n = 2.0 * (pixels_next[:, 0:1, :].float() / torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()) / (W - 1) - 1.0, 2.0 * (pixels_next[:, 1:2, :].float() / torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()) / (H - 1) - 1.0
    return torch.nan_to_num(F.grid_sample(img_next, torch.clamp(torch.cat([x_n, y_n], dim=1).view(B, 2, H, W).permute(0, 2, 3, 1), -2.0, 2.0), mode="bilinear", padding_mode="border", align_corners=True), 0.0), (((x_n > -1.0) & (x_n < 1.0) & (y_n > -1.0) & (y_n < 1.0)).view(B, 1, H, W).float() * ((depth > 0.01) & (pixels_next[:, 2:3, :].view(B, 1, H, W) > 0.01)).float())

def get_loss_weights(step):
    ramp = lambda s, e, v: 0.0 if step < s else (v if step > e else v * (step - s) / (e - s))
    return {"obj": 1.0, "box": 1.5, "mask": 1.0, "depth": 3.0 if step < 3000 else 1.5, "photo": ramp(1000, 3000, 1.0), "ego": ramp(100, 600, 3.0), "flow": ramp(300, 1000, 1.0), "cls": ramp(1000, 1001, 1.0), "anom": ramp(4000, 6000, 1.0), "smooth": 0.05, "gate": 0.05}

LOSS_EMA = {}
def get_ema_loss(name, current_val, alpha=0.95):
    global LOSS_EMA
    with torch.no_grad():
        val = current_val.detach()
        if name not in LOSS_EMA: LOSS_EMA[name] = torch.tensor(1.0, device=val.device)
        if val > 0.0: LOSS_EMA[name] = LOSS_EMA[name] * alpha + val * (1.0 - alpha)
        return torch.clamp(LOSS_EMA[name], min=1e-4) if val > 0.0 else torch.tensor(1.0, device=val.device)

def compute_instance_loss(preds, targets, step):
    B, device, num_scales = preds["objectness"][0].shape[0], preds["objectness"][0].device, len(preds["objectness"])
    loss_obj = loss_box = loss_mask = loss_cls = torch.tensor(0.0, device=device)
    w = get_loss_weights(step)

    for i in range(num_scales):
        p_obj, t_obj = preds["objectness"][i], targets["obj_dense"][i]
        loss_obj += focal_loss(p_obj, t_obj) + (focal_loss(preds["dense_objectness"][i], t_obj) * 0.5 if "dense_objectness" in preds else 0.0)
        pos_mask = t_obj[:, 0] > 0.5

        if w["box"] > 0:
            pb, tb, pdist = preds["boxes"][i].permute(0, 2, 3, 1), targets["bboxes_dense"][i].permute(0, 2, 3, 1), preds["box_dist"][i].permute(0, 2, 3, 1)
            l1_w = min(1.0, max(0.0, (step - 500) / 1000.0))
            giou = F.smooth_l1_loss(pb, tb, beta=1.0, reduction="none").mean(dim=-1) * (1 - l1_w) + giou_loss(pb, tb) * l1_w if step >= 500 else F.smooth_l1_loss(pb, tb, beta=1.0, reduction="none").mean(dim=-1)
            loss_box += ((giou * 1.5 + dfl_loss(pdist, tb, 32) * 0.5) * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

        if w["mask"] > 0:
            H_feat, W_feat, H, W = p_obj.shape[2], p_obj.shape[3], targets["seg_raw"].shape[1], targets["seg_raw"].shape[2]
            y_g, x_g = torch.meshgrid(torch.arange(H_feat, device=device), torch.arange(W_feat, device=device), indexing="ij")
            inst_ids = torch.gather(targets["seg_raw"].reshape(B, H * W), 1, (torch.clamp(y_g * (H // H_feat) + (H // H_feat) // 2, 0, H - 1).unsqueeze(0).expand(B, -1, -1) * W + torch.clamp(x_g * (H // H_feat) + (H // H_feat) // 2, 0, W - 1).unsqueeze(0).expand(B, -1, -1)).reshape(B, H_feat * W_feat)).reshape(B, H_feat, W_feat).long()
            pred_logits = torch.einsum("bchw,bcHW->bhwHW", preds["mask_coefficients"][i], preds["mask_prototypes"])
            gt_masks = (targets["seg_small"].unsqueeze(1).unsqueeze(2) == inst_ids.view(B, H_feat, W_feat, 1, 1)).float()
            if gt_masks.shape[-2:] != pred_logits.shape[-2:]: gt_masks = F.interpolate(gt_masks.flatten(0, 2).unsqueeze(1), size=pred_logits.shape[-2:], mode="nearest").squeeze(1).view_as(pred_logits)
            
            intersection, union = (torch.sigmoid(pred_logits) * gt_masks).sum(dim=(3, 4)), torch.sigmoid(pred_logits).sum(dim=(3, 4)) + gt_masks.sum(dim=(3, 4))
            bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks, reduction="none")
            loss_mask += (((1.0 - (2.0 * intersection + gt_masks.sum(dim=(3, 4)).clamp(min=1.0) * 0.01) / (union + gt_masks.sum(dim=(3, 4)).clamp(min=1.0) * 0.01)) * 2.0 + (0.25 * (1 - torch.exp(-bce)) ** 2 * bce).mean(dim=(3, 4))) * ((inst_ids > 0).float() * pos_mask.float())).sum() / ((inst_ids > 0).float() * pos_mask.float()).sum().clamp(min=1.0)

        if w.get("cls", 0) > 0 and "dense_classification" in preds and "cls_dense" in targets:
            gt_cls = targets["cls_dense"][i][:, 0].long()
            loss_cls += ((F.cross_entropy(preds["dense_classification"][i].permute(0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask) + F.cross_entropy(preds["classification"][i].permute(0, 2, 3, 1).flatten(0, 2), gt_cls.flatten(0, 2), reduction="none").view_as(pos_mask)) * 0.5 * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)

    return loss_obj, loss_box, loss_mask, loss_cls

def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", step=0):
    device, B, H, W = preds["depth"].device, *preds["depth"].shape
    w = get_loss_weights(step)
    loss_obj, loss_box, loss_mask, loss_cls = compute_instance_loss(preds, targets, step)
    loss_ego, loss_depth, loss_flow, loss_photo, loss_smooth = (torch.tensor(0.0, device=device) for _ in range(5))
    warped_img = None

    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        R_n_inv = quaternion_to_matrix(targets["cam_quat_next"]).transpose(1, 2)
        loss_ego = F.smooth_l1_loss(preds["ego_pose"], torch.cat([torch.bmm(R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)).squeeze(-1), matrix_to_6d(torch.bmm(R_n_inv, quaternion_to_matrix(targets["cam_quat_t"])))], dim=1))
        v_d_mask = (~targets["sky_mask"]).float()
        loss_depth = (F.smooth_l1_loss(preds["log_depth"], targets["log_depth"], reduction="none") * v_d_mask).sum() / v_d_mask.sum().clamp(min=1) + 0.5 * (F.smooth_l1_loss((preds["depth"][:, :, 1:] - preds["depth"][:, :, :-1]) * v_d_mask[:, :, 1:] * v_d_mask[:, :, :-1], (targets["depth"][:, :, 1:] - targets["depth"][:, :, :-1]) * v_d_mask[:, :, 1:] * v_d_mask[:, :, :-1], reduction="sum") + F.smooth_l1_loss((preds["depth"][:, 1:, :] - preds["depth"][:, :-1, :]) * v_d_mask[:, 1:, :] * v_d_mask[:, :-1, :], (targets["depth"][:, 1:, :] - targets["depth"][:, :-1, :]) * v_d_mask[:, 1:, :] * v_d_mask[:, :-1, :], reduction="sum")) / v_d_mask.sum().clamp(min=1)

    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        loss_flow = (F.smooth_l1_loss(preds["flow"], targets["flow_target"], reduction="none") * targets["has_next"].view(-1, 1, 1, 1).float()).sum() / (targets["has_next"].view(-1, 1, 1, 1).float().sum().clamp(min=1) * preds["flow"].shape[1] * H * W) if "has_next" in targets else F.smooth_l1_loss(preds["flow"], targets["flow_target"])

    if img_t is not None:
        loss_smooth = edge_aware_smoothness_loss(preds["depth"].unsqueeze(1), img_t)
        if img_next is not None:
            K, K_inv = generate_intrinsics(H, W, device)
            warped_img, v_w_mask = inverse_warp(img_next, preds["depth"].unsqueeze(1), preds["ego_pose"], K, K_inv)
            if w["photo"] > 0:
                p_loss = lambda p, t: 0.15 * F.l1_loss(p, t, reduction="none").mean(dim=1, keepdim=True) + 0.85 * ssim_loss(p, t).mean(dim=1, keepdim=True)
                w_loss, m = p_loss(warped_img, img_t), v_w_mask * (1 - targets["sky_mask"].float().unsqueeze(1)) * (p_loss(warped_img, img_t) < p_loss(img_next, img_t)).float() * (targets["has_next"].view(-1, 1, 1, 1).float() if "has_next" in targets else 1.0)
                loss_photo = (w_loss * m).sum() / m.sum().clamp(min=1)

    loss_anom, loss_gate = preds["feature_error"].mean(), preds["state_update_gate"].abs().mean() * 0.01

    tot = sum(w.get(k, 0) * (l / get_ema_loss(k.capitalize()[:3], l)) for k, l in zip(["obj", "box", "mask", "depth", "photo", "ego", "flow", "anom", "cls"], [loss_obj, loss_box, loss_mask, loss_depth, loss_photo, loss_ego, loss_flow, loss_anom, loss_cls])) + w.get("smooth", 0.05) * loss_smooth + w.get("gate", 0.05) * loss_gate
    return tot, {k: v.detach() for k, v in zip(["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls", "Tot"], [loss_obj, loss_box, loss_mask, loss_depth, loss_photo, loss_ego, loss_flow, loss_anom, loss_gate, loss_cls, tot])}, warped_img

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
                for param in self.model.segmenter.parameters(): param.requires_grad = False
                
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.lr)
        self.scaler = torch.amp.GradScaler(self.device.type) if self.device.type == "cuda" else None
        self.global_step, self.start_time, self.best_loss, self.epochs_no_improve = 0, time.time(), float("inf"), 0
        self.mode = "supervised"

    def _load_yolo_weights(self):
        if not os.path.exists(self.args.yolo_weights):
            urllib.request.urlretrieve(f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{self.args.yolo_weights}", self.args.yolo_weights)
        ckpt = torch.load(self.args.yolo_weights, map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
        tgt = self.model.state_dict()
        tgt.update({k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k: v for k, v in sd.items() if (k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k) in tgt and tgt[(k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k)].shape == v.shape})
        self.model.load_state_dict(tgt)

    def _setup_finetune(self):
        for param in self.model.segmenter.parameters(): param.requires_grad = False
        for m in [self.model.depth_decoder, self.model.pose_head, self.model.conv_gru, self.model.conv_gru_p4, self.model.conv_gru_p5, self.model.feature_predictor, self.model.state_update_gate_head, self.model.flow_head]:
            for p in m.parameters(): p.requires_grad = True
        self.model.segmenter.model[-1].obj_proj.requires_grad_(True); self.model.segmenter.model[-1].one2one_obj_proj.requires_grad_(True)
        if hasattr(self.model.segmenter.model[-1], "class_prompts"): self.model.segmenter.model[-1].class_prompts.requires_grad_(True)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.args.lr * 0.1)

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            if self.args.finetune_after_epoch and epoch > self.args.finetune_after_epoch and self.mode == "supervised":
                self.mode = "self_supervised"
                self._setup_finetune()

            epoch_loss = self._train_epoch(epoch)
            print(f"\n✅ Epoch {epoch} End | Avg Loss: {epoch_loss:.4f} | Mode: {self.mode}")
            torch.save(self.model.state_dict(), self.args.checkpoint.replace(".pth", f"_epoch_{epoch}.pth"))

            if epoch_loss < self.best_loss:
                self.best_loss, self.epochs_no_improve = epoch_loss, 0
                torch.save(self.model.state_dict(), self.args.checkpoint.replace(".pth", "_best.pth"))
                print(f"🌟 Best Model saved (Loss: {self.best_loss:.4f})")
            else:
                self.epochs_no_improve += 1
                if self.epochs_no_improve >= self.args.early_stop_patience:
                    print(f"\n🛑 Early Stopping Triggered!")
                    break

    def _train_epoch(self, epoch):
        loss_sum = 0.0
        for _ in range(self.args.steps_per_epoch):
            if (batch := self.prefetcher.next()) is None: continue
            loss_sum += self._train_chunk(batch)
            if self.global_step == 500 and self.mode == "supervised" and hasattr(self.model.segmenter.model[-1], "class_prompts"):
                self.model.segmenter.model[-1].class_prompts.requires_grad = True
            if self.mode == "supervised" and self.global_step in [self.args.unfreeze_step_1, self.args.unfreeze_step_2]:
                for n, p in self.model.segmenter.named_parameters():
                    if any(f"model.{i}." in n for i in (range(20, 23) if self.global_step == self.args.unfreeze_step_1 else range(16, 20))): p.requires_grad = True
        return loss_sum / self.args.steps_per_epoch

    def _extract_target_t(self, batch, step, max_t):
        tgt = {k: (v if k == "is_dynamic" else ([x[:, step] for x in v] if isinstance(v, list) else (v[:, step] if v is not None else None))) for k, v in batch.items() if k not in ("video", "flow")}
        tgt.update({"flow_target": batch["flow"][:, step] if step + 1 < max_t else torch.zeros_like(batch["flow"][:, 0]), "cam_pos_next": batch["cam_pos"][:, step + 1 if step + 1 < max_t else step], "cam_quat_next": batch["cam_quat"][:, step + 1 if step + 1 < max_t else step], "cam_pos_t": batch["cam_pos"][:, step], "cam_quat_t": batch["cam_quat"][:, step], "has_next": torch.full((batch["video"].shape[0],), step + 1 < max_t, device=self.device, dtype=torch.bool)})
        if "cls_dense" in tgt and (self.global_step < 1000 or step < 2): tgt["cls_dense"] = [torch.full_like(x, -100) for x in tgt["cls_dense"]] if isinstance(tgt["cls_dense"], list) else torch.full_like(tgt["cls_dense"], -100)
        return tgt

    def _train_chunk(self, batch):
        v_seq, t_max = batch["video"], batch["video"].shape[1]
        state, loss_acc = None, {k: 0.0 for k in ["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate", "Cls"]}
        total_loss = 0.0

        for c_start in range(0, t_max, self.args.seq_len):
            c_end = min(c_start + self.args.seq_len, t_max)
            c_vids = v_seq[:, c_start:c_end]
            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=self.device.type, enabled=(self.scaler is not None)):
                with contextlib.nullcontext() if (self.mode == "supervised" and self.global_step >= self.args.unfreeze_step_1) else torch.no_grad():
                    feats = [f.view(v_seq.shape[0], c_end - c_start, *f.shape[1:]) for f in self.model.extract_features(c_vids.reshape(-1, *c_vids.shape[2:]))]
                
                c_preds, c_tgts = [], []
                for i_step, step in enumerate(range(c_start, c_end)):
                    dt = torch.full((v_seq.shape[0],), 1.0 / 24.0 if step > 0 else 0.0, device=self.device)
                    out = self.model.forward_physics(*(f[:, i_step] for f in feats), dt, self.global_step, state, get_loss_weights, c_vids.shape[-2:])
                    state = out.pop("next_state")
                    c_preds.append(out); c_tgts.append(self._extract_target_t(batch, step, t_max))
                
                loss, l_dict, w_img = compute_physics_loss(concat_dicts(c_preds), concat_dicts(c_tgts), c_vids.flatten(0, 1), torch.cat([v_seq[:, min(s+1, t_max-1)] for s in range(c_start, c_end)], dim=0), self.mode, self.global_step)

            if self.scaler:
                self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); self.scaler.step(self.optimizer); self.scaler.update()
            else:
                loss.backward(); torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0); self.optimizer.step()

            state = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in state.items()}
            total_loss += loss.item()
            for k in loss_acc: loss_acc[k] += l_dict[k] * (c_end - c_start)

            if (self.global_step + 1) % self.args.vis_interval == 0:
                fp = save_visualization(c_vids[:, -1], c_tgts[-1], c_preds[-1], self.global_step + 1, w_img[-v_seq.shape[0]:] if w_img is not None else None)
                if wandb and fp: wandb.log({"Vis": wandb.Image(fp)}, step=self.global_step)
            
            self.global_step += 1
            if self.global_step % 10 == 0:
                print(f"[{time.time()-self.start_time:.1f}s] S{self.global_step} | Tot:{loss.item():.4f} | " + " ".join([f"{k}:{loss_acc[k]/(c_end-c_start):.2f}" for k in ["Obj", "Box", "Mask", "Depth", "Ego", "Flow", "Anom"]]))
                if wandb: wandb.log({**{f"Loss/{k}": loss_acc[k]/(c_end-c_start) for k in loss_acc}, "Loss/Total": loss.item(), "Step": self.global_step}, step=self.global_step)

        return total_loss

# =====================================================================
# Main 参数配置与执行入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--max_buffer_size", type=int, default=64)
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
    parser.add_argument("--finetune_after_epoch", type=int, default=0)
    args = parser.parse_args()

    if args.use_wandb and wandb:
        wandb.init(project="tao_not_42", config=vars(args))
    elif not args.use_wandb:
        wandb = None

    try:
        TAOTrainer(args, TAONot42VisionModel(), AsyncDataBuffer(max_buffer_size=args.max_buffer_size, batch_size=args.batch_size), CUDAPrefetcher(AsyncDataBuffer(max_buffer_size=args.max_buffer_size, batch_size=args.batch_size), torch.device(args.device), args.img_size)).train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")