"""
TAO-NOT-42 on Kubric MOVi-E (TFDS 官方极速直读版)
-- V12：完整重构 —— 逐实例分割 + 多帧深度 + 两阶段自监督 --
"""
import os
import argparse
import time
import random
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# =====================================================================
# 启用极致 TF32 计算加速
# =====================================================================
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
try:
    torch.set_float32_matmul_precision('high')
except Exception:
    pass
import threading
from collections import deque
import wandb
# =====================================================================
# GPU 环境配置
# =====================================================================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], 'GPU')
    except RuntimeError:
        pass
    import tensorflow_datasets as tfds

def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2 = x * x, y * y, z * z
    w2 = w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    matrix = torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2
    ], dim=-1).view(*q.shape[:-1], 3, 3)
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
class YOLOConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
    def forward(self, x): return self.net(x)
class Bottleneck(nn.Module):
    def __init__(self, channels, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.conv1 = YOLOConv(channels, channels, kernel_size=3)
        self.conv2 = YOLOConv(channels, channels, kernel_size=3)
    def forward(self, x):
        y = self.conv2(self.conv1(x))
        return x + y if self.shortcut else y
class C2f(nn.Module):
    def __init__(self, in_channels, out_channels, repeats=1):
        super().__init__()
        hidden = max(out_channels // 2, 1)
        self.stem = YOLOConv(in_channels, hidden * 2, kernel_size=1)
        self.blocks = nn.ModuleList(Bottleneck(hidden) for _ in range(repeats))
        self.out = YOLOConv(hidden * (2 + repeats), out_channels, kernel_size=1)
    def forward(self, x):
        parts = list(self.stem(x).chunk(2, dim=1))
        for block in self.blocks:
            parts.append(block(parts[-1]))
        return self.out(torch.cat(parts, dim=1))
class SPPF(nn.Module):
    def __init__(self, channels, pool_size=5):
        super().__init__()
        hidden = max(channels // 2, 1)
        self.conv1 = YOLOConv(channels, hidden, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=1, padding=pool_size // 2)
        self.conv2 = YOLOConv(hidden * 4, channels, kernel_size=1)
    def forward(self, x):
        x = self.conv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.conv2(torch.cat([x, y1, y2, y3], dim=1))
class TimeAwareConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        gate_channels = input_channels + hidden_channels
        self.update_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.decay_rate = nn.Parameter(torch.full((1, hidden_channels, 1, 1), -4.0))
    def forward(self, x, dt, state=None):
        if state is None:
            state = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        if state.shape[-2:] != x.shape[-2:]:
            state = F.interpolate(state, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma = F.softplus(self.decay_rate)
        dt_view = dt.view(-1, 1, 1, 1)
        decayed_state = state * torch.exp(-gamma * dt_view)
        gates_in = torch.cat([x, decayed_state], dim=1)
        update = torch.sigmoid(self.update_gate(gates_in))
        reset = torch.sigmoid(self.reset_gate(gates_in))
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * decayed_state], dim=1)))
        return (1.0 - update) * decayed_state + update * candidate

class FlowDecoder(nn.Module):
    def __init__(self, in_channels=256, hidden=128):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            YOLOConv(in_channels, hidden, kernel_size=3)
        )
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            YOLOConv(hidden, hidden // 2, kernel_size=3)
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            YOLOConv(hidden // 2, hidden // 4, kernel_size=3)
        )
        self.head = nn.Conv2d(hidden // 4, 2, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        return self.head(x)

class _FastSAMPredictionHead(nn.Module):
    def __init__(self, in_channels, hidden_channels=256):
        super().__init__()
        self.spatial_feature = YOLOConv(in_channels, hidden_channels, kernel_size=3)
        self.objectness = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.reg_max = 32  # DFL regression bins
        self.boxes = nn.Conv2d(hidden_channels, 4 * self.reg_max, kernel_size=1)
        nn.init.constant_(self.boxes.bias, 0.0)
        self.mask_coefficients = nn.Conv2d(hidden_channels, 32, kernel_size=1)
        self.grid_cache = {}
    def forward(self, x):
        feat_sp = self.spatial_feature(x)
        return {
            "features": feat_sp,
            "objectness": self.objectness(feat_sp),
            "boxes": self.boxes(feat_sp),
            "mask_coefficients": self.mask_coefficients(feat_sp)
        }
class FastSAMStyleSegmenter(nn.Module):
    def __init__(self, base=48, hidden=768):
        super().__init__()
        # 骨干网络各 Stage 定义：提取不同感受野与空间分辨率的特征
        self.stem = YOLOConv(3, base, stride=2)
        self.stage2 = nn.Sequential(YOLOConv(base, base * 2, stride=2), C2f(base * 2, base * 2, repeats=2))
        self.stage3 = nn.Sequential(YOLOConv(base * 2, base * 4, stride=2), C2f(base * 4, base * 4, repeats=4)) # P3 尺度（输入分辨率的 1/8）
        self.stage4 = nn.Sequential(YOLOConv(base * 4, base * 8, stride=2), C2f(base * 8, base * 8, repeats=4)) # P4 尺度（输入分辨率的 1/16）
        self.stage5 = nn.Sequential(YOLOConv(base * 8, hidden, stride=2), C2f(hidden, hidden, repeats=2), SPPF(hidden)) # P5 尺度（高层语义信息，1/32）
        
        # 💡 极简 FPN 融合模块：激活原本闲置的高层 Stage4/Stage5 特征，增强大尺度与全图上下文的物理理解力
        # 1. 将 Stage 5 抽象语义特征的通道数用 1x1 卷积压缩，并使用邻近插值上采样 2 倍至与 Stage 4 尺度一致
        self.up_5_to_4 = nn.Sequential(
            YOLOConv(hidden, base * 8, kernel_size=1),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        # 2. 对 Stage 4 的融合特征进行 3x3 卷积，消除上采样产生的混叠效应
        self.fuse_4 = YOLOConv(base * 8, base * 8, kernel_size=3)
        # 3. 类似地，将融合后的 P4 特征压缩通道并上采样 2 倍，与 P3 尺度对齐
        self.up_4_to_3 = nn.Sequential(
            YOLOConv(base * 8, base * 4, kernel_size=1),
            nn.Upsample(scale_factor=2.0, mode='nearest')
        )
        # 4. 对最终融合后的 P3 特征（实例提取的黄金分辨率）进行 3x3 卷积平滑
        self.fuse_3 = YOLOConv(base * 4, base * 4, kernel_size=3)

        self.mask_prototypes = nn.Sequential(
            YOLOConv(base * 4, base * 4, kernel_size=3),
            YOLOConv(base * 4, base * 2, kernel_size=3),
            nn.Conv2d(base * 2, 32, kernel_size=1),
        )
        self.prediction_head = _FastSAMPredictionHead(base * 4, hidden_channels=256)
    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.stage2(f1)
        p3 = self.stage3(f2)
        f4 = self.stage4(p3)
        f5 = self.stage5(f4) # 运行高层骨干推理
        
        # 💡 自顶向下特征侧边融合 (Feature Pyramid Fusion)
        f4_fused = self.fuse_4(f4 + self.up_5_to_4(f5))
        p3_fused = self.fuse_3(p3 + self.up_4_to_3(f4_fused))
        
        prototypes = self.mask_prototypes(p3_fused)
        preds = self.prediction_head(p3_fused)
        return (f1, f2, p3_fused), prototypes, preds

class EgoPoseHead(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        c3 = 64
        self.fc = nn.Sequential(
            nn.Linear(in_channels, c3),
            nn.SiLU(),
            nn.Linear(c3, 9) 
        )
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)
    def forward(self, x):
        pooled = self.pool(x).flatten(1)
        pose = self.fc(pooled) 
        t = torch.tanh(pose[:, :3]) * 1.0
        # 零初始化保证初始输出为0，加上理想单位正交基
        identity_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=pose.device)
        rot_6d = identity_6d + torch.tanh(pose[:, 3:]) * 0.5
        return torch.cat([t, rot_6d], dim=1)

class FeaturePredictorHead(nn.Module):
    def __init__(self, channels=256, action_dim=9):
        super().__init__()
        self.stem = YOLOConv(channels + action_dim, channels, kernel_size=1)
        self.net = nn.Sequential(
            Bottleneck(channels, shortcut=True),
            Bottleneck(channels, shortcut=True),
            YOLOConv(channels, channels, kernel_size=3)
        )
    def forward(self, state, action):
        action_map = action.view(action.shape[0], action.shape[1], 1, 1).expand(-1, -1, state.shape[2], state.shape[3])
        x = torch.cat([state, action_map], dim=1)
        return self.net(self.stem(x))

class DepthDecoder(nn.Module):
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48, ch_gru=256):
        super().__init__()
        self.temporal_fuse = nn.Sequential(
            YOLOConv(ch_p3 + ch_gru, ch_p3, kernel_size=1),
            Bottleneck(ch_p3, shortcut=True)
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            YOLOConv(ch_p3, ch_f2, kernel_size=3)
        )
        self.conv1 = YOLOConv(ch_f2 * 2, ch_f2, kernel_size=3)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            YOLOConv(ch_f2, ch_f1, kernel_size=3)
        )
        self.conv2 = YOLOConv(ch_f1 * 2, ch_f1, kernel_size=3)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            YOLOConv(ch_f1, ch_f1, kernel_size=3)
        )
        self.depth_out = nn.Sequential(
            YOLOConv(ch_f1, ch_f1 // 2, kernel_size=3),
            nn.Conv2d(ch_f1 // 2, 1, kernel_size=3, padding=1)
        )
    def forward(self, f1, f2, p3, gru_state=None):
        if gru_state is not None:
            if gru_state.shape[-2:] != p3.shape[-2:]:
                gru_state = F.interpolate(gru_state, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            p3 = self.temporal_fuse(torch.cat([p3, gru_state], dim=1))
        x = self.up1(p3)
        x = torch.cat([x, f2], dim=1)
        x = self.conv1(x)
        x = self.up2(x)
        x = torch.cat([x, f1], dim=1)
        x = self.conv2(x)
        x = self.up3(x)
        return self.depth_out(x)

def decode_dfl_boxes(pred_dist, reg_max=16):
    # pred_dist: (B, 4*reg_max, H, W)
    B, C, H, W = pred_dist.shape
    prob = F.softmax(pred_dist.view(B, 4, reg_max, H, W), dim=2)
    weights = torch.arange(reg_max, dtype=torch.float32, device=pred_dist.device)
    distances = (prob * weights.view(1, 1, reg_max, 1, 1)).sum(dim=2)  # (B, 4, H, W)
    return distances

class TAONot42VisionModel(nn.Module):
    def __init__(self, base_channels=48, hidden_channels=768):
        super().__init__()
        self.segmenter = FastSAMStyleSegmenter(base=base_channels, hidden=hidden_channels)
        self.depth_decoder = DepthDecoder(256, base_channels * 2, base_channels, ch_gru=256)
        self.conv_gru = TimeAwareConvGRUCell(256, 256)
        self.pose_head = EgoPoseHead(256)
        
        # 💡 Flow Head 接入 GRU：赋予光流以时序记忆，采用深层解码器
        self.flow_head = FlowDecoder(256)
        
        self.feature_predictor = FeaturePredictorHead(256)
        self.state_update_gate_head = nn.Sequential(nn.Linear(256 + 1, 64), nn.SiLU(), nn.Linear(64, 1))
        
    def forward(self, peripheral, dt, step, state=None):
        b, _, h, w = peripheral.shape
        state = state or {}
        (f1, f2, p3), mask_prototypes, preds = self.segmenter(peripheral)
        
        gru_state = state.get("gru", None)
        next_gru_state = self.conv_gru(preds["features"], dt, gru_state)
        
        depth_logits = self.depth_decoder(f1, f2, preds["features"], next_gru_state)
        depth_logits = F.interpolate(depth_logits, size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        log_depth_pred = depth_logits
        depth_pred = torch.exp(torch.clamp(log_depth_pred, min=-4.6, max=4.6)) 
        
        ego_pose = self.pose_head(next_gru_state)
        
        w = get_loss_weights(step)
        
        # 💡 极速旁路优化：如果当前步数不训练光流，直接跳过深层解码网络！
        if w["flow"] > 0:
            raw_flow = self.flow_head(next_gru_state)
            pred_flow = 1.5 * torch.tanh(raw_flow)
            preds["flow"] = pred_flow
        else:
            pred_flow = None
        
        prev_ego_pose = state.get("prev_ego", torch.zeros_like(ego_pose))
        if gru_state is not None and w["anom"] > 0:
            pred_current_feature = self.feature_predictor(gru_state, prev_ego_pose)
            feature_error_map = F.smooth_l1_loss(pred_current_feature, preds["features"], reduction='none').mean(dim=1)
        else:
            feature_error_map = torch.zeros(b, preds["features"].shape[2], preds["features"].shape[3], device=peripheral.device)
        
        selected_feature = preds["features"].mean(dim=[2, 3])
        gate_in = torch.cat([selected_feature, dt.view(-1, 1)], dim=-1)
        raw_gate = self.state_update_gate_head(gate_in)
        gate = torch.sigmoid(raw_gate).view(-1, 1, 1, 1)
        final_gru_state = gru_state * (1.0 - gate) + next_gru_state * gate if gru_state is not None else next_gru_state
        
        return {
            "objectness": preds["objectness"],
            "box_dist": preds["boxes"] if w["box"] > 0 else None,
            "boxes": decode_dfl_boxes(preds["boxes"], reg_max=32) if w["box"] > 0 else None,
            "mask_coefficients": preds["mask_coefficients"] if w["mask"] > 0 else None, 
            "mask_prototypes": mask_prototypes if w["mask"] > 0 else None,               
            "depth": depth_pred,                              
            "log_depth": log_depth_pred,                      
            "ego_pose": ego_pose,                             
            "flow": pred_flow,
            "features": preds["features"],
            "anomaly_map": feature_error_map,                 
            "feature_error": feature_error_map.mean(),
            "state_update_gate": gate.view(b),
            "next_state": {"gru": final_gru_state, "prev_ego": ego_pose},
        }

# =====================================================================
# 2. YOLO 权重加载
# =====================================================================
def load_yolo_backbone_weights(model, checkpoint_path):
    if not os.path.exists(checkpoint_path): return
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"].state_dict() if hasattr(ckpt["model"], "state_dict") else ckpt
    except Exception: return
    layer_specs = [
        ("model.0.", "segmenter.stem.net."),
        ("model.1.", "segmenter.stage2.0.net."), ("model.2.", "segmenter.stage2.1."),
        ("model.3.", "segmenter.stage3.0.net."), ("model.4.", "segmenter.stage3.1."),
        ("model.5.", "segmenter.stage4.0.net."), ("model.6.", "segmenter.stage4.1."),
        ("model.7.", "segmenter.stage5.0.net."), ("model.8.", "segmenter.stage5.1."),
        ("model.9.", "segmenter.stage5.2.")
    ]
    target_state = model.state_dict()
    updates = {}
    for src_prefix, tgt_prefix in layer_specs:
        for src_key in state_dict:
            if src_key.startswith(src_prefix):
                tgt_key = src_key.replace(src_prefix, tgt_prefix)
                tgt_key = tgt_key.replace("conv.", "0.").replace("bn.", "1.")
                if "cv1." in tgt_key: tgt_key = tgt_key.replace("cv1.", "stem.net.")
                if "cv2." in tgt_key: tgt_key = tgt_key.replace("cv2.", "out.net.")
                if "m." in tgt_key:
                    tgt_key = tgt_key.replace("m.", "blocks.")
                    tgt_key = tgt_key.replace(".cv1.", ".conv1.net.").replace(".cv2.", ".conv2.net.")
                if tgt_key in target_state and target_state[tgt_key].shape == state_dict[src_key].shape:
                    updates[tgt_key] = state_dict[src_key]
    target_state.update(updates)
    model.load_state_dict(target_state)
def freeze_backbone(model):
    for name, param in model.segmenter.named_parameters():
        if "stem" in name or "stage" in name: param.requires_grad = False

# =====================================================================
# 3. 经验回放池 (Replay Buffer) 与 GPU 数据流水线
# =====================================================================
class AsyncDataBuffer:
    def __init__(self, split='train', max_buffer_size=64, batch_size=16, max_samples=None):
        self.split = split
        self.max_buffer_size = max_buffer_size
        self.batch_size = batch_size
        self.max_samples = max_samples
        self.buffer = deque(maxlen=max_buffer_size)
        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock)
        
        print("\n" + "="*60)
        print(f"🚀 [异步管线] 正在启动后台独立 I/O 数据流缓冲池...")
        print(f"   >> 最大数据缓冲池: {max_buffer_size} 个序列 (滚动窗口)")
        print(f"   >> 动态批次抽样: 每次随机抽取 {batch_size} 条 (拒绝空转)")
        print("="*60 + "\n")
        
        self.thread = threading.Thread(target=self._fetch_loop, daemon=True)
        self.thread.start()
    def _fetch_loop(self):
        read_config = tfds.ReadConfig(
            interleave_cycle_length=16, 
            num_parallel_calls_for_interleave_files=tf.data.AUTOTUNE
        )
        ds = tfds.load("movi_e", data_dir="gs://kubric-public/tfds", split=self.split, read_config=read_config)
        ds = ds.repeat()
        
        def process_video_frames(x):
            return {
                'video': x['video'],               
                'segmentations': x['segmentations'], 
                'depth': x['depth'],
                'forward_flow': x['forward_flow'],
                'cam_pos': x['camera']['positions'],
                'cam_quat': x['camera']['quaternions']
            }
            
        ds = ds.map(process_video_frames, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
        
        for item in tfds.as_numpy(ds):
            pinned_item = {
                "video": torch.from_numpy(item['video']).pin_memory(),
                "segmentation": torch.from_numpy(item['segmentations'][..., 0]).pin_memory(),
                "depth": torch.from_numpy(item['depth'][..., 0]).pin_memory(),
                "cam_pos": torch.from_numpy(item['cam_pos']).pin_memory(),
                "cam_quat": torch.from_numpy(item['cam_quat']).pin_memory()
            }
            
            # Decode forward_flow from uint16 (Fallback logic works perfectly for Kubric MOVi-E)
            flow_np = item['forward_flow'].astype(np.float32)
            if 'metadata' in item and 'forward_flow_range' in item['metadata']:
                minv, maxv = item['metadata']['forward_flow_range']
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
                    raise RuntimeError("❌ 后台数据流线程异常崩溃，请检查网络或 TFDS 配置！")
                self.has_data.wait(timeout=5.0)
            batch_list = random.sample(self.buffer, self.batch_size)
        
        return {
            "video": [item['video'] for item in batch_list],
            "segmentation": [item['segmentation'] for item in batch_list],
            "depth": [item['depth'] for item in batch_list],
            "forward_flow": [item['forward_flow'] for item in batch_list],
            "cam_pos": [item['cam_pos'] for item in batch_list],
            "cam_quat": [item['cam_quat'] for item in batch_list]
        }

def process_batch_on_gpu(batch, device, target_size=256):
    video_raw = torch.stack([x.to(device, non_blocking=True) for x in batch["video"]])
    depth_raw_uint16 = torch.stack([x.to(device, non_blocking=True) for x in batch["depth"]]).float()
    seg_raw = torch.stack([x.to(device, non_blocking=True) for x in batch["segmentation"]])
    flow_raw = torch.stack([x.to(device, non_blocking=True) for x in batch["forward_flow"]]).float()
    cam_pos = torch.stack([x.to(device, non_blocking=True) for x in batch["cam_pos"]])
    cam_quat = torch.stack([x.to(device, non_blocking=True) for x in batch["cam_quat"]])
    
    B, T = video_raw.shape[:2]
    
    depth_raw_m = depth_raw_uint16 / 1000.0
    depth_raw_m[depth_raw_uint16 == 0] = 1096.0
    depth_raw_m = torch.clamp(depth_raw_m, 0.01, 1096.0)
    
    video = video_raw.permute(0, 1, 4, 2, 3).float() / 255.0
    
    if video.shape[-1] != target_size:
        video = F.interpolate(video.flatten(0, 1), size=(target_size, target_size), mode='bilinear', align_corners=False).view(B, T, 3, target_size, target_size)
        seg = F.interpolate(seg_raw.float().flatten(0, 1).unsqueeze(1), size=(target_size, target_size), mode='nearest').view(B, T, target_size, target_size)
        depth_m = F.interpolate(depth_raw_m.float().flatten(0, 1).unsqueeze(1), size=(target_size, target_size), mode='bilinear', align_corners=False).squeeze(1).view(B, T, target_size, target_size)
    else:
        seg = seg_raw.float()
        depth_m = depth_raw_m
    H, W = target_size, target_size
    seg_long = seg.long()
    
    depth_m_clamped = torch.clamp(depth_m, 0.01, 100.0)
    log_depth_target = torch.log(depth_m_clamped) 
    
    flow_norm = torch.clamp(flow_raw * 2.0 / target_size, -1.5, 1.5)
    if flow_norm.shape[2] != target_size:
        flow_norm = F.interpolate(flow_norm.flatten(0, 1).permute(0, 3, 1, 2), size=(target_size, target_size), mode='bilinear', align_corners=False)
        flow_norm = flow_norm.view(B, T, 2, target_size, target_size)
    else:
        flow_norm = flow_norm.permute(0, 1, 4, 2, 3)
    
    active_mask = (seg_long > 0)
    active_mask_float = active_mask.float()
    
    H_feat, W_feat = H // 8, W // 8
    seg_small = F.interpolate(seg.flatten(0, 1).unsqueeze(1), size=(H_feat, W_feat), mode='nearest').squeeze(1).view(B, T, H_feat, W_feat)
    
    bboxes_dense = torch.zeros(B, T, 4, H_feat, W_feat, device=device)
    obj_dense = torch.zeros(B, T, 1, H_feat, W_feat, device=device)
    y_grid = torch.arange(H, device=device, dtype=torch.int16).view(1, 1, 1, H, 1)
    x_grid = torch.arange(W, device=device, dtype=torch.int16).view(1, 1, 1, 1, W)
    
    max_uid = int(seg_long.max().item())
    if max_uid > 0:
        uids = torch.arange(1, max_uid + 1, device=device, dtype=torch.int16).view(-1, 1, 1, 1, 1)
        masks = (seg_long.to(torch.int16).unsqueeze(0) == uids)
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
        
        valid_mask = (true_area >= 10) & (box_area <= 4 * true_area) & valid_bt
        
        n_idx, b_idx, t_idx = torch.where(valid_mask)
        if len(n_idx) > 0:
            cy = torch.clamp(((ymin[n_idx, b_idx, t_idx] + ymax[n_idx, b_idx, t_idx]) / 2 / 8).long(), 0, H_feat - 1)
            cx = torch.clamp(((xmin[n_idx, b_idx, t_idx] + xmax[n_idx, b_idx, t_idx]) / 2 / 8).long(), 0, W_feat - 1)
            
            obj_dense[b_idx, t_idx, 0, cy, cx] = 1.0
            
            grid_x = (cx.float() * 8.0 + 4.0)
            grid_y = (cy.float() * 8.0 + 4.0)
            
            valid_boxes = torch.stack([
                torch.clamp((grid_x - xmin[n_idx, b_idx, t_idx].float()) / 8.0, min=1e-4),
                torch.clamp((grid_y - ymin[n_idx, b_idx, t_idx].float()) / 8.0, min=1e-4),
                torch.clamp((xmax[n_idx, b_idx, t_idx].float() - grid_x) / 8.0, min=1e-4),
                torch.clamp((ymax[n_idx, b_idx, t_idx].float() - grid_y) / 8.0, min=1e-4)
            ], dim=-1)
            bboxes_dense[b_idx, t_idx, :, cy, cx] = valid_boxes
    
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
    
    bboxes_global = torch.stack([xmin_g/W, ymin_g/H, xmax_g/W, ymax_g/H], dim=-1)
    empty = ~active_mask.view(B, T, -1).any(dim=-1)
    bboxes_global[empty] = torch.tensor([0.0, 0.0, 1.0, 1.0], device=device)
    
    return {
        "video": video,
        "seg_raw": seg,
        "mask": active_mask_float.unsqueeze(2),
        "bbox": bboxes_global,           
        "depth": depth_m_clamped,
        "log_depth": log_depth_target,
        "flow": flow_norm,
        "cam_pos": cam_pos,
        "cam_quat": cam_quat,
        "obj_dense": obj_dense,
        "bboxes_dense": bboxes_dense,
        "seg_small": seg_small
    }

# =====================================================================
# 4. NMS 实例提取与逐实例 Loss
# =====================================================================
def extract_instances(preds, score_thresh=0.3, nms_thresh=0.5, max_det=20):
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
        pixel_boxes = decoded_boxes_norm * torch.tensor([W_img, H_img, W_img, H_img], device=device)
        
        keep = torchvision.ops.nms(pixel_boxes, sel_scores, nms_thresh)[:max_det]
        
        coeffs = preds.get("mask_coefficients")
        protos = preds.get("mask_prototypes")
        
        if coeffs is not None and protos is not None:
            sel_coeffs = coeffs[b, :, valid.nonzero()[:, 0], valid.nonzero()[:, 1]].T
            kept_coeffs = sel_coeffs[keep]
            masks = torch.einsum("kp,phw->khw", kept_coeffs, protos[b])
            masks = F.interpolate(masks.unsqueeze(0), size=(H_img, W_img), mode='bilinear', align_corners=False)[0]
            
            # Box crop for masks
            boxes_pixel = pixel_boxes[keep]
            N_masks = masks.shape[0]
            rows = torch.arange(H_img, device=device).view(1, H_img, 1)
            cols = torch.arange(W_img, device=device).view(1, 1, W_img)
            x1, y1, x2, y2 = boxes_pixel.unbind(-1)
            mask_crop = (cols >= x1.view(N_masks, 1, 1)) & (cols < x2.view(N_masks, 1, 1)) & \
                        (rows >= y1.view(N_masks, 1, 1)) & (rows < y2.view(N_masks, 1, 1))
            masks = masks * mask_crop.float() - 10.0 * (~mask_crop).float()
            
            masks_bool = torch.sigmoid(masks) > 0.5
        else:
            masks_bool = None
        
        results.append({
            "scores": sel_scores[keep],
            "boxes": decoded_boxes_norm[keep],
            "masks": masks_bool
        })
    return results

def compute_instance_loss(preds, targets, step=0):
    device = preds["objectness"].device
    
    loss_obj = focal_loss(preds["objectness"], targets["obj_dense"])
    pos_mask = targets["obj_dense"][:, 0] > 0.5
    if pos_mask.sum() == 0:
        return loss_obj, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    
    loss_box = torch.tensor(0.0, device=device)
    loss_mask = torch.tensor(0.0, device=device)
    w = get_loss_weights(step)
    
    if w["box"] > 0:
        pred_boxes_pos = preds["boxes"].permute(0,2,3,1)[pos_mask]
        gt_boxes_pos = targets["bboxes_dense"].permute(0,2,3,1)[pos_mask]
        
        loss_giou = giou_loss_with_l1_warmup(pred_boxes_pos, gt_boxes_pos, step=step)
        pred_dist_pos = preds["box_dist"].permute(0,2,3,1)[pos_mask]
        loss_dfl = dfl_loss(pred_dist_pos, gt_boxes_pos, reg_max=32)
        
        loss_box = loss_giou * 1.5 + loss_dfl * 0.5
        
    if w["mask"] > 0:
        loss_mask = compute_per_instance_mask_loss(preds, targets, pos_mask)
    
    return loss_obj, loss_box, loss_mask

def compute_per_instance_mask_loss(preds, targets, pos_mask):
    B, _, H_feat, W_feat = preds["objectness"].shape
    device = preds["objectness"].device
    seg_raw = targets["seg_raw"]
    H, W = seg_raw.shape[1], seg_raw.shape[2]
    
    b_indices, y_indices, x_indices = torch.where(pos_mask)
    num_instances = b_indices.numel()
    if num_instances == 0:
        return torch.tensor(0.0, device=device)
    
    center_y = torch.clamp(y_indices * 8 + 4, 0, H - 1)
    center_x = torch.clamp(x_indices * 8 + 4, 0, W - 1)
    inst_ids = seg_raw[b_indices, center_y, center_x].long()
    
    valid_mask = inst_ids > 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
        
    b_indices = b_indices[valid_mask]
    y_indices = y_indices[valid_mask]
    x_indices = x_indices[valid_mask]
    inst_ids = inst_ids[valid_mask]
    num_instances = b_indices.numel()
    
    coeffs = preds["mask_coefficients"][b_indices, :, y_indices, x_indices]
    protos = preds["mask_prototypes"][b_indices]
    pred_logits_small = torch.einsum("np,nphw->nhw", coeffs, protos)
    
    seg_batch = targets["seg_small"][b_indices]
    gt_masks_small = (seg_batch == inst_ids.view(num_instances, 1, 1)).float()
    
    preds_sig = torch.sigmoid(pred_logits_small).flatten(1)
    targets_flat = gt_masks_small.flatten(1)
    intersection = (preds_sig * targets_flat).sum(dim=1)
    union = preds_sig.sum(dim=1) + targets_flat.sum(dim=1)
    
    pos_count = targets_flat.sum(dim=1).clamp(min=1.0)
    smooth = pos_count * 0.01
    loss_dice = (1.0 - (2. * intersection + smooth) / (union + smooth)).mean()
    
    bce = F.binary_cross_entropy_with_logits(pred_logits_small, gt_masks_small, reduction='none')
    p_t = torch.exp(-bce)
    loss_bce = (0.25 * (1 - p_t) ** 2 * bce).mean()
    
    return loss_dice * 2.0 + loss_bce * 1.0

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
    
    model.segmenter.prediction_head.objectness.requires_grad_(True)
    model.segmenter.prediction_head.boxes.requires_grad_(False)
    model.segmenter.prediction_head.mask_coefficients.requires_grad_(False)
    
    for param in model.flow_head.parameters():
        param.requires_grad = True

# =====================================================================
# 5. 论文级可视化
# =====================================================================
def depth_to_color(depth_map, d_min=None, d_max=None):
    if d_min is None: d_min = depth_map.min()
    if d_max is None: d_max = depth_map.max()
    if d_max > d_min:
        d_norm = (depth_map - d_min) / (d_max - d_min)
    else:
        d_norm = np.zeros_like(depth_map)
    d_uint8 = (np.clip(d_norm, 0, 1) * 255).astype(np.uint8)
    return cv2.applyColorMap(d_uint8, cv2.COLORMAP_MAGMA)

def flow_to_color(flow_np):
    flow_np = flow_np.astype(np.float32)
    # Subtract median to remove global camera motion and highlight relative parallax/object motion
    flow_np[..., 0] -= np.median(flow_np[..., 0])
    flow_np[..., 1] -= np.median(flow_np[..., 1])
    
    h, w = flow_np.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow_np[..., 0], flow_np[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    mag_max = np.max(mag)
    if mag_max > 1e-3:
        hsv[..., 2] = (mag / mag_max * 255).astype(np.uint8)
    else:
        hsv[..., 2] = 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def save_visualization(video_t, target_t, pred_t, step, warped_img=None, output_dir="vis_outputs"):
    os.makedirs(output_dir, exist_ok=True)
    img_tensor = video_t[0].permute(1, 2, 0).cpu().numpy()
    base_bgr = cv2.cvtColor((img_tensor * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    H, W = base_bgr.shape[:2]
    
    pred_canvas = base_bgr.copy()
    with torch.no_grad():
        instances = extract_instances(pred_t, score_thresh=0.3, nms_thresh=0.5)
    inst = instances[0]
    inst_count = 0
    if inst is not None:
        colors_list = [(0,0,255), (255,0,0), (0,255,255), (255,0,255), (0,165,255), (255,255,0)]
        inst_count = len(inst["scores"])
        for k in range(inst_count):
            color = colors_list[k % len(colors_list)]
            if inst["masks"] is not None:
                m = inst["masks"][k].cpu().numpy()
                pred_canvas[m] = pred_canvas[m] * 0.5 + np.array(color) * 0.5
            b = inst["boxes"][k].cpu().numpy()
            cv2.rectangle(pred_canvas, (int(b[0]*W), int(b[1]*H)), (int(b[2]*W), int(b[3]*H)), color, 2)
    cv2.putText(pred_canvas, "Prediction (Box & Mask)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    gt_canvas = base_bgr.copy()
    if "bboxes_dense" in target_t and "obj_dense" in target_t:
        obj_t = target_t["obj_dense"][0, 0].cpu().numpy()
        boxes_t = target_t["bboxes_dense"][0].cpu().numpy()
        y_idx, x_idx = np.where(obj_t > 0.5)
        for y, x in zip(y_idx, x_idx):
            b = boxes_t[:, y, x]
            cv2.rectangle(gt_canvas, (int(b[0]*W), int(b[1]*H)), (int(b[2]*W), int(b[3]*H)), (0, 255, 0), 2)
    cv2.putText(gt_canvas, "Ground Truth", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    anom_canvas = base_bgr.copy()
    anom_map = pred_t["anomaly_map"][0].cpu().detach().numpy()
    anom_max = max(float(np.max(anom_map)), 0.001)
    anom_norm = np.clip(anom_map / anom_max, 0, 1)
    anom_img = cv2.applyColorMap((anom_norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    anom_img = cv2.resize(anom_img, (W, H))
    anom_canvas = cv2.addWeighted(anom_canvas, 0.4, anom_img, 0.6, 0)
    cv2.putText(anom_canvas, "Anomaly Heatmap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    half_h, half_w = H // 2, W // 2
    
    if warped_img is not None:
        warp_np = warped_img[0].permute(1, 2, 0).cpu().detach().numpy()
        warp_bgr = cv2.cvtColor((np.clip(warp_np, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        warp_img = cv2.resize(warp_bgr, (half_w, half_h))
    else:
        warp_img = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    
    obj_map = torch.sigmoid(pred_t["objectness"][0, 0]).cpu().detach().numpy()
    obj_img = cv2.applyColorMap((np.clip(obj_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    obj_img = cv2.resize(obj_img, (half_w, half_h))
    
    gt_depth_np = target_t["depth"][0].cpu().numpy()
    pred_depth_np = pred_t["depth"][0].cpu().detach().numpy()
    d_min = min(gt_depth_np.min(), pred_depth_np.min())
    d_max = max(gt_depth_np.max(), pred_depth_np.max())
    gt_depth_img = cv2.resize(depth_to_color(gt_depth_np, d_min, d_max), (half_w, half_h))
    pred_depth_img = cv2.resize(depth_to_color(pred_depth_np, d_min, d_max), (half_w, half_h))
    
    gt_flow_np = target_t.get("flow_target", torch.zeros((1, 2, H, W)))[0].cpu().numpy()
    pred_flow = pred_t.get("flow")
    if pred_flow is not None:
        pred_flow_np = pred_flow[0].cpu().detach().numpy()
        pred_flow_img = cv2.resize(flow_to_color(pred_flow_np.transpose(1, 2, 0)), (half_w, half_h))
    else:
        pred_flow_img = np.zeros((half_h, half_w, 3), dtype=np.uint8)
    
    gt_flow_img = cv2.resize(flow_to_color(gt_flow_np.transpose(1, 2, 0)), (half_w, half_h))
    
    row1 = np.hstack([obj_img, warp_img])
    row2 = np.hstack([gt_depth_img, pred_depth_img])
    row3 = np.hstack([gt_flow_img, pred_flow_img])
    grid = np.vstack([row1, row2, row3])
    
    grid = cv2.resize(grid, (int(grid.shape[1] * H / grid.shape[0]), H))
    
    final_img = np.hstack([pred_canvas, gt_canvas, anom_canvas, grid])
    filepath = os.path.join(output_dir, f"vis_step_{step:05d}.jpg")
    cv2.imwrite(filepath, final_img)
    return filepath

# =====================================================================
# 6. 核心物理监督 Loss 与主循环
# =====================================================================
def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    K_inv = torch.inverse(K)
    return K, K_inv

def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    device = depth.device
    
    y, x = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing='ij')
    x = x.flatten().expand(B, -1)
    y = y.flatten().expand(B, -1)
    ones = torch.ones_like(x)
    pixels = torch.stack([x, y, ones], dim=1) 
    
    points_3d = torch.bmm(K_inv.expand(B, 3, 3), pixels.float()) 
    points_3d = points_3d * depth.view(B, 1, H*W)
    
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
    
    warped_img = F.grid_sample(img_next, grid, mode='bilinear', padding_mode='border', align_corners=True)
    warped_img = torch.nan_to_num(warped_img, 0.0)
    
    valid_mask = ((x_norm > -1.0) & (x_norm < 1.0) & (y_norm > -1.0) & (y_norm < 1.0)).view(B, 1, H, W).float()
    safe_depth_mask = ((depth > 0.01) & (z_next_raw.view(B, 1, H, W) > 0.01)).float()
    valid_mask = valid_mask * safe_depth_mask
    
    return warped_img, valid_mask

def edge_aware_smoothness_loss(depth, img):
    mean_depth = depth.mean(dim=[2, 3], keepdim=True).float()
    norm_depth = (depth.float() / torch.clamp(mean_depth, min=1e-4)).to(depth.dtype)
    grad_depth_x = torch.abs(norm_depth[:, :, :, :-1] - norm_depth[:, :, :, 1:])
    grad_depth_y = torch.abs(norm_depth[:, :, :-1, :] - norm_depth[:, :, 1:, :])
    grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), dim=1, keepdim=True)
    grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), dim=1, keepdim=True)
    grad_depth_x *= torch.exp(-grad_img_x)
    grad_depth_y *= torch.exp(-grad_img_y)
    return grad_depth_x.mean() + grad_depth_y.mean()

def dice_loss(preds, targets, smooth=1e-5):
    preds = torch.sigmoid(preds)
    preds = preds.flatten()
    targets = targets.flatten()
    intersection = (preds * targets).sum()
    dice = (2. * intersection + smooth) / (preds.sum() + targets.sum() + smooth)
    return 1.0 - dice

def focal_loss(preds_logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(preds_logits, targets, reduction='none')
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
    loss_left = F.cross_entropy(pred_dist.view(-1, reg_max), target_left.view(-1), reduction='none').view(-1, 4) * weight_left
    loss_right = F.cross_entropy(pred_dist.view(-1, reg_max), target_right.view(-1), reduction='none').view(-1, 4) * weight_right
    
    return (loss_left + loss_right).mean()

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
    return (1.0 - giou).mean()

def giou_loss_with_l1_warmup(preds, targets, step, warmup_steps=500):
    l1 = F.smooth_l1_loss(preds, targets, beta=1.0)
    if step < warmup_steps:
        return l1
    giou = giou_loss(preds, targets)
    alpha = min((step - warmup_steps) / 1000.0, 1.0)
    return l1 * (1 - alpha) + giou * alpha

def ssim_loss(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    x_pad = F.pad(x, (1, 1, 1, 1), mode='reflect')
    y_pad = F.pad(y, (1, 1, 1, 1), mode='reflect')
    mu_x = F.avg_pool2d(x_pad, 3, 1)
    mu_y = F.avg_pool2d(y_pad, 3, 1)
    sigma_x = F.avg_pool2d(x_pad ** 2, 3, 1) - mu_x ** 2
    sigma_y = F.avg_pool2d(y_pad ** 2, 3, 1) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x_pad * y_pad, 3, 1) - mu_x * mu_y
    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)

def get_loss_weights(step):
    def ramp(start, end, val):
        if step < start: return 0.0
        if step > end: return val
        return val * (step - start) / (end - start)
    return {
        "obj":   1.0,
        "box":   ramp(300, 800, 2.0),
        "mask":  ramp(1500, 2500, 1.0),
        "depth": 3.0 if step < 2000 else 1.5,
        "photo": ramp(3000, 5000, 1.0),
        "ego":   3.0,
        "flow":  ramp(500, 1500, 1.0),
        "anom":  ramp(4000, 6000, 1.0),
        "smooth": 0.05,
        "gate":   0.05,
    }

LOSS_EMA = {}
def get_ema_loss(name, current_val, alpha=0.95):
    val = current_val.item()
    if val == 0: return 1.0
    if name not in LOSS_EMA:
        LOSS_EMA[name] = val
    else:
        LOSS_EMA[name] = LOSS_EMA[name] * alpha + val * (1 - alpha)
    return max(LOSS_EMA[name], 1e-4)

def compute_physics_loss(preds, targets, img_t=None, img_next=None, mode="supervised", teacher_forcing_ego=None, step=0):
    device = preds["depth"].device
    B, H, W = preds["depth"].shape
    w = get_loss_weights(step)
    
    loss_obj, loss_box, loss_mask = compute_instance_loss(preds, targets, step=step)
    
    loss_ego = torch.tensor(0.0, device=device)
    if mode == "supervised" and "cam_pos_t" in targets and "cam_pos_next" in targets:
        c_mat_t = quaternion_to_matrix(targets["cam_quat_t"])
        c_mat_n = quaternion_to_matrix(targets["cam_quat_next"])
        R_n_inv = c_mat_n.transpose(1, 2)
        R_delta = torch.bmm(R_n_inv, c_mat_t)
        T_delta = torch.bmm(R_n_inv, (targets["cam_pos_t"] - targets["cam_pos_next"]).unsqueeze(-1)).squeeze(-1)
        gt_ego = torch.cat([T_delta, matrix_to_6d(R_delta)], dim=1)
        loss_ego = F.smooth_l1_loss(preds["ego_pose"], gt_ego)
    
    # Depth loss with sky exclusion
    loss_depth = torch.tensor(0.0, device=device)
    if mode == "supervised":
        raw_loss_depth = F.smooth_l1_loss(preds["log_depth"], targets["log_depth"], reduction='none')
        # Sky was clamped to 100.0, so anything >= 99.0 is sky
        valid_depth_mask = (targets["depth"] < 99.0).float()
        loss_depth = (raw_loss_depth * valid_depth_mask).sum() / valid_depth_mask.sum().clamp(min=1)
    
    loss_flow = torch.tensor(0.0, device=device)
    if w["flow"] > 0 and preds.get("flow") is not None and "flow_target" in targets:
        loss_flow = F.smooth_l1_loss(preds["flow"], targets["flow_target"])
    
    warped_img = None
    loss_photo = torch.tensor(0.0, device=device)
    loss_smooth = torch.tensor(0.0, device=device)
    
    if img_t is not None and img_next is not None and w["photo"] > 0:
        K, K_inv = generate_intrinsics(H, W, device)
        warped_img, valid_warp_mask = inverse_warp(img_next, preds["depth"].unsqueeze(1), preds["ego_pose"], K, K_inv)
        
        # Photo loss uses L1 + SSIM
        def photo_loss_fn(pred, tgt):
            l1 = F.l1_loss(pred, tgt, reduction='none').mean(dim=1, keepdim=True)
            ssim = ssim_loss(pred, tgt).mean(dim=1, keepdim=True)
            return 0.15 * l1 + 0.85 * ssim
            
        warp_loss = photo_loss_fn(warped_img, img_t)
        identity_loss = photo_loss_fn(img_next, img_t)
        
        auto_mask = (warp_loss < identity_loss).float()
        sky_mask_1 = (targets["seg_raw"] == 0).float().unsqueeze(1)
        
        mask = valid_warp_mask * (1 - sky_mask_1) * auto_mask
        loss_photo = (warp_loss * mask).sum() / mask.sum().clamp(min=1)
        loss_smooth = edge_aware_smoothness_loss(preds["depth"].unsqueeze(1), img_t)
    
    loss_anom = preds["feature_error"]
    loss_gate = F.smooth_l1_loss(preds["state_update_gate"], torch.zeros_like(preds["state_update_gate"]))
    
    norm_obj = loss_obj / get_ema_loss("Obj", loss_obj)
    norm_box = loss_box / get_ema_loss("Box", loss_box)
    norm_mask = loss_mask / get_ema_loss("Mask", loss_mask)
    norm_depth = loss_depth / get_ema_loss("Depth", loss_depth)
    norm_photo = loss_photo / get_ema_loss("Photo", loss_photo)
    norm_ego = loss_ego / get_ema_loss("Ego", loss_ego)
    norm_flow = loss_flow / get_ema_loss("Flow", loss_flow)
    norm_anom = loss_anom / get_ema_loss("Anom", loss_anom)
    
    total = (
        norm_obj * w["obj"] + norm_box * w["box"] + norm_mask * w["mask"] +
        norm_depth * w["depth"] + norm_photo * w["photo"] + loss_smooth * w["smooth"] +
        norm_ego * w["ego"] + norm_flow * w["flow"] + norm_anom * w["anom"] + loss_gate * w["gate"]
    )
    
    return total, {
        "Obj": loss_obj.detach(), "Box": loss_box.detach(), "Mask": loss_mask.detach(),
        "Depth": loss_depth.detach(), "Photo": loss_photo.detach(),
        "Ego": loss_ego.detach(), "Flow": loss_flow.detach(),
        "Anom": loss_anom.detach(), "Gate": loss_gate.detach()
    }, warped_img

def train_model(args):
    device = torch.device(args.device)
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    
    model = TAONot42VisionModel(base_channels=48, hidden_channels=768).to(device)
    if getattr(args, 'compile_model', False) and hasattr(torch, 'compile'):
        try:
            model.segmenter = torch.compile(model.segmenter, mode='reduce-overhead')
            model.depth_decoder = torch.compile(model.depth_decoder, mode='reduce-overhead')
            print("🚀 torch.compile 成功开启！")
        except Exception as e:
            print(f"⚠️ torch.compile 开启失败 (将使用正常模式): {e}")

    if args.yolo_weights:
        load_yolo_backbone_weights(model, args.yolo_weights)
        if args.freeze: freeze_backbone(model)
            
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(device.type) if device.type == 'cuda' else None
    
    buffer = AsyncDataBuffer(
        split='train', 
        max_buffer_size=args.max_buffer_size, 
        batch_size=args.batch_size, 
        max_samples=args.max_samples
    )
    
    if args.use_wandb: wandb.init(project=args.wandb_project, config=vars(args))
        
    model.train()
    mode = "supervised"
    print(f"\n🚀 开始 TAO-NOT-42 V12 训练 (Device: {device}, Mode: {mode})")
    
    global_step = 0
    start_time = time.time()
    
    best_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(1, args.epochs + 1):
        if args.finetune_after_epoch and epoch > args.finetune_after_epoch and mode == "supervised":
            mode = "self_supervised"
            setup_finetune_mode(model)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr * 0.1)
            if scaler is not None:
                scaler = torch.amp.GradScaler(device.type)
        
        print(f"\n" + "="*40)
        print(f"🌟 Epoch {epoch}/{args.epochs} [Mode: {mode}]")
        print("="*40)
        
        epoch_loss_sum = 0.0
        
        for step_in_epoch in range(args.steps_per_epoch):
            raw_batch = buffer.get_batch()
            batch = process_batch_on_gpu(raw_batch, device, args.img_size)
            
            videos = batch["video"]
            b, t, c, h, w = videos.shape
            state = None
            
            for chunk_start in range(0, t, args.seq_len):
                chunk_end = min(chunk_start + args.seq_len, t)
                optimizer.zero_grad(set_to_none=True)
                total_seq_loss = 0
                loss_dict_acc = {k: 0.0 for k in ["Obj", "Box", "Mask", "Depth", "Photo", "Ego", "Flow", "Anom", "Gate"]}
                
                for step in range(chunk_start, chunk_end):
                    x_t = videos[:, step]
                    x_next = videos[:, step+1] if step+1 < t else x_t
                    
                    time_t = torch.full((b,), step * 0.1, device=device)
                    dt_t = torch.full((b,), 1.0 / 24.0 if step > 0 else 0.0, device=device)
                    target_t = {k: v[:, step] for k, v in batch.items() if k != "video"}
                    if step > 0:
                        target_t["flow_target"] = batch["flow"][:, step - 1]
                    
                    if step + 1 < t:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step+1]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step+1]
                    else:
                        target_t["cam_pos_next"] = batch["cam_pos"][:, step]
                        target_t["cam_quat_next"] = batch["cam_quat"][:, step]
                    target_t["cam_pos_t"] = batch["cam_pos"][:, step]
                    target_t["cam_quat_t"] = batch["cam_quat"][:, step]
                    
                    with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                        out = model(x_t, dt_t, global_step, state)
                        state = out["next_state"]
                        loss, loss_dict, warped_img = compute_physics_loss(out, target_t, x_t, x_next, mode=mode, step=global_step)
                        
                    total_seq_loss += loss
                    for k in loss_dict_acc: loss_dict_acc[k] = loss_dict_acc[k] + loss_dict[k]
                    
                    if (global_step + 1) % args.vis_interval == 0 and step == chunk_end - 1:
                        filepath = save_visualization(x_t, target_t, out, global_step + 1, warped_img)
                        if args.use_wandb and filepath:
                            wandb.log({"Visualization": wandb.Image(filepath)}, step=global_step)
                            
                chunk_steps = chunk_end - chunk_start
                total_seq_loss = total_seq_loss / chunk_steps
                
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
                
                if global_step == args.unfreeze_step_1 and mode == "supervised":
                    print(f"\n🔓 [Step {global_step}] 解冻 Stage 5 高层语义...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage5" in name: param.requires_grad = True
                elif global_step == args.unfreeze_step_2 and mode == "supervised":
                    print(f"\n🔓 [Step {global_step}] 解冻 Stage 4 中层特征...")
                    for name, param in model.segmenter.named_parameters():
                        if "stage4" in name: param.requires_grad = True
                        
                global_step += 1
                epoch_loss_sum += total_seq_loss.item()
                
                if global_step % 10 == 0:
                    elapsed = time.time() - start_time
                    tot_val = total_seq_loss.item()
                    cs = chunk_steps
                    
                    print(f"[{elapsed:.1f}s] E{epoch} S{global_step} [{mode[:3]}] | Tot:{tot_val:.4f} | "
                          f"Obj:{loss_dict_acc['Obj'].item()/cs:.2f} Bx:{loss_dict_acc['Box'].item()/cs:.2f} "
                          f"Msk:{loss_dict_acc['Mask'].item()/cs:.2f} Dep:{loss_dict_acc['Depth'].item()/cs:.2f} "
                          f"Pht:{loss_dict_acc['Photo'].item()/cs:.2f} Ego:{loss_dict_acc['Ego'].item()/cs:.2f} "
                          f"Flw:{loss_dict_acc['Flow'].item()/cs:.2f} Ano:{loss_dict_acc['Anom'].item()/cs:.2f}")
                    
                    if args.use_wandb:
                        log_dict = {f"Loss/{k}": loss_dict_acc[k].item()/cs for k in loss_dict_acc}
                        log_dict.update({
                            "Loss/Total": tot_val,
                            "System/Step": global_step,
                            "System/Epoch": epoch,
                            "System/Mode": 0 if mode == "supervised" else 1,
                            "System/Buffer_Size": len(buffer.buffer)
                        })
                        wandb.log(log_dict, step=global_step)
        
        # --- Epoch 结束 ---
        avg_epoch_loss = epoch_loss_sum / args.steps_per_epoch
        print(f"\n✅ Epoch {epoch} 结束 | 平均 Loss: {avg_epoch_loss:.4f} | Mode: {mode}")
        
        epoch_ckpt_path = args.checkpoint.replace(".pth", f"_epoch_{epoch}.pth")
        torch.save(model.state_dict(), epoch_ckpt_path)
        print(f"💾 已保存: {epoch_ckpt_path}")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
            best_ckpt_path = args.checkpoint.replace(".pth", "_best.pth")
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"🌟 最佳模型 (Loss: {best_loss:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"⚠️ 未优化 ({epochs_without_improvement}/{args.early_stop_patience})")
            
        if epochs_without_improvement >= args.early_stop_patience:
            print(f"\n🛑 早停触发！")
            break
            
    print(f"🎉 训练完成。最佳: {args.checkpoint.replace('.pth', '_best.pth')}")

# =====================================================================
# Colab / GPU 训练参数配置
# =====================================================================
class ColabConfig:
    mode = "train"
    img_size = 256
    seq_len = 12
    
    batch_size = 6
    max_buffer_size = 64         
    max_samples = None           
    
    num_workers = 0
    vis_interval = 100            
    compile_model = False
    epochs = 100
    steps_per_epoch = 1000
    early_stop_patience = 3
    unfreeze_step_1 = 200
    unfreeze_step_2 = 1000
    
    lr = 1e-4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = "tao_not_42_weights.pth"
    yolo_weights = "yolov8m-seg.pt"
    freeze = True
    use_wandb = True
    wandb_project = "TAO-NOT-42"
    
    # Stage 2 自监督微调: 设置为 epoch 数 (例如 5) 以在该 epoch 后切换
    finetune_after_epoch = None

if __name__ == "__main__":
    args = ColabConfig()
    print("====== TAO-NOT-42 V12 配置 ======")
    for k, v in vars(ColabConfig).items():
        if not k.startswith("__"): print(f"  {k}: {v}")
    print("==================================")
    if args.mode == "train":
        train_model(args)