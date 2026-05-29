import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_blocks import *

# =====================================================================
# 时空与自定义任务预测头组件 (Custom Task-Specific Heads)
# =====================================================================


class SpatioTemporalMambaBlock(nn.Module):
    """时空 Mamba 模块，由项目代码定义。"""

    def __init__(self, channels, num_frequencies=16):
        super().__init__()
        # 此处使用 Mamba 逻辑，在 Mock 中会被注入
        from mamba_ssm import Mamba
        self.channels = channels
        self.conv3d = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(channels)
        self.act = nn.SiLU(inplace=True)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
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
        mamba_out = self.mamba(x_flat)
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
        self.depth_branch = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), Conv(
            ch_f1, ch_f1, 3), Conv(ch_f1, ch_f1 // 2, 3), nn.Conv2d(ch_f1 // 2, 1, 3, padding=1))
        self.flow_up = nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False)
        self.flow_conv = nn.Sequential(Conv(ch_f1 + pose_dim, ch_f1, 3), Conv(
            ch_f1, ch_f1 // 2, 3), nn.Conv2d(ch_f1 // 2, 2, 3, padding=1))

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


class TrackQueryModule(nn.Module):
    def __init__(self, feat_channels=128, num_queries=32, num_heads=4, nc=80, nm=32):
        super().__init__()
        from mamba_ssm import Mamba
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.query_mamba = Mamba(
            d_model=feat_channels, d_state=16, d_conv=4, expand=2)
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
        return {"track_boxes": self.box_head(q_temp), "track_classes": self.cls_head(q_temp), "track_alive": self.alive_head(q_temp), "track_masks": self.mask_head(q_temp)}
