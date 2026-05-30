import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_blocks import Conv, Bottleneck

# =====================================================================
# 核心细粒度基础模块 (Core Fine-grained Blocks)
# 遵循高度参数化设计原则
# =====================================================================

class SpatioTemporalMambaBlock(nn.Module):
    """时空 Mamba 模块，参数化版本。"""
    def __init__(self, c1, c2=None, num_frequencies=16):
        super().__init__()
        from mamba_ssm import Mamba
        c2 = c2 or c1
        self.channels = c1
        self.out_channels = c2
        self.conv3d = nn.Conv3d(c1, c1, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(c1)
        self.act = nn.SiLU(inplace=True)
        self.mamba = Mamba(d_model=c1, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(c1)
        
        self.register_buffer("frequencies", torch.exp(torch.linspace(-5, 3, num_frequencies)))
        self.time_mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, 64), 
            nn.SiLU(), 
            nn.Linear(64, c1)
        )
        self.gamma = nn.Parameter(torch.tensor([0.1]))
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x, t):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x3d = x.permute(0, 2, 1, 3, 4)
        x3d = self.act(self.bn3d(self.conv3d(x3d)))
        x3d = x3d.permute(0, 2, 1, 3, 4).contiguous()
        
        scaled_time = t.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        fourier_feats = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        time_embed = self.time_mlp(fourier_feats)
        x3d = x3d + time_embed.view(B, T, C, 1, 1)
        
        x_flat = x3d.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)
        mamba_out = self.mamba(x_flat)
        x_flat = self.norm(x_flat + mamba_out)
        out = x_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        
        # apply residual and project to c2
        res = x + self.gamma * out
        # B,T,C,H,W -> B*T, C, H, W for projection if needed
        res_bt = res.view(B*T, C, H, W)
        proj_bt = self.proj(res_bt)
        return proj_bt.view(B, T, self.out_channels, H, W)


class SpatioTemporalGRUFallback(nn.Module):
    """
    时空 ConvGRU 模块（RAFT 原始设计思想）。
    作为 Mamba 的向下兼容替代品，接口保持完全一致（接受 [B, T, C, H, W] 序列输入）。
    适用于环境不支持 mamba_ssm，或者在边缘设备上推理。
    """
    def __init__(self, c1, c2=None, num_frequencies=16):
        super().__init__()
        c2 = c2 or c1
        self.out_channels = c2
        self.channels = c1
        
        # 时间嵌入模块 (保持与 Mamba 接口一致)
        self.register_buffer("frequencies", torch.exp(torch.linspace(-5, 3, num_frequencies)))
        self.time_mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, 64), 
            nn.SiLU(), 
            nn.Linear(64, c1)
        )
        
        # ConvGRU 门控卷积核
        self.conv_z = nn.Conv2d(c1 * 2, c1, 3, padding=1)
        self.conv_r = nn.Conv2d(c1 * 2, c1, 3, padding=1)
        self.conv_q = nn.Conv2d(c1 * 2, c1, 3, padding=1)
        
        self.proj = nn.Conv2d(c1, c2, 1) if c1 != c2 else nn.Identity()

    def forward(self, x, t):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 频率空间的时间缩放嵌入
        scaled_time = t.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        fourier_feats = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        time_embed = self.time_mlp(fourier_feats) # [B, T, c1]
        
        out_sequence = []
        # 初始化隐藏状态 h_prev 为 0
        h = torch.zeros(B, C, H, W, device=x.device, dtype=x.dtype)
        
        # RNN 序列化解包，逐帧处理
        for i in range(T):
            x_i = x[:, i, :, :, :] + time_embed[:, i, :].view(B, C, 1, 1)
            
            hx = torch.cat([x_i, h], dim=1)
            z = torch.sigmoid(self.conv_z(hx))
            r = torch.sigmoid(self.conv_r(hx))
            
            hx_r = torch.cat([x_i, r * h], dim=1)
            q = torch.tanh(self.conv_q(hx_r))
            
            h = (1 - z) * h + z * q
            out_sequence.append(h)
            
        out = torch.stack(out_sequence, dim=1) # [B, T, C, H, W]
        
        out_bt = out.view(B*T, C, H, W)
        proj_bt = self.proj(out_bt)
        return proj_bt.view(B, T, self.out_channels, H, W)



class TemporalConditioning(nn.Module):
    """基于 FiLM 的时间条件注入模块。"""
    def __init__(self, c1):
        super().__init__()
        self.dt_proj = nn.Sequential(
            nn.Linear(1, c1),
            nn.SiLU(),
            nn.Linear(c1, c1 * 2)
        )
        
    def forward(self, x, dt):
        # x: [B, C, H, W]
        # dt: [B, 1]
        scale_shift = self.dt_proj(dt).unsqueeze(-1).unsqueeze(-1)
        scale, shift = scale_shift.chunk(2, dim=1)
        return x * (1 + scale) + shift


class LocalCorrelationVolume(nn.Module):
    """局部相关性计算模块 (RAFT-lite)。计算 x_t 和 warp(x_t_minus_1) 的相关性。"""
    def __init__(self, c1, search_radius=4):
        super().__init__()
        self.search_radius = search_radius
        self.proj = Conv(c1, c1 // 2, 1) # 降维计算内积
        
    def forward(self, feat_curr, feat_prev):
        # feat_curr, feat_prev: [B, C, H, W]
        f_c = self.proj(feat_curr)
        f_p = self.proj(feat_prev)
        B, C, H, W = f_c.shape
        # 简单实现：使用 unfold 取局部 patch 然后做点积 (B, (2r+1)^2, H, W)
        r = self.search_radius
        pad_fp = F.pad(f_p, (r, r, r, r))
        correlations = []
        for dy in range(2 * r + 1):
            for dx in range(2 * r + 1):
                f_p_shifted = pad_fp[:, :, dy:dy+H, dx:dx+W]
                corr = torch.sum(f_c * f_p_shifted, dim=1, keepdim=True)
                correlations.append(corr)
        return torch.cat(correlations, dim=1) # [B, (2r+1)^2, H, W]


class C2f_SE3Temporal(nn.Module):
    """C2f 风格的时空融合模块，支持 Delta T 条件注入和相关性特征拼接。"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        # x_t 和 h_prev 拼接后通道数为 c1 * 2，加上相关性维度 (假设 radius=4, 则是 81)
        corr_dim = (2 * 4 + 1) ** 2
        self.cv1 = Conv(c1 * 2 + corr_dim, c2, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.cond = TemporalConditioning(self.c)
        self.corr_module = LocalCorrelationVolume(c1, search_radius=4)

    def forward(self, x_t, h_prev, dt):
        # x_t: 当前特征, h_prev: 上一帧特征
        corr_feat = self.corr_module(x_t, h_prev)
        x_cat = torch.cat([x_t, h_prev, corr_feat], dim=1)
        
        y = list(self.cv1(x_cat).chunk(2, 1))
        for m in self.m:
            y_cond = self.cond(y[-1], dt)
            y.append(m(y_cond))
            
        out = self.cv2(torch.cat(y, 1))
        return out, out # returns (output, new_hidden_state)


