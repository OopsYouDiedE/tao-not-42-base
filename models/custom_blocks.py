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


# =====================================================================
# 简单的功能组合头模块 (Simple Combined Decoders)
# 无复杂跳转，仅参数化映射
# =====================================================================

class SE3TwistDecoder(nn.Module):
    """
    预测稀疏锚点处的局部物体级三维刚体运动旋量 twist (6 维) 和 相对中心偏移量 offset (2 维)。
    
    物理设计与自监督约束机理：
    - twist 前 3 维表示平移速度向量 v，后 3 维表示旋转角速度向量 omega。
    - 虽然解码器本身是全连接或卷积映射，但在下游的三维运动重投影计算中，其输出严格遵循刚体运动学方程：
      dX = v + omega x X1
    - 由于平移分量 v 的贡献与空间坐标 X1 无关，而旋转分量 omega 的贡献与 X1 呈叉乘的距离线性缩放关系，
      两者的物理响应特性完全不同。
    - 因此，当计算重投影损失（SSIM 光度一致性损失）时，梯度反向传播会根据物理投影的反投影坐标特征，
      自动且唯一地约束并迫使网络将前 3 维学习为平移速度，将后 3 维学习为旋转角速度，实现无监督物理语义的自动解耦。
    """
    def __init__(self, c1, c2=8):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(c1, max(c1 // 2, 32), 3),
            nn.Conv2d(max(c1 // 2, 32), c2, 1)
        )
        
    def forward(self, x):
        # x: [B, C, H, W]
        out = self.conv(x)
        # 返回刚体 twist 旋量 [v (0:3), omega (3:6)]，以及偏移量 [offset (6:8)]
        return out[:, :6, ...], out[:, 6:8, ...]

class DepthDecoder(nn.Module):
    """预测稀疏点位的逆深度 (1维)。"""
    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(c1, max(c1 // 4, 16), 3),
            nn.Conv2d(max(c1 // 4, 16), 1, 1)
        )
        
    def forward(self, x):
        return F.softplus(self.conv(x))

class CoverageMaskDecoder(nn.Module):
    """预测稀疏点位的 Mask 权重向量，用于软分配。"""
    def __init__(self, c1, mask_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(c1, max(c1 // 2, 64), 3),
            nn.Conv2d(max(c1 // 2, 64), mask_dim, 1)
        )
        
    def forward(self, x):
        return self.conv(x)

class UIMaskDecoder(nn.Module):
    """预测 UI 遮挡概率掩码 (1维)。"""
    def __init__(self, c1):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(c1, max(c1 // 4, 16), 3),
            nn.Conv2d(max(c1 // 4, 16), 1, 1)
        )
        
    def forward(self, x):
        return torch.sigmoid(self.conv(x))

class PrototypeMaskDecoder(nn.Module):
    """在最高分辨率/底层特征生成密集 Prototype 画布。"""
    def __init__(self, c1, mask_dim=32):
        super().__init__()
        # 接收底层特征如 P2
        self.net = nn.Sequential(
            Conv(c1, c1 // 2, 3),
            nn.Upsample(scale_factor=2.0, mode='bilinear', align_corners=False),
            Conv(c1 // 2, mask_dim, 1)
        )
        
    def forward(self, x):
        return self.net(x)

class GlobalEgoMotionDecoder(nn.Module):
    """预测全局相机的 SE(3) 自我运动，输出结构化且绝对正交的 SO(3)。"""
    def __init__(self, c1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(c1, 64),
            nn.SiLU(),
            nn.Linear(64, 9)
        )
        
    def forward(self, x):
        b = x.size(0)
        out = self.fc(self.pool(x).view(b, -1))
        t = torch.tanh(out[:, :3]) * 5.0
        rot6d_raw = out[:, 3:9]
        
        identity_6d = torch.tensor([1.0, 0, 0, 0, 1.0, 0], dtype=x.dtype, device=x.device)
        rot6d = identity_6d.unsqueeze(0) + torch.tanh(rot6d_raw) * 0.5
        
        R = rot6d_to_matrix(rot6d)
        T_4x4 = make_4x4_transform(R, t)
        
        return {
            "rot6d": rot6d,
            "R": R,
            "t": t,
            "T": T_4x4
        }

# =====================================================================
# 3D 几何投影与复合模块 (Geometry & Composer Blocks)
# =====================================================================

def make_4x4_transform(R, t):
    B = R.size(0)
    T = torch.eye(4, device=R.device, dtype=R.dtype).unsqueeze(0).repeat(B, 1, 1)
    T[:, :3, :3] = R
    T[:, :3, 3] = t.view(B, 3)
    return T

def rot6d_to_matrix(x, eps=1e-6):
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=eps)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(a2 - a2_proj, dim=-1, eps=eps)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)

def meshgrid(B, H, W, dtype, device):
    ys, xs = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype), indexing='ij')
    ones = torch.ones_like(xs)
    return torch.stack([xs, ys, ones], dim=0).unsqueeze(0).repeat(B, 1, 1, 1)

def backproject(depth, K_inv):
    B, _, H, W = depth.shape
    grid = meshgrid(B, H, W, depth.dtype, depth.device)
    grid_flat = grid.view(B, 3, -1)
    cam_coords = torch.bmm(K_inv, grid_flat)
    cam_coords = cam_coords.view(B, 3, H, W)
    return cam_coords * depth

def project(points, K):
    B, _, H, W = points.shape
    points_flat = points.view(B, 3, -1)
    pixel_coords = torch.bmm(K, points_flat)
    pixel_coords = pixel_coords.view(B, 3, H, W)
    z = pixel_coords[:, 2:3, :, :].clamp(min=1e-3)
    return pixel_coords[:, :2, :, :] / z

def transform_se3(points, T):
    B, _, H, W = points.shape
    points_flat = points.view(B, 3, -1)
    ones = torch.ones(B, 1, H * W, device=points.device, dtype=points.dtype)
    points_homo = torch.cat([points_flat, ones], dim=1)
    points_trans = torch.bmm(T, points_homo)
    return points_trans[:, :3, :].view(B, 3, H, W)

class RigidFlowProjector(nn.Module):
    """严格遵循 3D 几何投影等式：depth + camera SE(3) + K -> rigid flow"""
    def __init__(self):
        super().__init__()
        
    def forward(self, inv_depth, T_cam, K, K_inv):
        B, _, H, W = inv_depth.shape
        depth = 1.0 / (inv_depth + 1e-6)
        X1 = backproject(depth, K_inv)
        X2 = transform_se3(X1, T_cam)
        uv2 = project(X2, K)
        grid = meshgrid(B, H, W, inv_depth.dtype, inv_depth.device)[:, :2, :, :]
        flow = uv2 - grid
        return flow

class ObjectSE3Composer(nn.Module):
    """使用 Sparse Splatting 将对象锚点参数映射到密集运动场"""
    def __init__(self, tau=1.0):
        super().__init__()
        self.tau = tau
        
    def forward(self, sparse_twists, mask_weights, prototypes):
        B, C, H, W = prototypes.shape
        P = F.normalize(prototypes.view(B, C, -1).transpose(1, 2), dim=-1)
        
        _, _, Ha, Wa = mask_weights.shape
        N = Ha * Wa
        W_w = F.normalize(mask_weights.view(B, C, N).transpose(1, 2), dim=-1)
        
        A = torch.softmax((torch.bmm(P, W_w.transpose(1, 2))) / self.tau, dim=-1)
        
        anchor_twists = sparse_twists.view(B, 6, N).transpose(1, 2)
        dense_twist = torch.bmm(A, anchor_twists).transpose(1, 2).view(B, 6, H, W)
        
        dense_obj_mask, _ = A.max(dim=-1)
        dense_obj_mask = dense_obj_mask.view(B, 1, H, W)
        
        return dense_twist, dense_obj_mask

class ResidualFlowDecoder(nn.Module):
    """带振幅截断的安全残差流解码器"""
    def __init__(self, c1, max_residual=5.0):
        super().__init__()
        self.max_residual = max_residual
        self.net = nn.Sequential(
            Conv(c1, max(c1 // 2, 32), 3),
            nn.Conv2d(max(c1 // 2, 32), 2, 1)
        )
    def forward(self, x):
        return self.max_residual * torch.tanh(self.net(x))
