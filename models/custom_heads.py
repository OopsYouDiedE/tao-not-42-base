import torch
import torch.nn as nn
import torch.nn.functional as F

from models.yolo_blocks import *
from models.custom_blocks import *
# =====================================================================
# 时空与自定义任务预测头组件 (Custom Task-Specific Heads)
# =====================================================================


class SE3PhysicsHead(nn.Module):
    """
    分离式的参数预测与底层密集的分配重构 (Object-Centric Splatting)
    """
    def __init__(self, ch_list, prototype_ch=32):
        super().__init__()
        self.ch_list = ch_list
        self.prototype_ch = prototype_ch
        
        self.se3_decoders = nn.ModuleList([SE3TwistDecoder(c) for c in ch_list])
        self.mask_weight_decoders = nn.ModuleList([CoverageMaskDecoder(c, prototype_ch) for c in ch_list])
        self.ui_decoders = nn.ModuleList([UIMaskDecoder(c) for c in ch_list])
        
        self.ego_motion_head = GlobalEgoMotionDecoder(ch_list[-1])
        self.prototype_head = PrototypeMaskDecoder(ch_list[0], prototype_ch)
        
        self.object_composer = ObjectSE3Composer(tau=1.0)
        self.residual_decoder = ResidualFlowDecoder(ch_list[0])

    def forward(self, features):
        sparse_outputs = []
        for i, feat in enumerate(features):
            se3_twist, pos_offset = self.se3_decoders[i](feat) 
            mask_weights = self.mask_weight_decoders[i](feat)
            p_ui = self.ui_decoders[i](feat)
            
            sparse_outputs.append({
                "se3_twist": se3_twist,
                "pos_offset": pos_offset,
                "mask_weights": mask_weights,
                "p_ui": p_ui
            })
            
        se3_cam = self.ego_motion_head(features[-1])
        prototypes = self.prototype_head(features[0])
        
        # 在高分辨层执行 Object-Centric Splatting 密集场重构
        finest_twists = sparse_outputs[0]["se3_twist"]
        finest_masks = sparse_outputs[0]["mask_weights"]
        dense_obj_twist, dense_obj_mask = self.object_composer(finest_twists, finest_masks, prototypes)
        
        residual_flow = self.residual_decoder(features[0])
        
        return {
            "sparse_anchors": sparse_outputs, 
            "se3_cam": se3_cam,               
            "prototypes": prototypes,
            "dense_obj_twist": dense_obj_twist,
            "dense_obj_mask": dense_obj_mask,
            "residual_flow": residual_flow
        }


class UnifiedGeometryDecoder(nn.Module):
    """
    专门负责多帧双目/时序相关性的深度与遮挡解译。
    彻底废除自由光流（作弊光流）的预测，仅保留几何深度基底。
    """
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48):
        super().__init__()
        self.up1 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_p3, ch_f2, 3))
        self.conv1 = Conv(ch_f2 * 2, ch_f2, 3)
        self.up2 = nn.Sequential(nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_f2, ch_f1, 3))
        self.conv2 = Conv(ch_f1 * 2, ch_f1, 3)
        
        self.depth_branch = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), 
            Conv(ch_f1, ch_f1, 3), 
            Conv(ch_f1, ch_f1 // 2, 3), 
            nn.Conv2d(ch_f1 // 2, 1, 3, padding=1)
        )
        self.conf_branch = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), 
            Conv(ch_f1, ch_f1 // 2, 3), 
            nn.Conv2d(ch_f1 // 2, 1, 3, padding=1)
        )
        self.occ_branch = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear", align_corners=False), 
            Conv(ch_f1, ch_f1 // 2, 3), 
            nn.Conv2d(ch_f1 // 2, 1, 3, padding=1)
        )

    def forward(self, f1, f2, p3):
        x1 = self.conv1(torch.cat([self.up1(p3), f2], dim=1))
        x2 = self.conv2(torch.cat([self.up2(x1), f1], dim=1))
        
        inv_depth = torch.exp(torch.clamp(self.depth_branch(x2), min=-4.6, max=4.6))
        depth_conf = torch.sigmoid(self.conf_branch(x2))
        occ_logits = self.occ_branch(x2)
        
        return {
            "inv_depth": inv_depth,
            "depth_conf": depth_conf,
            "occ_logits": occ_logits
        }



class FeaturePredictorHead(nn.Module):
    def __init__(self, channels=256, action_dim=9):
        super().__init__()
        self.stem = Conv(channels + action_dim, channels, 1)
        self.net = nn.Sequential(
            Bottleneck(channels, channels), 
            Bottleneck(channels, channels), 
            Conv(channels, channels * 2, 3)
        )

    def forward(self, state, action):
        action_map = action.view(*action.shape, 1, 1).expand(-1, -1, state.shape[2], state.shape[3])
        out = self.net(self.stem(torch.cat([state, action_map], dim=1)))
        
        C = out.shape[1] // 2
        pred_feat = out[:, :C]
        uncertainty = F.softplus(out[:, C:]) + 1e-4
        
        return {
            "pred_feat": pred_feat,
            "uncertainty": uncertainty
        }


class TrackQueryModule(nn.Module):
    """持久化状态的时序追踪模块"""
    def __init__(self, feat_channels=128, num_queries=32, num_heads=4, nc=80, nm=32):
        super().__init__()
        self.num_queries = num_queries
        
        from mamba_ssm import Mamba
        self.query_mamba = Mamba(d_model=feat_channels, d_state=16, d_conv=4, expand=2)
            
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.box_prompt_encoder = nn.Sequential(
            nn.Linear(4, feat_channels),
            nn.LayerNorm(feat_channels)
        )
        self.query_norm = nn.LayerNorm(feat_channels)
        self.cross_attn = nn.MultiheadAttention(feat_channels, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(feat_channels)
        
        self.box_head = nn.Sequential(nn.Linear(feat_channels, 64), nn.SiLU(), nn.Linear(64, 4))
        self.cls_head = nn.Linear(feat_channels, nc)
        self.mask_head = nn.Linear(feat_channels, nm)
        self.alive_head = nn.Linear(feat_channels, 1)
        nn.init.constant_(self.alive_head.bias, -4.0)

    def forward(self, st_p3, prev_queries=None, box_prompts=None):
        B, T, C, H, W = st_p3.shape
        N = self.num_queries
        
        if prev_queries is None:
            if box_prompts is not None:
                # 强行注入真值框作为 Prompt（Teacher Forcing）
                queries = self.box_prompt_encoder(box_prompts)
            else:
                queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1).clone()
        else:
            queries = prev_queries.clone()
            
        query_seq = []
        for t in range(T):
            feat_flat = st_p3[:, t].flatten(2).permute(0, 2, 1)
            q_attn, _ = self.cross_attn(queries, feat_flat, feat_flat)
            queries = self.cross_attn_norm(queries + q_attn)
            query_seq.append(queries)
            
        next_queries = queries.detach()
        
        q_seq = torch.stack(query_seq, dim=1)
        q_flat = q_seq.permute(0, 2, 1, 3).reshape(B * N, T, C)
        
        q_temp = self.query_mamba(q_flat)
            
        q_temp = self.query_norm(q_flat + q_temp)
        q_temp = q_temp.view(B, N, T, C).permute(0, 2, 1, 3)
        
        raw_box = self.box_head(q_temp)
        center = torch.sigmoid(raw_box[..., :2])
        wh = F.softplus(raw_box[..., 2:]) + 1e-4
        valid_box = torch.cat([center, wh], dim=-1)
        
        alive_logits = self.alive_head(q_temp)
        
        return {
            "track_boxes": valid_box, 
            "track_classes": self.cls_head(q_temp), 
            "track_alive": alive_logits, 
            "track_masks": self.mask_head(q_temp)
        }, next_queries

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
    points_flat = points.view(B, 3, -1).float()
    K_f = K.float()
    pixel_coords = torch.bmm(K_f, points_flat)
    pixel_coords = pixel_coords.view(B, 3, H, W)
    
    # [FIX] 安全 Z 截断，采用 float32 运算，防止投影奇点（z<=0）时的溢出
    z = pixel_coords[:, 2:3, :, :].clamp(min=0.01)
    uv = pixel_coords[:, :2, :, :] / z
    
    # 截断到安全范围，防止极限拉扯导致转回 float16 时 inf
    uv = uv.clamp(min=-50000.0, max=50000.0)
    return uv.to(points.dtype)

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
        X2 = transform_se3(X1, torch.linalg.inv(T_cam.float()).to(T_cam.dtype))
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
        P = F.normalize(prototypes.view(B, C, -1).transpose(1, 2), dim=-1, eps=1e-4)
        
        _, _, Ha, Wa = mask_weights.shape
        N = Ha * Wa
        W_w = F.normalize(mask_weights.view(B, C, N).transpose(1, 2), dim=-1, eps=1e-4)
        
        A = torch.softmax((torch.bmm(P, W_w.transpose(1, 2))) / self.tau, dim=-1)
        
        anchor_twists = sparse_twists.view(B, 6, N).transpose(1, 2)
        dense_twist = torch.bmm(A, anchor_twists).transpose(1, 2).view(B, 6, H, W)
        
        dense_obj_mask, _ = A.max(dim=-1)
        dense_obj_mask = dense_obj_mask.view(B, 1, H, W)
        
        return dense_twist, dense_obj_mask

class ObjectRigidFlowProjector(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, inv_depth_resized, dense_twist, dense_obj_mask, residual_flow, flow_rigid, K, K_inv):
        h, w = inv_depth_resized.shape[-2:]
        depth = 1.0 / (inv_depth_resized + 1e-6)
        X1 = backproject(depth, K_inv)
        
        dense_twist_resized = F.interpolate(dense_twist, size=(h, w), mode="bilinear", align_corners=False)
        v = dense_twist_resized[:, :3, :, :]
        omega = dense_twist_resized[:, 3:6, :, :]
        
        dX = v + torch.cross(omega, X1, dim=1)
        X2_obj = X1 + dX
        
        uv2_obj = project(X2_obj, K)
        B_T = inv_depth_resized.shape[0]
        grid = meshgrid(B_T, h, w, inv_depth_resized.dtype, inv_depth_resized.device)[:, :2, :, :]
        flow_obj_rigid = uv2_obj - grid
        
        obj_mask = F.interpolate(dense_obj_mask, size=(h, w), mode="bilinear", align_corners=False)
        res_flow = F.interpolate(residual_flow, size=(h, w), mode="bilinear", align_corners=False)
        
        flow_final = flow_rigid + obj_mask * (flow_obj_rigid + res_flow)
        return flow_final

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
