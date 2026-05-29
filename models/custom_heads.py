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


# EgoPoseHead is replaced by GlobalEgoMotionDecoder in custom_blocks.py. 
# Code left empty here since SE3PhysicsHead handles it directly.


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
        self.query_norm = nn.LayerNorm(feat_channels)
        self.cross_attn = nn.MultiheadAttention(feat_channels, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(feat_channels)
        
        self.box_head = nn.Sequential(nn.Linear(feat_channels, 64), nn.SiLU(), nn.Linear(64, 4))
        self.cls_head = nn.Linear(feat_channels, nc)
        self.mask_head = nn.Linear(feat_channels, nm)
        self.alive_head = nn.Linear(feat_channels, 1)
        nn.init.constant_(self.alive_head.bias, -4.0)

    def forward(self, st_p3, prev_queries=None):
        B, T, C, H, W = st_p3.shape
        N = self.num_queries
        
        if prev_queries is None:
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
