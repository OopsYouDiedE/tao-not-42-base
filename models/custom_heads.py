import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba

from models.yolo_blocks import *




class SpatioTemporalMambaBlock(nn.Module):
    """时空 Mamba 模块，结合了 3D 卷积和 Mamba 状态空间模型。"""
    def __init__(self, channels, num_frequencies=16):
        super().__init__()
        self.channels = channels
        self.conv3d = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn3d = nn.BatchNorm3d(channels)
        self.act = nn.SiLU(inplace=True)
        self.mamba = Mamba(d_model=channels, d_state=16, d_conv=4, expand=2)
        self.norm = nn.LayerNorm(channels)

        # 时间编码频率
        self.register_buffer("frequencies", torch.exp(
            torch.linspace(-5, 3, num_frequencies)))
        self.time_mlp = nn.Sequential(
            nn.Linear(num_frequencies * 2, 64), nn.SiLU(), nn.Linear(64, channels))
        self.gamma = nn.Parameter(torch.tensor([0.1]))

    def forward(self, x, t):
        B, T, C, H, W = x.shape
        # 应用 3D 卷积提取时空特征
        x3d = x.permute(0, 2, 1, 3, 4)
        x3d = self.act(self.bn3d(self.conv3d(x3d)))
        x3d = x3d.permute(0, 2, 1, 3, 4).contiguous()

        # 计算傅里叶时间嵌入
        scaled_time = t.unsqueeze(-1) * self.frequencies.view(1, 1, -1)
        fourier_feats = torch.cat(
            [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
        time_embed = self.time_mlp(fourier_feats)

        # 将时间嵌入注入特征图
        x3d = x3d + time_embed.view(B, T, C, 1, 1)
        x_flat = x3d.permute(0, 3, 4, 1, 2).reshape(B * H * W, T, C)

        # 应用 Mamba 状态空间模型处理序列
        mamba_out = self.mamba(x_flat)
        x_flat = self.norm(x_flat + mamba_out)

        out = x_flat.view(B, H, W, T, C).permute(0, 3, 4, 1, 2).contiguous()
        return x + self.gamma * out


class UnifiedGeometryDecoder(nn.Module):
    """统一几何解码器，用于预测深度图和光流。"""
    def __init__(self, ch_p3=256, ch_f2=96, ch_f1=48, pose_dim=9):
        super().__init__()
        # 上采样和特征融合层
        self.up1 = nn.Sequential(nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_p3, ch_f2, 3))
        self.conv1 = Conv(ch_f2 * 2, ch_f2, 3)
        self.up2 = nn.Sequential(nn.Upsample(
            scale_factor=2.0, mode="bilinear", align_corners=False), Conv(ch_f2, ch_f1, 3))
        self.conv2 = Conv(ch_f1 * 2, ch_f1, 3)

        # 深度预测分支
        self.depth_branch = nn.Sequential(
            nn.Upsample(scale_factor=2.0, mode="bilinear",
                        align_corners=False),
            Conv(ch_f1, ch_f1, 3),
            Conv(ch_f1, ch_f1 // 2, 3),
            nn.Conv2d(ch_f1 // 2, 1, 3, padding=1)
        )

        # 光流预测分支
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

        # 深度图输出
        depth_out = self.depth_branch(x2)

        # 光流输出（如果需要）
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
    """自车姿态预测头，预测平移和 6D 旋转。"""
    def __init__(self, in_channels):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64), nn.SiLU(), nn.Linear(64, 9))
        nn.init.zeros_(self.fc[-1].weight)
        nn.init.zeros_(self.fc[-1].bias)

    def forward(self, x):
        pose = self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))
        # 初始旋转偏移
        rot_6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], device=pose.device,
                              dtype=pose.dtype) + torch.tanh(pose[:, 3:]) * 0.5
        return torch.cat([torch.tanh(pose[:, :3]) * 5.0, rot_6d], dim=1)


class FeaturePredictorHead(nn.Module):
    """特征预测头，用于预测下一时刻的特征，支持异常检测。"""
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
    """端到端追踪模块，使用 Query 机制和 Mamba 处理物体追踪。"""
    def __init__(self, feat_channels=128, num_queries=32, num_heads=4, nc=80, nm=32):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, feat_channels)
        self.query_mamba = Mamba(d_model=feat_channels, d_state=16, d_conv=4, expand=2)
        self.query_norm = nn.LayerNorm(feat_channels)

        self.cross_attn = nn.MultiheadAttention(
            feat_channels, num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(feat_channels)

        # 预测头
        self.box_head = nn.Sequential(
            nn.Linear(feat_channels, 64), nn.SiLU(), nn.Linear(64, 4), nn.Sigmoid())
        self.cls_head = nn.Linear(feat_channels, nc)
        self.mask_head = nn.Linear(feat_channels, nm)
        self.alive_head = nn.Linear(feat_channels, 1)
        nn.init.constant_(self.alive_head.bias, -4.0) # 初始置信度设低

    def forward(self, st_p3):
        B, T, C, H, W = st_p3.shape
        N = self.num_queries
        queries = self.query_embed.weight.unsqueeze(
            0).expand(B, -1, -1).clone()

        query_seq = []
        for t in range(T):
            # 时间步交叉注意力：Query 与 图像特征交互
            feat_flat = st_p3[:, t].flatten(2).permute(0, 2, 1)
            q_attn, _ = self.cross_attn(queries, feat_flat, feat_flat)
            queries = self.cross_attn_norm(queries + q_attn)
            query_seq.append(queries)

        # 使用 Mamba 处理 Query 的时间序列一致性
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

        self.attr_heads = nn.ModuleList([
            nn.Conv2d(x, 2, kernel_size=1) for x in ch
        ])

    def forward(self, x):
        proto_out, semseg = self.proto(x)

        def process_branch(cv2_list, cv3_list, cv5_list):
            boxes, scores, mc, obj, cls = [], [], [], [], []
            for i, f in enumerate(x):
                feat_box = cv2_list[i](f)
                feat_cls = cv3_list[i](f)
                feat_mask = cv5_list[i](f)

                bbox = F.softplus(self.lrpc[i].loc(feat_box)) + 1e-4
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

        attrs = []
        for i, f in enumerate(x):
            attrs.append(self.attr_heads[i](f))

        return {
            "features": x,
            "objectness": obj, "classification": cls, "boxes": boxes, "mask_coefficients": mc,
            "o2o_objectness": obj_o2o, "o2o_classification": cls_o2o, "o2o_boxes": boxes_o2o, "o2o_mask_coefficients": mc_o2o,
            "mask_prototypes": proto_out, "semseg": semseg, "attributes": attrs
        }

# =====================================================================
