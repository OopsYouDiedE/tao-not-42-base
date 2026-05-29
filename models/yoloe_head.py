import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.yolo_blocks import *

# =====================================================================
# 1. 辅助 Transformer 与 提示词嵌入组件
# =====================================================================

class SwiGLUFFN(nn.Module):
    """SwiGLU 前向反馈网络，用于 Transformer 架构。"""

    def __init__(self, gc, ec, e=4):
        super().__init__()
        self.w12 = nn.Linear(gc, e * ec)
        self.w3 = nn.Linear(e * ec // 2, ec)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)


class Residual(nn.Module):
    """残差连接封装。"""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return x + self.m(x)


class SAVPE(nn.Module):
    """Spatial-Aware Visual Prompt Embedding (空间感知视觉提示嵌入)。"""

    def __init__(self, ch, c3, embed):
        super().__init__()
        # cv1: 特征增强路径
        self.cv1 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3),
                Conv(c3, c3, 3),
                nn.Upsample(scale_factor=2**i) if i > 0 else nn.Identity()
            ) for i, x in enumerate(ch)
        )
        # cv2: 特征映射路径
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 1),
                nn.Upsample(scale_factor=2**i) if i > 0 else nn.Identity()
            ) for i, x in enumerate(ch)
        )
        self.c = 16
        self.cv3 = nn.Conv2d(3 * c3, embed, 1)
        self.cv4 = nn.Conv2d(3 * c3, self.c, 3, padding=1)
        self.cv5 = nn.Conv2d(1, self.c, 3, padding=1)
        self.cv6 = nn.Sequential(
            Conv(2 * self.c, self.c, 3), nn.Conv2d(self.c, self.c, 3, padding=1))

    def forward(self, x, vp):
        # 简化版推理逻辑，实际权重加载后将覆盖行为
        raise NotImplementedError("SAVPE forward pass is not fully implemented.")

# =====================================================================
# 2. 预测头组件 (Head Components)
# =====================================================================


class LRPCHead(nn.Module):
    """Lightweight Region Proposal and Classification Head (对齐官方 ultralytics 实现)。"""

    def __init__(self, vocab, pf, loc, enabled=True):
        super().__init__()
        self.vocab = self.conv2linear(vocab) if enabled and isinstance(vocab, nn.Conv2d) else vocab
        self.pf = pf
        self.loc = loc
        self.enabled = enabled

    @staticmethod
    def conv2linear(conv: nn.Conv2d) -> nn.Linear:
        """将 1×1 Conv2d 转换为等价的 Linear 层（对齐官方）。"""
        assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
        linear = nn.Linear(conv.in_channels, conv.out_channels)
        linear.weight.data = conv.weight.data.view(conv.out_channels, -1)
        linear.bias.data = conv.bias.data
        return linear

    def forward(self, cls_feat, loc_feat, conf=0.001):
        """处理分类与定位特征，生成检测框和类别分数。

        Args:
            cls_feat: 分类特征图 [B, c3, H, W]
            loc_feat: 定位特征图 [B, c2, H, W]（输入 self.loc 得到 box）
            conf: 置信度阈值（训练时设 0.0 以保留所有 anchor）

        Returns:
            (box_logits, cls_scores, mask): 与官方 LRPCHead.forward 输出格式一致。
        """
        if self.enabled:
            # 官方路径：pf 做 anchor 过滤，vocab Linear 做分类
            pf_score = self.pf(cls_feat)[0, 0].flatten(0)   # [H*W]
            mask = pf_score.sigmoid() > conf
            cls_flat = cls_feat.flatten(2).transpose(-1, -2)  # [B, H*W, c3]
            if conf > 0:
                cls_score = self.vocab(cls_flat[:, mask])     # [B, N_kept, nc]
            else:
                cls_score = self.vocab(cls_flat * mask.unsqueeze(-1).int())
            return self.loc(loc_feat), cls_score.transpose(-1, -2), mask  # loc:[B,4,H,W], score:[B,nc,N_kept]
        else:
            # Conv2d 版 vocab（lrpc[2]）：直接做空间卷积
            cls_score = self.vocab(cls_feat)                  # [B, nc, H, W]
            loc = self.loc(loc_feat)                          # [B, 4, H, W]
            mask = torch.ones(
                cls_feat.shape[2] * cls_feat.shape[3],
                device=cls_feat.device, dtype=torch.bool
            )
            return loc, cls_score.flatten(2), mask            # [B,4,H,W], [B,nc,H*W], all-True


class Proto26(nn.Module):
    """YOLOE-26 分割原型生成模块。

    注意：semseg_nc 默认 80（对齐官方 COCO 预训练权重 shape），
    与追踪/检测用的 nc=4585 解耦，避免权重加载 shape 不匹配。
    """

    def __init__(self, ch, npr=256, nm=32, nc=80, semseg_nc=80):
        super().__init__()
        self.cv1 = Conv(npr, npr, 3)
        self.upsample = nn.ConvTranspose2d(npr, npr, 2, 2, 0, bias=True)
        self.cv2 = Conv(npr, npr, 3)
        self.cv3 = Conv(npr, nm, 1)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], 1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], npr, 3)
        # semseg_nc=80 与官方预训练权重对齐；追踪目标使用 track_module 独立预测
        self.semseg = nn.Sequential(Conv(ch[0], npr, 3), Conv(
            npr, npr, 3), nn.Conv2d(npr, semseg_nc, 1))

    def forward(self, x):
        feat = x[0]
        for i, f in enumerate(self.feat_refine):
            feat = feat + \
                F.interpolate(f(x[i+1]), size=feat.shape[2:], mode="nearest")
        fused = self.feat_fuse(feat)
        proto = self.cv3(self.cv2(self.upsample(self.cv1(fused))))
        return proto, self.semseg(feat)

# =====================================================================
# 3. YOLOE 分割头 (YOLOESegment26) - 100% 对齐官方结构
# =====================================================================


class YOLOESegment26(nn.Module):
    """完全对齐 yoloe-26s-seg-pf 权重的预测头。"""

    def __init__(self, nc=4585, nm=32, npr=256, embed=512, ch=(), **kwargs):
        super().__init__()
        self.nc = nc
        self.nm = nm
        self.nl = len(ch)
        self.reg_max = 1
        self.register_buffer("stride", torch.tensor([8.0, 16.0, 32.0]))

        # 核心：Prompt-Free 变种中，Dense 预测头 (cv2, cv3) 显式为 None（O2M 分支已在发布权重中删除）
        self.cv2 = None
        self.cv3 = None
        self.cv4 = None
        self.dfl = nn.Identity()

        # 通道设置
        c2 = max((16, ch[0] // 4, self.reg_max * 4))
        c3 = 128  # 依据 s 缩放比

        # 端到端 (One-to-One) 预测路径
        self.one2one_cv2 = nn.ModuleList(nn.Sequential(
            Conv(x, c2, 3), Conv(c2, c2, 3)) for x in ch)
        self.one2one_cv3 = nn.ModuleList(nn.Sequential(
            nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
            nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
            nn.Conv2d(c3, embed, 1)
        ) for x in ch)
        self.one2one_cv4 = nn.ModuleList(BNContrastiveHead(embed) for _ in ch)

        # Transformer 组件
        self.reprta = Residual(SwiGLUFFN(embed, embed))
        self.savpe = SAVPE(ch, c3, embed)

        # 分割组件（semseg_nc=80 对齐官方预训练权重，追踪使用独立 track_module）
        self.proto = Proto26(ch, npr, nm, nc, semseg_nc=80)
        self.cv5 = nn.ModuleList(nn.Sequential(Conv(x, 32, 3), Conv(
            32, 32, 3), nn.Conv2d(32, 32, 1)) for x in ch)
        self.one2one_cv5 = nn.ModuleList(nn.Sequential(
            Conv(x, 32, 3), Conv(32, 32, 3), nn.Conv2d(32, 32, 1)) for x in ch)

        # 词表投影头（LRPC）
        # lrpc.0, lrpc.1: vocab 为 Linear(c3, nc)，对齐官方 shape (nc, c3)
        # lrpc.2: vocab 为 Conv2d(c3, nc, 1)，对齐官方 shape (nc, c3, 1, 1)
        self.lrpc = nn.ModuleList([
            LRPCHead(nn.Linear(c3, self.nc), nn.Conv2d(c3, 1, 1), nn.Conv2d(32, 4, 1)),
            LRPCHead(nn.Linear(c3, self.nc), nn.Conv2d(c3, 1, 1), nn.Conv2d(32, 4, 1)),
            LRPCHead(nn.Conv2d(c3, self.nc, 1), nn.Conv2d(c3, 1, 1), nn.Conv2d(32, 4, 1)),
        ])

    def forward(self, x):
        """前向传播。

        训练时：返回逐尺度特征图字典，供 compute_instance_loss 使用。
        推理时：执行 LRPC 解码路径，返回端到端检测结果，与 test_yoloe_bus.py 兼容。

        架构说明：
            one2one_cv3[i] 是三步 Sequential：
              step[0]: DWConv(x,x,3) + Conv(x,c3,1) → [B, c3=128, H, W]
              step[1]: DWConv(c3,c3,3) + Conv(c3,c3,1) → [B, c3=128, H, W]
              step[2]: Conv2d(c3, embed=512, 1) → [B, 512, H, W]
            lrpc[i].pf 接收 step[1] 输出（c3 维），而非最终 embed 维特征。

        Args:
            x (list[Tensor]): 三个尺度的特征图 [P3, P4, P5]。

        Returns:
            训练时：dict，含 objectness/boxes/box_dist/mask_coefficients/
                    mask_prototypes/classification（供损失函数消费）。
            推理时：((y_tensor, preds_dict), proto)，与官方 head 输出格式一致。
        """
        bs = x[0].shape[0]

        # ── 原型掩膜（训练推理均需要）──────────────────────────────────
        proto, _ = self.proto(x)   # [B, nm, H_p, W_p]

        if self.training:
            # ── 训练模式：逐尺度提取特征图，组装损失所需的 dict ──────────
            obj_maps, box_maps, mask_coef_maps, cls_maps = [], [], [], []
            for i in range(self.nl):
                # 分步执行 one2one_cv3[i]，取中间 c3 维特征供 lrpc 使用
                mid_feat = self.one2one_cv3[i][0](x[i])   # [B, c3=128, H, W]
                mid_feat = self.one2one_cv3[i][1](mid_feat) # [B, c3, H, W]
                # （step[2] 升到 embed=512，推理才需要，训练时节省计算）

                # 1. objectness：lrpc[i].pf 为 1-channel Conv2d，接收 c3 特征
                obj_logit = self.lrpc[i].pf(mid_feat)          # [B, 1, H, W]
                obj_maps.append(obj_logit)

                # 2. box（LRTB 距离）：one2one_cv2 → lrpc.loc（4-channel Conv2d）
                box_feat = self.one2one_cv2[i](x[i])            # [B, c2, H, W]
                box_raw  = self.lrpc[i].loc(box_feat)           # [B, 4, H, W]
                # softplus 保证距离严格为正（GIoU 要求）
                box_pos  = F.softplus(box_raw) + 1e-4
                box_maps.append(box_pos)

                # 3. classification（可选）：lrpc[i].vocab
                # 3. classification：所有 lrpc[i].vocab 均已由 conv2linear 转为 Linear(c3, nc)
                #    需要将空间维铺平，经过 Linear 后再还原
                cls_i = self.lrpc[i].vocab(
                    mid_feat.permute(0, 2, 3, 1)   # [B, H, W, c3]
                ).permute(0, 3, 1, 2)               # [B, nc, H, W]
                cls_maps.append(cls_i)

                # 4. mask coefficients：one2one_cv5
                mc = self.one2one_cv5[i](x[i])                  # [B, nm, H, W]
                mask_coef_maps.append(mc)

            return {
                "objectness":        obj_maps,       # list of [B, 1, H_i, W_i]
                "boxes":             box_maps,       # list of [B, 4, H_i, W_i]，LRTB 格式
                "box_dist":          box_maps,       # reg_max=1，与 boxes 等价（保留梯度）
                "mask_coefficients": mask_coef_maps, # list of [B, nm, H_i, W_i]
                "mask_prototypes":   proto,          # [B, nm, H_p, W_p]
                "classification":    cls_maps,       # list of [B, nc, H_i, W_i]
            }

        # ── 推理模式：LRPC 解码，输出与官方格式兼容 ─────────────────────
        boxes_out, scores_out, index_out = [], [], []
        for i in range(self.nl):
            mid_feat = self.one2one_cv3[i][0](x[i])
            mid_feat = self.one2one_cv3[i][1](mid_feat)
            loc_feat = self.one2one_cv2[i](x[i])
            box_out, score_out, idx = self.lrpc[i](mid_feat, loc_feat, conf=0.001)
            boxes_out.append(box_out.view(bs, self.reg_max * 4, -1))
            scores_out.append(score_out)
            index_out.append(idx)

        mc_flat = torch.cat(
            [self.one2one_cv5[i](x[i]).view(bs, 32, -1) for i in range(self.nl)], dim=2
        )
        index_cat = torch.cat(index_out)
        raw_boxes = torch.cat(boxes_out, 2)  # [B, 4, 8400]

        # ── 生成锚点并解码为原图物理坐标 (xyxy 格式) ──
        anchors, strides = self.make_anchors(x, self.stride)
        anchors = anchors.unsqueeze(0).transpose(1, 2)  # [1, 2, 8400]
        strides = strides.unsqueeze(0).transpose(1, 2)  # [1, 1, 8400]
        
        # dist2bbox: raw_boxes 是 ltrb 距离
        lt, rb = raw_boxes.chunk(2, dim=1)
        x1y1 = anchors - lt
        x2y2 = anchors + rb
        decoded_boxes = torch.cat((x1y1, x2y2), dim=1) * strides  # [B, 4, 8400]

        preds_dict = dict(
            boxes=decoded_boxes[..., index_cat],
            scores=torch.cat(scores_out, 2),
            feats=x,
            index=index_cat,
            mask_coefficient=mc_flat[..., index_cat],
        )

        # 拼接为 [B, 4+nc_kept+nm, N_kept] 的推理结果张量
        scores_sig = preds_dict["scores"].sigmoid()
        y = torch.cat([preds_dict["boxes"], scores_sig, preds_dict["mask_coefficient"]], dim=1)
        
        if getattr(self, "end2end", True):
            y = self.postprocess(y.permute(0, 2, 1))
            
        return (y, preds_dict), proto

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """端到端后处理，获取 top-k 检测结果。
        preds: [B, N, 4 + nc + nm] (最后一个维度格式：xyxy, scores, mask_coef)
        返回: [B, max_det, 6 + nm] (最后一个维度格式：xyxy, max_score, cls_idx, mask_coef)
        """
        boxes, scores, mask_coefficient = preds.split([4, self.nc, self.nm], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, getattr(self, "max_det", 300))
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        mask_coefficient = mask_coefficient.gather(dim=1, index=idx.repeat(1, 1, self.nm))
        return torch.cat([boxes, scores, conf, mask_coefficient], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int):
        """获取分数最高的前 max_det 个索引。"""
        batch_size, anchors, nc = scores.shape
        k = min(max_det, anchors)
        if getattr(self, "agnostic_nms", False):
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]
        return scores[..., None], (index % nc)[..., None].float(), idx

    @staticmethod
    def make_anchors(feats, strides, grid_cell_offset=0.5):
        """根据特征图尺寸生成中心点锚点。"""
        anchor_points, stride_tensor = [], []
        dtype, device = feats[0].dtype, feats[0].device
        for i, stride in enumerate(strides):
            _, _, h, w = feats[i].shape
            sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset
            sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        return torch.cat(anchor_points), torch.cat(stride_tensor)


class BNContrastiveHead(nn.Module):
    """带批归一化的对比学习头。"""

    def __init__(self, embed_dims):
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias
