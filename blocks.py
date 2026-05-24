import torch
import torch.nn as nn
import torch.nn.functional as F

def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p




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
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )
        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    def __init__(self, c: int, attn_ratio: float = 0.5, num_heads: int = 4, shortcut: bool = True):
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
        self.cv1 = nn.Sequential(Conv(c1, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, self.c * 2, 1, 1))
        self.cv2 = nn.Sequential(Conv(self.c * 2, c1, 1, 1), Conv(c1, c1, 3, 1, g=c1), Conv(c1, c1, 1, 1))
        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))
    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Bottleneck(nn.Module):
    def __init__(self, c1: int, c2: int, shortcut: bool = True, g: int = 1, k: tuple[int, int] = (3, 3), e: float = 0.5):
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
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k2(nn.Module):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, e2=1.0, g=1, shortcut=True, attn=False):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)
        if attn:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, e2) if c3k else Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e2) for _ in range(n-1))
            self.m.append(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64))
        else:
            self.m = nn.ModuleList(C3k(self.c, self.c, 2, shortcut, g, e2) if c3k else Bottleneck(self.c, self.c, shortcut, g, k=(3, 3), e=e2) for _ in range(n))
    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))

class TimeAwareConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, num_frequencies=8):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.num_frequencies = num_frequencies
        
        # 基频设置为 16s (Base Period T = 16s, Base Frequency = 2*pi / 16 = pi / 8)
        # 固定倍率为 2.0
        base_period = 16.0
        base_omega = (2.0 * torch.pi) / base_period
        self.register_buffer('frequencies', 2.0 ** torch.arange(num_frequencies) * base_omega)
        
        time_embed_dim = num_frequencies * 2
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, 64),
            nn.SiLU(),
            nn.Linear(64, hidden_channels * 2)
        )
        
        gate_channels = input_channels + hidden_channels
        self.update_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.reset_gate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)
        self.candidate = nn.Conv2d(gate_channels, hidden_channels, 3, padding=1)

    def forward(self, x, dt, state=None):
        if state is None:
            state = x.new_zeros(x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
        elif state.shape[-2:] != x.shape[-2:]:
            state = F.interpolate(state, size=x.shape[-2:], mode="bilinear", align_corners=False)
            
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
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * modulated_state], dim=1)))
        
        return (1.0 - update) * modulated_state + update * candidate

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
            Bottleneck(channels, channels, shortcut=True),
            Bottleneck(channels, channels, shortcut=True),
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
            Bottleneck(ch_p3, ch_p3, shortcut=True)
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


class Proto26(nn.Module):
    def __init__(self, ch=(), c_=256, c2=32, nc=80):
        super().__init__()
        self.cv1 = Conv(c_, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2, k=1)
        self.feat_refine = nn.ModuleList(Conv(x, ch[0], k=1) for x in ch[1:])
        self.feat_fuse = Conv(ch[0], c_, k=3)
        self.semseg = nn.Sequential(Conv(ch[0], c_, k=3), Conv(c_, c_, k=3), nn.Conv2d(c_, nc, 1))

    def forward(self, x):
        feat = x[0]
        for i, m in enumerate(self.feat_refine):
            up_feat = m(x[i + 1])
            up_feat = F.interpolate(up_feat, size=feat.shape[2:], mode='nearest')
            feat = feat + up_feat
        p = self.cv3(self.cv2(self.upsample(self.cv1(self.feat_fuse(feat)))))
        semseg = self.semseg(feat)
        return p, semseg

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
        
        self.obj_proj = nn.Conv2d(embed, 1, 1)
        self.one2one_obj_proj = nn.Conv2d(embed, 1, 1)
        
        # Open-Vocabulary Semantic Prompts: Active (Class 0) and Passive (Class 1)
        # Replacing the traditional classification conv with explicit prompt embeddings
        self.class_prompts = nn.Parameter(torch.randn(2, embed))

    def forward(self, x):
        proto_out, semseg = self.proto(x)
        boxes = [self.cv2[i](x[i]) for i in range(len(x))]
        scores = [self.cv3[i](x[i]) for i in range(len(x))]
        mc = [self.cv5[i](x[i]) for i in range(len(x))]
        
        boxes_o2o = [self.one2one_cv2[i](x[i]) for i in range(len(x))]
        scores_o2o = [self.one2one_cv3[i](x[i]) for i in range(len(x))]
        mc_o2o = [self.one2one_cv5[i](x[i]) for i in range(len(x))]
        
        features = x[0]
        obj_foreground = self.obj_proj(scores[0])
        obj_foreground_o2o = self.one2one_obj_proj(scores_o2o[0])
        
        # Explicit Dot Product Matching: Object Embeddings * Prompt Embeddings
        # scores[0] is (B, embed, H, W)
        # class_prompts is (2, embed)
        norm_scores = F.normalize(scores[0], p=2, dim=1)
        norm_prompts = F.normalize(self.class_prompts, p=2, dim=1)

        # 计算余弦相似度并乘以缩放温度系数 (如 10.0，使 sigmoid 激活具有更好的区分度)
        cls_scores = torch.einsum('b c h w, k c -> b k h w', norm_scores, norm_prompts) * 10.0
        cls_scores_o2o = torch.einsum('b c h w, k c -> b k h w', scores_o2o[0], self.class_prompts)
        
        return {
            'features': features,
            'objectness': obj_foreground,
            'classification': cls_scores,
            'boxes': boxes[0],
            'mask_coefficients': mc[0],
            'o2o_objectness': obj_foreground_o2o,
            'o2o_classification': cls_scores_o2o,
            'o2o_boxes': boxes_o2o[0],
            'o2o_mask_coefficients': mc_o2o[0],
            'mask_prototypes': proto_out
        }


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    def forward_fuse(self, x):
        return self.act(self.conv(x))

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
