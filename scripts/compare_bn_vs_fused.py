"""比较方案A（Conv+BN融合加载）和方案B（直接替换为Conv+bias）的数值差异与训练影响。"""
import sys; sys.path.insert(0, '.')
import tests.mock_mamba as mm; mm.inject_mock_mamba()
from models.tao_core import MyYOLOE
from models.yolo_blocks import Conv as OurConv
from ultralytics import YOLOE
import torch, torch.nn as nn
import cv2, numpy as np

off = YOLOE('yoloe-26s-seg-pf.pt').model.eval()

# === 方案A：保持 Conv+BN（BN融合近似加载）===
from tests.test_yoloe_bus import load_official_weights_to_ours
ours_bn = MyYOLOE().eval()
load_official_weights_to_ours(ours_bn, 'yoloe-26s-seg-pf.pt')

# === 方案B：将所有 Conv+BN 替换为 Conv+bias（结构等同官方）===
def build_fused_model(official_sd):
    model = MyYOLOE()
    for name, module in model.named_modules():
        if isinstance(module, OurConv):
            c1 = module.conv.in_channels
            c2 = module.conv.out_channels
            k  = module.conv.kernel_size
            s  = module.conv.stride
            p  = module.conv.padding
            g  = module.conv.groups
            d  = module.conv.dilation
            new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
            new_conv.to(module.conv.weight.device)
            module.conv = new_conv
            module.bn   = nn.Identity()
    # 直接加载：此时 key 完全相同，shape 完全匹配
    our_sd = model.state_dict()
    new_sd = dict(our_sd)
    matched = 0
    for k, v in official_sd.items():
        if k in our_sd and our_sd[k].shape == v.shape:
            new_sd[k] = v.clone()
            matched += 1
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model, matched

ckpt = torch.load('yoloe-26s-seg-pf.pt', map_location='cpu', weights_only=False)
official_sd = ckpt['model'].state_dict()
ours_fused, matched = build_fused_model(official_sd)
print(f"[方案B] 直接精确匹配: {matched}/{len(official_sd)} = {matched/len(official_sd)*100:.1f}%")

# === 推理对比 ===
img = cv2.imread('bus.jpg')
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1)
inp = torch.from_numpy(np.ascontiguousarray(img).astype(np.float32) / 255.0).unsqueeze(0)

routes = {12: [-1, 6], 15: [-1, 4], 18: [-1, 13], 21: [-1, 10]}

def run_backbone(model, x):
    y = []; xc = x.clone()
    for i, m in enumerate(model.model):
        if i == 23: break
        if i in routes:
            f = routes[i]
            xc = m([xc if j == -1 else y[j] for j in f])
        else:
            xc = m(xc)
        y.append(xc)
    return y[0], y[1], y[16], y[19], y[22]

with torch.no_grad():
    off_feats  = run_backbone(off, inp)
    bn_feats   = run_backbone(ours_bn, inp)
    fuse_feats = run_backbone(ours_fused, inp)

names = ['f1(L0)', 'f2(L1)', 'P3(L16)', 'P4(L19)', 'P5(L22)']
print("\n=== 数值精度对比 ===")
print(f"{'层':<10} | {'方案A BN融合 max_diff':>22} | {'方案B 直接复制 max_diff':>24} | {'A vs B':>12}")
print('-' * 78)
for name, fo, fa, fb in zip(names, off_feats, bn_feats, fuse_feats):
    da = (fo - fa).abs().max().item()
    db = (fo - fb).abs().max().item()
    ab = (fa - fb).abs().max().item()
    print(f"{name:<10} | {da:>22.2e} | {db:>24.2e} | {ab:>12.2e}")

# === 参数量对比 ===
print("\n=== 参数量对比 ===")
pa = sum(p.numel() for p in ours_bn.parameters())
pb = sum(p.numel() for p in ours_fused.parameters())
print(f"方案A (Conv+BN): {pa:,} 参数")
print(f"方案B (Conv+bias): {pb:,} 参数")
print(f"差异: {pa - pb:,} 个（BN 的 weight/bias/running_mean/var）")

# === 推理速度对比（简单 benchmark）===
print("\n=== 推理速度对比（100次前向，CPU）===")
import time

with torch.no_grad():
    # 预热
    for _ in range(5):
        run_backbone(ours_bn, inp)
        run_backbone(ours_fused, inp)
    
    t0 = time.perf_counter()
    for _ in range(30):
        run_backbone(ours_bn, inp)
    ta = (time.perf_counter() - t0) / 30 * 1000

    t0 = time.perf_counter()
    for _ in range(30):
        run_backbone(ours_fused, inp)
    tb = (time.perf_counter() - t0) / 30 * 1000

print(f"方案A (Conv+BN):  {ta:.1f} ms/inference")
print(f"方案B (Conv+bias): {tb:.1f} ms/inference")
print(f"方案B 加速: {(ta-tb)/ta*100:.1f}%")

# === 梯度流对比（训练场景）===
print("\n=== 梯度流健康度对比（单次反向传播）===")
ours_bn.train()
ours_fused.train()

test_input = torch.randn(1, 3, 256, 256, requires_grad=False)

# 方案A 梯度
loss_a = run_backbone(ours_bn, test_input)[2].mean()
loss_a.backward()
grads_a = []
for n, p in ours_bn.named_parameters():
    if p.grad is not None and 'conv.weight' in n and p.grad.abs().max() > 0:
        grads_a.append(p.grad.abs().mean().item())

# 方案B 梯度
loss_b = run_backbone(ours_fused, test_input)[2].mean()
loss_b.backward()
grads_b = []
for n, p in ours_fused.named_parameters():
    if p.grad is not None and 'conv.weight' in n and p.grad.abs().max() > 0:
        grads_b.append(p.grad.abs().mean().item())

import statistics
if grads_a and grads_b:
    print(f"方案A Conv grad mean: {statistics.mean(grads_a):.2e}  std: {statistics.stdev(grads_a):.2e}  max: {max(grads_a):.2e}  min: {min(grads_a):.2e}")
    print(f"方案B Conv grad mean: {statistics.mean(grads_b):.2e}  std: {statistics.stdev(grads_b):.2e}  max: {max(grads_b):.2e}  min: {min(grads_b):.2e}")
    ratio = statistics.stdev(grads_b) / (statistics.stdev(grads_a) + 1e-10)
    print(f"梯度方差比 B/A: {ratio:.2f}x  {'（方案B梯度更不均匀）' if ratio > 1.5 else '（梯度均匀性相近）'}")
