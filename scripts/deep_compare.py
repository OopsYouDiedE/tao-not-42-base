import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ultralytics import YOLOE
import tests.mock_mamba
tests.mock_mamba.inject_mock_mamba()
from models.tao_core import MyYOLOE
from models.yolo_blocks import C3k2, C3k, Conv

def get_module_summary(net):
    summary = []
    for i, m in enumerate(net.model):
        m_type = m.__class__.__name__
        info = {"type": m_type}
        if m_type == "C3k2":
            info["n"] = len(m.m)
            # Check shortcut in first bottleneck
            if hasattr(m.m[0], "m") and isinstance(m.m[0].m, torch.nn.Sequential):
                info["shortcut"] = m.m[0].m[0].add if hasattr(m.m[0].m[0], "add") else "N/A"
            elif hasattr(m.m[0], "add"):
                info["shortcut"] = m.m[0].add
        elif m_type == "Conv":
            info["c2"] = m.conv.out_channels
        summary.append(info)
    return summary

print("--- 骨干网络与 Head 结构深度对比 ---")

# Official
model_off = YOLOE("yoloe-26s-seg-pf.pt").model
summary_off = get_module_summary(model_off)

# Ours
net = MyYOLOE()
summary_ours = get_module_summary(net)

for i in range(len(summary_off)):
    s_off = summary_off[i]
    s_ours = summary_ours[i]
    
    diffs = []
    if s_off["type"] != s_ours["type"]:
        diffs.append(f"类型不同: {s_off['type']} vs {s_ours['type']}")
    else:
        for k in s_off:
            if k in s_ours and s_off[k] != s_ours[k]:
                diffs.append(f"{k} 不同: {s_off[k]} vs {s_ours[k]}")
    
    if diffs:
        print(f"Layer {i:<2}: {' | '.join(diffs)}")

# Head check (Layer 23)
head_ours = net.model[23]
head_off = model_off.model[23]

print("\n--- Head (Layer 23) 详细差异 ---")
print(f"我们的代码中的模块: {dir(head_ours)}")
print(f"官方权重中的模块: {dir(head_off)}")

attrs_ours = set([n for n, _ in head_ours.named_children()])
attrs_off = set([n for n, _ in head_off.named_children()])

print(f"\n我们在代码中多出的模块 (官方没有): {sorted(list(attrs_ours - attrs_off))}")
print(f"官方权重中特有的模块 (我们没有): {sorted(list(attrs_off - attrs_ours))}")

# Check specific crucial attributes
print(f"\nHead.cv2 状态: 我们={type(head_ours.cv2)} | 官方={type(head_off.cv2)}")
print(f"Head.cv3 状态: 我们={type(head_ours.cv3)} | 官方={type(head_off.cv3)}")
