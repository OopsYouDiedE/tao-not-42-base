import torch
import cv2
import tests.mock_mamba
tests.mock_mamba.inject_mock_mamba()

from models.tao_core import MyYOLOE
from ultralytics import YOLOE

img = cv2.imread("bus.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (640, 640))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0

official_model = YOLOE("yoloe-26s-seg-pf.pt")
official_net = official_model.model
official_net.eval()
with torch.no_grad():
    y_off = []
    x = img_tensor
    for m in official_net.model:
        if m.f != -1:
            x = y_off[m.f] if isinstance(m.f, int) else [x if j == -1 else y_off[j] for j in m.f]
        x = m(x)
        y_off.append(x if m.i in official_net.save else None)

net = MyYOLOE()
from models.yolo_blocks import Bottleneck, C3k2, C3k
for i, m in enumerate(net.model):
    if isinstance(m, C3k2) and hasattr(m, 'm') and len(m.m) > 0 and isinstance(m.m[0], C3k):
        c3k = m.m[0]
        if len(c3k.m) == 1:
            c_ = c3k.m[0].cv1.conv.in_channels
            c2_out = c3k.m[0].cv2.conv.out_channels
            c3k.m.append(Bottleneck(c2_out, c2_out, shortcut=True, g=1, k=(3, 3), e=1.0))

import torch.nn as nn
for name, module in net.named_modules():
    if module.__class__.__name__ == 'Conv':
        c1, c2 = module.conv.in_channels, module.conv.out_channels
        k, s, p = module.conv.kernel_size, module.conv.stride, module.conv.padding
        g, d = module.conv.groups, module.conv.dilation
        new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
        parts = name.split('.')
        if len(parts) > 1:
            parent = net
            for p_name in parts[:-1]:
                parent = getattr(parent, p_name)
            module.conv = new_conv
            module.bn = nn.Identity()
        else:
            module.conv = new_conv
            module.bn = nn.Identity()

net.model[-1] = official_net.model[-1]

ckpt = torch.load("yoloe-26s-seg-pf.pt", map_location='cpu', weights_only=False)
state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']
mapped_state_dict = {
    k.replace("model.model.", "model.") if k.startswith("model.model.") else k: v
    for k, v in state_dict.items()
}
model_state = net.state_dict()
loaded = {k: v for k, v in mapped_state_dict.items() if k in model_state and model_state[k].shape == v.shape}
net.load_state_dict(loaded, strict=False)
net.eval()

with torch.no_grad():
    x = img_tensor
    y_ours = []
    for i, m in enumerate(net.model):
        if i in net.routes:
            f = net.routes[i]
            x = m([x if j == -1 else y_ours[j] for j in f] if isinstance(f, list) else (y_ours[f] if f != -1 else x))
        else:
            x = m(x)
        y_ours.append(x)

# Print differences up to layer 22
for i in range(23):
    if y_off[i] is not None and y_ours[i] is not None:
        diff = torch.max(torch.abs(y_off[i] - y_ours[i])).item()
        print(f"Layer {i} Diff: {diff:.8f}")

EOF
