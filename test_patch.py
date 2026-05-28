import torch
import torch.nn as nn
import tests.mock_mamba
tests.mock_mamba.inject_mock_mamba()
from models.tao_core import MyYOLOE
from models.yolo_blocks import Bottleneck, C3k2, C3k

net = MyYOLOE()

# 1. Patch backbone C3k2
for i, m in enumerate(net.model):
    if isinstance(m, C3k2) and hasattr(m, 'm') and len(m.m) > 0 and isinstance(m.m[0], C3k):
        c3k = m.m[0]
        if len(c3k.m) == 1:
            c_ = c3k.m[0].cv1.conv.in_channels
            c2_out = c3k.m[0].cv2.conv.out_channels
            c3k.m.append(Bottleneck(c2_out, c2_out, shortcut=True, g=1, k=(3, 3), e=1.0))

# 2. Convert Conv layers to match checkpoint (fuse BN)
for name, module in net.named_modules():
    if module.__class__.__name__ == 'Conv':
        c1 = module.conv.in_channels
        c2 = module.conv.out_channels
        k = module.conv.kernel_size
        s = module.conv.stride
        p = module.conv.padding
        g = module.conv.groups
        d = module.conv.dilation
        new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
        parts = name.split('.')
        if len(parts) > 1:
            parent = net
            for p_name in parts[:-1]:
                parent = getattr(parent, p_name)
            module.conv = new_conv
            module.bn = nn.Identity()

# 3. Patch YOLOESegment26 (model.23)
head = net.model[23]
del head.cv2
del head.cv3
del head.attr_heads

ckpt = torch.load("yoloe-26s-seg-pf.pt", map_location='cpu', weights_only=False)
state_dict = ckpt['model'].state_dict() if hasattr(ckpt['model'], 'state_dict') else ckpt['model']

class DummyContainer(nn.Module):
    pass

head.savpe = DummyContainer()
head.reprta = DummyContainer()

for k, v in state_dict.items():
    new_k = k.replace("model.model.", "model.") if k.startswith("model.model.") else k
    if new_k.startswith("model.23.savpe.") or new_k.startswith("model.23.reprta."):
        sub_k = new_k.replace("model.23.", "")
        parts = sub_k.split('.')
        curr = head
        for part in parts[:-1]:
            if not hasattr(curr, part):
                setattr(curr, part, DummyContainer())
            curr = getattr(curr, part)
        setattr(curr, parts[-1], nn.Parameter(torch.zeros_like(v)))

model_state = net.state_dict()
loaded = {}
skipped_shape = []
unexpected = []

for k, v in state_dict.items():
    new_k = k.replace("model.model.", "model.") if k.startswith("model.model.") else k
    if new_k not in model_state:
        unexpected.append(new_k)
        continue
    if model_state[new_k].shape != v.shape:
        skipped_shape.append((new_k, tuple(v.shape), tuple(model_state[new_k].shape)))
        continue
    loaded[new_k] = v

missing = [k for k in model_state.keys() if k not in loaded]

print("Truly loaded:", len(loaded))
print("Missing:", len(missing))
print("Shape mismatched:", len(skipped_shape))
print("Unexpected:", len(unexpected))

if missing: print("Missing ex:", missing[:5])
if unexpected: print("Unexpected ex:", unexpected[:5])
