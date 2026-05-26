import sys
import types
import torch

# Inject mock mamba_ssm module for Windows environments where compilation fails
mamba_mock = types.ModuleType("mamba_ssm")
class MockMamba(torch.nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)
    def forward(self, x, *args, **kwargs):
        return self.proj(x)
mamba_mock.Mamba = MockMamba
sys.modules["mamba_ssm"] = mamba_mock

import time
from all import TAONot42VisionModel, get_loss_weights, compute_physics_loss

def run_mock():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running mock test on {device}")
    
    B, T, img_size = 2, 12, 256
    model = TAONot42VisionModel().to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # --- 计算权重大小 ---
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size_mb = total_params * 4 / (1024 ** 2) # float32 is 4 bytes
    print(f"\\n--- 模型参数信息 ---")
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
    print(f"静态权重占用内存: {param_size_mb:.2f} MB\\n")
    
    print("Extracting features with Segmenter...")
    video_input = torch.randn(B * T, 3, img_size, img_size, device=device)
    with torch.no_grad():
        feats = model.extract_features(video_input)
    
    feats = [f.view(B, T, *f.shape[1:]) for f in feats]
    dt = torch.full((B, T), 1.0 / 24.0, device=device)
    c_vids_shape = (img_size, img_size)
    
    for f in feats:
        f.requires_grad = True

    print("Running forward_physics with Profiler...")
    
    # 借助内建 Profiler 计算 FLOPs 和内存
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA] if device.type == 'cuda' else [torch.profiler.ProfilerActivity.CPU],
        profile_memory=True,
        record_shapes=True,
        with_flops=True
    ) as prof:
        preds = model.forward_physics(*feats, dt, step=0, get_loss_weights_fn=get_loss_weights, original_shape=c_vids_shape)
        
    print(f"\\n--- Profiler 前向传播统计 (Top 5 内存消耗) ---")
    print(prof.key_averages().table(sort_by="cpu_memory_usage" if device.type == 'cpu' else "cuda_memory_usage", row_limit=5))
    
    BT = B * T
    tgts = {
        "obj_dense": [torch.ones(BT, 1, img_size//s, img_size//s, device=device) for s in [8, 16, 32]],
        "bboxes_dense": [torch.ones(BT, 4, img_size//s, img_size//s, device=device) for s in [8, 16, 32]],
        "seg_raw": torch.zeros(BT, img_size, img_size, device=device),
        "seg_small": torch.zeros(BT, img_size//8, img_size//8, device=device),
        "depth": torch.ones(BT, img_size, img_size, device=device),
        "log_depth": torch.zeros(BT, img_size, img_size, device=device),
        "sky_mask": torch.zeros(BT, img_size, img_size, dtype=torch.bool, device=device),
        "cam_pos_t": torch.zeros(BT, 3, device=device),
        "cam_quat_t": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(BT, 4),
        "cam_pos_next": torch.zeros(BT, 3, device=device),
        "cam_quat_next": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).expand(BT, 4),
        "has_next": torch.ones(BT, dtype=torch.bool, device=device),
        "flow_target": torch.zeros(BT, 2, img_size, img_size, device=device),
        "cls_dense": [torch.zeros(BT, 1, img_size//s, img_size//s, device=device) for s in [8, 16, 32]]
    }
    
    img_t = torch.randn(BT, 3, img_size, img_size, device=device)
    img_next = torch.randn(BT, 3, img_size, img_size, device=device)
    
    print("\\nComputing loss...")
    loss, l_dict, _ = compute_physics_loss(preds, tgts, img_t, img_next, "supervised", step=0)
    
    print("Running backward...")
    loss.backward()
    
    print("\\nSuccess! Forward and Backward pass completed perfectly.")

if __name__ == "__main__":
    run_mock()
