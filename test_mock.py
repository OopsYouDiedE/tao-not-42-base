import sys
import os
import torch
import numpy as np

# 0. 强制执行严格的 CUDA 约束以进行测试
assert torch.cuda.is_available(), "严禁在非 CUDA 环境中进行测试！"

# 1. 在导入核心代码库前注入模块化 mock
import tests.mock_mamba
import tests.mock_scipy
tests.mock_mamba.inject_mock_mamba()
tests.mock_scipy.inject_mock_scipy()

# 2. 导入核心代码库模块
import train
from models import TAONot42VisionModel
from utils import get_loss_weights, compute_physics_loss, save_visualization, compute_track_loss
from dataset import process_batch_on_gpu
from trainer import TAOTrainer
from tests.mock_data import get_movi_e_or_fallback

def load_yoloe_weights(model, path="yoloe-26s-seg-pf.pt"):
    import torch
    import torch.nn as nn
    import os
    import urllib.request
    
    if not os.path.exists(path):
        weights_url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{path}"
        print(f"正在从 {weights_url} 下载权重...")
        urllib.request.urlretrieve(weights_url, path)
        print("下载完成。")

    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Conv':
            c1 = module.conv.in_channels
            c2 = module.conv.out_channels
            k = module.conv.kernel_size
            s = module.conv.stride
            p = module.conv.padding
            g = module.conv.groups
            d = module.conv.dilation
            
            new_conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
            new_conv.to(module.conv.weight.device)
            module.conv = new_conv
            module.bn = nn.Identity()
        elif module.__class__.__name__ == 'PSABlock':
            if hasattr(module, 'add_norm1'): module.add_norm1 = torch.nn.Identity()
            if hasattr(module, 'add_norm2'): module.add_norm2 = torch.nn.Identity()

    try:
        from ultralytics import YOLO
        ul_model = YOLO(path)
        sd = ul_model.model.state_dict()
    except ImportError:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        sd = ckpt["model"].state_dict() if isinstance(ckpt, dict) and "model" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)

    tgt = model.state_dict()
    loaded_keys = {k for k, v in sd.items() if (k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k) in tgt and tgt[(k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k)].shape == v.shape}
    print(f"====================================================")
    print(f"[YOLO] Successfully loaded {len(loaded_keys)}/{len(sd)} keys from '{path}'!")
    print(f"====================================================")
    tgt.update({k.replace("model.model.", "segmenter.model.").replace("model.", "segmenter.model.") if k.startswith("model.") else k: v for k, v in sd.items() if k in loaded_keys})
    model.load_state_dict(tgt)

class DummyTrainer:
    """Minimal self substitute to call TAOTrainer helper methods directly."""
    def __init__(self, device, global_step):
        self.device = device
        self.global_step = global_step

def test_all_stages():
    """测试所有训练阶段，从基础检测到端到端追踪。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"====================================================")
    print(f"[RUN] 正在 {device.type.upper()} 上运行多阶段真实物理测试")
    print(f"====================================================")
    
    # 1. 实例化视觉模型
    model = TAONot42VisionModel().to(device)
    load_yoloe_weights(model, 'yoloe-26s-seg-pf.pt')
    model.train()
    
    # 模型参数摘要
    total_params = sum(p.numel() for p in model.parameters())
    print(f"总模型参数量: {total_params / 1e6:.2f} M")
    print(f"静态权重内存占用: {total_params * 4 / (1024**2):.2f} MB\n")
    
    # 2. 获取真实的 MOVi-E 样本或回退方案
    B, T, img_size = 1, 24, 256
    v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np = get_movi_e_or_fallback("movi_e_sample_0000.npz", B, T, img_size, img_size)
    
    # 准备匹配数据集输出格式的 Batch 字典
    batch = {
        "video": torch.from_numpy(v_np),
        "depth": torch.from_numpy(d_np),
        "segmentation": torch.from_numpy(s_np),
        "forward_flow": torch.from_numpy(f_np),
        "cam_pos": torch.from_numpy(cp_np),
        "cam_quat": torch.from_numpy(cq_np),
        "is_dynamic": [torch.from_numpy(x) for x in id_np],
        "category": [torch.from_numpy(x) for x in cat_np],
        "velocities": [torch.from_numpy(x) for x in vel_np],
        "angular_velocities": [torch.from_numpy(x) for x in avel_np],
        "visibility": [torch.from_numpy(x) for x in vis_np],
    }
    
    # 运行 GPU 预处理流水线
    print("正在运行 GPU Batch 预处理...")
    gpu_batch = process_batch_on_gpu(batch, device, img_size)
    
    # 3. 定义涵盖所有 5 个核心阶段的课程训练步骤
    stages = {
        1: ("阶段 1: 聚焦检测与深度", 50),
        2: ("阶段 2: 引入相机姿态", 250),
        3: ("阶段 3: 引入光流", 500),
        4: ("阶段 4: 光度误差与类别学习", 1500),
        5: ("阶段 5: 激活异常自监督", 5000)
    }
    
    # 提取预处理后的输入序列
    v_seq = gpu_batch["video"]
    t_max = v_seq.shape[1]
    
    # 为光度计算 Mock 未来帧
    img_next = torch.zeros_like(v_seq)
    for t in range(t_max):
        img_next[:, t] = v_seq[:, min(t + 1, t_max - 1)]
    
    print("\n开始多阶段验证...\n")
    
    for stage_id, (stage_name, step) in stages.items():
        print(f"----------------------------------------------------")
        print(f"[RUN] 正在测试课程 {stage_name} (Step = {step})")
        print(f"----------------------------------------------------")
        
        # 获取当前步的激活损失权重
        lw = get_loss_weights(step)
        active_losses = [k for k, v in lw.items() if v > 0]
        print(f"激活的损失组件: {active_losses}")
        
        # 从分割器提取特征
        with torch.no_grad():
            feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        
        # 为特征启用梯度
        for f in feats:
            f.requires_grad = True
            
        dt = torch.full((B, T), 1.0 / 24.0, device=device)
        
        # 前向传播
        preds = model.forward_physics(
            *feats, dt, step=step, 
            get_loss_weights_fn=get_loss_weights, 
            original_shape=(img_size, img_size)
        )
        
        # 4. 直接复用 TAOTrainer._extract_target_chunk 提取目标
        dummy_trainer = DummyTrainer(device, step)
        tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=t_max)
        
        # 损失计算
        loss, l_dict, w_img = compute_physics_loss(
            preds, tgts, 
            v_seq.flatten(0, 1), 
            img_next.flatten(0, 1), 
            mode="supervised", 
            step=step
        )
        
        # 反向传播
        loss.backward()
        print(f"损失计算完成: {loss.item():.4f}")
        for l_name, l_val in l_dict.items():
            if lw.get(l_name.lower()[:4], 0.0) > 0 or l_name == "Tot":
                print(f"  - {l_name} 损失: {l_val.item():.4f}")
                
        # 5. 在阶段 5 生成可视化输出以供人工检查
        if stage_id == 5:
            print("\n正在生成可视化验证网格...")
            # Slice the second frame (index 1) from the flattened B*T tensors,
            # matching TAOTrainer visualization slice logic.
            vis_frame_idx = 1  # second frame: has both t-1 and t+1 neighbours
            def slice_vis_frame(v):
                if v is None: return None
                if isinstance(v, list):
                    return [x[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1] if (x.dim() > 0 and x.shape[0] == B * T) else (x[-B:] if x.dim() > 0 else x) for x in v]
                if v.dim() == 0: return v
                if v.shape[0] == B * T:
                    return v[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1]
                return v[-B:]

            vis_dir = "vis_outputs"
            fp = save_visualization(
                v_seq[:, vis_frame_idx],
                {k: slice_vis_frame(v) for k, v in tgts.items()},
                {k: slice_vis_frame(v) for k, v in preds.items()},
                step=step,
                warped_img=slice_vis_frame(w_img) if w_img is not None else None,
                output_dir=vis_dir
            )
            abs_fp = os.path.abspath(fp)
            print(f"[SUCCESS] Visualization saved successfully at: {abs_fp}")
            
    print("\n====================================================")
    print("[SUCCESS] SUCCESS: All 5 curriculum stages verified!")
    print("====================================================")

    # -----------------------------------------------------------------
    # Stage 6: End-to-End Tracking Verification
    # -----------------------------------------------------------------
    print("\n----------------------------------------------------")
    print("[RUN] Stage 6: End-to-End Tracking Module Verification")
    print("----------------------------------------------------")

    # Re-run forward at step=1000 (track loss is active: ramp(500,2000,1.0))
    track_step = 1000
    with torch.no_grad():
        feats6 = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
    for f in feats6:
        f.requires_grad = True

    dt6 = torch.full((B, T), 1.0 / 24.0, device=device)
    preds6 = model.forward_physics(
        *feats6, dt6, step=track_step,
        get_loss_weights_fn=get_loss_weights,
        original_shape=(img_size, img_size)
    )

    # --- Shape verification ---
    N = 32
    assert "track_boxes"   in preds6, "track_boxes missing from preds"
    assert "track_classes" in preds6, "track_classes missing from preds"
    assert "track_alive"   in preds6, "track_alive missing from preds"
    assert "track_masks"   in preds6, "track_masks missing from preds"

    tb = preds6["track_boxes"]
    tc = preds6["track_classes"]
    ta = preds6["track_alive"]
    tm = preds6["track_masks"]

    assert tb.shape == (B, T, N, 4),   f"track_boxes shape mismatch: {tb.shape}"
    assert tc.shape == (B, T, N, 4585),  f"track_classes shape mismatch: {tc.shape}"
    assert ta.shape == (B, T, N, 1),   f"track_alive shape mismatch: {ta.shape}"
    assert tm.shape == (B, T, N, 32),  f"track_masks shape mismatch: {tm.shape}"
    print(f"  track_boxes:   {list(tb.shape)}  OK")
    print(f"  track_classes: {list(tc.shape)}  OK")
    print(f"  track_alive:   {list(ta.shape)}  OK")
    print(f"  track_masks:   {list(tm.shape)}  OK")

    # --- Value verification ---
    alive_prob = ta.sigmoid()
    print(f"  track_alive sigmoid mean (expect <0.5 initially): {alive_prob.mean().item():.4f}")
    assert alive_prob.mean().item() < 0.5, "Initial alive probability too high (bias init failed?)"

    assert tb.min().item() >= 0.0 and tb.max().item() <= 1.0, \
        f"track_boxes out of [0,1]: min={tb.min().item():.4f}, max={tb.max().item():.4f}"
    print(f"  track_boxes range: [{tb.min().item():.4f}, {tb.max().item():.4f}]  (expected [0,1])  OK")

    # --- Loss and gradient verification ---
    dummy_trainer6 = DummyTrainer(device, track_step)
    tgts6 = TAOTrainer._extract_target_chunk(dummy_trainer6, gpu_batch, c_start=0, c_end=T, max_t=t_max)

    track_loss_val = compute_track_loss(preds6, tgts6, track_step)
    assert torch.isfinite(track_loss_val), f"compute_track_loss returned non-finite: {track_loss_val}"
    print(f"  compute_track_loss: {track_loss_val.item():.4f}  (finite)  OK")

    track_loss_val.backward()
    grad_norms = [f.grad.norm().item() for f in feats6 if f.grad is not None]
    assert len(grad_norms) > 0, "No gradients flowed back through tracking loss!"
    print(f"  Gradient norms through feats: {[f'{g:.4f}' for g in grad_norms]}  OK")

    print("\n====================================================")
    print("[SUCCESS] SUCCESS: Stage 6 Tracking Verification PASSED!")
    print("====================================================")

if __name__ == "__main__":
    test_all_stages()
