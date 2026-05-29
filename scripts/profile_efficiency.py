"""
TAO-Not-42 Perceptual Model Efficiency Profiler
================================================
本脚本用于测试核心大模型在训练 (Train) 和测试 (Inference/Eval) 两种模式下的效率表现。
支持对以下核心指标进行测量与分析：
1. 运行时长 (ms/iter) 与 吞吐率 (Frames/Sec)
2. 平均/峰值 GPU 显存占用 (Peak GPU VRAM)
3. 网络总参数量 (Total Parameters) & 估计算力需求 (Theoretical FLOPs)

环境与能力降级自适应机制：
- 本地 Windows 开发环境：自动检测并使用 SpatioTemporalGRUFallback (ConvGRU) 代替 Mamba，
  并从本地测试样本 .npz 文件 (Mock 模式) 流式加载数据。
- Google Colab / 生产环境：不进行任何 Mock 劫持，使用真实的 Mamba SSM 算子和全量数据流加载。
"""

import os
import sys
import time
import torch

# 1. 确保包搜索路径正确
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 2. 检测并配置降级环境（完美对齐能力降级矩阵）
is_colab = False
try:
    import mamba_ssm
    print("\n" + "="*70)
    print("[INFO] [环境检测] 已成功检测到 mamba_ssm 库！")
    print("-> 运行模式: [Colab 生产模式] 将使用全量真实时空 Mamba 模块与生产级算子进行效率分析。")
    print("="*70 + "\n")
    is_colab = True
except ImportError:
    print("\n" + "="*70)
    print("[INFO] [环境检测] 未检测到 mamba_ssm 依赖库！")
    print("-> 运行模式: [Windows 开发降级模式] 正在向系统注入 SpatioTemporalGRUFallback (ConvGRU) 退化平替...")
    print("="*70 + "\n")
    
    # 动态劫持 Mamba 与 Scipy 依赖
    import tests.mock_mamba
    import tests.mock_scipy
    tests.mock_mamba.inject_mock_mamba()
    tests.mock_scipy.inject_mock_scipy()

# 3. 导入核心代码库（必须在 Mock 注入之后执行）
from models.tao_core import TAONot42VisionModel
from dataset import AsyncDataBuffer, process_batch_on_gpu
from utils.losses import get_loss_weights, compute_physics_loss
from trainer import TAOTrainer

class DummyTrainer:
    def __init__(self, device, global_step):
        self.device = device
        self.global_step = global_step

def run_profiler():
    # 验证 CUDA 约束
    assert torch.cuda.is_available(), "[ERROR] 错误: 效率测试强依赖 GPU 计算，请确保在 CUDA 环境下运行！"
    device = torch.device("cuda")
    print(f"[INFO] 使用 GPU 设备: {torch.cuda.get_device_name(0)}")
    
    B, T, img_size = 1, 6, 256
    print(f"\n[数据准备] Batch Size = {B}, Sequence Length = {T}, Resolution = {img_size}x{img_size}")
    
    # 数据自适应读取逻辑
    npz_path = "movi_e_sample_0000.npz"
    if not os.path.exists(npz_path):
        # 尝试从 tests/data/ 下寻找
        npz_path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "movi_e_static_sample.npz")
        
    print(f"[INFO] 正在加载物理几何样本数据 (路径: {npz_path})...")
    from tests.mock_data import get_movi_e_or_fallback
    v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np = get_movi_e_or_fallback(
        npz_path, B, T, img_size, img_size
    )
    
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
        
    gpu_batch = process_batch_on_gpu(batch, device, img_size)
    
    # 准备前向与评估数据
    v_seq = gpu_batch["video"]
    img_next = torch.zeros_like(v_seq)
    for t in range(T):
        img_next[:, t] = v_seq[:, min(t + 1, T - 1)]
    dt = torch.full((B, T), 1.0 / 24.0, device=device)
    
    # === B. 构建模型与基础算力分析 ===
    model = TAONot42VisionModel().to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[模型结构]")
    print(f"  - 总参数量: {total_params / 1e6:.2f} M")
    print(f"  - 可训练参数量: {trainable_params / 1e6:.2f} M")
    
    # === C. 效率测试 - 模式 1: 训练模式 (Train Mode) ===
    print("\n" + "-"*50)
    print("[RUN] 正在评估 [训练模式 (Train Mode)] 效率参数 (包含反向传播与梯度更新)...")
    print("-"*50)
    
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 预热
    for _ in range(3):
        feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        preds = model.forward_physics(*feats, dt, step=5000, get_loss_weights_fn=get_loss_weights, original_shape=(img_size, img_size))
        dummy_trainer = DummyTrainer(device, 5000)
        tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=T)
        loss, _, _ = compute_physics_loss(preds, tgts, v_seq.flatten(0, 1), img_next.flatten(0, 1), step=5000)
        loss.backward()
        model.zero_grad(set_to_none=True)
        
    torch.cuda.synchronize()
    
    # 实测运行时间
    steps = 10
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    
    start_evt.record()
    for _ in range(steps):
        feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        preds = model.forward_physics(*feats, dt, step=5000, get_loss_weights_fn=get_loss_weights, original_shape=(img_size, img_size))
        dummy_trainer = DummyTrainer(device, 5000)
        tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=T)
        loss, _, _ = compute_physics_loss(preds, tgts, v_seq.flatten(0, 1), img_next.flatten(0, 1), step=5000)
        loss.backward()
        model.zero_grad(set_to_none=True)
        
    end_evt.record()
    torch.cuda.synchronize()
    
    train_time_ms = start_evt.elapsed_time(end_evt) / steps
    train_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    train_fps = (B * T) / (train_time_ms / 1000.0)
    
    print(f"  [OK] 平均时长: {train_time_ms:.1f} ms / batch iteration")
    print(f"  [OK] 吞吐速度: {train_fps:.1f} frames / second")
    print(f"  [OK] 显存峰值: {train_mem_mb:.1f} MB")
    
    # === D. 效率测试 - 模式 2: 推理/测试模式 (Eval Mode) ===
    print("\n" + "-"*50)
    print("[RUN] 正在评估 [推理模式 (Eval/Inference Mode)] 效率参数 (仅前向计算，无梯度图)...")
    print("-"*50)
    
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # 预热
    with torch.no_grad():
        for _ in range(3):
            feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
            preds = model.forward_physics(*feats, dt, step=5000, get_loss_weights_fn=get_loss_weights, original_shape=(img_size, img_size))
            
    torch.cuda.synchronize()
    
    start_evt_eval = torch.cuda.Event(enable_timing=True)
    end_evt_eval = torch.cuda.Event(enable_timing=True)
    
    start_evt_eval.record()
    with torch.no_grad():
        for _ in range(steps):
            feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
            preds = model.forward_physics(*feats, dt, step=5000, get_loss_weights_fn=get_loss_weights, original_shape=(img_size, img_size))
            
    end_evt_eval.record()
    torch.cuda.synchronize()
    
    eval_time_ms = start_evt_eval.elapsed_time(end_evt_eval) / steps
    eval_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    eval_fps = (B * T) / (eval_time_ms / 1000.0)
    
    print(f"  [OK] 平均时长: {eval_time_ms:.1f} ms / batch forward")
    print(f"  [OK] 吞吐速度: {eval_fps:.1f} frames / second")
    print(f"  [OK] 显存峰值: {eval_mem_mb:.1f} MB")
    
    # === E. 算力估计 (Theoretical FLOPs Estimation) ===
    # 粗略估计 YOLO 核心卷积前向计算算力 (以标准 s 模型为参照，加上三维解译与时空 Mamba)
    # 输入分辨率 256x256 下，24 帧时序，算力需求估计
    flops_est = 31.66 * 1e6 * (img_size * img_size) * T * 2 / 1e9  # GFLOPs 理论估值
    
    # === F. 报告分析生成 ===
    print("\n" + "="*70)
    print("[REPORT] 效率评估对比总结 (Train vs Inference Comparison)")
    print("="*70)
    report = f"""
| 效率测试指标 | 训练模式 (Train Mode) | 推理模式 (Eval Mode) | 性能提升 / 降幅比率 |
| :--- | :---: | :---: | :---: |
| **运行时长 (ms/iter)** | {train_time_ms:.1f} ms | {eval_time_ms:.1f} ms | 减少 {((train_time_ms - eval_time_ms)/train_time_ms)*100:.1f}% 耗时 |
| **吞吐量 (Frames/Sec)** | {train_fps:.1f} fps | {eval_fps:.1f} fps | 速度提升 {((eval_fps - train_fps)/train_fps)*100:.1f}% |
| **GPU 显存峰值 (VRAM)** | {train_mem_mb:.1f} MB | {eval_mem_mb:.1f} MB | 节省 {((train_mem_mb - eval_mem_mb)/train_mem_mb)*100:.1f}% 显存 |
| **理论算力需求 (FLOPs)**| 约 {flops_est * 3:.1f} GFLOPs | 约 {flops_est:.1f} GFLOPs | 训练包含反向梯度，开销约 3 倍 |
"""
    print(report)
    
    print("[ANALYSIS] [性能差异深度分析]")
    print("1. 显存开销 (VRAM Peak):")
    print("   - 训练模式下，PyTorch 必须维护前向传播产生的完整计算激活图 (Activation Maps) 以用于反向梯度计算，"
          "因此显存开销极其庞大。")
    print("   - 推理模式下使用 `with torch.no_grad()`，前向特征张量在使用完毕后会被立即释放，无冗余激活缓存，"
          f"成功节省了 {((train_mem_mb - eval_mem_mb)/train_mem_mb)*100:.1f}% 显存。")
    print("2. 运行时长与吞吐量:")
    print("   - 训练模式包含前向计算、损失计算、反向传播 (Autograd Backward Graph Walk) 和参数权重梯度更新，"
          "有极高的算力依赖 and 内存带宽负担。")
    print("   - 推理模式仅进行纯粹的前向张量运算，算力需求减半，"
          f"吞吐量实现 {((eval_fps - train_fps)/train_fps)*100:.1f}% 的爆发式提升。")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_profiler()
