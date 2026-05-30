import sys
import os
import shutil

# Add parent path to path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np

# 1. Enforce CUDA constraint
assert torch.cuda.is_available(), "严禁在非 CUDA 环境中进行模型与可视化测试！"
device = torch.device("cuda")

# 2. Inject modular mocks
import tests.mock_mamba
import tests.mock_scipy
tests.mock_mamba.inject_mock_mamba()
tests.mock_scipy.inject_mock_scipy()

# 3. Import core modules
from models import TAONot42VisionModel
from utils import get_loss_weights, save_visualization
from dataset import process_batch_on_gpu
from tests.mock_data import get_movi_e_or_fallback

def main():
    print("====================================================")
    print("[测试] 开始对可视化数据函数进行真值与未训练模型预测集成测试")
    print("====================================================")
    
    # Instantiate visual model (untrained)
    model = TAONot42VisionModel().to(device)
    # 使用 train() 模式，确保 YOLOESegment26 分割头以 Training 模式运行并返回训练多尺度预测字典，
    # 这样才能完美对应和展示我们实际训练的多尺度预测通道（Objectness, Box 距离, Mask Prototypes 等）
    model.train()
    
    # Load dataset
    B, T, img_size = 1, 24, 256
    npz_path = os.path.join(os.path.dirname(__file__), "data", "movi_e_static_sample.npz")
    print(f"正在从以下路径加载真实 NPZ 样本: {npz_path}")
    
    v_np, d_np, s_np, f_np, cp_np, cq_np, id_np, cat_np, vel_np, avel_np, vis_np, col_np = get_movi_e_or_fallback(npz_path, B, T, img_size, img_size)
    
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
    
    # Run GPU preprocessing
    print("正在对批次执行 GPU 预处理...")
    gpu_batch = process_batch_on_gpu(batch, device, img_size)
    
    v_seq = gpu_batch["video"] # Shape [B, T, C, H, W]
    
    # Prepare forward inputs
    print("提取视觉特征...")
    with torch.no_grad():
        feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        
        dt = torch.full((B, T), 1.0 / 24.0, device=device)
        print("模型前向推理中...")
        preds = model.forward_physics(
            *feats, dt, step=5000, 
            get_loss_weights_fn=get_loss_weights, 
            original_shape=(img_size, img_size)
        )
        
        # Build targets dictionary matching training structure
        # (Using same logic as TAOTrainer to extract target chunk)
        from trainer import TAOTrainer
        class DummyTrainer:
            def __init__(self, device, global_step):
                self.device = device
                self.global_step = global_step
        dummy_trainer = DummyTrainer(device, 5000)
        tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=T)
        
        # Compute warped image if warp loss components are present
        w_img = preds.get("warped_img", None)
        
        # 按照用户要求：在第 2 帧到第 20 帧之间（即 0-indexed 的 1 至 19）随机选择一帧进行测试
        import random
        vis_frame_idx = random.randint(1, 19)
        print(f"\n[随机帧选择] 已随机抽取帧索引: {vis_frame_idx} (代表第 {vis_frame_idx + 1} 帧) 进行可视化测试！\n")
        
        def slice_vis_frame(v):
            if v is None:
                return None
            if isinstance(v, dict):
                return {k: slice_vis_frame(val) for k, val in v.items()}
            if isinstance(v, list):
                # Process list elements by slicing the B*T dimension
                res = []
                for x in v:
                    if x is None:
                        res.append(None)
                    elif x.dim() > 0 and x.shape[0] == B * T:
                        res.append(x[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1])
                    elif x.dim() > 0:
                        res.append(x[-B:])
                    else:
                        res.append(x)
                return res
            if v.dim() == 0:
                return v
            if v.shape[0] == B * T:
                return v[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1]
            return v[-B:]

        vis_dir = "vis_outputs"
        print(f"调用 save_visualization 写入可视化图像到 {vis_dir}...")
        fp = save_visualization(
            v_seq[:, vis_frame_idx],
            {k: slice_vis_frame(v) for k, v in tgts.items()},
            {k: slice_vis_frame(v) for k, v in preds.items()},
            step=99999, # step id for file name matching
            warped_img=slice_vis_frame(w_img) if w_img is not None else None,
            output_dir=vis_dir
        )
        
        abs_fp = os.path.abspath(fp)
        print(f"\n[成功] 可视化图像保存成功：{abs_fp}")
        
        # Copy to the artifact directory to present to the user via absolute path
        artifact_dir = r"C:\Users\iii\.gemini\antigravity\brain\936eaad8-2dda-4b65-bfbc-ba438a1c9ec0"
        if os.path.exists(artifact_dir):
            shutil.copy(abs_fp, os.path.join(artifact_dir, "vis_step_99999.jpg"))
            print(f"[成功] 图像已复制到 Artifact 目录: {os.path.join(artifact_dir, 'vis_step_99999.jpg')}")
            
if __name__ == "__main__":
    main()
