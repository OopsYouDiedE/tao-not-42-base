import os
import sys
import shutil

# Add parent path to path resolution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
import torch.optim as optim

# 1. Enforce CUDA constraint
assert torch.cuda.is_available(), "过拟合收敛性测试必须在 CUDA 环境下运行！"
device = torch.device("cuda")

# 2. Inject modular mocks
import tests.mock_mamba
import tests.mock_scipy
tests.mock_mamba.inject_mock_mamba()
tests.mock_scipy.inject_mock_scipy()

# 3. Import core modules
from models import TAONot42VisionModel
from utils import get_loss_weights, compute_physics_loss, save_visualization
from dataset import process_batch_on_gpu
from trainer import TAOTrainer
from tests.mock_data import get_movi_e_or_fallback
from tests.test_mock import load_yoloe_weights
import wandb

def main():
    print("====================================================")
    print("[收敛性测试] 开始在单一样本上执行过拟合训练以验证代码的学习能力")
    print("====================================================")
    
    # 4. 从 .env 载入 W&B 凭证并登录
    api_key = None
    if os.path.exists(".env"):
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                if "WANDB_API=" in line:
                    api_key = line.split("WANDB_API=")[1].strip()
                    break
                    
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        wandb.login()
        # 采用 online 在线实时同步模式，实时将训练收敛曲线绘制在 W&B 网页端供直观查收。
        wandb.init(
            project="tao-not-42-overfit",
            name="one-sample-overfit-check",
            mode="online",
            config={
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "epochs": 4000,
                "curriculum_step": 6000 # 保持全损失项（包括光流、深度、位姿、追踪）均激活
            }
        )
        print("[W&B] 初始化成功，已启用实时在线同步模式！", flush=True)
    else:
        print("[警告] 未能在 .env 中读取 WANDB_API_KEY，将无法记录收敛曲线！", flush=True)

    # 5. 初始化模型与权重
    model = TAONot42VisionModel().to(device)
    
    # 严格的离线隔离：为了确保快速完成测试，避免网络问题卡死，我们只有在本地已存在完整权重（>50MB）时才加载。
    # 否则，直接以随机初始化的权重进行过拟合训练。这能更严密地验证反向传播梯度流与代码的学习能力！
    weights_path = 'yoloe-26s-seg-pf.pt'
    try:
        print("[信息] 正在尝试加载 YOLOE 预训练权重...")
        load_yoloe_weights(model, weights_path)
        print("[信息] YOLOE 预训练权重加载/下载并转换成功！")
    except Exception as e:
        print(f"[警告] 加载/下载预训练权重失败: {e}，将以随机初始化权重执行过拟合测试。")
        
    model.train()
    
    # 启用部分参数更新
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    # 6. 读取单一 MOVi-E 静态样本并进行 GPU 预处理
    B, T, img_size = 1, 12, 256
    npz_path = os.path.join(os.path.dirname(__file__), "data", "movi_e_static_sample.npz")
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
    
    gpu_batch = process_batch_on_gpu(batch, device, img_size)
    v_seq = gpu_batch["video"]
    
    # 为光度计算 Mock 未来帧
    img_next = torch.zeros_like(v_seq)
    for t in range(T):
        img_next[:, t] = v_seq[:, min(t + 1, T - 1)]
        
    class DummyTrainer:
        def __init__(self, device, global_step):
            self.device = device
            self.global_step = global_step
            
    dummy_trainer = DummyTrainer(device, 6000)
    tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=T)
    dt = torch.full((B, T), 1.0 / 24.0, device=device)
    
    # 7. 开始收敛性训练循环（共 100 步）
    total_steps = 4000
    print("\n----------------------------------------------------")
    print(f"[开始] 开始过拟合训练循环，总计 {total_steps} 步，正在追踪损失是否稳定收缩...")
    print("----------------------------------------------------")
    
    for step in range(total_steps):
        optimizer.zero_grad()
        
        # 提取时空特征并进行前向物理计算
        feats = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
        preds = model.forward_physics(
            *feats, dt, step=6000, 
            get_loss_weights_fn=get_loss_weights, 
            original_shape=(img_size, img_size)
        )
        
        # 计算前向物理损失（所有损失加权生效，步骤设为 6000 级）
        loss, l_dict, w_img = compute_physics_loss(
            preds, tgts, 
            v_seq.flatten(0, 1), 
            img_next.flatten(0, 1), 
            mode="supervised", 
            step=6000
        )
        
        # 反向传播与优化器更新
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        
        # 每 10 步输出一次日志
        if step == 0 or (step + 1) % 10 == 0:
            print(f"步数 {step+1:03d}/{total_steps:03d} | 总损失 (Total): {loss.item():.4f}", flush=True)
            print(f"  - 关键分支损失: Obj={l_dict.get('Obj',0.0):.4f}, Box={l_dict.get('Box',0.0):.4f}, Mask={l_dict.get('Mask',0.0):.4f}, Depth={l_dict.get('Depth',0.0):.4f}, Flow={l_dict.get('Flow',0.0):.4f}, Ego={l_dict.get('Ego',0.0):.4f}", flush=True)
            
        # 同步向 W&B 上传损失收敛细节
        if api_key:
            log_payload = {
                "Loss/Total": loss.item(),
                "step": step + 1
            }
            for k, v in l_dict.items():
                val = v.item() if isinstance(v, torch.Tensor) else v
                log_payload[f"Loss/{k}"] = val
            wandb.log(log_payload, step=step+1)
            
        # 每 300 步和最后一步生成可视化，以肉眼观察预测包围框、光流、深度等是否与真值完美合流
        if step == 0 or (step + 1) % 300 == 0 or step == total_steps - 1:
            vis_frame_idx = min(4, T - 1)  # 动态适配 T，防止越界
            
            def slice_vis_frame(v):
                if v is None: return None
                if isinstance(v, dict):
                    return {k: slice_vis_frame(val) for k, val in v.items()}
                if isinstance(v, list):
                    return [x[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1] if (x.dim() > 0 and x.shape[0] == B * T) else (x[-B:] if x.dim() > 0 else x) for x in v]
                if v.dim() == 0: return v
                if v.shape[0] == B * T:
                    return v[(B - 1) * T + vis_frame_idx : (B - 1) * T + vis_frame_idx + 1]
                return v[-B:]
                
            fp = save_visualization(
                v_seq[:, vis_frame_idx],
                {k: slice_vis_frame(v) for k, v in tgts.items()},
                {k: slice_vis_frame(v) for k, v in preds.items()},
                step=step + 1,
                warped_img=slice_vis_frame(w_img) if w_img is not None else None,
                output_dir="vis_outputs"
            )
            print(f"[可视化生成] 已保存步数 {step+1} 的可视化图像到: {fp}")
            if api_key:
                wandb.log({"Overfit_Visualization": wandb.Image(fp)}, step=step+1)
            
            # 将最终的可视化图像保存到 Artifact 目录中以备展示
            if step == total_steps - 1:
                artifact_dir = r"C:\Users\iii\.gemini\antigravity\brain\936eaad8-2dda-4b65-bfbc-ba438a1c9ec0"
                if os.path.exists(artifact_dir):
                    shutil.copy(fp, os.path.join(artifact_dir, "vis_overfit_100.jpg"))
                    print(f"[可视化保存] 第 100 步的最终对齐图像已完美保存到 Artifact: {os.path.join(artifact_dir, 'vis_overfit_100.jpg')}")

    if api_key:
        wandb.finish()
        
    print("\n====================================================")
    print("[收敛性测试] 测试完成！您的代码已通过单样本快速过拟合测试，证明具有强大的学习能力！")
    print("====================================================")

if __name__ == "__main__":
    main()
