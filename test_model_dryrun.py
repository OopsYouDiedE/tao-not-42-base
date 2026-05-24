import torch
import sys
import os
# Add the project directory to sys.path so we can import from models
sys.path.append(r"c:\Users\zznZZ\Desktop\tao-not-42-base")

from models import TAONot42VisionModel

def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")
    print("⏳ 初始化 TAONot42VisionModel (模块化 YOLO11 风格)...")
    
    model = TAONot42VisionModel(base_channels=48, hidden_channels=768).to(device)
    print("✅ 模型初始化成功！")
    
    # 模拟输入数据 (BatchSize=2, Channels=3, H=256, W=256)
    b, c, h, w = 2, 3, 256, 256
    peripheral = torch.rand(b, c, h, w).to(device)
    dt = torch.full((b,), 1.0/24.0, device=device)
    step = 0
    state = None
    
    def dummy_loss_weights_fn(step):
        return {"flow": 1, "box": 1, "mask": 1, "anom": 1}
        
    print(f"⏳ 正在进行一次前向传播 (Forward Pass) 尺寸: {peripheral.shape}...")
    try:
        out = model(peripheral, dt, step, state, get_loss_weights_fn=dummy_loss_weights_fn)
        print("✅ 前向传播成功完成！没有报错！")
        print("\n📊 模型输出特征:")
        for k, v in out.items():
            if isinstance(v, torch.Tensor):
                print(f"  - {k}: {v.shape}")
            elif isinstance(v, dict):
                print(f"  - {k}: dict with keys {list(v.keys())}")
            else:
                print(f"  - {k}: {type(v)}")
    except Exception as e:
        import traceback
        print("❌ 前向传播中途报错:")
        traceback.print_exc()

if __name__ == "__main__":
    test()
