# -*- coding: utf-8 -*-
"""
Colab L4 GPU 诊断与调试脚本 (colab_debug.py)
用于深度验证硬件、编译算子、模型物理梯度流动以及 YOLO 预训练权重的加载状态。
"""

import os
import sys
import time

def print_section(title):
    print("\n" + "=" * 60)
    print(f" 🔍 {title}")
    print("=" * 60)

def debug_diagnose():
    # 1. 硬件与 CUDA 环境诊断
    print_section("第 1 步：硬件与 PyTorch/CUDA 环境诊断")
    import torch
    print(f"Python 版本: {sys.version}")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 可用性: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU 设备名称: {torch.cuda.get_device_name(0)}")
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"CUDNN 版本: {torch.backends.cudnn.version()}")
        # 验证显存状态
        r = torch.cuda.memory_reserved(0) / 1024**3
        a = torch.cuda.memory_allocated(0) / 1024**3
        print(f"已分配显存: {a:.3f} GB / 已保留显存: {r:.3f} GB")
    else:
        raise RuntimeError("❌ 核心致命错误：未检测到 CUDA 环境！请在 Colab 菜单中选择 运行时 -> 更改运行时类型 并切换至 GPU (推荐 L4)。")

    # 2. 深度算子与底层编译依赖包校验
    print_section("第 2 步：深度算子与编译依赖校验 (causal_conv1d, mamba_ssm)")
    
    try:
        import causal_conv1d
        print(f"✅ causal_conv1d 导入成功！版本: {getattr(causal_conv1d, '__version__', '未知')}")
    except ImportError as e:
        print(f"❌ causal_conv1d 导入失败: {e}")
        print("提示：请确保已安装与当前 PyTorch/CUDA 版本兼容的预编译 whl。")
        raise e

    try:
        import mamba_ssm
        print(f"✅ mamba_ssm 导入成功！版本: {getattr(mamba_ssm, '__version__', '未知')}")
    except ImportError as e:
        print(f"❌ mamba_ssm 导入失败: {e}")
        print("提示：请确保已安装与当前 PyTorch/CUDA 版本兼容的预编译 whl。")
        raise e

    # 3. 运行 Mamba 算子微型 Forward/Backward 物理验证
    print_section("第 3 步：运行 Mamba 算子 GPU 前向/反向梯度流物理验证")
    try:
        from mamba_ssm import Mamba
        device = torch.device("cuda")
        # 声明一个极其微型的 Mamba 算子
        mamba_block = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).to(device)
        test_input = torch.randn(2, 8, 64, device=device, requires_grad=True)
        test_output = mamba_block(test_input)
        loss = test_output.sum()
        loss.backward()
        print("✅ Mamba 算子 GPU Forward & Backward 双向验证成功，梯度流动正常！")
    except Exception as e:
        print(f"❌ Mamba 算子 GPU 验证失败！错误信息：{type(e).__name__}: {e}")
        raise e

    # 4. TAONot42VisionModel 实例化与 YOLO 权重吸收折叠校验
    print_section("第 4 步：检测骨干网络实例化与预训练权重偏置折叠")
    try:
        from models.tao_core import TAONot42VisionModel
        model = TAONot42VisionModel().to(device)
        print("✅ 成功实例化 TAONot42VisionModel。")
        
        # 引入并运行测试中的 load_yoloe_weights 以校验偏置折叠与跨设备计算
        from tests.test_mock import load_yoloe_weights
        print("正在尝试自动下载与折叠加载 yoloe-26s-seg-pf.pt 权重...")
        load_yoloe_weights(model, "yoloe-26s-seg-pf.pt")
        print("✅ YOLO 先验权重跨设备偏置折叠与无损吸收成功！")
    except Exception as e:
        print(f"❌ 骨干网络初始化或权重加载失败！错误信息：{type(e).__name__}: {e}")
        raise e

    # 5. 全流程自监督课程训练 Mock 测试 (测试 test_mock.py 的完整逻辑)
    print_section("第 5 步：运行全课程多阶段自监督梯度流动测试 (integration_check)")
    try:
        from tests.test_mock import test_all_stages
        print("正在调用内置的多阶段课程自监督测试，这需要约 10 秒时间...")
        test_all_stages()
        print("\n✅ 全流程课程自监督检测、深度、姿态、光流、异常检测及端到端追踪验证完美通过！")
    except Exception as e:
        print(f"❌ 多阶段课程自监督测试失败！错误信息：{type(e).__name__}: {e}")
        raise e

if __name__ == "__main__":
    print("=" * 60)
    print("🚀 开始运行 Colab L4 GPU 深度调试与诊断方案 🚀")
    print("=" * 60)
    start_time = time.time()
    try:
        debug_diagnose()
        print("\n" + "=" * 60)
        print(f"🎉 恭喜！Colab L4 GPU 深度环境与模型物理梯度流动诊断 100% 成功！(耗时: {time.time() - start_time:.2f}s)")
        print("您现在可以极其放心地开始在大数据集上进行生产级训练与微调！")
        print("=" * 60)
    except Exception as e:
        print("\n" + "=" * 60)
        print(f"⚠️ 诊断发现异常！请在本地 VS Code 调试器的调用栈中仔细阅读上述错误提示进行定位。")
        print("=" * 60)
        sys.exit(1)
