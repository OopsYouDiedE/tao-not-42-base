# TAO-Not-42 时空物理感知与端到端追踪视觉大模型 (tao-not-42-base)

本仓库提供了一个高度集成、实时且边缘优化的时空感知网络系统。它将先进的 YOLO 二维特征与目标分割底座、时空状态空间混合（Mamba）、三维绝对深度估计与自运动姿态重投影（Warping）、以及持久化查询（Persistent Queries）实例时空追踪深度整合为一个统一的多任务估计系统。

---

## 📂 项目结构概览

为了保证代码库的整洁性与可读性，项目采用了**“每个代码文件夹（包括根目录）仅包含一个 README”**的扁平化规范，且所有系统性的设计与数学机理文档均作为独立的专题文件存放在知识库目录 `knowledge/` 中：

```
tao-not-42-base/
├── readme.md (本文件：整机系统与运行指南)
├── GEMINI.md (本地测试与数据生成指南)
├── AGENTS.md (智能体开发与架构决策日志)
├── test_mock.py (单卡端到端 5 阶段课程物理仿真与 Stage 6 追踪测试)
├── train.py (主训练运行入口)
├── trainer.py (6阶段渐进式多课程损失调度训练管理器)
├── dataset.py (高保真视频流式数据加载器)
│
├── 📂 models/ (核心神经网络模型定义)
│   ├── README.md (介绍本目录下文件结构)
│   ├── tao_core.py (主模型集成：TAONot42VisionModel 与 YOLOEBackbone)
│   ├── custom_heads.py (自研时空 Mamba 混合、绝对几何解码、EgoPose、FeaturePredictor、追踪等特定任务头)
│   ├── yoloe_head.py (官方 YOLOE 对齐的目标检测分割预测头与 LRPC 门控)
│   └── yolo_blocks.py (YOLO 核心基础卷积与自注意力算子)
│
├── 📂 utils/ (辅助计算、物理重投影与损失计算工具箱)
│   ├── README.md (介绍本目录下文件结构)
│   ├── label_generator.py (基于 GPU 极值约简的高效非阻塞边界框生成器)
│   ├── geometry.py (相机 3D 反投影与 warp 重投影计算)
│   ├── losses.py (SSIM 混合光度损失、Edge-aware 平滑损失及 Tracklet-Aware 追踪匹配损失)
│   └── visualization.py (高保真拼图异步分离渲染与自动 ABI 降级 NMS 导出)
│
├── 📂 scripts/ (调试、对比与样本导出脚本)
│   ├── README.md (介绍各调试脚本的物理作用)
│   └── *.py
│
├── 📂 tests/ (测试与 Fallback 仿真 Mock 包)
│   ├── README.md (介绍本地仿真球模拟器、Mamba 退化等 Mock)
│   └── *.py
│
└── 📂 knowledge/ (专题化系统知识库)
    ├── README.md (知识库导航索引入口)
    ├── yolo.md (YOLO 视觉金字塔、分割预测头、权重精确折叠数学等价性专题)
    ├── dataset.md (异步缓存、异步 Stream 预取、GPU scatter 并行提取专题)
    ├── custom_heads.md (自研 Mamba 时序、深度光流联合估计、6D 连续自运动回归专题)
    └── system_integration.md (整机前向组装、6阶段课程损失调度、物理重投影与 Tracklet-Aware 匈牙利匹配专题)
```

---

## 🚀 核心技术亮点与专题文献

若要深入理解各模块的数学模型及实现，请直接查阅 `knowledge/` 下对应的专题知识文档：

1. **零 PCIe 带宽同步瓶颈**：
   通过 GPU 异步解码与 PCIe 总线带宽折半压缩传输，配合我们在 [utils/label_generator.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/label_generator.py) 中实现的 **`scatter_reduce_` 和 `scatter_add_` 并行算子** 直接在 GPU 显存上生成真值框，实现**显存分配开销为 0** 且零主从 CPU-GPU 握手同步。
   👉 详细设计请参阅：[数据流与预处理专题 (knowledge/dataset.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/dataset.md)
   
2. **三维几何自监督约束**：
   无需任何人工绝对深度或光流标注。模型通过 Ego-Pose 位姿头预测相机连续三维运动，在 [utils/geometry.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/geometry.py) 的 `inverse_warp` 中反投影构建出 3D 稠密点云，实现帧与帧之间基于 **L1 与 SSIM 的自监督光度重投影一致性损失** 计算。
   👉 详细公式请参阅：[系统整机集成与自监督损失专题 (knowledge/system_integration.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/system_integration.md)
   
3. **时序稳定 Tracklet-Aware 匈牙利追踪**：
   在 Chunk 时序序列上持久化维护实例目标和 32 个 Query 的绑定状态，仅对新产生的 GT 实例触发匹配，抑制 ID Switch。在循环外**单次发射向量化** Smooth L1 与 BCE 损失，兼顾高稳定追踪与高速运行。
   👉 详细实现请参阅：[系统整机集成与自监督损失专题 (knowledge/system_integration.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/system_integration.md)

---

## 🛠 极速上手与运行指南

### 1. 验证 YOLO 网络对齐与官方权重迁移 (零对齐测试)
运行官方预训练权重与我们重构的骨干+LRPC 头部在真实图像上的推理对齐测试：
```bash
python tests/test_yoloe_bus.py
```
*(预期结果：模型无缝迁移 291/298 个键值，与官方前向数值输出绝对对齐，完美检测并分割出客车和行人目标)*

### 2. 验证多阶段课程联合训练与物理追踪闭环 (端到端仿真测试)
运行本地高保真物理仿真与 Chunk 序列级别的 6 阶段课程自诊断测试：
```bash
python test_mock.py
```
*(预期结果：自动通过 CUDA 环境检测，调用 tests/mock_data.py 进行球物理轨迹生成，无NaN顺利跑通前 5 阶段课程，可视化落盘，并以 Gradient norms OK 通过第 6 阶段端到端追踪梯度流检测)*
