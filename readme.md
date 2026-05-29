# TAO-Not-42 时空物理感知与端到端追踪视觉大模型 (tao-not-42-base)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 简介

**TAO-Not-42** 提供了一个高度集成、实时且边缘优化的时空感知网络系统。它将先进的 YOLO 二维特征与目标分割底座、时空状态空间混合（Mamba）、三维绝对深度估计与自运动姿态重投影（Warping）、以及持久化查询（Persistent Queries）实例时空追踪深度整合为一个统一的多任务估计系统。

该项目致力于解决复杂动态场景下的端到端物理追踪与感知问题，通过创新的架构实现了极高的 GPU 效率和零 PCIe 同步瓶颈。

## ✨ 核心特性

- 🚀 **极高 GPU 效率**：通过 scatter 极值约简与并行化边界框生成，实现零 CPU-GPU 同步阻塞。
- 👁️ **三维几何自监督约束**：无需人工绝对深度和光流标注，支持基于 Ego-Pose 与光度重投影一致性损失的自监督训练。
- 🎯 **时序稳定 Tracklet-Aware 追踪**：创新的持久化 Query 与 ID 绑定机制，配合单次发射向量化损失，大幅抑制 ID Switch 并保持高效运行。
- 🧩 **模块化架构设计**：完全解耦的 YOLO 骨干、物理预测头与追踪模块，支持极简的配置与替换。

## 📦 安装指南

### 环境要求

- 操作系统：Linux / Windows (仅支持 CUDA 环境)
- Python 3.8 或以上版本
- 带有 CUDA 11.8+ 支持的 PyTorch

### 快速安装

克隆仓库并安装依赖：

```bash
git clone https://github.com/your-username/tao-not-42-base.git
cd tao-not-42-base
pip install -r requirements.txt
```

*(注意：核心生产环境严格要求运行在 CUDA 下，暂不支持 CPU 推理或回退。)*

## 🚀 快速开始

本项目提供了开箱即用的测试脚本以验证您的环境与模型性能：

### 1. 验证模型架构与权重对齐
执行零 Prompt 预测，验证官方预训练权重与重构骨干前向输出的绝对对齐：
```bash
python tests/test_yoloe_bus.py
```

### 2. 端到端闭环物理仿真训练
运行高保真物理仿真与多阶段课程学习追踪测试，验证模型训练逻辑：
```bash
python test_mock.py
```
*(注：该脚本将自动下载所需数据，验证完整的 6 阶段学习流程)*

## 🏗 项目结构与开发文档

我们采用了清晰的模块化结构，更详尽的架构设计、数学机理和核心技术解析已分类整理在 `knowledge/` 目录中：

- [knowledge/yolo.md](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/yolo.md)：YOLO 视觉底座、分割预测头与权重精确折叠专题
- [knowledge/dataset.md](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/dataset.md)：数据流、异步加载与并行预处理专题
- [knowledge/custom_heads.md](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/custom_heads.md)：时空 Mamba 混合、三维物理几何头部专题
- [knowledge/system_integration.md](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/system_integration.md)：系统整机集成、自监督损失与课程学习专题

**核心源码结构：**
- `models/` - 核心神经网络（`tao_core.py`, `yoloe_head.py` 等）
- `utils/` - 辅助工具、3D 几何工具与损失函数（`geometry.py`, `losses.py` 等）
- `tests/` - 测试包与离线 Mock 数据模拟器

更多关于开发规范、架构决策的内容，请参阅：
- [AGENTS.md](file:///c:/Users/iii/Desktop/tao-not-42-base/AGENTS.md) - 智能体活动与架构决策日志
- [GEMINI.md](file:///c:/Users/iii/Desktop/tao-not-42-base/GEMINI.md) - 本地测试与数据生成指南

## 🤝 贡献指南

我们欢迎所有形式的贡献（包括提交 Issue，提出 PR，或是完善文档）。在提交代码前，请确保：
1. 所有的文档必须使用中文更新。
2. 核心代码禁止包含本地 Mock 逻辑，所有兼容降级逻辑仅限于 `tests/` 目录下。

## 📄 许可证

本项目采用 MIT License 协议进行开源。
