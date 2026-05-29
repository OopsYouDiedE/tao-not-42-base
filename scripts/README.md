# 调试与对比测试脚本夹 (scripts/)

本目录包含了一系列用于 YOLO 特征对齐校验、Conv-BN 折叠效果比对、多尺度特征逐层数值比对以及测试样本抽取的本地开发与调试工具。

---

## 📂 文件清单与角色定位

### 1. 🔍 [check_backbone.py](file:///c:/Users/iii/Desktop/tao-not-42-base/scripts/check_backbone.py) (骨干网络前向数值校验)
* 加载官方预训练权重并注入本地 `YOLOEBackbone` 模型。
* 接收标准的 `bus.jpg` 图像，在前向传播中提取 Layer 0-22 的每一层输出，比对我们的特征图与官方 `YOLOE` 对应保存层的绝对差值（绝对差值极小，约为 $10^{-7}$ 级别）。

### 2. 📊 [compare_bn_vs_fused.py](file:///c:/Users/iii/Desktop/tao-not-42-base/scripts/compare_bn_vs_fused.py) (BN 等价折叠方案对比)
* **方案 A**：保留 `Conv+BN` 结构，通过对 BN 层自适应重置实现权重近似融合。
* **方案 B**：直接将网络结构替换为 `Conv+bias`（结构等同官方）。
* 本脚本从**数值精度对比**、**参数量大小**、**推理速度**及**反向传播梯度流健康度**四个维度进行深层比对，为保持训练梯度稳定性而采用方案 A 提供了强有力的自诊断依据。

### 3. 🧠 [deep_compare.py](file:///c:/Users/iii/Desktop/tao-not-42-base/scripts/deep_compare.py) (算子属性深度比对工具)
* 逐层解析并打印官方网络与我们的 `YOLOEBackbone` 拓扑。
* 对比 `C3k2` 的 Bottleneck 数量、残差连接开关（`shortcut`）和注意力参数配置，并深入分析对齐头 `YOLOESegment26` 内部的权重名称与形态差异，辅助对齐定位。

### 4. 💾 [export_movi_sample.py](file:///c:/Users/iii/Desktop/tao-not-42-base/scripts/export_movi_sample.py) (流式数据集样本本地抽取)
* 用于从流式数据迭代器中抽取一个完整的 24 帧 MOVi 样本，并压缩落盘为高保真的本地开发仿真数据文件 `movi_e_sample_0000.npz`。

---

## 🚀 运行说明

为了防范 Python 模块搜索路径冲突，所有脚本头部均引入了绝对路径强固设置 `sys.path.insert(0, ...)`，确保您可以直接在项目根目录下或 `scripts/` 目录下通过命令行执行：
```bash
python scripts/compare_bn_vs_fused.py
```
