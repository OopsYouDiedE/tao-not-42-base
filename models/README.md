# 模型文件夹 (models/)

本目录包含整个 `tao-not-42-base` 核心视觉大模型的所有神经网络结构定义与模块化分支。

---

## 📂 文件清单与角色定位

### 1. 🔗 [tao_core.py](file:///c:/Users/iii/Desktop/tao-not-42-base/models/tao_core.py) (核心集成模型)
* **`YOLOEBackbone`**：完全对齐官方 `yoloe-26s` 的 23 层特征提取金字塔（FPN/PAN），充当模型的“空间视觉底座”。
* **`TAONot42VisionModel`**：系统的主集成网络，负责串联二维视觉提取、时空特征混合、自运动姿态估计、绝对几何估计、未来特征异常预测及实例跨帧追踪这六大子计算模块的前向拼装与流控。

### 2. 🕒 [custom_heads.py](file:///c:/Users/iii/Desktop/tao-not-42-base/models/custom_heads.py) (自研特定预测头)
* **`SpatioTemporalMambaBlock`**：利用 Mamba (或时序分组空洞卷积退化模块 `TemporalConvFallback`) 在 Chunk 时间轴上混合特征。
* **`UnifiedGeometryDecoder`**：并行联合输出绝对单目深度图与稠密像素级光流图。
* **`EgoPoseHead`**：基于 6D 连续表示法正交化估计相机自身的相对运动变换矩阵。
* **`FeaturePredictorHead`**：用于物理异常自监督的特征动力学预测头。
* **`TrackQueryModule`**：管理 32 个持久化时序查询向量，求解实例跨帧绑定。

### 3. 🧠 [yoloe_head.py](file:///c:/Users/iii/Desktop/tao-not-42-base/models/yoloe_head.py) (官方 YOLOESegment 对齐头部)
* **`YOLOESegment26`**：官方 s 缩放比目标分割检测头。
* **`LRPCHead`**：轻量级类别建议与分类投影，集成了免交互 Objectness 门控过滤机制（PF 门控）。
* **`Proto26`**：掩膜原型图发生器，输出 32 通道 1/4 分辨率原型图。
* **`BNContrastiveHead` & `SAVPE`**：批归一化特征对比学习头与空间感知视觉提示嵌入（用于开放词表与多模态交互）。

### 4. 🛠 [yolo_blocks.py](file:///c:/Users/iii/Desktop/tao-not-42-base/models/yolo_blocks.py) (YOLO 底层基础算子)
* 包含了官方标准和增强的底层通用运算积木（如 `Conv` 自动对齐、`DWConv` 深度可分离卷积、`C3k2` 密集跨接增强块、`C2PSA` 金字塔自注意力机制和 `SPPF` 快速空间金字塔池化等）。

---

## 🛠 设计准则

1. **分离设计，解耦放置**：原生/对齐的 YOLO 算子与 FPN 金字塔保存在 `yoloe_head.py` 和 `yolo_blocks.py` 中；我们特有的时空几何物理预测分支存放在 `custom_heads.py` 中。
2. **零阻断集成**：顶层 `tao_core.py` 作为纽带实现两者的零阻断级联。
