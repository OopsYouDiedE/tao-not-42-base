# MOVi-E 与 YOLOE-26 整合知识库 (Consolidated Knowledge Base - MOVi-E & YOLOE-26)

本知识库详细阐述了 Kubric MOVi-E 数据集规格、数据加载管线以及 YOLOE-26 模型架构的当前代码设计与实现细节。

---

## 1. Kubric MOVi-E 数据集与数据流管线

项目采用 **Kubric MOVi-E** 数据集，该数据集专为物理直觉学习、目标发现以及自主运动（Ego-motion）与独立物体运动的分离而设计。

### 1.1 MOVi-E 数据集特性
- **动态相机（Dynamic Camera）**：相机在半球形外壳上以恒定速度进行线性平移。
- **动态物体（Dynamic Objects）**：包含地面的 10-20 个障碍物，以及 1-3 个独立运动的活跃物体。
- **自主运动解耦（Ego-motion Decoupling）**：解决了将相机视角变化（自主运动视差）与独立物体运动进行分离的核心挑战。

### 1.2 双通道异步数据管线与 GPU 效率优化
为了最大化 GPU 吞吐量并彻底消除主循环中的 CPU-GPU 同步卡顿，代码实现了一套极致优化的异步数据管线：
1. **CPU 异步数据缓冲区 (`AsyncDataBuffer`)**：
   - 运行在后台守护线程中，通过 TFDS 异步读取视频片段（`movi_e/256x256`）。
   - 将原始数据转换为 PyTorch 格式，并推入双端队列 `deque` 循环缓冲区（容量为 64）。
   - 随机采样 batch 以增加训练多样性。
2. **GPU 并行预取器 (`CUDAPrefetcher`) 与 Batch CPU 合并堆叠**：
   - **CPU 侧合并 Stacking**：在 CPU 端对已执行 `pin_memory` 的张量先进行 `torch.stack()` 拼接，然后整体以单次非阻塞传输 `to(device, non_blocking=True)` 发送至显存，大幅减少了 PCIe 传输和碎片化 CUDA Kernel 的启动开销。
   - **GPU 侧 GT 追踪框预计算**：在 GPU 端（`process_batch_on_gpu` 内）利用矩阵并行直接生成 ground truth Tracking Bounding Boxes（`track_gt_boxes` 形状为 `[B, T, MAX_INSTANCES, 4]`，`track_gt_valid` 形状为 `[B, T, MAX_INSTANCES]`），彻底消除了损失计算中对 `.tolist()`、`nonzero()` 和极高频 `.min()` / `.max()` 的同步依赖。
   - 使用专用的 `torch.cuda.Stream` 与主训练循环重叠（Overlap），在 GPU 上并行执行尺寸缩放、填充与归一化。
3. **去同步化追踪损失与 Trainer 流水线**：
   - **批量化代价矩阵**：在 `compute_track_loss` 中，通过一次 `torch.cdist` 在 GPU 上批量算出整批次的代价矩阵，然后仅执行**单次** `.cpu().numpy()` 下发至 CPU 进行匈牙利图匹配，握手开销由每步 2400+ 次缩减为 1 次。
   - **Detached 累加**：主循环中的 Loss 在 GPU 侧执行 `loss.detach()` 离散累计（完全不占用梯度图），仅在 Epoch 结束或每 10 步打印日志时，才执行标量标定转换（`.item()`），从而确保 PyTorch 主线程的算子能持续流畅地下发。

### 1.3 核心物理量规格与预处理
所有 batch 变量均在 `process_batch_on_gpu` 中被标准化为目标维度（256x256）：

#### 视频帧 (`video`)
- **数据类型**：`uint8` RGB 时序图像。
- **预处理**：缩放到 `float32` 范围 `[0.0, 1.0]`，并使用双线性插值进行尺寸缩放。

#### 绝对深度 (`depth`)
- **数据类型**：`uint16` 编码格式。
- **解码公式**：
  $$\text{depth\_m} = \frac{\text{depth\_encoded}}{65535.0} \times (\text{depth\_range}[1] - \text{depth\_range}[0]) + \text{depth\_range}[0]$$
- **预处理**：深度值为 0 的无限远/天空区域被提取为天空掩膜，其绝对深度被设为 `100.0` 米，真实值被截断在 `[0.01, 100.0]` 范围内。网络预测对数深度 `log_depth`，并映射在 `[-4.6, 4.6]` 区间。

#### 稠密前向光流 (`forward_flow`)
- **数据类型**：`uint16` 压缩像素偏移量。
- **解码公式**：
  $$\text{flow\_px} = \frac{\text{flow\_encoded}}{65535.0} \times (\text{flow\_range}[1] - \text{flow\_range}[0]) + \text{flow\_range}[0]$$
- **归一化**：通过 `flow_raw * 2.0 / target_size` 转换为图像相对坐标系，并截断在 `[-1.5, 1.5]` 范围内以滤除 transient 渲染噪声。

#### 相机与 Ego Pose (`cam_pos`, `cam_quat`)
- **绝对位姿**：`cam_pos` 代表绝对 $(X, Y, Z)$ 坐标，`cam_quat` 代表绝对旋转四元数。
- **四元数表达**：解析顺序为 $(w, x, y, z)$，代表正确的视角方向。
- **相对位姿 (Ego-Pose)**：训练损失计算相邻帧之间的相对平移（$\Delta T$）与旋转矩阵（$\Delta R$）。$\Delta R$ 被扁平化为 6D 连续流形表达，整体组成 9D 相对位姿向量。
- **相机内参**：公式为 $f_x = f_y = \frac{35.0}{32.0} \times W$（当 $W=256$ 时为 $280.0$），光心位于中心 $(W/2, H/2)$，用于逆向投影（Inverse Warping）与光度误差（Photometric Loss）计算。

#### 分割与动态标签 (`segmentation`, `is_dynamic`)
- **实例分割掩膜**：为每个独立物体提供唯一的整数 ID 掩膜。GPU 预处理由此提取 dense 边界框以监督检测头。
- **实例动态性**：静态背景/刚体被映射为类别 0，主动/动态移动的物体映射为类别 1，以解耦 ego-motion 视差与真正动态物理位移。

---

## 2. YOLOE-26 架构与权重载入

实现于 `models/tao_core.py` 与 `models/custom_heads.py` 中的 YOLOE-26 是为物理特征预测与时序连续性进行定制和精简的高效版本。

### 2.1 YOLOESegment26 模型结构
包含以下模块：
- **YOLO-style Backbone 与 FPN/PAN**：捕获 P3、P4、P5 的多尺度特征。
- **时空处理模块**：多尺度特征通过 `SpatioTemporalMambaBlock` 模块融入时序上下文。
- **双分支预测**：提供 dense（多锚点）与 one-to-one（无锚点）的分支输出，分别预测目标概率、定位框与掩膜系数。
- **LRPCLayer 预测层**：
  - `vocab`：包含一个 4585 维的类别语义空间（完整保留零样本能力）。
  - `pf`：生成 objectness 与 prompt-free 门控。
  - `loc`：预测边界框偏移坐标。
- **Proto26 掩膜原型层**：融合多尺度特征生成掩膜原型图与语义分割图。

*注：当前本地实现不包含 RepRTA、SAVPE 等外部文本编码模块，不破坏 4585 维语义空间。*

### 2.2 权重部分加载逻辑
模型从预训练权重 `yoloe-26s-seg-pf.pt` 中部分加载初始化：
1. 本地 Conv 模块（包含 `conv + bn`）在载入前被动态替换为无 BN 且带 `bias=True` 的 Bare Conv 模块，以适配预训练权重的层布局。
2. 权重 state_dict 中的官方路径（如 `model.*` 或 `model.model.*`）被自动映射为本地的 `segmenter.model.*`。
3. 严格匹配机制：仅当参数名称映射成功且张量形状完全一致时才会载入。
4. **加载成功率**：可成功加载 **243 / 298 个键**（81.5%）。诸如 RepRTA、SAVPE 等在本地被精简或改造的结构参数被安全跳过。

---

## 3. 项目核心类（Class）分门别类详细索引 (Structured Class Directory)

为了便于开发者极速查询和深度掌握代码实现，项目为各文件夹下的每一个核心 Class 均构建了**专有的、极其细致的独立知识文档**。点击下方超链接可一键直达对应 Class 的说明：

### 📂 [dataset/](file:///c:/Users/iii/Desktop/tao-not-42-base/dataset.py) — 异步数据加载与 GPU 预处理
- 🚀 [AsyncDataBuffer 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/dataset/AsyncDataBuffer.md) — 后台异步多模态 TFDS 视频帧缓冲区。
- ⚡ [CUDAPrefetcher 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/dataset/CUDAPrefetcher.md) — GPU 并行预取、PCIe 压缩传输与 GPU 端异步并行解码器。

### 📂 [trainer/](file:///c:/Users/iii/Desktop/tao-not-42-base/trainer.py) — 课程训练器
- 📊 [TAOTrainer 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/trainer/TAOTrainer.md) — 多阶段自适应课程损失调度与在线自诊断评估系统。

### 📂 [models/tao_core/](file:///c:/Users/iii/Desktop/tao-not-42-base/models/tao_core.py) — 核心架构与大模型集成
- 🧠 [MyYOLOE 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/tao_core/MyYOLOE.md) — 完全对齐官方 yoloe-26s 拓扑的多尺度特征骨干网络。
- 🦅 [TAONot42VisionModel 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/tao_core/TAONot42VisionModel.md) — 时空物理自监督与端到端追踪视觉大模型整机集成。

### 📂 [models/custom_heads/](file:///c:/Users/iii/Desktop/tao-not-42-base/models/custom_heads.py) — 项目自定义时空与物理预测头
- 🕒 [SpatioTemporalMambaBlock 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/custom_heads/SpatioTemporalMambaBlock.md) — 时空特征混合与傅里叶时间嵌入状态空间单元。
- 📐 [UnifiedGeometryDecoder 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/custom_heads/UnifiedGeometryDecoder.md) — 融入 Ego-Pose 自视差解耦的光流与绝对深度并行预测器。
- 🚗 [EgoPoseHead 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/custom_heads/EgoPoseHead.md) — 连续 6D 三维旋转矩阵回归头。
- 🔮 [FeaturePredictorHead 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/custom_heads/FeaturePredictorHead.md) — 自监督物理特征动力学异常映射预测器。
- 🔗 [TrackQueryModule 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/custom_heads/TrackQueryModule.md) — 持久化时空 Queries 端到端自监督追踪器。

### 📂 [models/yoloe_head/](file:///c:/Users/iii/Desktop/tao-not-42-base/models/yoloe_head.py) — 官方对齐预测与实例分割头
- 🎯 [YOLOESegment26 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/YOLOESegment26.md) — 双轨运行、LRTB 坐标映射与 Top-K NMS-Free 推理头。
- 🎛️ [LRPCHead 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/LRPCHead.md) — 词表映射层折叠与 PF 门控粗选过滤层。
- 🧩 [Proto26 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/Proto26.md) — 爱因斯坦求和实例分割高分辨率原型图生成层。
- 🧲 [BNContrastiveHead 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/BNContrastiveHead.md) — 归一化特征高维余弦空间对比度分类分支。
- 🗺️ [SAVPE 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/SAVPE.md) — 空间感知多尺度视觉提示词嵌入交互层。
- 🌀 [SwiGLUFFN_Residual 详细文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yoloe_head/SwiGLUFFN_Residual.md) — Transformer 双通道非线性门控反馈残差组合层。

### 📂 [models/yolo_blocks/](file:///c:/Users/iii/Desktop/tao-not-42-base/models/yolo_blocks.py) — 底层核心算子积木块
- 🧱 [YOLO_Blocks 整合文档](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/models/yolo_blocks/YOLO_Blocks.md) — SPPF 快速空间池化、C3k2 特征密集流动块、DWConv 等 12 个积木类合集。
