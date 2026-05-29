# TAO-Not-42 视觉大模型知识库 (Knowledge Base)

本知识库采用**专项专题化**架构进行组织，所有的专题文档直接存放在本 `knowledge/` 目录下，且每个子目录只属于一个代码模块的说明文档。这有助于以系统、连贯的视角呈现整个项目的技术细节、数学公式与实现机制。

---

## 📂 专题导航与目录结构

### 1. 🧠 [YOLO 视觉底座与权重对齐专题 (yolo.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/yolo.md)
* **核心内容**：
  * **YOLOEBackbone**：对齐 `yoloe-26s` 拓扑的 23 层多尺度金字塔特征提取器。
  * **YOLOESegment26 & LRPCHead**：4585 维大词表开放式分类与实例分割头，含 PF 门控机制。
  * **底层基础算子**：`C3k2`、`C2PSA`、`SPPF`、`DWConv` 等算子数学机理。
  * **权重迁移算法**：如何利用 **Conv-BN 折叠折算技术** 实现与官方带偏置卷积权重的比特级无损加载（相似度达 1.000000）。

### 2. 💾 [数据流、异步加载与并行预处理专题 (dataset.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/dataset.md)
* **核心内容**：
  * **AsyncDataBuffer**：后台独立守护线程，无限保活流式读取与双端随机采样缓冲机制。
  * **CUDAPrefetcher**：基于 CUDA Stream 的计算与传输重叠（Overlap），PCIe 带宽折半压缩传输（Original uint16）与 GPU 并行解码。
  * **process_batch_on_gpu**：使用高并行 `scatter_reduce_` 和 `scatter_add_` 算子直接在 GPU 上快速完成边界框提取，实现**动态显存分配降为 0** 且零 CPU 同步。

### 3. 📐 [自研时空混合与三维物理几何头部专题 (custom_heads.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/custom_heads.md)
* **核心内容**：
  * **SpatioTemporalMambaBlock**：Mamba 状态空间模型与傅里叶时间嵌入，以及缺失环境下的 `SpatioTemporalGRUFallback` ConvGRU 退化保护。
  * **UnifiedGeometryDecoder**：多尺度单目绝对深度估计与稠密光流预测解译器。
  * **GlobalEgoMotionDecoder**：全局相机 6D 位姿旋转矩阵与平移估计头部。
  * **FeaturePredictorHead**：时空动力学特征预测头（异常自监督核心）。
  * **TrackQueryModule**：32 个持久化时序查询向量的实例追踪建模。

### 4. 🔗 [系统整机集成、自监督损失与课程学习专题 (system_integration.md)](file:///c:/Users/iii/Desktop/tao-not-42-base/knowledge/system_integration.md)
* **核心内容**：
  * **TAONot42VisionModel**：主模型前向计算、时空混合、几何解码、自监督计算与追踪子任务串联。
  * **TAOTrainer**：6 阶段渐进式课程训练调度器（检测 -> 姿态 -> 光流 -> 光度 -> 异常 -> 追踪）与评估自诊断体系。
  * **自监督物理一致性损失**：SSIM 光度重投影损失（`inverse_warp` 反投影机制）、Edge-Aware 深度平滑损失、光学光流一致性损失。
  * **Tracklet-Aware 追踪损失**：时序跨帧 ID 持久化绑定与向量化匈牙利匹配损失计算。

---

39: ## 🛠 设计原则
40: 
41: 1. **扁平化结构**：所有专题知识文档直接放置在 `knowledge/` 目录下，根除多层级目录碎片化。
42: 2. **源码级对齐**：所有技术文档中的数学公式、张量形状（Shape）及参数名称，必须与项目根目录下的实际 Python 代码（`models/`、`utils/`、`trainer.py` 等）完全对齐。
43: 3. **中文更新规范**：根据项目活动规则要求，整个知识库的所有专题文档均采用专业严谨的中文进行更新与维护。
44: 
45: ---
