# 自研时空混合与三维物理几何头部专题 (custom_heads.md)

本专题深入剖析了为应对相机高速自运动、场景复杂运动及实例时序追踪而专门设计的自定义网络分支。这些自研头部网络与 YOLO 原生网络解耦放置，构成了整个时空物理感知系统的“核心动力臂”。

---

## 1. 时空特征混合模块 (SpatioTemporalMambaBlock)

时空特征混合模块承载着序列维度（Chunk 级别）的时序信息交互与长距离几何关系捕获的物理职责。

### 1.1 傅里叶时间嵌入与自适应融合
* **绝对时间戳映射**：接收绝对时间戳 $t_{\text{abs}} \in [B, T, 1]$。利用**傅里叶正弦/余弦投影（Fourier Temporal Embedding）**将其投射到高维隐空间（128 维），让网络能够感知并抵御游戏的不稳定帧率（$\Delta t$）。
* **多尺度自适应混合**：多尺度图像特征（P3, P4, P5）首先在 Spatial 维度进行平均下采样，输入 `SpatioTemporalMambaBlock` 完成序列维度的跨帧状态流动。

### 1.2 Mamba 状态空间交互与 Mamba-Fallback 保护
* **厚重代码的巧思**：该模块将 5D 视频流（[B, T, C, H, W]）的空间维度压平，结合 `3D Conv` 进行局部空间特征混合，使 Mamba 把屏幕上的每个像素点视作一个独立的“时间旅行者”，扫过它的历史序列。它准确的定位是“具有时频缩放能力的全局时序上下文融合器”。
* 为保障部署兼容性，我们在 `models/custom_blocks.py` 中实现了 **`SpatioTemporalGRUFallback` (ConvGRU 退化保护模块)**。当 Mamba 不可用时，自动无缝回退到原汁原味的 RAFT ConvGRU 架构推演。

---

## 2. 尺度约束深度与动态遮罩解译器 (UnifiedGeometryDecoder)

本模块旨在依靠多帧时序特征与双帧相关性，求解场景的精确几何结构。**系统绝不是”纯单目”框架**。

### 2.1 彻底废除”自由光流作弊”
* **严谨的几何推导**：光流严格由物理链组装，绝不是卷积的自由输出：
  1. 刚性基础流（相机运动 + 深度）：`Flow_rigid = Project( SE3_cam * Backproject(Depth, K), K )`
  2. 对象刚体流：`Flow_obj = Project( X1 + v + ω×X1, K )`（v、ω 由 `SE3TwistDecoder` 输出，经 `tanh` 硬界定 v≤2m/帧、ω≤1rad/帧）
  3. 最终物理流：`Flow_final = Flow_rigid + obj_mask * (Flow_obj + Flow_residual)`

### 2.2 三大数值稳定性不变量

| 不变量 | 实现位置 | 作用 |
|-------|---------|------|
| **有界操作数**（Invariant 1）| `SE3TwistDecoder.forward`：`tanh × scale` | 消除向投影除法 1/Z² 传递无界梯度的根因 |
| **梯度解耦**（Invariant 2）| `_run_geometry_decoding`：`inv_depth.detach()` 传入投影器 | depth 由 GT-depth loss 训练，flow loss 梯度只进入 twist + residual，-1/Z² 放大器架构级消除 |
| **fp32 几何**（Invariant 3）| `_run_geometry_decoding`：`torch.autocast(enabled=False)` | 透视除法动态范围超 fp16 上限；fp32 保证 BN running_var 不被前向溢出永久污染 |

### 2.3 尺度约束深度的来源
* **绝对米制尺度的注入**：深度绝对米制尺度**唯一来源于训练阶段注入的游戏引擎相机真实运动（Ego-Motion GT）**。

---

## 3. 相机全局自运动解译器 (GlobalEgoMotionDecoder)

`GlobalEgoMotionDecoder` 用于在连续时间序列中估计全局相机自身的旋转与平移参数。

### 3.1 真正的 6D 正交化 (Gram-Schmidt SO(3))
为了保证回归的数值稳定性并避免万向节死锁，回归输出的旋转分量表示两个三维向量 $a_1, a_2 \in \mathbb{R}^3$，通过 **Gram-Schmidt 正交化**构造出绝对正交且行列式为 1 的标准 $3 \times 3$ 旋转矩阵 $R \in SO(3)$：
1. $b_1 = \text{Normalize}(a_1)$
2. $a_{2\_proj} = (b_1 \cdot a_2) b_1$
3. $b_2 = \text{Normalize}(a_2 - a_{2\_proj})$
4. $b_3 = b_1 \times b_2$
5. $R = [b_1, b_2, b_3]$

### 3.2 位姿构建
输出中包含了平移向量 $T$ 和 6D 旋转向量，用于下游计算全局 SE(3) 相机运动。

---

## 4. 实例追踪决策单元 (TrackQueryModule)

`TrackQueryModule` 实现了在时空特征上的高度稳定实例目标绑定与时序跟踪。

### 4.1 Persistent Memory (持久化状态绑定)
* 追踪网络内部维护了 32 个可学习的查询向量（Queries）。
* **跨 Chunk 状态保存**：Queries 不会在每次 `forward()` 时被重置，而是作为网络的持久隐藏状态（Persistent State），在处理序列时跨越帧与 Chunk 边界传递，实现真正的时间线身份一致性绑定。

### 4.2 追踪参数的数学激活约束
* **GIoU 边界框正值保障**：边界框的宽高预测使用严格的 `Softplus(x) + 1e-4`，杜绝出现负宽高的几何谬误。
* **存活率输出**：目标的存活概率 (`track_alive`) 使用明确的 `Sigmoid` 激活，确保数值处在 $[0, 1]$ 之间。

---

## 5. SOTA 时空网络架构重构：SE(3) 稀疏覆盖掩码机制

为解决游戏环境下的高速运动和边缘运动模糊问题，我们在 `SE3PhysicsHead` 中引入了“对象中心化覆盖（Object-Centric Splatting）”机制：

### 5.1 抛弃双线性插值 (Bilinear Interpolation)
传统的双线性插值会跨越物体边界，导致前后景光流与深度互相粘连，产生严重违反物理刚体常理的“幽灵效应”。

### 5.2 真正落地的 Sparse Splatting 密集场拼装
* **稀疏管辖点 (Sparse Anchors)**：网络的高层（如 YOLO P3）输出极其精简但高维的数据：
  1. $\xi_{obj}$ (SE3 局部残差运动) 与 $Z$ (深度)
  2. 中心偏移量 $(\Delta x, \Delta y)$
  3. **高维度的覆盖掩码权重向量 (Coverage Mask Weight $W$)**
* **原型画布拼装 (Prototype Canvas $P$)**：网络底层并行输出一张极其轻量的 32 通道密集特征画布。
* **Assignment 软分配矩阵计算**：
  在前向传播中，严格执行数学点积与 Softmax 分配：`Assignment = Softmax( P @ W^T )`。
  随后，稀疏的 SE(3) 参数与分配矩阵做矩阵乘法，拼装出全图密集的物理运动场。这使得同一刚体表面的光流绝对平滑一致，而在边界处发生锐利断崖式的像素级物理分割！
