# 系统整机集成、半监督几何重投影与课程学习专题 (system_integration.md)

本专题系统性地介绍了大模型的整机网络前向组装（TAONot42VisionModel）、多阶段自适应课程训练控制（TAOTrainer）、结合双帧尺度约束的几何重投影损失，以及面向高稳定性追踪的 Tracklet-Aware 匈牙利时空损失。

---

## 1. 统一视觉大模型整机集成 (TAONot42VisionModel)

`TAONot42VisionModel` 是整个感知大模型的大脑，负责将底层的二维图像特征提取（`YOLOEBackbone`）、时空序列动态特征混合（`SpatioTemporalMambaBlock` / `SpatioTemporalGRUFallback`）、绝对三维几何解析（`SE3PhysicsHead` / `UnifiedGeometryDecoder`）、相机自运动估计（`GlobalEgoMotionDecoder`）、自监督异常检测（`FeaturePredictorHead`）以及时空追踪（`TrackQueryModule`）完美地编织在一起。

### 1.1 系统架构与几何投影闭环
在 `forward_physics` 阶段，系统执行严密的计算闭环：
1. **时空特征交互**：输入视频流，通过骨干网络提取多尺度特征，送入时空 Mamba 模块进行状态时序流动。
2. **尺度约束深度与 SE(3) 解析**：利用时序和相关性特征，输出场景的尺度约束深度与刚体/非刚体掩码。**深度的绝对米制尺度唯一来源于训练时的相机真值监督**。
3. **强制几何反投影推导光流**：严格禁止网络随意预测光流作弊。所有的物理光流均由公式计算得出：
   `Flow_rigid = Project( SE3_cam * Backproject(Depth, K), K )`
   `Final_Flow = Flow_rigid + Flow_residual * Dynamic_Mask`
4. **实例追踪与状态持久化**：依靠 32 个跨 Chunk 持久化存在的 Queries 状态执行跨越时间线的物体锁定。

---

## 2. 多阶段自适应课程训练器 (TAOTrainer)

物理常识与三维几何关系的无监督学习极难在单一阶段或单一损失下直接收敛。`TAOTrainer` 担当“交响乐指挥家”，负责多阶段课程训练。

### 2.1 三阶段动态课程调度 (Curriculum Loss Scheduling)
1. **阶段 1 (0-2000步)**：聚焦静态物体与基础视差，启动深度分支与刚性特征。
2. **阶段 2 (2000-5000步)**：引入全局相机真实运动监督（Ego Pose GT），建立起几何重构的绝对尺度锚点，计算刚体光流误差。
3. **阶段 3 (5000+步)**：激活残差光流、异常预测与 Tracklet-Aware 时空追踪，实现完整的物理系统解耦。

---

## 3. 半监督混合几何重投影训练与防毒

在依赖前后多帧序列建立的三维空间重投影等价关系中，为了保证在极具挑战的游戏环境中稳健收敛，我们引入了两项巧妙的防毒机制：

### 3.1 相机真值注入与 UI 数据投毒 (Data Augmentation)
* **相机真值强监督**：在训练阶段，直接利用游戏引擎提取的两帧间相机真实运动 GT $\xi_{cam\_gt}$ 对相机的 SE3 分支施加 L1/L2 强监督。这直接打破了单目推断极难避免的**深度与自车运动尺度模糊性（Scale Ambiguity）**，迫使网络专注求解真实的深度几何。
* **合成假 UI 增强（“数据投毒”）**：由于原始数据集无 UI，我们在 Dataloader 阶段随机在画面中绘制彩色几何图块（假血条等），强迫网络训练出极其灵敏的掩码 $p_{ui}$。

### 3.2 物理反投影 Warping 与自动异常拦截 (Photometric Loss)
* **三维投影建立 (`inverse_warp`)**：通过给定的相机内参 $K$、求解到的尺度约束深度和位姿 $R, T$，将当前帧像素反投回 3D 点云，经过刚体变换后重新正投影到原图，执行双线性插值进行图像重构。
* **SSIM 混合光度一致性损失与拦截**：
  $$\mathcal{L}_{\text{photo}} = (1 - p_{ui}) \cdot \min\left(\mathcal{L}_{\text{fwd\_photo}}, \mathcal{L}_{\text{bwd\_photo}}\right)$$
  1. **自动遮挡剔除 (Auto-Masking)**：前向与后向重构误差取最小值 $\min()$，自动剔除物体遮挡带来的谬误。
  2. **UI 干扰截断**：误差乘以 $(1 - p_{ui})$，任何落在假 UI 或 HUD 界面的像素，其重投影误差都会被拦截清零，保证传回 Backbone 的每一滴梯度都纯净无瑕。

---

## 4. 异常特征检测与持久化时序追踪损失

### 4.1 FeaturePredictorHead (基于不确定性的异常发现)
真正的物理异常检测不仅是特征相减。网络同时预测出一个**不确定性方差图 (Uncertainty Map)**。
异常分数计算公式升级为：`Anomaly_Score = Norm(Pred_Feature - True_Feature) / Predicted_Uncertainty`。
在遮挡、模糊等网络原本就不确定的区域，方差增大，抑制错误的高分异常，大幅提升检测可靠性。

### 4.2 时序稳定的 Tracklet-Aware 匈牙利匹配
* **Persistent Memory (跨帧锁死)**：凡是上一时刻已经匹配的实例，其 ID 映射关系将强行锁定并继承其隐含状态，仅对新产生的 GT 实例触发新一轮的匈牙利线性匹配关联。
* **激活保障**：所有追踪宽高的预测输出经过严密的 `Softplus(x) + 1e-4` 处理，完全杜绝了 GIoU 损失计算崩盘。
