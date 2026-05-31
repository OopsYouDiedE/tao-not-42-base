# TAO-Not-42 训练收敛指标分析指南 (Training Metrics & Convergence Guide)

本文档基于 `utils/losses.py` 的底层数学实现，为你提供各项物理 Loss 和几何 Loss 的收敛基准线。
在后续的云端全量训练中，你可以通过对比这些阈值来判断模型是否“训练得足够好”。

## 1. 实例发现模块 (Instance Discovery)

| 损失项 (Loss) | 计算方式原理 | 期望收敛目标 | 说明 |
| :--- | :--- | :--- | :--- |
| **Obj Loss** | `Focal Loss` (二分类交叉熵 + 难样本挖掘) | **< 0.05** | 衡量模型能否分清“哪里是物体，哪里是背景”。过拟合测试中通常能降到 `0.06` 以下。如果高于 `0.2`，说明画面中存在大量的幽灵检测（Ghost Detections）或漏检。 |
| **Box Loss** | `GIoU Loss * 1.5 + DFL * 0.5` | **< 0.20** | GIoU 越接近 0 代表边界框和真值完全重合。初期随机值往往在 `3.0` 以上，收敛极好时应在 `0.1` 左右。 |
| **Mask Loss** | `Dice Loss * 2.0 + Focal BCE` | **< 0.10** | 像素级的掩膜重合度。Dice Loss 最优值为 0。收敛后应降至 `0.05 ~ 0.1`，此时实例的轮廓已经相当锐利和准确。 |

## 2. 三维几何与物理投影模块 (3D Geometry & Physics)

| 损失项 (Loss) | 计算方式原理 | 期望收敛目标 | 说明 |
| :--- | :--- | :--- | :--- |
| **Depth Loss** | `SmoothL1(log_D) + SmoothL1(∂D/∂x) + SmoothL1(∂D/∂y)` | **< 0.10** | 在对数空间计算主项，额外加 x/y 方向梯度监督确保深度边缘锐利。`0.1` 的 Log L1 误差约等价于 10% 相对误差（10 米处测成 9~11 米）。 |
| **Ego Loss** | 对相机平移 `t` 和旋转 `rot6d` 的 SmoothL1 回归 | **< 0.10** | 衡量相机自身运动轨迹的估计精度。 |
| **Flow Loss** | `Smooth L1`（归一化光流，tanh 软约束至 ±2） | **< 0.01** | 光流严格由物理链推导（depth→backproject→transform→project），深度梯度在架构上已与 flow loss 解耦。Flow 只训练有界 twist + 残差解码器，EPE 目标 < 10 px（256×256 画面）。 |

## 3. 时序追踪与高级感知 (Tracking & High-level Perception)

| 损失项 (Loss) | 计算方式原理 | 期望收敛目标 | 说明 |
| :--- | :--- | :--- | :--- |
| **Track Loss** | Sinkhorn 匹配 + 时序框重叠度 | **< 0.25** | 衡量物体在跨帧时 ID 的一致性。它比单帧的 Box Loss 稍大是正常的，因为它必须在剧烈运动中维持 ID 绑定。 |
| **Anom Loss** | 特征空间异常度量 | **< 0.10** | 预测下一帧物理特征与真实特征的残差，用于判定物理世界的“意外事件”（如碰撞）。 |

---

### 💡 训练质检黄金准则：

当前有效 loss 项（`get_loss_weights` 常量字典）：`obj`、`box`、`mask`、`depth`、`ego`、`flow`、`attr`、`anom`、`track`。已移除：`photo`（GT 直接监督下冗余）、`cls`（vocab 冻结）、`gate`（正则化头已删）、`smooth`（与 depth 梯度监督重复）。

在观察终端或 W&B 曲线时：
1. **初期暴降（前 1000 步）**：`Box` 和 `Mask` 从 3.0+ 迅速降到 0.5 左右；`Depth` 同步收敛，DepthAbsRel 应在 1000 步内降至 0.3 以下。
2. **中期对齐（1000–5000 步）**：`Obj` 趋近 0.05，`Flow` 同步训练（不再延迟启动）。Flow 只监督有界 twist + 残差，如果 EPE 长期 > 100 px 不降，说明 twist 仍未建立几何对应，检查 SE3PhysicsHead 梯度。
3. **后期精化（5000 步以后）**：`Depth Loss` 破 `0.10`，`Flow Loss` 破 `0.01`，EPE < 10 px，三维投影物理链完全对齐。
