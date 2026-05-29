# 辅助计算与物理损失工具箱 (utils/)

本目录包含了项目所有用于标签提取、物理三维几何投影、物理一致性损失函数计算以及训练可视化的通用数学与并行算子库。

---

## 📂 文件清单与角色定位

### 1. 💾 [label_generator.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/label_generator.py) (非阻塞 GPU 标签生成器)
* **核心算子**：`process_batch_on_gpu()` 
* **物理职责**：完全运行在 GPU 上，利用高度并行的极值约简 `scatter_reduce_` (基于 amin/amax) 直接提取实例的 Ground-Truth 边界框，并使用 `scatter_add_` 统计实例面积。**动态显存分配降为 0**，实现了全程零 CPU-GPU 同步阻断，打通了 PCIe 数据总线传输的极致吞吐。

### 2. 📐 [geometry.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/geometry.py) (物理重投影几何库)
* **核心方法**：`inverse_warp()`、`generate_intrinsics()`
* **物理职责**：实现了三维相机的反投影建模。根据相机内参、估计的绝对深度将像素点云重构为三维点，随后根据预测自运动 Ego-Pose 旋转和平移，重新投影到未来帧像素网格上以执行双线性插值采样（Warping），提供自监督光度误差信号。

### 3. 🔗 [losses.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/losses.py) (自监督物理与追踪损失函数库)
* **核心方法**：`compute_track_loss()`、`compute_physics_loss()`
* **物理职责**：
  * 计算 SSIM 与 L1 混合的光度重投影损失、Edge-Aware 二阶深度平滑损失、光学流一致性损失。
  * **Tracklet-Aware 追踪损失**：在 Chunk 序列尺度上持久化维护 GT 与 Query 之间的匈牙利绑定关系（抑制 ID Switch 跳变），在 CPU 上收集切片索引后在 GPU 上**单次向量化发射** Smooth L1 与 BCE 损失算子。

### 4. 📊 [visualization.py](file:///c:/Users/iii/Desktop/tao-not-42-base/utils/visualization.py) (训练周期性可视化与落盘)
* **核心方法**：`save_vis_grid()`、`draw_boxes()`
* **物理职责**：在不阻碍主训练线程的前提下，异步 detach 梯度，将训练中的输入图像、实例分割图、预测绝对深度图、预测密集光流图拼接成高保真的可视化网格，并自动处理 `torchvision` ABI 异常降级回退到内置的 NMS。
