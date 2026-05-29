# 测试与仿真调试包 (tests/)

本目录包含了项目所有的本地 Mock 仿真器、环境降级 Fallback 处理以及与官方 YOLO 权重对齐的零 Prompt 目标分割测试。

---

## 📂 文件清单与角色定位

### 1. 🎯 [test_yoloe_bus.py](file:///c:/Users/iii/Desktop/tao-not-42-base/tests/test_yoloe_bus.py) (YOLOE 分割与分类零对齐测试)
* 构建并初始化标准的 `YOLOEBackbone` 网络与官方分割头。
* 自适应折叠并载入 `yoloe-26s-seg-pf.pt` 的 291 个参数键值。
* 对标准的 `bus.jpg` 图像执行前向推理，输出多尺度边界框、类别概率、置信度，并渲染出高保真分割标注拼图 `vis_aligned_bus.jpg`，实现了与官方推理 **100% 毫无差别的数值一致性**。

### 2. 💾 [mock_data.py](file:///c:/Users/iii/Desktop/tao-not-42-base/tests/mock_data.py) (高保真物理仿真球模拟器)
* **核心类**：`MockPhysicsDataGenerator`
* **物理职责**：在本地脱机且缺少 MOVi-E 大数据集的环境中，本仿真器在三维空间中构建了带有物理定律的运动球（包含惯性漂移、速度反弹与遮挡计算），为多阶段课程训练在线流式生成高保真的图像序列、绝对深度、连续相机位姿（Ego-Pose）及像素级光流图。

### 3. 🕒 [mock_mamba.py](file:///c:/Users/iii/Desktop/tao-not-42-base/tests/mock_mamba.py) (时序混合 Fallback 控制)
* 当开发机或服务器未安装复杂的 `mamba_ssm` 时，由本模块提供动态 Monkey Patch 注入，将时序交互无缝路由到我们实现的 `TemporalConvFallback` 分组时间空洞卷积中，不阻塞自监督主流程开发。

### 4. 🧮 [mock_scipy.py](file:///c:/Users/iii/Desktop/tao-not-42-base/tests/mock_scipy.py) (匈牙利匹配求解 Fallback 控制)
* 当本地缺少 `scipy.optimize` 时，由本模块提供动态 Monkey Patch，执行贪心匹配回退算子，不破坏多课程联合训练整体架构运行。

---

## 🚀 运行说明

所有测试及 Mock 仿真在被 `test_mock.py` 载入时，会自动断言 `torch.cuda.is_available()` 满足 CUDA 物理硬件约束，并自动执行注入。
您可以直接在根目录运行检测对齐测试：
```bash
python tests/test_yoloe_bus.py
```
