# 项目 Mock 测试指南

本文档介绍了 `tao-not-42-base` 仓库中提供的两个主要 Mock 测试。

## Mock 1: YOLOE 预测测试
**文件:** `tests/test_yoloe_bus.py`

### 描述
该测试通过在标准的 `bus.jpg` 图像上执行推理来验证 YOLOE 网络实现。它执行以下步骤：
1.  **网络构建:** 构建 `YOLOEBackbone` 网络。
2.  **权重加载:** 从 `yoloe-26s-seg-pf.pt` 加载权重。它会自动重命名键值，以确保与自定义模型结构的“全量匹配”。
3.  **推理:** 下载并处理 `bus.jpg`，然后运行零 Prompt 预测。
4.  **验证:** 输出检测到的边界框（Boxes）和分类得分（Classification scores）的形状。

### 如何运行
```bash
python tests/test_yoloe_bus.py
```

---

## Mock 2: 自定义数据集训练测试
**文件:** `tests/test_mock.py`

### 描述
该测试使用来自 MOVi-E 数据集的单个无限重复样本来模拟完整的训练课程。
- **参数:** Batch Size = 1, Sequence Length = 24。
- **数据:** 使用 `movi_e_sample_0000.npz`。
- **阶段:** 执行所有 5 个课程阶段（检测、姿态、光流、光度误差、异常检测）以及第 6 阶段（追踪）。

### 数据准备
如果缺少 `movi_e_sample_0000.npz`，脚本将提供说明。您可以通过在 Google Colab 中运行以下代码来生成它：

```python
!pip install tensorflow-datasets imageio
import tensorflow_datasets as tfds
import numpy as np
ds = tfds.load('movi_e/256x256', data_dir='gs://kubric-public/tfds', split='test')
sample = next(iter(tfds.as_numpy(ds.take(1))))
np.savez_compressed('movi_e_sample_0000.npz', **sample)
```
运行后，下载生成的 `.npz` 文件并将其放置在项目根目录中。

### 如何运行
```bash
python tests/test_mock.py
```
*(注意: 需要 CUDA 支持)*
