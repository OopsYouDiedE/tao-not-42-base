# BNContrastiveHead (带批归一化的特征对比度头)

`BNContrastiveHead` 是 YOLOE 分割预测头中专用于特征表示与对比学习特征映射的辅助分支。它通过引入批归一化约束（Batch Normalization）和可学习尺度因子，确保了高维特征的表示稳定性。

---

## 1. 设计初衷与位置

在开放词表（Open-vocabulary）或自监督对比表示学习中，直接对特征进行余弦相似度（Cosine Similarity）投影往往会由于缺少特征尺度的动态调适，导致预测在训练初期极易陷入全零解或梯度崩溃状态。

`BNContrastiveHead` 处于多尺度特征金字塔的最终对比映射处：
- 接收 $512$ 维的实例特征，以及对比目标锚点特征矩阵。
- 利用内置的归一化与动态尺度激活器，输出极具鲁棒性的局部空间特征对比得分。

---

## 2. 类接口与参数说明

### 构造函数

```python
def __init__(self, embed_dims):
```

| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| `embed_dims` | `int` | 对比学习隐空间的通道维度（通常为 512）。 |

---

## 3. 前向计算公式与自适应对比 (Adaptive Contrastive calculation)

在 `forward(x, w)` 阶段，它执行以下极致规范的高阶投影逻辑：

1. **特征归一化 (Feature Normalization)**：
   对特征图 $x \in [B, C, H, W]$ 运行 `self.norm` (2D 批归一化层)。这强制限制了每个通道的激活均值为 0，方差为 1，有效防止了由于某些通道过度激活引起的高维余弦偏置：
   $$x_{\text{norm}} = \text{BatchNorm}(x)$$
2. **权重归一化 (Weight L2-Normalization)**：
   对目标锚点矩阵 $w \in [B, K, C]$（其中 $K$ 代表类别数或提示词数）在最后一维（特征维）执行正规的 L2 标准归一化，使其映射在超球面上：
   $$\hat{w} = \frac{w}{\|w\|_2}$$
3. **爱因斯坦求和空间对比矩阵计算**：
   利用高效的 `torch.einsum` 并行算子，计算三维特征与超球面词表之间的稠密空间点积，输出 $K$ 通道的三维对比度得分：
   $$\text{score}_{\text{raw}} = \text{einsum}("bchw,bkc \to bkhw", x_{\text{norm}}, \hat{w})$$
4. **动态温标调整与偏置**：
   利用可学习的逆温标数 `self.logit_scale`（初始值为 -1.0）与固定的低偏置 `self.bias`（初始值为 -10.0）进行缩放，输出具有超高数值健壮性的最终分类/对比概率矩阵：
   $$\text{score}_{\text{final}} = \text{score}_{\text{raw}} \times \exp(\text{logit\_scale}) + \text{bias}$$
