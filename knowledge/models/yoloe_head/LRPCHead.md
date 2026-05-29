# LRPCHead (轻量区域建议与分类预测层)

`LRPCHead`（Lightweight Region Proposal and Classification Head）是 YOLOE 预测分支中的核心执行算子。它直接承载了分类特征与定位特征的语义映射，并在推理态负责空间网格的轻量化置信度过滤。

---

## 1. 设计初衷与位置

在标准的密集预测（Dense Prediction）网络中，为每个空间网格直接回归数千维分类的计算量是极度庞大的。`LRPCHead` 创造性地引入了 **PF 门控机制（Prompt-free Filter）**：
- **过滤冗余**：利用超轻量的单通道卷积 `pf` 首先在分类特征图上预测每一个格点存在目标的置信度得分（Objectness）。
- **按需投影**：在评估模式下，只对分数大于 `conf` 阈值（默认 0.001）的少部分存活网格，通过分类层（`vocab`）映射到 $4585$ 维的语义类别空间。这使得计算量暴降 95% 以上，实现了在大词表场景下的极其流畅推理。

---

## 2. 权重折叠与 conv2linear 正确性保障

在官方 `yoloe-26s-seg-pf.pt` 预训练权重中，为了便于某些部署设备上的并行执行，前两尺度（Scale 0 与 Scale 1）的分类层在导出时，其参数布局被转换为了 `nn.Linear` 权重（形状为 `[nc, c3]`），而第三尺度（Scale 2）则保留了 `1x1 Conv2d` 卷积结构（形状为 `[nc, c3, 1, 1]`）。

若直接加载，会导致前两尺度产生 Shape 维度不匹配而崩溃。
`LRPCHead` 内部实现了一个极其鲁棒的静态转换方法 `conv2linear`：

```python
@staticmethod
def conv2linear(conv: nn.Conv2d) -> nn.Linear:
    assert isinstance(conv, nn.Conv2d) and conv.kernel_size == (1, 1)
    linear = nn.Linear(conv.in_channels, conv.out_channels)
    linear.weight.data = conv.weight.data.view(conv.out_channels, -1)
    linear.bias.data = conv.bias.data
    return linear
```

在模型初始化时：
- 对前两个尺度传入的分类 Conv 层，自动调用此方法正交折叠并转化为 `nn.Linear`。
- 保证了结构上与官方预训练权重的参数 100% 毫无差别的数值对齐。

---

## 3. 类接口与参数说明

### 构造函数

```python
def __init__(self, vocab, pf, loc, enabled=True):
```

| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| `vocab` | `nn.Module` | 类别语义投影层（可以是 Linear 或 Conv2d）。维数对应 $4585$ 类。 |
| `pf` | `nn.Conv2d` | 单通道 $1 \times 1$ 卷积，用于回归格点置信度分数。 |
| `loc` | `nn.Conv2d` | 四通道 $1 \times 1$ 卷积，用以估计网格中心至边界的绝对距离。 |
| `enabled` | `bool` | 门控开关。当前官方版本默认 `True`，即激活 Linear 解码过滤路径。 |

---

## 4. 前向逻辑与数据重排

在前向传播时，根据 `self.enabled` 进行双轨分支处理：

- **当 `enabled=True`（对齐官方模式）**：
  1. `pf_score = self.pf(cls_feat)`，经过 `sigmoid` 判断哪些网格满足 `conf` 限制，产生 `mask` 布尔掩膜。
  2. 若 `conf > 0`（推理态）：
     将分类特征图铺平为 `[B, H*W, C]`，并只提取 `mask` 为真的存活特征切片送入 `self.vocab`（Linear）进行映射，输出 `[B, N_kept, nc]` 的紧凑分类矩阵。
  3. 若 `conf = 0`（训练态）：
     保留所有梯度图，用带有掩膜值的特征图通过 `vocab` 输出空间全网格分类得分。
- **当 `enabled=False`**：
  直接对全图空间进行 Conv2d 分类回归，作为原生 YOLO Dense 机制。
