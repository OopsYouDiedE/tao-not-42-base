# CUDAPrefetcher (GPU 异步预取器与并行解码器)

`CUDAPrefetcher` 是专门为极致发挥英伟达 GPU 带宽、减少 PCIe 总线延迟而定制的并行计算辅助类。它实现了基于 CUDA Stream 的计算与传输重叠（Overlap），同时承载了 GPU 端并行解码与目标框自适应计算的职责。

---

## 1. 设计初衷与位置

传统的 PyTorch `DataLoader` 在将数据从 CPU 发送至 GPU 时，通常是在训练的主线程中同步执行 `batch = batch.to(device)`。这会产生两大严重弊端：
1. **PCIe 阻塞**：传输期间 GPU 核心处于空闲等待状态，产生明显的 PCIe 泡泡。
2. **CPU 侧解码开销过大**：将密集图像、`uint16` 绝对深度图与稠密光流图在后台 CPU 线程中频繁解码，极易引起 CPU 过载，导致数据流水线饥饿。

`CUDAPrefetcher` 通过在 GPU 端实现**异步解码**和 **CUDA 专属流传输** 解决了以上问题：

```
主 CUDA 流 (计算)     : |--- Forward & Backward (Batch N) ---|--- Forward & Backward (Batch N+1) ---|
                       \                                     \
预取 CUDA 流 (传输)   :  \--- Pinned Transfer (Batch N+1) ---|--- Pinned Transfer (Batch N+2) ---|
```

---

## 2. 类接口与参数说明

### 构造函数

```python
def __init__(self, async_buffer, device, batch_size=1, target_size=(256, 256)):
```

| 参数 | 类型 | 描述 |
| :--- | :--- | :--- |
| `async_buffer` | `AsyncDataBuffer` | 绑定的 CPU 异步缓存实例。 |
| `device` | `torch.device` | 目标计算 GPU 设备（通常为 `cuda:0`）。 |
| `batch_size` | `int` | 单次提取与预取的批次大小（当前物理课程默认为 1）。 |
| `target_size` | `tuple` | 缩放的目标图像宽高分辨率。 |

---

## 3. 核心机制：三剑客优化

`CUDAPrefetcher` 内部实现了以下三个层面的极致优化：

### 3.1 预取与 CUDA 异步流重叠 (`preload`)
`CUDAPrefetcher` 内部维护了 `self.stream = torch.cuda.Stream()`，这是一个独立的预取专用计算流：
- 在主流正在运行当前的 Forward/Backward 时，预取流异步下发 `non_blocking=True` 的显存拷贝指令。
- 采用双缓冲区机制（包含当前已就绪的 `self.next_batch` 与后台正在传输的样本），使得训练主线程在进入下一轮迭代时，GPU 显存中早已准备好了完整的时序数据。

### 3.2 PCIe 总线传输带宽折半 (Compressed Transmission)
- **绝对深度与光流以原始 `uint16` 压缩传输**：以前将深度图和光流图在 CPU 解码为 `float32` 后传输，显存数据量极大（$256 \times 256 \times 24 \times 4$ 字节）。
- `CUDAPrefetcher` 限制了在 PCIe 上只传输原始的 `uint16` 格式。传输的数据体积**压缩了整整 50%**，极大缓解了 PCIe 硬件带宽瓶颈。
- 传输完成后，由 GPU CUDA 核心在专属流中**并行、极速地执行 float32 反归一化与还原解码**。

### 3.3 GPU 侧无同步边界框计算 (`process_batch_on_gpu`)
- **零显存分配开销**：删除了原本的实例 seg 空间掩膜提取，引入了高度并行的 `scatter_reduce_` 和 `scatter_add_` 算子，直接在 1D 展平坐标图上由 GPU 计算出地面真值边界框（`track_gt_boxes`）。
- **零 CPU 反馈同步**：去除了所有 `.cpu().numpy()` 阻断，边界框和实例 presence mask 全程保留在 GPU 寄存器与显存上。

---

## 4. 核心数据解码公式与操作

### 4.1 深度图 GPU 异步解码
```python
depth_raw = depth_encoded.float() / 65535.0
depth_m = depth_raw * (depth_range[1] - depth_range[0]) + depth_range[0]
```
对于天空或深度未知的远端像素，`CUDAPrefetcher` 会并行设置：
- `depth_m[depth_encoded == 0] = 100.0`
- 最终对数深度映射关系：`log_depth = torch.log(depth_m)`

### 4.2 光流 GPU 异步解码与通道修正
MOVi-E 原始光流以 `(dy, dx)` 排列。`CUDAPrefetcher` 执行如下转置与图像线性比例缩放：
```python
# 通道重置为标准的 (dx, dy)
flow_px = flow_encoded.float() / 65535.0
flow_px = flow_px * (flow_range[1] - flow_range[0]) + flow_range[0]
flow_std = flow_px[..., [1, 0]]  # y,x -> x,y
# 分辨率线性变换比率
flow_scaled = flow_std * (target_size / original_size)
```

---

## 5. 使用范例与生命周期管理

```python
prefetcher = CUDAPrefetcher(async_buffer, device, batch_size=1)

# 获取下一批异步载入且解码好的数据
batch = prefetcher.next()
while batch is not None:
    # 核心前向与损失计算（已是纯 GPU Tensor 且异步对齐）
    outputs = model(batch["video"])
    loss = compute_loss(outputs, batch)
    ...
    batch = prefetcher.next()
```
- **异步生命周期保护**：由于 `next()` 包含异步 stream 拷贝，`CUDAPrefetcher` 使用了 `torch.cuda.current_stream().wait_stream(self.stream)` 握手，确保主流在读取 tensor 时其内存拷贝已 100% 结束，绝对防范异步内存写污染或空读。
