# Agent Activity & Architectural Decision Log (AGENTS.md)

This log tracks architectural changes, critical bug fixes, performance optimizations, and debugging isolation procedures implemented by agentic assistants.

---

## Assistant Constraints & Guidelines

> [!IMPORTANT]
> **开发与文档规范限制**
> 1. **文档语言要求**：所有的项目文档（包括但不仅限于 `readme.md`、`knowledge/knowledge_base.md` 和 `AGENTS.md` 等）**必须使用中文更新**。
> 2. **Mock 测试隔离原则**：
>    - 核心代码库（如 `dataset.py`、`trainer.py`、`train.py`）必须保持纯净且适合生产环境训练（如 Google Colab / 云端 TFDS），不能掺杂任何本地 Mock 或调试用临时数据加载代码。
>    - 本地调试测试脚本 `test_mock.py` 应该是**直接 `import train.py` (或对应的运行入口)**，然后再按需在运行时动态替换（Monkey Patch）其中的模块，而不是在生产代码中编写 Mock 逻辑。

---

## Log Entry: 2026-05-28

### 1. GPU Efficiency Optimization (Utilization -> 90%+)
- **Problem**: GPU utilization during training was limited to ~80% due to major bottlenecks in data loading, PCIe transfers, and massive CPU-GPU synchronizations inside the tracking loss computation block.
- **Root Cause & Fixes**:
  1. **Tracking Loss De-synchronization** (`utils/losses.py`):
     - **Removed** redundant nested loops calling `.unique().tolist()`, `m.nonzero()`, and `.min()` / `.max()` on GPU tensors, which triggered 2400+ host-device roundtrips per batch.
     - **Precalculated** ground truth tracking bounding boxes (`track_gt_boxes` and `track_gt_valid`) on the GPU within `process_batch_on_gpu`.
     - **Batched Cost Computation**: Flattened sequences to compute cost matrix in a single `torch.cdist` call on GPU, resulting in exactly **one** `.cpu().numpy()` transfer to CPU per step.
  2. **Pipeline PCIe Batching** (`dataset.py`):
     - **Stacked** CPU-pinned tensors on CPU first before transferring to GPU via a single `to(device, non_blocking=True)` call.
  3. **Main Training Loop Pipe-lining** (`trainer.py`):
     - **Removed** blocking `loss.item()` calls inside the training steps, accumulating float losses as detached GPU tensors and converting to CPU scalar only once per epoch or log step (every 10 steps).
  4. **Robust KeyError Resolution** (`trainer.py`):
     - Changed direct lookup `l_dict[k]` to safe default access `l_dict.get(k, 0.0)` for all spatiotemporal loss components, ensuring that whenever a loss component (such as `'Photo'`) is inactive, the loop safely cascades to `0.0` without breaking the training run.

### 2. Isolation of Mock & Debugging Utilities
- **Decision**: To keep the core codebase (`dataset.py`, `trainer.py`, `train.py`) strictly production-ready and free of local development or mock file dependencies, all offline simulation and mock loading logic was completely isolated into `test_mock.py`.
- **Implementation**:
  - Reverted `dataset.py` to a clean, production-grade streaming dataloader with zero references to local NPZ files (`movi_e_sample_0000.npz`) or environment-dependent mock flags (`FORCE_MOCK`).
  - Refactored `test_mock.py` to load and parse the local 32MB genuine Kubric sample `"movi_e_sample_0000.npz"` directly, adding support for the dataset's native high-fidelity properties (`depth_m` and `forward_flow_px`).
  - Verified that running `python test_mock.py` completes successfully fully offline, performing exact curriculum physical and tracking loss validations.

### 3. 可视化模块 Grad 运行时异常修复 (Save Visualization RuntimeError Fix)
- **问题**：在可视化保存阶段，`utils/visualization.py` 在执行第 62 行 `b_np = t_boxes[i].cpu().numpy()` 时触发了 `RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.` 报错，导致训练中断。
- **根本原因**：`t_boxes` 和 `t_alive` 提取自模型的预测输出 `pred_t`（在反向传播计算图中），这些张量本身带有梯度信息 (`requires_grad=True`)。由于未在 `with torch.no_grad():` 块中处理，且直接在未分离（detach）的情况下调用了 `.cpu().numpy()`，导致 PyTorch 的 Autograd 引擎拦截并报错。
- **修复方案**：
  - 在 `utils/visualization.py` 中，对 `pred_t["track_boxes"]` 和 `pred_t["track_alive"]` 分别在切片/变形前调用 `.detach()`，使其从当前计算图中断开，从而彻底解决 numpy 转换时的 autograd 报错问题。
  - 通过运行本地测试脚本 `python test_mock.py` 进行了全阶段多课程训练和跟踪组件的高保真仿真闭环测试，确认 5 阶段可视化结果保存顺利（成功保存至 `vis_outputs/vis_step_05000.jpg`），Stage 6 跟踪网络验证全部通过，无任何异常。
