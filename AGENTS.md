# 智能体活动与架构决策日志 (AGENTS.md)

本日志记录了智能体助手实施的架构更改、关键错误修复、性能优化以及调试隔离程序。

---

## 助手约束与指南

> [!IMPORTANT]
> **开发与文档规范限制**
> 1. **文档语言要求**：所有的项目文档（包括但不仅限于 `readme.md`、`knowledge/knowledge_base.md` 和 `AGENTS.md` 等）**必须使用中文更新**。
> 2. **Mock 测试隔离原则**：
>    - 核心代码库（如 `dataset.py`、`trainer.py`、`train.py`）必须保持纯净且适合生产环境训练（如 Google Colab / 云端 TFDS），不能掺杂任何本地 Mock 或调试用临时数据加载代码。
>    - 本地调试测试脚本 `test_mock.py` 应该是**直接 `import train.py` (或对应的运行入口)**，然后再按需在运行时动态替换（Monkey Patch）其中的模块，而不是在生产代码中编写 Mock 逻辑。
> 3. **禁止非 CUDA 环境测试与移除 CPU 兼容**：
>    - 禁止在没有 CUDA 的环境中运行测试或核心模型。
>    - 核心生产代码库（如 `dataset.py`、`trainer.py`、`train.py`、`models/custom_heads.py`、`utils/losses.py`、`utils/visualization.py`）必须彻底移除任何针对 CPU 运行或环境缺包（如未安装 `mamba_ssm`、`scipy` 或 `torchvision`）的兼容与降级（fallback）处理逻辑，导入失败必须直接报错。
>    - 所有的 CPU 兼容和降级模拟逻辑**必须且仅允许**保存在独立的 `tests/` 目录下的 Mock 代码中。
> 4. **测试目录结构规范**：
>    - `tests/conftest.py`：全局拦截器与 Mock 注入器（当测试环境没有 `mamba_ssm` 时自动注入）。
>    - `tests/mock_mamba.py`：Mamba 算子的分组卷积平替。
>    - `tests/data/movi_e_static_sample.npz`：本地微型数据集。
>    - `tests/unit/`：纯粹的单元测试（无需网络，无需 GPU，如几何、损失函数计算）。
>    - `tests/integration/`：集成测试（覆盖离线数据流、Mamba Mock 拦截下的完整前向与反向梯度流动）。
> 5. **TAO-Not-42 文档治理 SOP**：
>    - **SSOT 原则 (Single Source of Truth) 与契约化注释**：代码是逻辑的唯一事实来源。禁止在 Markdown 中描述易变的微观逻辑。所有的 Class/Function 必须使用 Google/NumPy 风格的 Docstring 明确声明 Shape 和 Dtype 契约；Markdown 仅解释宏观系统架构、算法设计思想与物理直觉。
>    - **变更隔离与同步红线**：任何对 `models/` 或 `utils/` 的目录结构、类名、核心方法签名进行修改的提交，**必须包含**对 `knowledge/` 对应 Markdown 专题的同步更新。
>    - **物理废弃声明**：重构替代的废弃模块（如 `EgoPoseHead`）必须**物理删除**原文件，严禁仅留空并写注释，同时在变更日志中显式描述迁移路线。
>
> [!NOTE]
> **🛠 环境与能力降级矩阵 (Capability Matrix)**
> 
> | 硬件环境 | `mamba_ssm` | 数据源 (TFDS) | 预期能力边界 | 测试覆盖率支持 |
> | :--- | :--- | :--- | :--- | :--- |
> | **Linux + CUDA (生产)** | ✅ 已安装 | ✅ 在线下载 | **全量端到端闭环训练**，支持时空长序列。 | 100% E2E 测试 |
> | **Windows + CUDA (开发)** | ❌ 未安装 | ❌ 离线 NPZ | **核心算法与几何验证**。Mamba 退化为 1D 卷积，数据流读取本地 Mock。 | >85% 单元与集成测试 |
> | **CPU Only** | - | - | 🚫 **不支持**。视觉底层算子与 Scatter 极值约简强依赖 CUDA。 | 0% |
