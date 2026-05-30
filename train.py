import argparse
import os
import torch

try:
    import wandb
except ImportError:
    wandb = None

from trainer import TAOTrainer
from dataset import AsyncDataBuffer, CUDAPrefetcher
from models.tao_core import TAONot42VisionModel

# Main 参数配置与执行入口
# =====================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--seq_len", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_buffer_size", type=int, default=64)
    parser.add_argument("--vis_interval", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)
    parser.add_argument("--early_stop_patience", type=int, default=3)
    # 彻底取消任何硬设置 step 的解冻，参数解冻静态由 --freeze 控制。保留参数仅为兼容命令行
    parser.add_argument("--unfreeze_step_1", type=int, default=1000, help="[已废弃] 骨干解冻步数（现已彻底移除动态解冻，由 --freeze 静态控制）")
    parser.add_argument("--unfreeze_step_2", type=int, default=2000, help="[已废弃] 骨干解冻步数（现已彻底移除动态解冻，由 --freeze 静态控制）")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda",
                        help="训练使用的 GPU 设备类型。生产环境必须为 CUDA。")
    parser.add_argument("--checkpoint", type=str,
                        default="tao_not_42_weights.pth")
    parser.add_argument("--yolo_weights", type=str,
                        default="yoloe-26s-seg-pf.pt")
    parser.add_argument("--use_wandb", dest="use_wandb", action="store_true", default=True,
                        help="启用 Weights & Biases 跟踪；默认开启。")
    parser.add_argument("--no_wandb", dest="use_wandb", action="store_false",
                        help="关闭 Weights & Biases 跟踪。")
    parser.add_argument("--wandb_project", type=str, default="tao_not_42")
    parser.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                        choices=["online", "offline", "disabled"],
                        help="WandB 模式。online 会正常上传；offline 本地记录；disabled 完全关闭。")
    parser.add_argument("--wandb_init_timeout", type=int, default=20,
                        help="WandB 初始化超时秒数，避免网络/登录问题卡住训练。")
    parser.add_argument("--data_timeout_sec", type=int, default=180,
                        help="等待 TFDS/预取 batch 的最长秒数；超时会报出明确错误而不是静默卡住。")
    parser.add_argument("--no_benchmark", action="store_true", default=False,
                        help="关闭 cuDNN benchmark（适合需要极速启动的单步调试阶段）。")
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--finetune_after_epoch", type=int, default=0)
    parser.add_argument("--offline_path", type=str, default=None,
                        help="指定离线 NPZ 数据集路径（例如 tests/data/movi_e_static_sample.npz），启用后不依赖在线下载。")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("严格要求 CUDA 环境！核心视觉组件无法在 CPU 上运行。")

    if args.device.startswith("cuda") and torch.cuda.is_available():
        if not args.no_benchmark:
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.use_wandb and args.wandb_mode != "disabled" and wandb:
        try:
            os.environ.setdefault("WANDB_SILENT", "true")
            os.environ.setdefault("WANDB_INIT_TIMEOUT", str(args.wandb_init_timeout))
            try:
                settings = wandb.Settings(start_method="thread", init_timeout=args.wandb_init_timeout)
            except TypeError:
                settings = wandb.Settings(start_method="thread")
            wandb.init(
                project=args.wandb_project,
                config=vars(args),
                mode=args.wandb_mode,
                anonymous="allow",
                settings=settings,
            )
            print(f"[W&B] 跟踪已启用：项目={args.wandb_project}, 模式={args.wandb_mode}")
        except Exception as e:
            print(f"[W&B] 初始化失败，在没有跟踪的情况下继续：{type(e).__name__}: {e}")
            wandb = None
    else:
        wandb = None

    try:
        data_buffer = AsyncDataBuffer(
            max_buffer_size=args.max_buffer_size, batch_size=args.batch_size,
            wait_timeout_sec=args.data_timeout_sec, offline_path=args.offline_path)
        prefetcher = CUDAPrefetcher(
            data_buffer, torch.device(args.device), args.img_size,
            wait_timeout_sec=args.data_timeout_sec)
        model = TAONot42VisionModel()

        trainer = TAOTrainer(args, model, prefetcher)
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")
