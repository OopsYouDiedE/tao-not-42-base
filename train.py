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
    parser.add_argument("--unfreeze_step_1", type=int, default=1000)
    parser.add_argument("--unfreeze_step_2", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--finetune_after_epoch", type=int, default=0)
    parser.add_argument("--torch_num_threads", type=int, default=0,
                        help="CPU 线程上限；0 表示保持 PyTorch 默认值。主要用于本地 CPU smoke test，GPU 训练通常无需设置。")
    args = parser.parse_args()

    if args.torch_num_threads > 0:
        torch.set_num_threads(args.torch_num_threads)

    if args.device.startswith("cuda") and torch.cuda.is_available():
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
            print(f"[W&B] tracking enabled: project={args.wandb_project}, mode={args.wandb_mode}")
        except Exception as e:
            print(f"[W&B] init failed, continuing without tracking: {type(e).__name__}: {e}")
            wandb = None
    else:
        wandb = None

    try:
        data_buffer = AsyncDataBuffer(
            max_buffer_size=args.max_buffer_size, batch_size=args.batch_size,
            wait_timeout_sec=args.data_timeout_sec)
        prefetcher = CUDAPrefetcher(
            data_buffer, torch.device(args.device), args.img_size,
            wait_timeout_sec=args.data_timeout_sec)
        model = TAONot42VisionModel()

        trainer = TAOTrainer(args, model, data_buffer, prefetcher)
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")
