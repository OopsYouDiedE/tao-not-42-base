import argparse
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
    parser.add_argument("--use_wandb", action="store_true", default=False)
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

    if args.use_wandb and wandb:
        wandb.init(project="tao_not_42", config=vars(args))
    elif not args.use_wandb:
        wandb = None

    try:
        data_buffer = AsyncDataBuffer(
            max_buffer_size=args.max_buffer_size, batch_size=args.batch_size)
        prefetcher = CUDAPrefetcher(
            data_buffer, torch.device(args.device), args.img_size)
        model = TAONot42VisionModel()

        trainer = TAOTrainer(args, model, data_buffer, prefetcher)
        trainer.train()
    except KeyboardInterrupt:
        print("\n🛑 训练被用户中断。")
