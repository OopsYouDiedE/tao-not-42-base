import os
import time
import queue
import random
import argparse
import threading
import contextlib
import urllib.request
from collections import deque

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:
    _lsa = None

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

try:
    import google.colab
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    from mamba_ssm import Mamba
    import tensorflow as tf
    import tensorflow_datasets as tfds
else:
    Mamba = None
    tf = None
    tfds = None

try:
    import wandb
except ImportError:
    wandb = None

# =====================================================================
from trainer import *
from dataset import *
from models.tao_core import *

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
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--freeze", action="store_true", default=False)
    parser.add_argument("--finetune_after_epoch", type=int, default=0)
    args = parser.parse_args()

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
