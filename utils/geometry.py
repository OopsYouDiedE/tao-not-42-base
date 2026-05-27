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

def quaternion_to_matrix(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, zw, xz, yw, yz, xw = x * y, z * w, x * z, y * w, y * z, x * w
    return torch.stack([
        w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw),
        2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw),
        2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2,
    ], dim=-1).view(*q.shape[:-1], 3, 3)


def matrix_to_6d(matrix):
    return matrix[..., :2].reshape(*matrix.shape[:-2], 6)


def six_d_to_matrix(d6):
    x_raw, y_raw = d6[..., 0:3], d6[..., 3:6]
    x = F.normalize(x_raw, dim=-1)
    y = F.normalize(y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x, dim=-1)
    return torch.stack([x, y, torch.cross(x, y, dim=-1)], dim=-1)


def generate_intrinsics(H, W, device):
    fx = fy = 35.0 / 32.0 * W
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
                     device=device, dtype=torch.float32)
    return K, torch.inverse(K)


def inverse_warp(img_next, depth, pose, K, K_inv):
    B, _, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(
        W, device=depth.device), indexing="ij")

    pixels = torch.stack([x.flatten().expand(
        B, -1), y.flatten().expand(B, -1), torch.ones_like(x.flatten().expand(B, -1))], dim=1)

    pose_rot = six_d_to_matrix(pose[:, 3:])
    pose_trans = pose[:, :3].unsqueeze(2)

    # 3D points
    points_3d = torch.bmm(K_inv.expand(B, 3, 3),
                          pixels.float()) * depth.view(B, 1, H * W)
    # Transform to next frame
    points_next = torch.bmm(pose_rot, points_3d) + pose_trans
    # Project back to 2D
    pixels_next = torch.bmm(K.expand(B, 3, 3), points_next)

    depth_next = torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()
    x_n = 2.0 * (pixels_next[:, 0:1, :].float() / depth_next) / (W - 1) - 1.0
    y_n = 2.0 * (pixels_next[:, 1:2, :].float() / depth_next) / (H - 1) - 1.0

    grid = torch.cat([x_n, y_n], dim=1).view(B, 2, H, W).permute(0, 2, 3, 1)
    grid = torch.clamp(grid, -2.0, 2.0)

    warped = F.grid_sample(img_next, grid, mode="bilinear",
                           padding_mode="border", align_corners=True)
    warped = torch.nan_to_num(warped, 0.0)

    valid_mask = ((x_n > -1.0) & (x_n < 1.0) & (y_n > -1.0)
                  & (y_n < 1.0)).view(B, 1, H, W).float()
    depth_mask = ((depth > 0.01) & (
        pixels_next[:, 2:3, :].view(B, 1, H, W) > 0.01)).float()

    return warped, valid_mask * depth_mask


