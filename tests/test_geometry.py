import torch
import torch.nn.functional as F
import pytest
from utils.geometry import quaternion_to_matrix, matrix_to_6d, six_d_to_matrix, generate_intrinsics, inverse_warp, compute_rigid_flow

def test_quaternion_to_matrix():
    # Identity quaternion (w, x, y, z)
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mat = quaternion_to_matrix(q)
    assert torch.allclose(mat, torch.eye(3).unsqueeze(0))
    
    # 90 degrees around x axis (w=cos(45), x=sin(45))
    val = 0.70710678
    q = torch.tensor([[val, val, 0.0, 0.0]])
    mat = quaternion_to_matrix(q)
    expected = torch.tensor([[
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0]
    ]])
    assert torch.allclose(mat, expected, atol=1e-5)

def test_six_d_to_matrix():
    # Standard basis
    d6 = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    mat = six_d_to_matrix(d6)
    assert torch.allclose(mat, torch.eye(3).unsqueeze(0))

def test_generate_intrinsics():
    device = torch.device("cpu")
    K, K_inv = generate_intrinsics(256, 256, device)
    assert K.shape == (3, 3)
    assert K_inv.shape == (3, 3)
    assert torch.allclose(torch.matmul(K, K_inv), torch.eye(3))
    
    K2, K_inv2 = generate_intrinsics(256, 256, device, focal_length=0.7, sensor_width=36.0)
    assert K2.shape == (1, 3, 3)
    assert torch.allclose(torch.matmul(K2[0], K_inv2[0]), torch.eye(3))

def test_inverse_warp_and_rigid_flow():
    device = torch.device("cpu")
    B, C, H, W = 1, 1, 64, 64
    depth = torch.ones(B, 1, H, W, device=device) * 10.0
    
    pose_rot6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=device)
    pose_trans = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    pose = torch.cat([pose_trans, pose_rot6d], dim=-1)
    
    K, K_inv = generate_intrinsics(H, W, device)
    
    img_next = torch.rand(B, 3, H, W, device=device)
    
    warped, valid_mask = inverse_warp(img_next, depth, pose, K, K_inv)
    
    assert warped.shape == (B, 3, H, W)
    assert valid_mask.shape == (B, 1, H, W)
    
    flow = compute_rigid_flow(depth, pose, K, K_inv)
    assert flow.shape == (B, 2, H, W)
