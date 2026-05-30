import torch
import pytest
import math
from models.custom_heads import ObjectRigidFlowProjector, RigidFlowProjector
from utils.geometry import generate_intrinsics
from models.custom_heads import make_4x4_transform, rot6d_to_matrix

def test_rigid_flow_projection_analytical():
    B, T, h, w = 1, 1, 256, 256
    device = "cpu"
    
    K, K_inv = generate_intrinsics(h, w, device=device, focal_length=[35.0], sensor_width=[32.0])
    fx = K[0, 0, 0].item()
    
    # 构造一个 10 米远的纯平墙壁
    depth_val = 10.0
    inv_depth = torch.full((B*T, 1, h, w), 1.0 / depth_val, device=device)
    
    # 构造相机的移动：X 轴平移 0.1 米
    tx = 0.1
    t = torch.tensor([[tx, 0.0, 0.0]], device=device)
    # 纯平移无旋转
    R = torch.eye(3, device=device).unsqueeze(0)
    T_cam = make_4x4_transform(R, t)  # [B*T, 4, 4]
    
    projector = RigidFlowProjector()
    flow_rigid = projector(inv_depth, T_cam, K, K_inv)
    
    # 理论计算：
    # 如果相机向右（+X）平移了 tx
    # 对于世界坐标中的点，相当于物体向左（-X）移动了 tx
    # 像素位移 dx = fx * (-tx) / Z
    dx_expected = fx * (-tx) / depth_val
    dy_expected = 0.0
    
    flow_x_pred = flow_rigid[0, 0, h//2, w//2].item()
    flow_y_pred = flow_rigid[0, 1, h//2, w//2].item()
    
    assert math.isclose(flow_x_pred, dx_expected, abs_tol=1e-3), \
        f"Flow X mismatch: Expected {dx_expected}, got {flow_x_pred}"
    assert math.isclose(flow_y_pred, dy_expected, abs_tol=1e-3), \
        f"Flow Y mismatch: Expected {dy_expected}, got {flow_y_pred}"

def test_object_rigid_flow_projector_analytical():
    B, T, h, w = 1, 1, 256, 256
    device = "cpu"
    
    K, K_inv = generate_intrinsics(h, w, device=device, focal_length=[35.0], sensor_width=[32.0])
    fx = K[0, 0, 0].item()
    
    # 构造 10 米远墙壁
    depth_val = 10.0
    inv_depth_resized = torch.full((B*T, 1, h, w), 1.0 / depth_val, device=device)
    
    # 构造背景 flow 为 0
    flow_rigid = torch.zeros((B*T, 2, h, w), device=device)
    
    # 构造对象的移动：物体 X 轴平移 0.1 米 (v=[0.1, 0, 0])
    v_x = 0.1
    dense_twist = torch.zeros((B*T, 6, h, w), device=device)
    dense_twist[:, 0, :, :] = v_x
    
    # 假设整个屏幕都是对象
    dense_obj_mask = torch.ones((B*T, 1, h, w), device=device)
    residual_flow = torch.zeros((B*T, 2, h, w), device=device)
    
    projector = ObjectRigidFlowProjector()
    flow_final = projector(inv_depth_resized, dense_twist, dense_obj_mask, residual_flow, flow_rigid, K, K_inv)
    
    # 理论计算：
    # 物体坐标 dX = v + w x X1
    # 因为 v=[0.1, 0, 0]，w=0，所以 X2_obj = X1 + [0.1, 0, 0]^T
    # 相机的 X1 坐标的 X 为 (u - cx) / fx * Z
    # 在屏幕中心 u=cx，所以 X1_X = 0。X2_X = 0.1
    # 投影回来 u2 = fx * X2_X / Z + cx = fx * 0.1 / 10.0 + cx
    # flow_x = u2 - cx = fx * 0.1 / 10.0
    dx_expected = fx * v_x / depth_val
    dy_expected = 0.0
    
    flow_x_pred = flow_final[0, 0, h//2, w//2].item()
    flow_y_pred = flow_final[0, 1, h//2, w//2].item()
    
    assert math.isclose(flow_x_pred, dx_expected, abs_tol=1e-3), \
        f"Object Flow X mismatch: Expected {dx_expected}, got {flow_x_pred}"
    assert math.isclose(flow_y_pred, dy_expected, abs_tol=1e-3), \
        f"Object Flow Y mismatch: Expected {dy_expected}, got {flow_y_pred}"

