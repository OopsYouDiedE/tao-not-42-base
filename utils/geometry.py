import torch
import torch.nn.functional as F

# =====================================================================

def quaternion_to_matrix(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
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


def generate_intrinsics(
    H,
    W,
    device,
    focal_length=None,
    sensor_width=None,
    dtype=torch.float32,
):
    if focal_length is None or sensor_width is None:
        fx = fy = 35.0 / 32.0 * W
        cx, cy = W / 2.0, H / 2.0
        K = torch.tensor(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
            device=device,
            dtype=dtype,
        )
        return K, torch.inverse(K)

    focal_length = torch.as_tensor(focal_length, device=device, dtype=dtype).view(-1)
    sensor_width = torch.as_tensor(sensor_width, device=device, dtype=dtype).view(-1)

    fx = focal_length / sensor_width.clamp(min=1e-6) * float(W)
    fy = fx
    cx = torch.full_like(fx, W / 2.0)
    cy = torch.full_like(fx, H / 2.0)

    K = torch.zeros((fx.numel(), 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = fx
    K[:, 1, 1] = fy
    K[:, 0, 2] = cx
    K[:, 1, 2] = cy
    K[:, 2, 2] = 1.0

    return K, torch.linalg.inv(K)


def inverse_warp(img_next, depth, pose, K, K_inv, depth_is_distance=True):
    B, _, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(
        W, device=depth.device), indexing="ij")

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    if K_inv.dim() == 2:
        K_inv = K_inv.unsqueeze(0).expand(B, -1, -1)

    pixels = torch.stack(
        [
            x.flatten().expand(B, -1),
            y.flatten().expand(B, -1),
            torch.ones_like(x.flatten()).expand(B, -1),
        ],
        dim=1,
    ).to(depth.dtype)

    pose_rot = six_d_to_matrix(pose[:, 3:])
    pose_trans = pose[:, :3].unsqueeze(2)

    rays = torch.bmm(K_inv, pixels)

    if depth_is_distance:
        # MOVi depth 是 camera center distance。
        # 需要 distance / ||ray|| 得到沿 z=1 ray 的尺度。
        ray_norm = torch.linalg.vector_norm(rays, dim=1, keepdim=True).clamp(min=1e-6)
        z_scale = depth.view(B, 1, H * W) / ray_norm
    else:
        z_scale = depth.view(B, 1, H * W)

    points_3d = rays * z_scale

    # 变换到下一帧
    points_next = torch.bmm(pose_rot, points_3d) + pose_trans
    # 投影回 2D
    pixels_next = torch.bmm(K, points_next)

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


def compute_rigid_flow(depth, pose, K, K_inv, depth_is_distance=True):
    B, _, H, W = depth.shape
    y, x = torch.meshgrid(torch.arange(H, device=depth.device), torch.arange(
        W, device=depth.device), indexing="ij")

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, -1, -1)
    if K_inv.dim() == 2:
        K_inv = K_inv.unsqueeze(0).expand(B, -1, -1)

    pixels = torch.stack(
        [
            x.flatten().expand(B, -1),
            y.flatten().expand(B, -1),
            torch.ones_like(x.flatten()).expand(B, -1),
        ],
        dim=1,
    ).to(depth.dtype)

    pose_rot = six_d_to_matrix(pose[:, 3:])
    pose_trans = pose[:, :3].unsqueeze(2)

    rays = torch.bmm(K_inv, pixels)

    if depth_is_distance:
        ray_norm = torch.linalg.vector_norm(rays, dim=1, keepdim=True).clamp(min=1e-6)
        z_scale = depth.view(B, 1, H * W) / ray_norm
    else:
        z_scale = depth.view(B, 1, H * W)

    points_3d = rays * z_scale
    points_next = torch.bmm(pose_rot, points_3d) + pose_trans
    pixels_next = torch.bmm(K, points_next)

    depth_next = torch.clamp(pixels_next[:, 2:3, :], min=0.01).float()
    x_n = 2.0 * (pixels_next[:, 0:1, :].float() / depth_next) / (W - 1) - 1.0
    y_n = 2.0 * (pixels_next[:, 1:2, :].float() / depth_next) / (H - 1) - 1.0

    x_n_prev = 2.0 * (pixels[:, 0:1, :] / (W - 1)) - 1.0
    y_n_prev = 2.0 * (pixels[:, 1:2, :] / (H - 1)) - 1.0

    flow_n = torch.cat([x_n - x_n_prev, y_n - y_n_prev], dim=1).view(B, 2, H, W)
    return flow_n
