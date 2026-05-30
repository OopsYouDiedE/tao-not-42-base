import torch
import pytest
from models.tao_core import TAONot42VisionModel
from utils.geometry import generate_intrinsics

@pytest.fixture
def dummy_model():
    model = TAONot42VisionModel()
    import torch.nn as nn
    for m in model.modules():
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.zeros_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    model.eval()
    return model

@pytest.fixture
def identical_frames_batch():
    B, T, C, H, W = 1, 2, 3, 256, 256
    
    # 构造绝对相等的帧 (T=2)
    img = torch.rand(B, 1, C, H, W)
    video = img.expand(-1, T, -1, -1, -1).clone()
    
    K, K_inv = generate_intrinsics(H, W, device=video.device, focal_length=[35.0], sensor_width=[32.0])
    
    # 构造假 Batch
    batch = {
        "video": video,
        "camera_focal_length": torch.tensor([K[0, 0, 0]]),
        "camera_sensor_width": torch.tensor([32.0])
    }
    return batch, K, K_inv

def test_physics_identity(dummy_model, identical_frames_batch):
    batch, K, K_inv = identical_frames_batch
    v_seq = batch["video"]
    B, T, C, H, W = v_seq.shape
    
    c_vids = v_seq
    extracted = dummy_model.extract_features(c_vids.flatten(0, 1))
    feats = [f.view(B, T, *f.shape[1:]) for f in extracted]
    
    dt = torch.full((B, T), 1.0 / 24.0)
    
    # 我们只关心 flow_final_norm 和 ego_pose
    with torch.no_grad():
        preds = dummy_model.forward_physics(
            *feats, dt, step=10, 
            get_loss_weights_fn=lambda s: {"flow": 1.0, "anom": 1.0, "track": 1.0, "cls": 0.0},
            original_shape=(H, W),
            tgts={}, K=K, K_inv=K_inv
        )
    
    flow_pred = preds["flow"]
    ego_pose = preds["ego_pose"]
    
    # 1. 验证 Ego-Motion 平移极小
    t_pred = ego_pose["t"]
    assert torch.allclose(t_pred, torch.zeros_like(t_pred), atol=1e-2), f"Identity test failed: Ego translation {t_pred} is not 0"
    
    # 2. 验证 Ego-Motion 旋转角极小
    rot6d_pred = ego_pose["rot6d"]
    # 期望的单位旋转 6D
    identity_6d = torch.tensor([[1.0, 0, 0, 0, 1.0, 0]]).expand_as(rot6d_pred)
    assert torch.allclose(rot6d_pred, identity_6d, atol=1e-2), f"Identity test failed: Ego rot6d {rot6d_pred} is not identity"
    
    # 3. 验证端到端刚体光流严格趋近 0
    # 因为输入特征几乎完全相同（实际上是绝对相同），网络不应该预测出巨大的光流
    # 这里允许 0.1 的误差是因为网络可能有随机初始化的偏置项打破了绝对零
    flow_max_abs = flow_pred.abs().max()
    assert flow_max_abs < 0.1, f"Identity test failed: Max flow magnitude {flow_max_abs} is too large for identical frames"
