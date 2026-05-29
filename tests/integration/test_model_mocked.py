import pytest
import torch
import numpy as np
from models import TAONot42VisionModel
from utils.losses import compute_physics_loss, get_loss_weights
from dataset import process_batch_on_gpu
from trainer import TAOTrainer

class DummyTrainer:
    def __init__(self, device, global_step):
        self.device = device
        self.global_step = global_step

def test_model_forward_backward():
    # Enforce CUDA if available, but allow CPU if not specified.
    # But since RULE[AGENTS.md] bans CPU testing for the main training but this is unit integration test,
    # let's assert CUDA is available to follow project rules.
    assert torch.cuda.is_available(), "严禁在非 CUDA 环境中进行模型与训练集成测试！"
    device = torch.device("cuda")
    
    # 1. Instantiate the model
    model = TAONot42VisionModel().to(device)
    model.train()
    
    # 2. Build a minimal synthetic batch matching dataset structure
    B, T, H, W = 1, 2, 256, 256
    
    batch = {
        "video": torch.randint(0, 256, (B, T, H, W, 3), dtype=torch.uint8),
        "depth": torch.rand(B, T, H, W) * 10.0,
        "segmentation": torch.zeros(B, T, H, W, dtype=torch.int16),
        "forward_flow": torch.rand(B, T, H, W, 2),
        "cam_pos": torch.rand(B, T, 3),
        "cam_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(B, T, 1),
        "is_dynamic": [torch.tensor([1, 0])] * B,
        "category": [torch.tensor([0, 1])] * B,
        "velocities": [torch.rand(2, 3)] * B,
        "angular_velocities": [torch.rand(2, 3)] * B,
        "visibility": [torch.tensor([1.0, 1.0])] * B,
    }
    
    # Process batch
    gpu_batch = process_batch_on_gpu(batch, device, H)
    
    # Prepare forward inputs
    v_seq = gpu_batch["video"]
    img_next = torch.zeros_like(v_seq)
    for t in range(T):
        img_next[:, t] = v_seq[:, min(t + 1, T - 1)]
        
    # Forward pass
    features = [f.view(B, T, *f.shape[1:]) for f in model.extract_features(v_seq.flatten(0, 1))]
    dt = torch.full((B, T), 1.0 / 24.0, device=device)
    
    preds = model.forward_physics(
        *features, dt, step=5000, 
        get_loss_weights_fn=get_loss_weights, 
        original_shape=(H, W)
    )
    
    # Extract target chunk to flatten target dimensions matching predictions
    dummy_trainer = DummyTrainer(device, 5000)
    tgts = TAOTrainer._extract_target_chunk(dummy_trainer, gpu_batch, c_start=0, c_end=T, max_t=T)
    
    # Compute loss for step 5000 (all losses active)
    loss, loss_dict, _ = compute_physics_loss(
        preds, tgts, img_t=v_seq.flatten(0, 1), img_next=img_next.flatten(0, 1), step=5000
    )
    
    assert torch.isfinite(loss)
    
    # Backward pass to ensure gradients flow correctly
    loss.backward()
    
    # Verify that gradients exist for model parameters
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), "Gradients contain NaN or Inf values."
            break
            
    assert has_grad, "No gradients computed during backward pass."
