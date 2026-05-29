import torch
import pytest
from utils.losses import focal_loss, giou_loss, edge_aware_smoothness_loss, compute_track_loss

def test_focal_loss():
    preds = torch.tensor([[0.0, 10.0], [-10.0, 0.0]])
    targets = torch.tensor([[0.0, 1.0], [0.0, 0.0]])
    loss = focal_loss(preds, targets)
    assert loss >= 0
    assert torch.isfinite(loss)

def test_giou_loss():
    # Format: L, T, R, B
    preds = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    loss = giou_loss(preds, targets)
    assert torch.allclose(loss, torch.tensor([0.0]), atol=1e-5)
    
    # Completely disjoint
    preds = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    targets = torch.tensor([[1.0, 1.0, 1.0, 1.0]]) + 10.0
    loss2 = giou_loss(preds, targets)
    assert loss2 > loss

def test_edge_aware_smoothness_loss():
    depth = torch.ones(1, 1, 32, 32)
    img = torch.ones(1, 3, 32, 32)
    loss = edge_aware_smoothness_loss(depth, img)
    assert torch.allclose(loss, torch.tensor(0.0))

def test_compute_track_loss():
    B, T, N = 1, 2, 32
    preds = {
        "track_boxes": torch.rand(B, T, N, 4),
        "track_alive": torch.rand(B, T, N, 1)
    }
    targets = {
        "track_gt_boxes": torch.rand(B, T, 10, 4),
        "track_gt_valid": torch.ones(B, T, 10, dtype=torch.bool)
    }
    
    loss = compute_track_loss(preds, targets, step=100)
    assert torch.isfinite(loss)
    assert loss >= 0
    
    # Empty case
    targets_empty = {
        "track_gt_boxes": torch.rand(B, T, 10, 4),
        "track_gt_valid": torch.zeros(B, T, 10, dtype=torch.bool)
    }
    loss_empty = compute_track_loss(preds, targets_empty, step=100)
    assert torch.isfinite(loss_empty)
