import os
import time
import pytest
import torch
from dataset import AsyncDataBuffer

def test_async_data_buffer_offline():
    # Use the static sample copied into tests/data
    offline_path = os.path.join(os.path.dirname(__file__), "..", "data", "movi_e_static_sample.npz")
    assert os.path.exists(offline_path), f"Mock data not found at: {offline_path}"
    
    # Initialize buffer
    buffer = AsyncDataBuffer(split="test", max_buffer_size=5, batch_size=1, offline_path=offline_path)
    
    # Wait for the thread to load at least one item
    start_time = time.time()
    while len(buffer.buffer) == 0 and time.time() - start_time < 10.0:
        time.sleep(0.1)
        
    try:
        assert len(buffer.buffer) > 0, "Buffer failed to load offline data."
        item = buffer.buffer[0]
        
        # Verify keys
        assert "video" in item
        assert "depth" in item
        assert "cam_pos" in item
        assert "cam_quat" in item
        assert "forward_flow" in item
        
        # Check shapes, dimensions, and types (Hard contracts)
        assert isinstance(item["video"], torch.Tensor)
        assert item["video"].dim() == 4  # [T, H, W, C]
        assert item["video"].dtype in (torch.uint8, torch.float32)

        assert isinstance(item["depth"], torch.Tensor)
        assert item["depth"].dim() == 3  # [T, H, W]
        assert item["depth"].dtype == torch.float32

        assert isinstance(item["cam_pos"], torch.Tensor)
        assert item["cam_pos"].dim() == 2  # [T, 3]
        assert item["cam_pos"].dtype == torch.float32

        assert isinstance(item["cam_quat"], torch.Tensor)
        assert item["cam_quat"].dim() == 2  # [T, 4]
        assert item["cam_quat"].dtype == torch.float32

        assert isinstance(item["forward_flow"], torch.Tensor)
        assert item["forward_flow"].dim() == 4  # [T, H, W, 2]
        assert item["forward_flow"].dtype == torch.float32

        # 物理合理区间校验
        # 1. 逆深度处于 [0.01, 100.0]m 的合理物理区间
        assert torch.all(item["depth"] >= 0.01), f"Depth below 0.01m: min={item['depth'].min().item()}"
        assert torch.all(item["depth"] <= 100.0), f"Depth above 100.0m: max={item['depth'].max().item()}"

        # 2. 四元数校验（L2 范数在误差范围内等于 1.0）
        quat_norms = torch.linalg.vector_norm(item["cam_quat"], dim=1)
        assert torch.allclose(quat_norms, torch.ones_like(quat_norms), atol=1e-5), f"Quaternion not normalized: norms={quat_norms}"

    finally:
        buffer.stop()
