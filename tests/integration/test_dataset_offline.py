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
        
        # Check shapes/types
        assert isinstance(item["video"], torch.Tensor)
        assert item["video"].dim() == 4  # [T, H, W, C]
    finally:
        buffer.stop()
