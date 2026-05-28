import sys
import types
import torch

class MockMamba(torch.nn.Module):
    def __init__(self, d_model, *args, **kwargs):
        super().__init__()
        self.proj = torch.nn.Linear(d_model, d_model)
    def forward(self, x, *args, **kwargs):
        return self.proj(x)

def inject_mock_mamba():
    """Inject MockMamba into sys.modules and custom_heads to bypass missing mamba_ssm."""
    mamba_mock = types.ModuleType("mamba_ssm")
    mamba_mock.Mamba = MockMamba
    sys.modules["mamba_ssm"] = mamba_mock

    import models.custom_heads
    models.custom_heads.Mamba = MockMamba
