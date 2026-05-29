import sys
import types
import pytest

@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session):
    """在所有测试开始前，执行全局依赖劫持"""
    try:
        import mamba_ssm
    except ImportError:
        print("\n[环境降级] 未检测到 mamba_ssm，正在向 sys.modules 注入 MockMamba...")
        
        # 兼容相对和绝对导入
        try:
            from .mock_mamba import MockMamba
        except ImportError:
            from tests.mock_mamba import MockMamba
            
        mamba_mock = types.ModuleType("mamba_ssm")
        mamba_mock.Mamba = MockMamba
        sys.modules['mamba_ssm'] = mamba_mock
        sys.modules['mamba_ssm.Mamba'] = MockMamba
