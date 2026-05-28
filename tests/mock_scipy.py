import sys
import types
import numpy as np

def inject_mock_scipy():
    """Inject mock scipy.optimize.linear_sum_assignment if scipy is missing, prior to importing modules."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:
        scipy_mock = types.ModuleType("scipy")
        scipy_opt_mock = types.ModuleType("scipy.optimize")
        scipy_opt_mock.linear_sum_assignment = lambda cost: (
            np.arange(min(cost.shape)), np.arange(min(cost.shape)))
        scipy_mock.optimize = scipy_opt_mock
        sys.modules["scipy"] = scipy_mock
        sys.modules["scipy.optimize"] = scipy_opt_mock
