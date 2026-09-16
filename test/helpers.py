"""Shared helpers for AnyTensor tests (importable under pytest importlib mode)."""

from __future__ import annotations

import numpy as np

from anytensor import backends

loaded_backends: dict = {
    "numpy": backends.NumpyBackend(),
}

try:
    import jax  # noqa: F401

    _b = backends.JaxBackend()
    loaded_backends[_b.framework_name] = _b
except ImportError:
    pass

try:
    import torch  # noqa: F401

    _b = backends.TorchBackend()
    loaded_backends[_b.framework_name] = _b
except ImportError:
    pass

try:
    import tensorflow as tf  # noqa: F401

    _b = backends.TensorflowBackend()
    loaded_backends[_b.framework_name] = _b
except ImportError:
    pass

BACKENDS = list(loaded_backends)


def close(x, y, *, equal_nan: bool = False):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.allclose(x, y, equal_nan=equal_nan)
