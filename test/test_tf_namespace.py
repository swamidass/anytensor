"""TensorFlow namespace shim coverage (AAC has no TF backend yet)."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at
from anytensor.namespace import array_namespace, _tensorflow_namespace
from helpers import close, loaded_backends

pytestmark = pytest.mark.skipif(
    "tensorflow" not in loaded_backends,
    reason="TensorFlow not installed",
)


@pytest.fixture
def tf_x():
    b = loaded_backends["tensorflow"]
    return b.from_numpy(np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32))


def test_tf_namespace_helpers(tf_x):
    xp = array_namespace(tf_x)
    assert xp is _tensorflow_namespace()
    assert xp.bool is loaded_backends["tensorflow"].tf.bool
    assert close(np.asarray(xp.asarray([1, 2], dtype=xp.float32, copy=True)), np.array([1, 2], dtype=np.float32))
    y = xp.astype(tf_x, xp.float64, copy=False)
    assert y.dtype == loaded_backends["tensorflow"].tf.float64
    flat = xp.reshape(tf_x[:2], (2, 1))
    assert tuple(xp.permute_dims(flat).shape) == (1, 2)
    cs = xp.cumulative_sum(tf_x[:2], dtype=xp.float64)
    assert cs.dtype == loaded_backends["tensorflow"].tf.float64
    with pytest.raises(NotImplementedError):
        xp.cumulative_sum(tf_x[:2], include_initial=True)
    assert xp.isdtype(xp.float32, ("floating", "real floating"))
    assert xp.isdtype(xp.float32, "numeric")
    assert not xp.isdtype(xp.float32, "complex floating")
    with pytest.raises(ValueError):
        xp.isdtype(xp.float32, "nope")
    out = np.asarray(at.nan_to_num(tf_x))
    assert out[0] == 1.0 and out[1] == 0.0 and np.isfinite(out[2]) and np.isfinite(out[3])


def test_tf_namespace_rejects_mixed_frameworks(tf_x):
    torch = pytest.importorskip("torch")
    with pytest.raises(TypeError, match="Cannot mix TensorFlow"):
        array_namespace(tf_x, torch.tensor([1.0]))


def test_tf_array_namespace_scalars_only():
    xp = array_namespace(1.0, 2)
    import array_api_compat.numpy as npx

    assert xp is npx or xp.__name__.endswith("numpy")
