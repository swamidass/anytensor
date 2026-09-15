"""Backend contract tests: version floors and known behavioral patches."""

from __future__ import annotations

import numpy as np
import pytest

from anytensor import backends


def test_numpy_segment_reduce_empty_min_max_sentinel():
    b = backends.NumpyBackend()
    x = np.array([1.0, 2.0], dtype=np.float64)
    seg = np.array([0, 0], dtype=np.int64)
    # segment 1 empty → dtype max for min-reduce (JAX/TF empty-segment semantics)
    out = b.segment_reduce(x, seg, 2, "min", sorted=False)
    assert out.shape == (2,)
    assert out[0] == 1.0
    assert out[1] == np.finfo(np.float64).max


def test_torch_from_numpy_does_not_force_grad():
    torch = pytest.importorskip("torch")
    b = backends.TorchBackend()
    t = b.from_numpy(np.array([1.0, 2.0], dtype=np.float64))
    assert isinstance(t, torch.Tensor)
    assert t.requires_grad is False


def test_torch_segment_reduce_int_dtype_and_int64_ids():
    torch = pytest.importorskip("torch")
    b = backends.TorchBackend()
    x = b.from_numpy(np.array([1, 5, 3], dtype=np.int64))
    # int32 ids must be accepted (cast internally to int64 for scatter)
    seg = torch.tensor([0, 0, 1], dtype=torch.int32)
    out = b.segment_reduce(x, seg, 2, "max", sorted=False)
    assert out.dtype == torch.int64
    assert list(b.to_numpy(out)) == [5, 3]


def test_torch_segment_min_empty_uses_dtype_max_not_float_inf():
    torch = pytest.importorskip("torch")
    b = backends.TorchBackend()
    x = b.from_numpy(np.array([1, 2], dtype=np.int64))
    seg = torch.tensor([0, 0], dtype=torch.int64)
    out = b.segment_reduce(x, seg, 2, "min", sorted=False)
    # empty segment stays at integer max, not float('inf') cast junk
    assert out.dtype == torch.int64
    assert int(out[1].item()) == torch.iinfo(torch.int64).max


def test_jax_array_type_detected():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    b = backends.JaxBackend()
    x = jnp.arange(3)
    assert b.is_appropriate_type(x)
    assert backends.get_backend(x).framework_name == "jax"
    # Prefer jax.Array when available
    if hasattr(jax, "Array"):
        assert isinstance(x, jax.Array)


def test_tf_sorted_segment_arity():
    """sorted=True uses tf.math.segment_* which takes only (data, ids)."""
    tf = pytest.importorskip("tensorflow")
    b = backends.TensorflowBackend()
    x = b.from_numpy(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    # must be sorted non-decreasing segment ids
    seg = tf.constant([0, 0, 1], dtype=tf.int32)
    out = b.segment_reduce(x, seg, num_segments=2, reduction="sum", sorted=True)
    assert list(b.to_numpy(out)) == [3.0, 3.0]


def test_backend_version_floors_when_imported():
    """Installed backends below declared floors should fail clearly at init."""
    # Smoke: constructing installed backends does not raise (versions OK).
    backends.NumpyBackend()
    for name, cls in [
        ("torch", backends.TorchBackend),
        ("jax", backends.JaxBackend),
        ("tensorflow", backends.TensorflowBackend),
    ]:
        try:
            __import__(name if name != "tensorflow" else "tensorflow")
        except ImportError:
            continue
        cls()  # should enforce floor internally without cryptic later errors
