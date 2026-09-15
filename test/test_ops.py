"""Cross-backend AnyTensor tests: NumPy reference vs each loaded backend."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import anytensor as at
from anytensor import backends

loaded_backends: dict = {
    "numpy": backends.NumpyBackend(),
}

try:
    import jax  # noqa: F401

    b = backends.JaxBackend()
    loaded_backends[b.framework_name] = b
except ImportError:
    pass

try:
    import torch  # noqa: F401

    b = backends.TorchBackend()
    loaded_backends[b.framework_name] = b
except ImportError:
    pass

try:
    import tensorflow as tf  # noqa: F401

    b = backends.TensorflowBackend()
    loaded_backends[b.framework_name] = b
except ImportError:
    pass

BACKENDS = list(loaded_backends)


def close(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.allclose(x, y)


def _same_framework_type(ref, out):
    """Results keep the framework array type (0-d arrays, never bare scalars)."""
    return type(ref) is type(out)


def _run_unary(backend_name, op_name, x_np, **kwargs):
    backend = loaded_backends[backend_name]
    op = getattr(at, op_name)
    bx = backend.from_numpy(x_np)
    by = op(bx, **kwargs)
    y = op(x_np, **kwargs)
    assert close(backend.to_numpy(by), np.asarray(y))
    assert _same_framework_type(bx, by)
    # Full reductions: 0-d arrays so callers can use .shape / .dtype / methods.
    if op_name in {"sum", "min", "max", "mean", "prod"} and kwargs.get("axes") is None:
        assert by.ndim == 0
        assert y.ndim == 0
        assert type(y) is type(x_np)


def _run_binary(backend_name, op_name, a_np, b_np, **kwargs):
    backend = loaded_backends[backend_name]
    op = getattr(at, op_name)
    ba = backend.from_numpy(a_np)
    bb = backend.from_numpy(b_np)
    by = op(ba, bb, **kwargs)
    y = op(a_np, b_np, **kwargs)
    assert close(backend.to_numpy(by), np.asarray(y))
    assert _same_framework_type(ba, by)


# --- segment ops ----------------------------------------------------------

segment_ops = (
    "segment_sum segment_max segment_min segment_normalize "
    "segment_mean segment_count segment_variance"
).split()


@pytest.mark.parametrize(
    "op,ndims,backend",
    list(itertools.product(segment_ops, [1, 2, 3], BACKENDS)),
)
def test_segment_ops(backend, op, ndims):
    backend_impl = loaded_backends[backend]
    fn = getattr(at, op)

    d = [5, 3, 6][:ndims]
    numel = int(np.prod(d))
    x = np.arange(numel, dtype=np.float64).reshape(*d)
    seg_id = np.array([0, 1, 1, 0, 1])
    num_segments = 2

    bx = backend_impl.from_numpy(x)
    bseg_id = backend_impl.from_numpy(seg_id)

    if op == "segment_count":
        by = fn(bseg_id, num_segments)
        y = fn(seg_id, num_segments)
    else:
        by = fn(bx, bseg_id, num_segments)
        y = fn(x, seg_id, num_segments)

    assert close(backend_impl.to_numpy(by), y)
    if op != "segment_count":
        assert _same_framework_type(bx, by)
    assert len(x.shape) == ndims


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_softmax_and_partition_softmax(backend):
    backend_impl = loaded_backends[backend]
    logits = np.array([1.0, 2.0, 3.0, 1.0, 2.0], dtype=np.float64)
    seg_id = np.array([0, 0, 0, 1, 1])
    partitions = np.array([3, 2])

    blogits = backend_impl.from_numpy(logits)
    bseg = backend_impl.from_numpy(seg_id)
    bpart = backend_impl.from_numpy(partitions)

    s = at.segment_softmax(blogits, bseg, 2)
    p = at.partition_softmax(blogits, bpart, sum_partitions=5)
    assert close(backend_impl.to_numpy(s), at.segment_softmax(logits, seg_id, 2))
    assert close(backend_impl.to_numpy(p), at.partition_softmax(logits, partitions, 5))
    assert close(backend_impl.to_numpy(s), backend_impl.to_numpy(p))


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_min_max_or_constant_empty(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, 2.0], dtype=np.float64)
    seg_id = np.array([0, 0])
    bx = backend_impl.from_numpy(x)
    bseg = backend_impl.from_numpy(seg_id)
    # segment 1 empty -> constant
    ymin = at.segment_min_or_constant(bx, bseg, 2, constant=0.0)
    ymax = at.segment_max_or_constant(bx, bseg, 2, constant=0.0)
    assert close(backend_impl.to_numpy(ymin), np.array([1.0, 0.0]))
    assert close(backend_impl.to_numpy(ymax), np.array([2.0, 0.0]))


# --- ordinary unary / binary ops -----------------------------------------

UNITARY_OPS = "sum min max mean prod exp log cumsum sqrt rsqrt".split()


@pytest.mark.parametrize("op,backend", list(itertools.product(UNITARY_OPS, BACKENDS)))
def test_unitary_ops(backend, op):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
    _run_unary(backend, op, x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_cumsum_axis0_2d(backend):
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    _run_unary(backend, "cumsum", x)
    # Pin default axis=0 (not NumPy's flatten-on-omitted-axis behavior).
    y = at.cumsum(x)
    expected = np.array([[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]])
    assert close(y, expected)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reductions_with_axes(backend):
    x = np.arange(12.0).reshape(3, 4)
    _run_unary(backend, "sum", x, axes=0)
    _run_unary(backend, "mean", x, axes=1)


@pytest.mark.parametrize("backend", BACKENDS)
def test_elementwise_maximum_minimum(backend):
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    _run_binary(backend, "maximum", a, b)
    _run_binary(backend, "minimum", a, b)


@pytest.mark.parametrize("backend", BACKENDS)
def test_scalar_operand_upcast(backend):
    """Python scalars in binary ops promote to 0-d arrays on the array's backend."""
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, 5.0, 3.0])
    bx = backend_impl.from_numpy(x)
    by = at.maximum(bx, 2.0)
    assert type(by) is type(bx)
    assert close(backend_impl.to_numpy(by), np.maximum(x, 2.0))
    s = at.sum(x)
    assert type(s) is np.ndarray and s.ndim == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_where_clip_astype(backend):
    from array_api_compat import array_namespace

    x = np.array([4.0, 9.0, 16.0])
    backend_impl = loaded_backends[backend]
    bx = backend_impl.from_numpy(x)
    by = at.where(bx > 5, bx, 0.0)
    assert close(backend_impl.to_numpy(by), np.where(x > 5, x, 0.0))
    assert close(backend_impl.to_numpy(at.clip(bx, 5.0, 10.0)), np.clip(x, 5.0, 10.0))
    xp = array_namespace(bx)
    casted = at.astype(bx, xp.int64)
    # JAX may truncate int64→int32 unless x64 is enabled; require an integer dtype.
    assert xp.isdtype(casted.dtype, "integral")


@pytest.mark.parametrize("backend", BACKENDS)
def test_reshape_transpose_concatenate_stack(backend):
    x = np.arange(6.0).reshape(2, 3)
    backend_impl = loaded_backends[backend]
    bx = backend_impl.from_numpy(x)
    assert close(backend_impl.to_numpy(at.reshape(bx, (3, 2))), x.reshape(3, 2))
    assert close(backend_impl.to_numpy(at.transpose(bx, (1, 0))), x.T)
    assert close(
        backend_impl.to_numpy(at.concatenate([bx, bx], axis=0)),
        np.concatenate([x, x], axis=0),
    )
    assert close(backend_impl.to_numpy(at.stack([bx, bx], axis=0)), np.stack([x, x], axis=0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_like_and_arange(backend):
    backend_impl = loaded_backends[backend]
    like = backend_impl.from_numpy(np.zeros((2, 3), dtype=np.float64))
    z = at.zeros_like(like)
    o = at.ones_like(like)
    f = at.full_like(like, 3.0)
    assert close(backend_impl.to_numpy(z), np.zeros((2, 3)))
    assert close(backend_impl.to_numpy(o), np.ones((2, 3)))
    assert close(backend_impl.to_numpy(f), np.full((2, 3), 3.0))
    r = at.arange(0, 5, like=like)
    assert close(backend_impl.to_numpy(r), np.arange(0, 5))


@pytest.mark.parametrize("backend", BACKENDS)
def test_repeat_total_repeat_length(backend):
    backend_impl = loaded_backends[backend]
    idx = backend_impl.from_numpy(np.arange(2))
    lengths = backend_impl.from_numpy(np.array([2, 1]))
    out = at.repeat(idx, lengths, total_repeat_length=3)
    assert close(backend_impl.to_numpy(out), np.array([0, 0, 1]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_matmul(backend):
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    _run_binary(backend, "matmul", a, b)
