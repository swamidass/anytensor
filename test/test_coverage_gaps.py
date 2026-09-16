"""Targeted coverage for public API branches (non-fuzz gate)."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at
from anytensor.semantics import empty_segment_identity
from helpers import close


def test_constructors_default_to_numpy():
    z = at.zeros((2, 3), dtype=np.float32)
    o = at.ones((2,), dtype=np.float32)
    f = at.full((2,), 7.0, dtype=np.float32)
    a = at.arange(4, dtype=np.int64)
    assert z.shape == (2, 3) and float(z[0, 0]) == 0.0
    assert list(np.asarray(o)) == [1.0, 1.0]
    assert list(np.asarray(f)) == [7.0, 7.0]
    assert list(np.asarray(a)) == [0, 1, 2, 3]


def test_arange_like_dtype_and_device_kwarg():
    like = np.array([0.0], dtype=np.float32)
    out = at.arange(0, 3, dtype=np.int32, like=like)
    assert list(np.asarray(out)) == [0, 1, 2]
    # device= on NumPy namespace is ignored via TypeError fallback
    out2 = at.arange(2, like=like, device="cpu")
    assert list(np.asarray(out2)) == [0, 1]


def test_transpose_default_axes_and_scalar_namespace():
    x = np.arange(6.0).reshape(2, 3)
    assert at.transpose(x).shape == (3, 2)
    # scalars-only path in _xp
    assert float(at.maximum(1.5, 2.5)) == 2.5


def test_promote_guards_and_mask_cast():
    with pytest.raises(TypeError):
        at.promote()(lambda x: x)
    with pytest.raises(TypeError):
        at.promote_scalars()(lambda x: x)
    with pytest.raises(TypeError):
        at.take(np.array([1.0, 2.0]), np.array([0.5, 1.5]))
    # non-bool mask is cast
    out = at.where(np.array([0, 1]), np.array([10.0, 20.0]), np.array([1.0, 2.0]))
    assert list(np.asarray(out)) == [1.0, 20.0]


def test_repeat_total_repeat_length():
    x = np.array([1.0, 2.0, 3.0])
    out = at.repeat(x, np.array([2, 0, 1]), total_repeat_length=3)
    assert close(np.asarray(out), np.array([1.0, 1.0, 3.0]))
    empty = at.repeat(x, np.array([0, 0, 0]), total_repeat_length=0)
    assert np.asarray(empty).shape == (0,)
    truncated = at.repeat(x, np.array([2, 2, 0]), total_repeat_length=3)
    assert close(np.asarray(truncated), np.array([1.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        at.repeat(x, np.array([1, 0, 0]), total_repeat_length=5)
    with pytest.raises(NotImplementedError):
        at.repeat(x, 2, total_repeat_length=6, axis=0)
    # plain repeat with axis
    assert at.repeat(np.array([[1.0], [2.0]]), 2, axis=0).shape[0] == 4


def test_iinfo_and_align_none():
    x = np.array([1, 2], dtype=np.int32)
    assert at.iinfo(x).bits == 32
    a, b = at.align_arrays(None, np.array([1.0]))
    assert a is None


def test_segment_softmax_infers_num_segments_and_2d_or_constant():
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    seg = np.array([0, 0, 1], dtype=np.int64)
    out = at.segment_softmax(logits, seg)  # num_segments inferred
    assert out.shape == (3,)
    assert close(float(np.sum(out[:2])), 1.0)
    x2 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    seg2 = np.array([0, 1], dtype=np.int64)
    m = at.segment_min_or_constant(x2, seg2, 3, constant=-9.0)
    assert m.shape == (3, 2)
    assert close(np.asarray(m[2]), np.array([-9.0, -9.0]))


def test_ones_full_like_inherit_dtype():
    like = np.array([1.0, 2.0], dtype=np.float32)
    assert at.ones((2,), like=like).dtype == like.dtype
    assert at.full((2,), 3.0, like=like).dtype == like.dtype


def test_repeat_without_axis():
    out = at.repeat(np.array([1.0, 2.0]), 2)
    assert close(np.asarray(out), np.array([1.0, 1.0, 2.0, 2.0]))


def test_promote_scalars_success_and_role_edges():
    @at.promote_scalars("x", "y")
    def add(x, y):
        return x + y

    assert float(add(np.array(1.0), np.array(2.0))) == 3.0

    @at.promote(indices="index")
    def only_index(indices):
        return indices

    assert list(np.asarray(only_index(np.array([0, 1])))) == [0, 1]

    @at.promote(x="data", ghost="mask")
    def with_ghost_role(x):
        return x

    assert float(with_ghost_role(np.array(1.0))) == 1.0


def test_promote_result_type_typeerror(monkeypatch):
    import array_api_compat.numpy as xp

    def boom(*_a, **_k):
        raise TypeError("no result_type")

    monkeypatch.setattr(xp, "result_type", boom)
    # Still runs; dtype promotion is skipped when result_type fails.
    out = at.maximum(np.array([1], dtype=np.int32), np.array([2], dtype=np.int32))
    assert int(np.asarray(out)[0]) == 2


def test_arange_device_typeerror_fallback(monkeypatch):
    import array_api_compat.numpy as xp

    real = xp.arange

    def flaky(*args, **kwargs):
        if "device" in kwargs:
            raise TypeError("device not supported")
        return real(*args, **kwargs)

    monkeypatch.setattr(xp, "arange", flaky)
    out = at.arange(3, like=np.array([0.0]), device="cpu")
    assert list(np.asarray(out)) == [0, 1, 2]


def test_partition_softmax_and_semantics_edges():
    logits = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    parts = np.array([2, 1], dtype=np.int64)
    out = at.partition_softmax(logits, parts)
    assert out.shape == (3,)
    assert close(float(np.sum(out[:2])), 1.0)
    out2 = at.partition_softmax(logits, parts, sum_partitions=3)
    assert out2.shape == (3,)

    with pytest.raises(ValueError):
        empty_segment_identity(np.float32, "mean", xp=np)

    class KindF:
        kind = "f"

    assert empty_segment_identity(KindF(), "min", xp=np) == np.inf
    assert empty_segment_identity(KindF(), "max", xp=np) == -np.inf
