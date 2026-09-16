"""Targeted coverage for public API branches (non-fuzz gate)."""

from __future__ import annotations

import sys

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


def test_segment_softmax_requires_num_segments_and_2d_or_constant():
    logits = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    seg = np.array([0, 0, 1], dtype=np.int64)
    out = at.segment_softmax(logits, seg, 2)
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


def test_normalize_shape_dim_and_promote_shape_roles():
    from anytensor.backends import UnknownSize
    from anytensor.core import _asarray, _normalize_shape_dim, _xp

    assert _normalize_shape_dim(None) is None
    u = UnknownSize()
    assert _normalize_shape_dim(u) is u
    assert _normalize_shape_dim(np.int64(4)) == 4
    assert _normalize_shape_dim(3.0) == 3
    with pytest.raises(TypeError, match="bool"):
        _normalize_shape_dim(True)
    with pytest.raises(TypeError, match="bool"):
        _normalize_shape_dim(np.bool_(True))
    with pytest.raises(TypeError, match="integral"):
        _normalize_shape_dim(1.5)
    assert _normalize_shape_dim(np.array(7)) == 7

    xp = _xp(np.array(1.0))
    assert _asarray(xp, None) is None

    @at.promote(n="shape")
    def take_n(n):
        return n

    assert take_n(None) is None
    assert take_n(3) == 3
    with pytest.raises(TypeError, match="scalar size"):
        take_n(np.array([1, 2]))
    with pytest.raises(TypeError, match="integral"):
        take_n(np.array(1.5))


def test_mean_empty_and_repeat_guards():
    from anytensor.core import (
        _host_concrete_int,
        _leading_dim_is_concrete,
        _pad_or_slice_leading,
        _repeats_are_host_concrete,
    )

    empty = at.mean(np.array([], dtype=np.float64))
    assert np.isnan(float(np.asarray(empty)))

    with pytest.raises(TypeError, match="integral"):
        at.repeat(np.array([1.0]), np.array([1.5]))

    assert _leading_dim_is_concrete(3) is True
    assert _leading_dim_is_concrete(object()) is False
    assert _host_concrete_int(3) == 3
    assert _host_concrete_int(object()) is None
    assert _repeats_are_host_concrete(2) is True
    assert _repeats_are_host_concrete(np.array([1, 2])) is True

    x = np.array([1.0, 2.0, 3.0])
    assert close(_pad_or_slice_leading(np, x, 5), np.array([1.0, 2.0, 3.0, 0.0, 0.0]))
    assert close(_pad_or_slice_leading(np, x, 2), np.array([1.0, 2.0]))

    torch = pytest.importorskip("torch")
    from anytensor.namespace import array_namespace

    tx = torch.tensor([1.0, 2.0, 3.0])
    xp = array_namespace(tx)
    padded = _pad_or_slice_leading(xp, tx, 5)
    assert list(padded.detach().cpu().numpy()) == [1.0, 2.0, 3.0, 0.0, 0.0]
    sliced = _pad_or_slice_leading(xp, tx, 2)
    assert list(sliced.detach().cpu().numpy()) == [1.0, 2.0]

    tf = pytest.importorskip("tensorflow")
    fx = tf.constant([1.0, 2.0, 3.0])
    xp = array_namespace(fx)
    tp = _pad_or_slice_leading(xp, fx, 5)
    assert list(np.asarray(tp)) == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_tensorflow_namespace_helpers():
    tf = pytest.importorskip("tensorflow")
    from anytensor.namespace import array_namespace

    ns = array_namespace(tf.constant([1.0, 2.0]))
    assert list(np.asarray(ns.repeat(tf.constant([1.0, 2.0]), 2))) == [1.0, 1.0, 2.0, 2.0]
    assert list(np.asarray(ns.repeat(tf.constant([[1.0], [2.0]]), 2, axis=0))).count(1.0) == 2
    assert list(np.asarray(ns.arange(3))) == [0, 1, 2]
    assert list(np.asarray(ns.arange(0, 3, 1, dtype=tf.int32))) == [0, 1, 2]
    assert list(np.asarray(ns.full((2,), 7.0))) == [7.0, 7.0]
    assert list(np.asarray(ns.full((2,), 7.0, dtype=tf.float32))) == [7.0, 7.0]


def test_enable_typecheck_installs_hook():
    # Idempotent: may already be installed by pytest conftest.
    at.enable_typecheck()


def test_enable_torchscript_returns_false_without_torch():
    """``enable_torchscript`` is a no-op until ``torch`` is imported."""
    from anytensor import segment

    was = segment._TORCHSCRIPT_ENABLED
    segment._TORCHSCRIPT_ENABLED = False
    torch_mod = sys.modules.pop("torch", None)
    torch_subs = {
        k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("torch.")
    }
    try:
        assert segment.enable_torchscript() is False
    finally:
        if torch_mod is not None:
            sys.modules["torch"] = torch_mod
        sys.modules.update(torch_subs)
        segment._TORCHSCRIPT_ENABLED = was
        if was:
            assert segment.enable_torchscript() is True


def test_repeat_host_concrete_and_pad_fallback(monkeypatch):
    from anytensor import core as core_mod
    from anytensor.core import _repeats_are_host_concrete

    class FakeRep:
        shape = (object(),)

        def __getitem__(self, i):
            return 1

    assert _repeats_are_host_concrete(FakeRep()) is False

    class FakeRepItems:
        shape = (2,)

        def __getitem__(self, i):
            return object()

    assert _repeats_are_host_concrete(FakeRepItems()) is False

    monkeypatch.setattr(core_mod, "_repeats_are_host_concrete", lambda _r: False)
    out = at.repeat(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]), total_repeat_length=3)
    assert close(np.asarray(out), np.array([1.0, 2.0, 3.0]))

    @at.promote(n="shape")
    def take_n(n):
        return n

    assert int(take_n(np.array(5, dtype=np.int64))) == 5
