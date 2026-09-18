"""Targeted coverage for public API branches (non-fuzz gate)."""

from __future__ import annotations

import gc
import sys
import weakref

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
    out = at.partition_softmax(logits, parts, 3)
    assert out.shape == (3,)
    assert close(float(np.sum(out[:2])), 1.0)
    ids = at.partition_ids(parts, 3)
    assert list(np.asarray(ids)) == [0, 0, 1]
    out2 = at.segment_softmax(logits, ids, 2)
    assert close(np.asarray(out2), np.asarray(out))
    out3 = at.partition_softmax(logits, parts, total_length=3)
    assert out3.shape == (3,)
    with pytest.raises(TypeError):
        at.partition_softmax(logits, parts)
    with pytest.raises(TypeError, match="total_length"):
        at.partition_softmax(logits, parts, None)
    with pytest.raises(TypeError, match="total_length"):
        at.partition_ids(parts, None)
    with pytest.raises(TypeError, match="num_segments"):
        at.segment_sum(logits, np.array([0, 0, 1]), None)

    with at.cache():
        ids_a = at.partition_ids(parts, 3)
        ids_b = at.partition_ids(parts, 3)
        assert ids_a is ids_b
        with at.cache():
            assert at.partition_ids(parts, 3) is ids_a
        nsum = np.int64(3)
        ids_t = at.partition_ids(parts, nsum)
        assert at.partition_ids(parts, nsum) is ids_t
        nsum_o = np.array(3)
        ids_o = at.partition_ids(parts, nsum_o)
        assert at.partition_ids(parts, nsum_o) is ids_o
    ids_c = at.partition_ids(parts, 3)
    assert ids_c is not ids_a

    with pytest.raises(ValueError):
        empty_segment_identity(np.float32, "mean", xp=np)

    class KindF:
        kind = "f"

    assert empty_segment_identity(KindF(), "min", xp=np) == np.inf
    assert empty_segment_identity(KindF(), "max", xp=np) == -np.inf


def test_cache_weakrefs_and_partition_softmax(monkeypatch):
    from anytensor import segment

    logits = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    parts = np.array([2, 1], dtype=np.int64)
    repeats = {"n": 0}
    real_repeat = segment.repeat

    def counting_repeat(*args, **kwargs):
        repeats["n"] += 1
        return real_repeat(*args, **kwargs)

    monkeypatch.setattr(segment, "repeat", counting_repeat)
    at.partition_softmax(logits, parts, 3)
    at.partition_softmax(logits, parts, 3)
    assert repeats["n"] == 2
    with at.cache():
        at.partition_softmax(logits, parts, 3)
        at.partition_softmax(logits, parts, 3)
        assert at.partition_ids(parts, 3) is not None
        ids = at.partition_ids(parts, 3)
        at.partition_softmax(logits, parts, 3)
        assert at.partition_ids(parts, 3) is ids
    assert repeats["n"] == 3

    with at.cache():
        wr, ids_live = _partition_ids_then_drop()
        ids_wr = weakref.ref(ids_live)
        gc.collect()
        assert wr() is None
        assert at.cache["partition"] == {}
        del ids_live
        gc.collect()
        assert ids_wr() is None
        other = np.array([2, 1], dtype=np.int64)
        ids_other = at.partition_ids(other, 3)
        assert list(np.asarray(ids_other)) == [0, 0, 1]

        gone = _Gone()
        dead = weakref.ref(gone)
        del gone
        gc.collect()
        assert dead() is None
        ns = at.cache["partition"]
        key = (id(other), ("i", 3))
        ns[key] = (dead, ids_other)
        ids_fresh = at.partition_ids(other, 3)
        assert ids_fresh is not ids_other

        stale = np.array([1, 2], dtype=np.int64)
        ns[key] = (weakref.ref(stale), ids_fresh)
        ids_ok = at.partition_ids(other, 3)
        assert ids_ok is not ids_fresh
        assert list(np.asarray(ids_ok)) == [0, 0, 1]


def test_cache_callback_does_not_pin_cache():
    """GC callbacks must not keep the cache root or namespace alive after the block exits."""
    from anytensor import _cache

    parts = np.array([2, 1], dtype=np.int64)
    with at.cache():
        ids = at.partition_ids(parts, 3)
        root_wr = weakref.ref(_cache._CACHE.get())
        ns_wr = weakref.ref(at.cache["partition"])
    gc.collect()
    assert root_wr() is None
    assert ns_wr() is None
    assert list(np.asarray(ids)) == [0, 0, 1]
    wr = weakref.ref(parts)
    del parts, ids
    gc.collect()
    assert wr() is None


def test_purge_cache_entry_noop_when_cache_is_dead():
    from anytensor import _cache

    ns = _cache._WeakMap()
    key = (id(ns), ("i", 1), ("i", 1))
    ns[key] = "held"
    ns_ref = weakref.ref(ns)
    _cache._purge_cache_entry(ns_ref, key)
    assert key not in ns
    del ns
    gc.collect()
    assert ns_ref() is None
    _cache._purge_cache_entry(ns_ref, key)


class _Gone:
    pass


def _partition_ids_then_drop():
    live = np.array([2, 1], dtype=np.int64)
    wr = weakref.ref(live)
    ids_live = at.partition_ids(live, 3)
    return wr, ids_live


def test_cache_strongref_when_weakref_fails(monkeypatch):
    from anytensor import _cache

    real_ref = _cache.weakref.ref

    def selective(obj, callback=None):
        if isinstance(obj, np.ndarray):
            raise TypeError("cannot create weak reference")
        return real_ref(obj, callback)

    monkeypatch.setattr(_cache.weakref, "ref", selective)
    parts = np.array([2, 1], dtype=np.int64)
    with at.cache():
        ids_a = at.partition_ids(parts, 3)
        ids_b = at.partition_ids(parts, 3)
        assert ids_a is ids_b
        assert list(np.asarray(ids_a)) == [0, 0, 1]


def test_cache_decorator_enable_disable_purge():
    parts = np.array([2, 1], dtype=np.int64)
    other = np.array([1, 2], dtype=np.int64)

    @at.cache
    def twice(p):
        a = at.partition_ids(p, 3)
        b = at.partition_ids(p, 3)
        return a, b

    a, b = twice(parts)
    assert a is b
    assert at.partition_ids(parts, 3) is not a

    @at.cache()
    def twice_paren(p):
        a = at.partition_ids(p, 3)
        return a, at.partition_ids(p, 3)

    c, d = twice_paren(parts)
    assert c is d

    at.cache.enable()
    at.cache.enable()
    try:
        e = at.partition_ids(parts, 3)
        assert at.partition_ids(parts, 3) is e
        with at.cache():
            assert at.partition_ids(parts, 3) is e
        assert at.partition_ids(parts, 3) is e
        f = at.partition_ids(other, 3)
        at.cache.purge("partition", parts)
        assert at.partition_ids(parts, 3) is not e
        assert at.partition_ids(other, 3) is f
        at.cache.purge_cache("partition", other)
        assert at.partition_ids(other, 3) is not f
    finally:
        at.cache.disable()
    g = at.partition_ids(parts, 3)
    assert g is not e
    at.cache.disable()
    at.cache.purge("partition", parts)

    with at.cache():
        h = at.partition_ids(parts, 3)
        at.cache.disable()
        assert at.partition_ids(parts, 3) is not h

    with at.cache():
        held = at.partition_ids(parts, 3)
        at.cache.enable()
    try:
        assert at.partition_ids(parts, 3) is held
    finally:
        at.cache.disable()


def test_cache_is_dict_of_dicts():
    parts = np.array([2, 1], dtype=np.int64)
    with pytest.raises(KeyError):
        at.cache["partition"]
    assert "partition" not in at.cache
    with at.cache():
        assert "partition" in at.cache
        assert dict(at.cache["partition"]) == {}
        ids = at.partition_ids(parts, 3)
        ns = at.cache["partition"]
        assert len(ns) == 1
        _, stored = next(iter(ns.values()))
        assert stored is ids
        at.cache["other"]["k"] = "v"
        assert "other" in at.cache
        assert at.cache["other"]["k"] == "v"
        at.cache.purge("other", parts)
        assert at.cache["other"]["k"] == "v"
        at.cache["other"][(id(parts),)] = "x"
        at.cache.purge("other", parts)
        assert at.cache["other"]["k"] == "v"
        assert (id(parts),) not in at.cache["other"]
        at.cache.purge("partition", parts)
        assert dict(at.cache["partition"]) == {}
        assert "missing" not in at.cache
        at.cache.purge("missing", parts)
        at.cache.purge("partition", parts)
    assert "partition" not in at.cache


def test_cache_purges_wrong_size_ids(monkeypatch):
    from anytensor import segment

    parts = np.array([2, 1], dtype=np.int64)
    with at.cache():
        at.partition_ids(parts, 3)
        ns = at.cache["partition"]
        key = (id(parts), ("i", 3))
        wrong = np.array([0, 0], dtype=np.int64)
        ns[key] = (weakref.ref(parts), wrong)
        with pytest.warns(UserWarning, match="length 2 != total_length 3"):
            fresh = at.partition_ids(parts, 3)
        assert list(np.asarray(fresh)) == [0, 0, 1]
        assert fresh is not wrong
        assert at.partition_ids(parts, 3) is fresh

        nsum = np.array(3)
        held = at.partition_ids(parts, nsum)
        nsum.fill(2)
        with pytest.warns(UserWarning, match="length 3 != total_length 2"):
            resized = at.partition_ids(parts, nsum)
        assert resized is not held
        assert int(np.asarray(resized).shape[0]) == 2

        monkeypatch.setattr(segment, "_host_concrete_int", lambda _v: None)
        stale = np.array([0], dtype=np.int64)
        ns[(id(parts), ("i", 3))] = (weakref.ref(parts), stale)
        assert at.partition_ids(parts, 3) is stale

        calls = {"n": 0}

        def once_int(_v):
            calls["n"] += 1
            return 3 if calls["n"] == 1 else None

        monkeypatch.setattr(segment, "_host_concrete_int", once_int)
        assert at.partition_ids(parts, 3) is stale


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
    import array_api_compat.numpy as xp_np

    assert close(_pad_or_slice_leading(xp_np, x, 5), np.array([1.0, 2.0, 3.0, 0.0, 0.0]))
    assert close(_pad_or_slice_leading(xp_np, x, 2), np.array([1.0, 2.0]))

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
    assert list(np.asarray(ns.clip(tf.constant([0.0, 5.0]), min=1.0, max=2.0))) == [1.0, 2.0]


def test_enable_typecheck_installs_hook():
    # Idempotent: may already be installed by pytest conftest.
    at.enable_typecheck()


def test_enable_torchscript_returns_false_without_torch():
    """``enable_torchscript`` returns False until ``torch`` is imported; it still queues the helper."""
    from anytensor import segment

    was = segment._TORCHSCRIPT_ENABLED
    segment._TORCHSCRIPT_ENABLED = False
    torch_mod = sys.modules.pop("torch", None)
    torch_subs = {
        k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("torch.")
    }
    try:
        assert segment.enable_torchscript() is False
        segment._TORCHSCRIPT_ENABLED = True
        assert segment._enable_torchscript(object()) is True
    finally:
        if torch_mod is not None:
            sys.modules["torch"] = torch_mod
        sys.modules.update(torch_subs)
        segment._TORCHSCRIPT_ENABLED = was
        if was:
            assert segment.enable_torchscript() is True
            assert segment._enable_torchscript(torch_mod) is True


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
