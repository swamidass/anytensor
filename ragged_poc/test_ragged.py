"""Tests for ragged_poc (not part of the coverage gate)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from ragged_poc import (
    Ragged,
    concatenate,
    lengths_to_row_ids,
    lengths_to_row_splits,
    row_ids_to_lengths,
    row_splits_to_lengths,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None
HAS_TF = importlib.util.find_spec("tensorflow") is not None


def _r_feat(
    values=None,
    lengths=None,
):
    """Default fixture: lengths [2, 0, 3], values (5, 2)."""
    if values is None:
        values = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
            dtype=np.float32,
        )
    if lengths is None:
        lengths = np.array([2, 0, 3], dtype=np.int64)
    return Ragged.from_lengths(values, lengths)


def _r_simple():
    """lengths [2, 1], values (3, 2)."""
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    return Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))


# ---------------------------------------------------------------------------
# Partition helpers
# ---------------------------------------------------------------------------


def test_lengths_roundtrip():
    lengths = np.array([2, 0, 3], dtype=np.int64)
    ids = lengths_to_row_ids(lengths)
    np.testing.assert_array_equal(ids, [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(row_ids_to_lengths(ids, nrows=3), lengths)
    splits = lengths_to_row_splits(lengths)
    np.testing.assert_array_equal(splits, [0, 2, 2, 5])
    np.testing.assert_array_equal(row_splits_to_lengths(splits), lengths)


def test_lengths_helpers_empty_and_singleton():
    empty = np.array([], dtype=np.int64)
    np.testing.assert_array_equal(lengths_to_row_ids(empty), [])
    np.testing.assert_array_equal(lengths_to_row_splits(empty), [0])

    one = np.array([4], dtype=np.int64)
    np.testing.assert_array_equal(lengths_to_row_ids(one), [0, 0, 0, 0])
    np.testing.assert_array_equal(lengths_to_row_splits(one), [0, 4])

    zeros = np.array([0, 0, 0], dtype=np.int64)
    np.testing.assert_array_equal(lengths_to_row_ids(zeros), [])
    np.testing.assert_array_equal(row_ids_to_lengths(np.array([], dtype=np.int64), 3), zeros)


def test_lengths_to_row_ids_total_length():
    lengths = np.array([2, 1], dtype=np.int64)
    ids = lengths_to_row_ids(lengths, total_length=3)
    np.testing.assert_array_equal(ids, [0, 0, 1])


# ---------------------------------------------------------------------------
# Constructors / derived views
# ---------------------------------------------------------------------------


def test_from_lengths_sum_and_softmax():
    values = np.array([1.0, 3.0, 1.0, 1.0, 2.0], dtype=np.float32)
    lengths = np.array([2, 0, 3], dtype=np.int64)
    r = Ragged.from_lengths(values, lengths)

    np.testing.assert_allclose(r.sum(), [4.0, 0.0, 4.0])
    sm = r.softmax()
    np.testing.assert_allclose(sm.values[:2].sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(sm.values[2:].sum(), 1.0, rtol=1e-5)
    assert sm.row_ids is r.row_ids


def test_from_row_ids_unsorted():
    values = np.array([10.0, 1.0, 2.0], dtype=np.float32)
    row_ids = np.array([2, 0, 0], dtype=np.int64)
    r = Ragged.from_row_ids(values, row_ids, nrows=3)
    np.testing.assert_allclose(r.sum(), [3.0, 0.0, 10.0])
    np.testing.assert_array_equal(r.lengths(), [2, 0, 1])


def test_from_row_splits():
    values = np.arange(5.0)
    splits = np.array([0, 2, 2, 5], dtype=np.int64)
    r = Ragged.from_row_splits(values, splits)
    np.testing.assert_array_equal(r.row_ids, [0, 0, 2, 2, 2])
    np.testing.assert_allclose(r.mean(), [0.5, 0.0, 3.0])


def test_with_values_reuses_row_ids_object():
    r = _r_simple()
    ids = r.row_ids
    out = r.with_values(r.values * 2)
    assert out.row_ids is ids
    assert out.nrows == r.nrows
    np.testing.assert_allclose(out.values, r.values * 2)


def test_same_structure_and_logical_ndim():
    r = _r_simple()
    assert r.same_structure(r)
    assert r.same_structure(r.with_values(r.values + 1))
    assert not r.same_structure(np.ones(3))
    other = Ragged.from_lengths(r.values.copy(), np.array([1, 2], dtype=np.int64))
    assert not r.same_structure(other)
    assert r.logical_ndim == 3  # batch + ragged + F
    flat = Ragged.from_lengths(np.array([1.0, 2.0]), np.array([2], dtype=np.int64))
    assert flat.logical_ndim == 2


def test_row_splits_from_unsorted_counts_not_storage_order():
    values = np.array([10.0, 1.0, 2.0], dtype=np.float32)
    row_ids = np.array([2, 0, 0], dtype=np.int64)
    r = Ragged.from_row_ids(values, row_ids, nrows=3)
    # lengths correct; splits describe a contiguous reordering, not storage
    np.testing.assert_array_equal(r.lengths(), [2, 0, 1])
    np.testing.assert_array_equal(r.row_splits(), [0, 2, 2, 3])


def test_map_escape_hatch():
    r = _r_simple()
    ids = r.row_ids
    out = r.map(np.square)
    assert isinstance(out, Ragged) and out.row_ids is ids
    np.testing.assert_allclose(out.values, np.square(r.values))


# ---------------------------------------------------------------------------
# Reductions
# ---------------------------------------------------------------------------


def test_reduce_axis_dispatch():
    values = np.array([[1.0, 3.0], [5.0, 7.0], [2.0, 4.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))

    np.testing.assert_allclose(r.sum(axis=0), [[6.0, 10.0], [2.0, 4.0]])
    np.testing.assert_allclose(r.sum(), [[6.0, 10.0], [2.0, 4.0]])

    out = r.sum(axis=1)
    assert isinstance(out, Ragged) and out.row_ids is r.row_ids
    np.testing.assert_allclose(out.values, [4.0, 12.0, 6.0])

    full = r.sum(axis=None)
    assert not isinstance(full, Ragged)
    np.testing.assert_allclose(full, values.sum())

    out = r.mean(axis=-1)
    assert isinstance(out, Ragged)
    np.testing.assert_allclose(out.values, [2.0, 6.0, 3.0])

    with pytest.raises(ValueError, match="out of bounds"):
        r.sum(axis=2)


def test_min_max_count_mean_segment():
    values = np.array([1.0, 5.0, 2.0, 8.0, 3.0], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 0, 3], dtype=np.int64))

    np.testing.assert_allclose(r.min(), [1.0, np.inf, 2.0])  # empty → +inf
    np.testing.assert_allclose(r.max(), [5.0, -np.inf, 8.0])  # empty → -inf
    np.testing.assert_allclose(r.mean(), [3.0, 0.0, 13.0 / 3.0])
    np.testing.assert_allclose(r.count(), [2.0, 0.0, 3.0])

    # feature axis keep ragged
    vf = np.array([[1.0, 9.0], [3.0, 1.0], [2.0, 4.0]], dtype=np.float32)
    rf = Ragged.from_lengths(vf, np.array([2, 1], dtype=np.int64))
    mn = rf.min(axis=1)
    assert isinstance(mn, Ragged)
    np.testing.assert_allclose(mn.values, [1.0, 1.0, 2.0])
    mx = rf.max(axis=-1)
    np.testing.assert_allclose(mx.values, [9.0, 3.0, 4.0])


def test_with_values_normalize():
    values = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 2], dtype=np.int64))
    n = r.normalize()
    np.testing.assert_allclose(n.values, [0.5, 0.5, 0.5, 0.5])
    assert n.row_ids is r.row_ids


def test_softmax_empty_row_skipped():
    values = np.array([1.0, 3.0, 2.0], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 0, 1], dtype=np.int64))
    sm = r.softmax()
    np.testing.assert_allclose(sm.values[:2].sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(sm.values[2], 1.0, rtol=1e-5)
    assert sm.nrows == 3
    np.testing.assert_array_equal(sm.lengths(), [2, 0, 1])


def test_empty_all_rows():
    values = np.zeros((0,), dtype=np.float32)
    lengths = np.array([0, 0], dtype=np.int64)
    r = Ragged.from_lengths(values, lengths, total_length=0)
    np.testing.assert_allclose(r.sum(), [0.0, 0.0])
    np.testing.assert_allclose(r.count(), [0.0, 0.0])


# ---------------------------------------------------------------------------
# at.* / NumPy / Torch dispatch
# ---------------------------------------------------------------------------


def test_at_and_numpy_exp_same_partition_object():
    import anytensor as at

    values = np.array([0.0, 1.0, 2.0], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    ids = r.row_ids

    at_out = at.exp(r)
    assert isinstance(at_out, Ragged)
    assert at_out.row_ids is ids
    np.testing.assert_allclose(at_out.values, np.exp(values))

    np_out = np.exp(r)
    assert isinstance(np_out, Ragged)
    assert np_out.row_ids is ids
    np.testing.assert_allclose(np_out.values, np.exp(values))


def test_at_binary_and_unary_more():
    import anytensor as at

    r = _r_simple()
    ids = r.row_ids

    for fn, expected in [
        (at.exp, np.exp(r.values)),
        (at.sqrt, np.sqrt(r.values)),
        (at.log, np.log(r.values)),
    ]:
        out = fn(r)
        assert isinstance(out, Ragged) and out.row_ids is ids
        np.testing.assert_allclose(out.values, expected, rtol=1e-5)

    # two ragged same partition through at.maximum
    other = r.with_values(r.values * 0.5)
    out = at.maximum(r, other)
    assert isinstance(out, Ragged) and out.row_ids is ids
    np.testing.assert_allclose(out.values, r.values)


def test_numpy_sum_returns_dense():
    r = Ragged.from_lengths(
        np.array([1.0, 2.0, 3.0]), np.array([2, 1], dtype=np.int64)
    )
    out = np.sum(r)
    assert not isinstance(out, Ragged)
    np.testing.assert_allclose(out, 6.0)


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_exp_passthrough():
    import torch

    values = torch.tensor([0.0, 1.0, 2.0])
    row_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    r = Ragged.from_row_ids(values, row_ids, nrows=2)
    out = torch.exp(r)
    assert isinstance(out, Ragged)
    assert out.row_ids is row_ids
    torch.testing.assert_close(out.values, torch.exp(values))


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
def test_torch_add_and_matmul():
    import torch

    values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    row_ids = torch.tensor([0, 0, 1], dtype=torch.long)
    r = Ragged.from_row_ids(values, row_ids, nrows=2)
    out = r + 1.0
    assert isinstance(out, Ragged)
    torch.testing.assert_close(out.values, values + 1.0)
    w = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    mm = r @ w
    assert isinstance(mm, Ragged)
    torch.testing.assert_close(mm.values, values @ w)


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_jax_pytree_values_only():
    import jax
    import jax.numpy as jnp

    values = jnp.asarray([1.0, 2.0, 3.0])
    row_ids = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    r = Ragged.from_row_ids(values, row_ids, nrows=2)
    # tree_map touches values only; row_ids stay the same object in aux
    doubled = jax.tree.map(lambda x: x * 2, r)
    assert isinstance(doubled, Ragged)
    np.testing.assert_allclose(np.asarray(doubled.values), [2.0, 4.0, 6.0])
    # nrows / partition metadata preserved
    assert int(doubled.nrows) == 2
    np.testing.assert_array_equal(np.asarray(doubled.row_ids), [0, 0, 1])


# ---------------------------------------------------------------------------
# Broadcast / arithmetic
# ---------------------------------------------------------------------------


def test_magic_broadcast_arithmetic():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    ids = r.row_ids

    out = r + 1.0
    assert isinstance(out, Ragged) and out.row_ids is ids
    np.testing.assert_allclose(out.values, values + 1.0)

    out = 2.0 * r
    np.testing.assert_allclose(out.values, values * 2.0)

    bias = np.array([0.5, -0.5], dtype=np.float32)
    out = r + bias
    np.testing.assert_allclose(out.values, values + bias)

    other = r.with_values(values * 10)
    out = r + other
    assert out.row_ids is ids
    np.testing.assert_allclose(out.values, values + values * 10)

    out = -r
    np.testing.assert_allclose(out.values, -values)


def test_size_one_ragged_axis_broadcasts():
    """Dense size-1 on ragged axis broadcasts over variable row lengths."""
    r = _r_simple()  # B=2, N=3, F=2
    # Explicit (B, 1, F)
    batch_bias = np.array([[[10.0, 0.0]], [[100.0, 0.0]]], dtype=np.float32)
    assert batch_bias.shape == (2, 1, 2)
    out = r + batch_bias
    assert out.values.shape[0] == 3  # still packed
    np.testing.assert_allclose(
        out.values, [[11.0, 2.0], [13.0, 4.0], [105.0, 6.0]]
    )

    # (1, 1, F) — size-1 batch and ragged
    feat = np.array([[[1.0, -1.0]]], dtype=np.float32)
    out = r * feat
    np.testing.assert_allclose(out.values, [[1.0, -2.0], [3.0, -4.0], [5.0, -6.0]])

    # (1, F) left-pads to (1, 1, F)
    out = r + np.array([[1.0, 2.0]], dtype=np.float32)
    np.testing.assert_allclose(out.values, r.values + np.array([1.0, 2.0]))

    # scalar 0-d array
    out = r - np.array(1.0, dtype=np.float32)
    np.testing.assert_allclose(out.values, r.values - 1.0)


def test_logical_broadcast_packed_execution():
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))

    batch_bias = np.array([[[10.0, 0.0]], [[100.0, 0.0]]], dtype=np.float32)
    out = r + batch_bias
    assert out.values.shape[0] == 3
    np.testing.assert_allclose(
        out.values, [[11.0, 2.0], [13.0, 4.0], [105.0, 6.0]]
    )

    row_scale = np.array([[[2.0]], [[3.0]]], dtype=np.float32)
    out = r * row_scale
    np.testing.assert_allclose(
        out.values, [[2.0, 4.0], [6.0, 8.0], [15.0, 18.0]]
    )

    with pytest.raises(ValueError, match="ragged axis"):
        _ = r + np.ones((2, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="ragged axis"):
        _ = r + values


def test_broadcast_to_packed_direct():
    r = _r_simple()
    packed = r.broadcast_to_packed(
        np.array([[[10.0, 1.0]], [[20.0, 2.0]]], dtype=np.float32)
    )
    np.testing.assert_allclose(
        packed, [[10.0, 1.0], [10.0, 1.0], [20.0, 2.0]]
    )
    # scalar passthrough
    assert r.broadcast_to_packed(3.0) == 3.0
    # rank too high
    with pytest.raises(ValueError, match="exceeds logical"):
        r.broadcast_to_packed(np.ones((1, 1, 1, 2)))


def test_broadcast_with_empty_rows():
    """Size-1 ragged broadcast still works when some batch rows are empty."""
    values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 0, 0], dtype=np.int64))
    bias = np.array([[[10.0, 0.0]], [[100.0, 0.0]], [[1000.0, 0.0]]], dtype=np.float32)
    out = r + bias
    # only packed entries get the take — empty rows contribute nothing
    np.testing.assert_allclose(out.values, [[11.0, 2.0], [13.0, 4.0]])
    assert out.nrows == 3
    np.testing.assert_array_equal(out.lengths(), [2, 0, 0])


def test_reverse_binary_ops():
    r = _r_simple()
    ids = r.row_ids

    out = 10.0 - r
    assert isinstance(out, Ragged) and out.row_ids is ids
    np.testing.assert_allclose(out.values, 10.0 - r.values)

    out = 12.0 / r
    np.testing.assert_allclose(out.values, 12.0 / r.values)

    out = 2 ** r
    np.testing.assert_allclose(out.values, 2 ** r.values)

    batch = np.array([[[10.0, 20.0]], [[30.0, 40.0]]], dtype=np.float32)
    # Prefer Ragged on the left so broadcast_to_packed runs (ndarray - ragged
    # may not reverse-dispatch through NumPy).
    out = (-r) + batch
    np.testing.assert_allclose(
        out.values, [[9.0, 18.0], [7.0, 16.0], [25.0, 34.0]]
    )
    out = r.__rsub__(batch)
    np.testing.assert_allclose(
        out.values, [[9.0, 18.0], [7.0, 16.0], [25.0, 34.0]]
    )


def test_magic_mismatched_partition_raises():
    a = Ragged.from_lengths(np.array([1.0, 2.0]), np.array([2], dtype=np.int64))
    b = Ragged.from_lengths(np.array([3.0, 4.0]), np.array([1, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="same partition"):
        _ = a + b


def test_magic_matmul():
    import anytensor as at

    values = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    w = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)

    out = r @ w
    assert isinstance(out, Ragged) and out.row_ids is r.row_ids
    np.testing.assert_allclose(out.values, values @ w)

    at_out = at.matmul(r, w)
    assert isinstance(at_out, Ragged)
    np.testing.assert_allclose(at_out.values, out.values)

    # left matmul: (k,) @ (N, k) is not typical; use (2, 2) @ values.T style via rmatmul on features
    # rmatmul: other @ values — e.g. batch of weights applied from the left on last dims
    left = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    # values is (3, 2); left @ values.T wouldn't go through __rmatmul__ on leading.
    # __rmatmul__ does other @ values → (2,2) @ (3,2) fails; use (1,2) broadcast? skip shape fail
    with pytest.raises(ValueError, match="matmul between two Ragged"):
        _ = r @ r
    with pytest.raises(ValueError, match="matmul between two Ragged"):
        _ = r.__rmatmul__(r)


def test_rmatmul_preserves_leading():
    """``other @ values`` only when leading length stays ``N``."""
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    # (N, N) @ (N, F) keeps packed leading axis
    a = np.eye(3, dtype=np.float32) * 2.0
    out = r.__rmatmul__(a)
    assert isinstance(out, Ragged) and out.row_ids is r.row_ids
    np.testing.assert_allclose(out.values, a @ values)


def test_magic_pow_mod_bitwise_compare():
    values = np.array([2, 3, 4], dtype=np.int64)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    ids = r.row_ids

    out = r**2
    assert isinstance(out, Ragged) and out.row_ids is ids
    np.testing.assert_array_equal(out.values, values**2)

    out = r % 3
    np.testing.assert_array_equal(out.values, values % 3)

    out = r ^ 1
    np.testing.assert_array_equal(out.values, values ^ 1)

    out = r & 1
    np.testing.assert_array_equal(out.values, values & 1)

    out = r | 1
    np.testing.assert_array_equal(out.values, values | 1)

    out = r << 1
    np.testing.assert_array_equal(out.values, values << 1)

    out = r >> 1
    np.testing.assert_array_equal(out.values, values >> 1)

    out = ~r
    np.testing.assert_array_equal(out.values, ~values)

    cmp = r > 2
    assert isinstance(cmp, Ragged) and cmp.row_ids is ids
    np.testing.assert_array_equal(cmp.values, values > 2)

    eq = r == 3
    np.testing.assert_array_equal(eq.values, values == 3)

    ne = r != 3
    np.testing.assert_array_equal(ne.values, values != 3)

    np.testing.assert_array_equal((r < 3).values, values < 3)
    np.testing.assert_array_equal((r <= 3).values, values <= 3)
    np.testing.assert_array_equal((r >= 3).values, values >= 3)

    q, rem = divmod(r, 3)
    assert isinstance(q, Ragged) and isinstance(rem, Ragged)
    np.testing.assert_array_equal(q.values, values // 3)
    np.testing.assert_array_equal(rem.values, values % 3)

    # reverse divmod / floordiv / mod
    q2, rem2 = divmod(10, r)
    np.testing.assert_array_equal(q2.values, 10 // values)
    np.testing.assert_array_equal(rem2.values, 10 % values)

    r2 = r
    r2 += 1
    assert isinstance(r2, Ragged) and r2.row_ids is ids
    np.testing.assert_array_equal(r2.values, values + 1)


def test_unary_pos_abs_neg():
    values = np.array([-1.0, 2.0, -3.0], dtype=np.float32)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    np.testing.assert_allclose((+r).values, values)
    np.testing.assert_allclose(abs(r).values, np.abs(values))
    np.testing.assert_allclose((-r).values, -values)


def test_inplace_ops_return_new_ragged():
    values = np.array([2, 4, 6], dtype=np.int64)
    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    ids = r.row_ids
    base = r

    r = base
    r *= 2
    assert r.row_ids is ids
    np.testing.assert_array_equal(r.values, [4, 8, 12])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r //= 2
    np.testing.assert_array_equal(r.values, [1, 2, 3])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r %= 3
    np.testing.assert_array_equal(r.values, [2, 1, 0])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r **= 2
    np.testing.assert_array_equal(r.values, [4, 16, 36])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r &= 2
    np.testing.assert_array_equal(r.values, [2, 0, 2])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r |= 1
    np.testing.assert_array_equal(r.values, [3, 5, 7])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r ^= 1
    np.testing.assert_array_equal(r.values, [3, 5, 7])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r <<= 1
    np.testing.assert_array_equal(r.values, [4, 8, 12])

    r = Ragged.from_lengths(values.copy(), np.array([2, 1], dtype=np.int64))
    r >>= 1
    np.testing.assert_array_equal(r.values, [1, 2, 3])

    rf = Ragged.from_lengths(
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([2, 1], dtype=np.int64),
    )
    rf /= 2.0
    np.testing.assert_allclose(rf.values, [0.5, 1.0, 1.5])
    rf = Ragged.from_lengths(
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
        np.array([2, 1], dtype=np.int64),
    )
    w = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    rf @= w
    np.testing.assert_allclose(rf.values, [[2.0, 0.0], [0.0, 3.0], [2.0, 3.0]])


def test_higher_feature_rank_broadcast():
    """values (N, H, D) → logical (B, R, H, D)."""
    values = np.arange(12.0, dtype=np.float32).reshape(3, 2, 2)
    r = Ragged.from_lengths(values, np.array([2, 1], dtype=np.int64))
    assert r.logical_ndim == 4

    # (H, D) left-pads to (1, 1, H, D)
    bias = np.ones((2, 2), dtype=np.float32)
    out = r + bias
    np.testing.assert_allclose(out.values, values + 1.0)

    # (B, 1, H, D)
    batch = np.array(
        [[[[10.0, 10.0], [10.0, 10.0]]], [[[100.0, 100.0], [100.0, 100.0]]]],
        dtype=np.float32,
    )
    assert batch.shape == (2, 1, 2, 2)
    out = r + batch
    np.testing.assert_allclose(
        out.values,
        [
            [[10.0, 11.0], [12.0, 13.0]],
            [[14.0, 15.0], [16.0, 17.0]],
            [[108.0, 109.0], [110.0, 111.0]],
        ],
    )


# ---------------------------------------------------------------------------
# Indexing — always Ragged
# ---------------------------------------------------------------------------


def test_indexing_always_ragged():
    r = _r_feat()

    row0 = r[0]
    assert isinstance(row0, Ragged)
    assert row0.nrows == 1
    np.testing.assert_allclose(row0.values, [[1.0, 10.0], [2.0, 20.0]])
    np.testing.assert_array_equal(row0.row_ids, [0, 0])

    sub = r[0:2]
    assert isinstance(sub, Ragged) and sub.nrows == 2
    np.testing.assert_allclose(sub.values, [[1.0, 10.0], [2.0, 20.0]])

    fancy = r[[2, 0]]
    assert fancy.nrows == 2
    np.testing.assert_allclose(
        fancy.values,
        [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0], [1.0, 10.0], [2.0, 20.0]],
    )
    np.testing.assert_array_equal(fancy.row_ids, [0, 0, 0, 1, 1])

    batch_mask = np.array([True, False, True])
    masked = r[batch_mask]
    assert masked.nrows == 2
    np.testing.assert_array_equal(masked.lengths(), [2, 3])

    vmask = np.array([True, False, True, True, False])
    vm = r[vmask]
    assert vm.nrows == 3
    np.testing.assert_allclose(vm.values, [[1.0, 10.0], [3.0, 30.0], [4.0, 40.0]])

    with pytest.raises(IndexError, match="ragged index"):
        _ = r[:, 0]

    dense_rows = Ragged.from_lengths(
        r.values[:4], np.array([2, 2], dtype=np.int64)
    )
    tok = dense_rows[:, 0]
    assert isinstance(tok, Ragged) and tok.nrows == 2
    np.testing.assert_array_equal(tok.lengths(), [1, 1])
    np.testing.assert_allclose(tok.values, [[1.0, 10.0], [3.0, 30.0]])

    feat = dense_rows[:, :, 1]
    assert isinstance(feat, Ragged)
    np.testing.assert_allclose(feat.values, [10.0, 20.0, 30.0, 40.0])

    feat2 = dense_rows[..., 1]
    np.testing.assert_allclose(feat2.values, feat.values)


def test_indexing_negative_batch_and_empty_select():
    r = _r_feat()
    last = r[-1]
    assert last.nrows == 1
    np.testing.assert_array_equal(last.lengths(), [3])
    np.testing.assert_allclose(last.values, [[3.0, 30.0], [4.0, 40.0], [5.0, 50.0]])

    empty = r[0:0]
    assert isinstance(empty, Ragged) and empty.nrows == 0
    assert empty.values.shape[0] == 0

    empty_fancy = r[np.array([], dtype=np.int64)]
    assert empty_fancy.nrows == 0

    with pytest.raises(IndexError, match="out of range"):
        _ = r[3]
    with pytest.raises(IndexError, match="out of range"):
        _ = r[-4]
    with pytest.raises(IndexError, match="out of range"):
        _ = r[[0, 5]]


def test_indexing_ragged_slice_and_negative_token():
    values = np.arange(10.0).reshape(5, 2)
    r = Ragged.from_lengths(values, np.array([3, 2], dtype=np.int64))

    mid = r[:, 1:3]
    assert mid.nrows == 2
    np.testing.assert_array_equal(mid.lengths(), [2, 1])
    np.testing.assert_allclose(mid.values, [[2.0, 3.0], [4.0, 5.0], [8.0, 9.0]])

    last_tok = r[:, -1]
    assert isinstance(last_tok, Ragged)
    np.testing.assert_array_equal(last_tok.lengths(), [1, 1])
    np.testing.assert_allclose(last_tok.values, [[4.0, 5.0], [8.0, 9.0]])

    # empty slice within rows
    none = r[:, 5:6]
    assert none.nrows == 2
    np.testing.assert_array_equal(none.lengths(), [0, 0])
    assert none.values.shape[0] == 0


def test_indexing_bool_errors_and_ellipsis():
    r = _r_feat()

    with pytest.raises(IndexError, match="matches neither"):
        _ = r[np.array([True, False])]

    with pytest.raises(IndexError, match="1-D"):
        _ = r[np.array([[True, False], [False, True]])]

    with pytest.raises(IndexError, match="too many indices"):
        _ = r[0, 0, 0, 0]

    with pytest.raises(IndexError, match="only one ellipsis"):
        _ = r[..., ...]

    # full slice no-op identity-ish
    full = r[:]
    assert full.nrows == r.nrows
    np.testing.assert_allclose(full.values, r.values)

    # empty batch kept when value-masking
    mask_all_false = np.zeros(5, dtype=bool)
    empty_vals = r[mask_all_false]
    assert empty_vals.nrows == 3
    np.testing.assert_array_equal(empty_vals.lengths(), [0, 0, 0])


def test_indexing_batch_then_feature():
    r = _r_feat()
    # select batch 2 then feature 0
    out = r[2, :, 0]
    assert isinstance(out, Ragged) and out.nrows == 1
    np.testing.assert_allclose(out.values, [3.0, 4.0, 5.0])


def test_indexing_unsupported_ragged_key():
    r = _r_simple()
    with pytest.raises(TypeError, match="unsupported ragged-axis"):
        _ = r[:, [0, 1]]


def test_indexing_empty_tensor():
    r = Ragged.from_lengths(
        np.zeros((0, 2), dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        total_length=0,
    )
    out = r[:, :]
    assert out.nrows == 2
    assert out.values.shape[0] == 0
    empty_batch = r[[ ]]
    assert empty_batch.nrows == 0


# ---------------------------------------------------------------------------
# Concatenate
# ---------------------------------------------------------------------------


def test_concatenate_batch_axis():
    import anytensor as at

    a = Ragged.from_lengths(
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.array([2], dtype=np.int64),
    )
    b = Ragged.from_lengths(
        np.array([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
        np.array([1, 0, 2], dtype=np.int64),
    )
    out = at.concatenate([a, b], axis=0)
    assert isinstance(out, Ragged)
    assert out.nrows == 4
    np.testing.assert_array_equal(out.lengths(), [2, 1, 0, 2])
    np.testing.assert_allclose(
        out.values,
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    )
    np.testing.assert_array_equal(out.row_ids, [0, 0, 1, 3, 3])

    # module helper + np.concatenate
    out2 = concatenate([a, b])
    np.testing.assert_array_equal(out2.row_ids, out.row_ids)
    out3 = np.concatenate([a, b])
    assert isinstance(out3, Ragged)
    np.testing.assert_allclose(out3.values, out.values)


def test_concatenate_feature_axis_matching_partition():
    import anytensor as at

    lengths = np.array([2, 1], dtype=np.int64)
    left = Ragged.from_lengths(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        lengths,
    )
    right = left.with_values(
        np.array([[10.0], [20.0], [30.0]], dtype=np.float32)
    )
    # logical (B, R, F) → feature axis=2; partitions match via shared row_ids
    out = at.concatenate([left, right], axis=2)
    assert isinstance(out, Ragged)
    assert out.row_ids is left.row_ids
    np.testing.assert_allclose(
        out.values,
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
    )
    # negative feature axis
    out_neg = at.concatenate([left, right], axis=-1)
    np.testing.assert_allclose(out_neg.values, out.values)


def test_concatenate_feature_mismatched_partition_raises():
    import anytensor as at

    a = Ragged.from_lengths(
        np.array([[1.0], [2.0]], dtype=np.float32),
        np.array([2], dtype=np.int64),
    )
    b = Ragged.from_lengths(
        np.array([[3.0], [4.0]], dtype=np.float32),
        np.array([1, 1], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="exact matching ragged partition"):
        at.concatenate([a, b], axis=2)


def test_concatenate_ragged_axis_raises():
    import anytensor as at

    r = _r_simple()
    with pytest.raises(ValueError, match="ragged axis"):
        at.concatenate([r, r], axis=1)


def test_concatenate_empty_and_errors():
    import anytensor as at

    with pytest.raises(ValueError, match="at least one"):
        concatenate([])

    empty = Ragged.from_lengths(
        np.zeros((0, 2), dtype=np.float32),
        np.array([0, 0], dtype=np.int64),
        total_length=0,
    )
    a = _r_simple()
    out = at.concatenate([a, empty], axis=0)
    assert out.nrows == a.nrows + 2
    np.testing.assert_allclose(out.values, a.values)
    np.testing.assert_array_equal(out.lengths()[:2], a.lengths())
    np.testing.assert_array_equal(out.lengths()[2:], [0, 0])


def test_concatenate_feature_equal_values_distinct_ids_eager():
    """Eager: equal row_ids values (distinct objects) still match."""
    import anytensor as at

    ids = np.array([0, 0, 1], dtype=np.int64)
    a = Ragged.from_row_ids(
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        ids,
        nrows=2,
    )
    b = Ragged.from_row_ids(
        np.array([[10.0], [20.0], [30.0]], dtype=np.float32),
        ids.copy(),
        nrows=2,
    )
    out = at.concatenate([a, b], axis=2)
    np.testing.assert_allclose(
        out.values,
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
    )


# ---------------------------------------------------------------------------
# Concatenate under tracing (all backends)
# ---------------------------------------------------------------------------


def _to_numpy(x):
    if hasattr(x, "numpy") and callable(x.numpy):
        out = x.numpy()
        return out if isinstance(out, np.ndarray) else np.asarray(out)
    return np.asarray(x)


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_concatenate_jax_jit_batch_and_feature():
    import jax
    import jax.numpy as jnp
    import anytensor as at

    ids_a = jnp.asarray([0, 0], dtype=jnp.int32)
    ids_b = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    ids_f = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    nrows_a, nrows_b, nrows_f = 1, 2, 2

    @jax.jit
    def batch_cat(va, vb):
        a = Ragged(va, ids_a, nrows=nrows_a)
        b = Ragged(vb, ids_b, nrows=nrows_b)
        out = at.concatenate([a, b], axis=0)
        return out.values, out.row_ids

    va = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    vb = jnp.asarray([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=jnp.float32)
    vals, ids = batch_cat(va, vb)
    np.testing.assert_allclose(
        _to_numpy(vals),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    )
    np.testing.assert_array_equal(_to_numpy(ids), [0, 0, 1, 1, 2])

    @jax.jit
    def feat_cat(vl, vr):
        left = Ragged(vl, ids_f, nrows=nrows_f)
        right = left.with_values(vr)  # shared row_ids under trace
        out = at.concatenate([left, right], axis=2)
        return out.values

    vl = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    vr = jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32)
    np.testing.assert_allclose(
        _to_numpy(feat_cat(vl, vr)),
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
    )


@pytest.mark.skipif(not HAS_JAX, reason="jax not installed")
def test_concatenate_jax_jit_feature_distinct_ids_raises():
    import jax
    import jax.numpy as jnp
    import anytensor as at

    @jax.jit
    def feat_bad(vl, vr):
        left = Ragged(vl, jnp.asarray([0, 0, 1], dtype=jnp.int32), nrows=2)
        right = Ragged(vr, jnp.asarray([0, 0, 1], dtype=jnp.int32), nrows=2)
        return at.concatenate([left, right], axis=2).values

    vl = jnp.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=jnp.float32)
    vr = jnp.asarray([[10.0], [20.0], [30.0]], dtype=jnp.float32)
    with pytest.raises(ValueError, match="shared row_ids|under tracing|cannot compare"):
        feat_bad(vl, vr)


@pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")
def test_concatenate_tf_function_batch_and_feature():
    import tensorflow as tf
    import anytensor as at

    ids_a = tf.constant([0, 0], dtype=tf.int32)
    ids_b = tf.constant([0, 0, 1], dtype=tf.int32)
    ids_f = tf.constant([0, 0, 1], dtype=tf.int32)

    @tf.function(autograph=False)
    def batch_cat(va, vb):
        a = Ragged(va, ids_a, nrows=1)
        b = Ragged(vb, ids_b, nrows=2)
        out = at.concatenate([a, b], axis=0)
        return out.values, out.row_ids

    va = tf.constant([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    vb = tf.constant(
        [[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], dtype=tf.float32
    )
    vals, ids = batch_cat(va, vb)
    np.testing.assert_allclose(
        _to_numpy(vals),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    )
    np.testing.assert_array_equal(_to_numpy(ids), [0, 0, 1, 1, 2])

    @tf.function(autograph=False)
    def feat_cat(vl, vr):
        left = Ragged(vl, ids_f, nrows=2)
        right = left.with_values(vr)
        return at.concatenate([left, right], axis=2).values

    vl = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
    vr = tf.constant([[10.0], [20.0], [30.0]], dtype=tf.float32)
    np.testing.assert_allclose(
        _to_numpy(feat_cat(vl, vr)),
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
    )


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
@pytest.mark.skipif(
    __import__("os").environ.get("CI") == "true",
    reason="torch.compile disabled on CI runners (dynamo SIGSEGV)",
)
def test_concatenate_torch_compile_batch_and_feature():
    import torch
    import anytensor as at

    ids_a = torch.tensor([0, 0], dtype=torch.long)
    ids_b = torch.tensor([0, 0, 1], dtype=torch.long)
    ids_f = torch.tensor([0, 0, 1], dtype=torch.long)

    def batch_cat(va, vb):
        a = Ragged(va, ids_a, nrows=1)
        b = Ragged(vb, ids_b, nrows=2)
        out = at.concatenate([a, b], axis=0)
        return out.values, out.row_ids

    compiled_batch = torch.compile(batch_cat, fullgraph=False)
    va = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    vb = torch.tensor([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
    vals, ids = compiled_batch(va, vb)
    np.testing.assert_allclose(
        _to_numpy(vals),
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]],
    )
    np.testing.assert_array_equal(_to_numpy(ids), [0, 0, 1, 1, 2])

    def feat_cat(vl, vr):
        left = Ragged(vl, ids_f, nrows=2)
        right = left.with_values(vr)
        return at.concatenate([left, right], axis=2).values

    compiled_feat = torch.compile(feat_cat, fullgraph=False)
    vl = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    vr = torch.tensor([[10.0], [20.0], [30.0]])
    np.testing.assert_allclose(
        _to_numpy(compiled_feat(vl, vr)),
        [[1.0, 2.0, 10.0], [3.0, 4.0, 20.0], [5.0, 6.0, 30.0]],
    )