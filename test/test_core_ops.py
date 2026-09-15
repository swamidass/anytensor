"""Intended semantics for ordinary AnyTensor ops (Array API thin dispatch)."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at


def test_cumsum_defaults_to_axis_0_on_2d():
    """cumsum must not flatten like bare NumPy; always reduce along axis 0 by default."""
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = at.cumsum(x)
    expected = np.array([[1.0, 2.0], [4.0, 6.0], [9.0, 12.0]])
    np.testing.assert_allclose(y, expected)


def test_sum_with_axes():
    x = np.arange(12.0).reshape(3, 4)
    np.testing.assert_allclose(at.sum(x, axes=0), x.sum(axis=0))
    np.testing.assert_allclose(at.sum(x, axes=(0, 1)), x.sum())


def test_mean_prod_reshape_transpose_concatenate_stack():
    x = np.arange(6.0).reshape(2, 3)
    np.testing.assert_allclose(at.mean(x, axes=0), x.mean(axis=0))
    np.testing.assert_allclose(at.prod(x, axes=1), x.prod(axis=1))
    np.testing.assert_allclose(at.reshape(x, (3, 2)), x.reshape(3, 2))
    np.testing.assert_allclose(at.transpose(x, (1, 0)), x.T)
    np.testing.assert_allclose(at.concatenate([x, x], axis=0), np.concatenate([x, x], axis=0))
    np.testing.assert_allclose(at.stack([x, x], axis=0), np.stack([x, x], axis=0))


def test_elementwise_maximum_minimum_distinct_from_reduce():
    a = np.array([1.0, 5.0, 3.0])
    b = np.array([4.0, 2.0, 3.0])
    np.testing.assert_allclose(at.maximum(a, b), np.maximum(a, b))
    np.testing.assert_allclose(at.minimum(a, b), np.minimum(a, b))
    # Reduce max/min still available under the same names with axes=
    np.testing.assert_allclose(at.max(a, axes=0), 5.0)
    np.testing.assert_allclose(at.min(a, axes=0), 1.0)


def test_sqrt_rsqrt_where_clip_astype():
    x = np.array([4.0, 9.0, 16.0])
    np.testing.assert_allclose(at.sqrt(x), np.sqrt(x))
    np.testing.assert_allclose(at.rsqrt(x), 1.0 / np.sqrt(x))
    np.testing.assert_allclose(at.where(x > 5, x, 0.0), np.where(x > 5, x, 0.0))
    np.testing.assert_allclose(at.clip(x, 5.0, 10.0), np.clip(x, 5.0, 10.0))
    y = at.astype(x, np.int64)
    assert y.dtype == np.int64


def test_zeros_ones_full_like_and_arange():
    like = np.zeros((2, 3), dtype=np.float64)
    z = at.zeros_like(like)
    o = at.ones_like(like)
    f = at.full_like(like, 3.0)
    assert z.shape == (2, 3) and np.all(z == 0)
    assert o.shape == (2, 3) and np.all(o == 1)
    assert f.shape == (2, 3) and np.all(f == 3)
    r = at.arange(0, 5, like=like)
    np.testing.assert_array_equal(np.asarray(r), np.arange(0, 5))


def test_repeat_with_total_repeat_length():
    # JAX-style: lengths [2, 1] over 2 rows -> [0, 0, 1] when values are arange(2)
    lengths = np.array([2, 1])
    idx = np.arange(2)
    out = at.repeat(idx, lengths, total_repeat_length=3)
    np.testing.assert_array_equal(np.asarray(out), np.array([0, 0, 1]))


def test_matmul():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    np.testing.assert_allclose(at.matmul(a, b), a @ b)
