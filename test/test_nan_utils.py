"""Tests for NaN / finiteness utilities."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at
from helpers import BACKENDS, close, loaded_backends


@pytest.mark.parametrize("backend", BACKENDS)
def test_specials_from_backend_attrs(backend):
    import math

    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    assert at.inf(x) == b.inf and math.isinf(at.inf(x))
    assert at.ninf(x) == b.ninf
    assert math.isnan(at.nan(x)) and math.isnan(b.nan)
    assert at.pi(x) == b.pi and at.e(x) == b.e
    assert at.dtype("bool", like=x) is b.bool
    assert at.dtype("float32", like=x) is b.float32
    assert at.finfo(x).eps == b.finfo(x.dtype).eps
    y = at.zeros((2,), dtype=at.dtype("bool", like=x), like=x)
    assert list(b.to_numpy(y)) == [False, False]
    z = b.from_numpy(np.array([1.0, np.nan], dtype=np.float32))
    out = b.to_numpy(at.fill_nan(z, at.ninf(z)))
    assert close(out, np.array([1.0, -np.inf], dtype=np.float32), equal_nan=True)


@pytest.mark.parametrize("backend", BACKENDS)
def test_is_nan_finite_inf(backend):
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32))
    assert list(b.to_numpy(at.is_nan(x))) == [False, True, False, False]
    assert list(b.to_numpy(at.is_finite(x))) == [True, False, False, False]
    assert list(b.to_numpy(at.is_inf(x))) == [False, False, True, True]
    # Array API short-name aliases
    assert list(b.to_numpy(at.isnan(x))) == [False, True, False, False]
    assert list(b.to_numpy(at.isfinite(x))) == [True, False, False, False]
    assert list(b.to_numpy(at.isinf(x))) == [False, False, True, True]


@pytest.mark.parametrize("backend", BACKENDS)
def test_fill_nan(backend):
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, np.nan, np.inf], dtype=np.float32))
    out = b.to_numpy(at.fill_nan(x, -1.0))
    assert close(out, np.array([1.0, -1.0, np.inf], dtype=np.float32), equal_nan=True)
    # alias
    assert close(
        b.to_numpy(at.nan_fill(x, -1.0)),
        np.array([1.0, -1.0, np.inf], dtype=np.float32),
        equal_nan=True,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_fill_nan_mask(backend):
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, np.nan, np.inf], dtype=np.float32))
    filled, mask = at.fill_nan_mask(x, -1.0)
    assert close(
        b.to_numpy(filled),
        np.array([1.0, -1.0, np.inf], dtype=np.float32),
        equal_nan=True,
    )
    assert list(b.to_numpy(mask)) == [False, True, False]
    # alias + equivalence to composing fill_nan / is_nan
    filled2, mask2 = at.nan_fill_mask(x, -1.0)
    assert close(b.to_numpy(filled2), b.to_numpy(at.fill_nan(x, -1.0)), equal_nan=True)
    assert list(b.to_numpy(mask2)) == list(b.to_numpy(at.is_nan(x)))


@pytest.mark.parametrize("backend", BACKENDS)
def test_nan_to_num(backend):
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, np.nan, np.inf, -np.inf], dtype=np.float32))
    out = b.to_numpy(at.nan_to_num(x, nan=-1.0, posinf=2.0, neginf=-2.0))
    assert close(out, np.array([1.0, -1.0, 2.0, -2.0], dtype=np.float32))


@pytest.mark.parametrize("backend", BACKENDS)
def test_equal_nan(backend):
    b = loaded_backends[backend]
    a = b.from_numpy(np.array([1.0, np.nan, np.inf], dtype=np.float32))
    c = b.from_numpy(np.array([1.0, np.nan, np.inf], dtype=np.float32))
    d = b.from_numpy(np.array([2.0, np.nan, -np.inf], dtype=np.float32))
    assert list(b.to_numpy(at.equal_nan(a, c))) == [True, True, True]
    assert list(b.to_numpy(at.equal_nan(a, d))) == [False, True, False]
