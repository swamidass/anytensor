"""Contract: empty-segment identities match :mod:`anytensor.semantics` on every backend."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at
from anytensor.semantics import empty_segment_identity
from helpers import BACKENDS, close, loaded_backends


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_empty_float_segment_identities(backend, dtype):
    """Empty float slots: min→+inf, max→-inf, sum→0."""
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([1.0, 2.0], dtype=dtype))
    seg = b.from_numpy(np.array([0, 0], dtype=np.int64))
    # segment 1 empty
    smin = b.to_numpy(at.segment_min(x, seg, 2))
    smax = b.to_numpy(at.segment_max(x, seg, 2))
    ssum = b.to_numpy(at.segment_sum(x, seg, 2))
    assert smin[0] == dtype(1.0) and np.isposinf(smin[1])
    assert smax[0] == dtype(2.0) and np.isneginf(smax[1])
    assert close(ssum, np.array([3.0, 0.0], dtype=dtype))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_int_segment_identities(backend):
    """Empty int slots: min→iinfo.max, max→iinfo.min, sum→0 (of the result dtype)."""
    b = loaded_backends[backend]
    # int32 host avoids JAX x64 truncation surprises while still testing integrals.
    x = b.from_numpy(np.array([1, 5], dtype=np.int32))
    seg = b.from_numpy(np.array([0, 0], dtype=np.int32))
    smin = b.to_numpy(at.segment_min(x, seg, 2))
    smax = b.to_numpy(at.segment_max(x, seg, 2))
    ssum = b.to_numpy(at.segment_sum(x, seg, 2))
    imax = np.iinfo(smin.dtype).max
    imin = np.iinfo(smax.dtype).min
    assert int(smin[0]) == 1 and int(smin[1]) == imax
    assert int(smax[0]) == 5 and int(smax[1]) == imin
    assert list(ssum) == [6, 0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_inf_only_segment_is_not_empty_identity(backend):
    """A segment whose only value is +inf must stay +inf for min (not finfo.max)."""
    b = loaded_backends[backend]
    x = b.from_numpy(np.array([np.inf], dtype=np.float32))
    seg = b.from_numpy(np.array([0], dtype=np.int64))
    out = b.to_numpy(at.segment_min(x, seg, 1))
    assert np.isposinf(out[0])


def test_empty_segment_identity_helper():
    assert empty_segment_identity(np.float32, "sum", xp=np) == 0
    assert empty_segment_identity(np.float32, "min", xp=np) == np.inf
    assert empty_segment_identity(np.float32, "max", xp=np) == -np.inf
    assert empty_segment_identity(np.int64, "min", xp=np) == np.iinfo(np.int64).max
    assert empty_segment_identity(np.int64, "max", xp=np) == np.iinfo(np.int64).min
