"""Hypothesis fuzz tests (NumPy reference; optional backend agreement)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import anytensor as at
from helpers import BACKENDS, close, loaded_backends

_settings = settings(
    max_examples=500,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# Hypothesis forbids allow_nan with min/max — compose specials explicitly.
_finite = st.floats(
    min_value=-50,
    max_value=50,
    allow_nan=False,
    allow_infinity=False,
)
_real = st.one_of(
    _finite,
    st.just(float("nan")),
    st.just(float("inf")),
    st.just(float("-inf")),
)


def _seg_examples():
    """Include empty (n=0) and short vectors; values may be NaN / ±inf."""
    return st.integers(0, 8).flatmap(
        lambda n: st.tuples(
            st.lists(_real, min_size=n, max_size=n).map(
                lambda xs: np.asarray(xs, dtype=np.float64)
            ),
            st.lists(st.integers(0, 5), min_size=n, max_size=n).map(
                lambda ids: np.asarray(ids, dtype=np.int64)
            ),
            st.integers(1, 8),
        )
    )


@_settings
@given(data=_seg_examples())
def test_fuzz_segment_sum_count_mean_numpy(data):
    x, seg, num_segments = data
    if seg.size:
        num_segments = max(num_segments, int(seg.max()) + 1)
    total = at.segment_sum(x, seg, num_segments)
    counts = at.segment_count(seg, num_segments)
    mean = at.segment_mean(x, seg, num_segments)
    assert total.shape == (num_segments,)
    assert counts.shape == (num_segments,)
    for i in range(num_segments):
        mask = seg == i
        if not np.any(mask):
            assert total[i] == 0.0
            assert counts[i] == 0.0
            assert mean[i] == 0.0
        else:
            assert close(total[i], x[mask].sum(), equal_nan=True)
            assert np.isclose(counts[i], mask.sum())
            assert close(mean[i], x[mask].mean(), equal_nan=True)


@_settings
@given(xs=st.lists(_real, min_size=0, max_size=12))
def test_fuzz_reductions_with_nan_inf_and_empty(xs):
    x = np.asarray(xs, dtype=np.float64)
    # Empty: sum/prod/mean/cumsum ok; min/max need nonempty.
    if x.size == 0:
        assert close(at.sum(x), np.sum(x), equal_nan=True)
        assert close(at.prod(x), np.prod(x), equal_nan=True)
        assert close(at.mean(x), np.mean(x), equal_nan=True)
        assert close(at.cumsum(x), np.cumsum(x), equal_nan=True)
        return
    for op in (at.sum, at.min, at.max, at.mean, at.prod):
        out = op(x)
        assert isinstance(out, np.ndarray) and out.ndim == 0
    assert close(at.sum(x), np.sum(x), equal_nan=True)
    assert close(at.mean(x), np.mean(x), equal_nan=True)
    assert close(at.min(x), np.min(x), equal_nan=True)
    assert close(at.max(x), np.max(x), equal_nan=True)


@_settings
@given(a=st.lists(_real, min_size=0, max_size=10), b=st.lists(_real, min_size=0, max_size=10))
def test_fuzz_maximum_matches_numpy_nan_inf(a, b):
    n = min(len(a), len(b))
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    assert close(at.maximum(aa, bb), np.maximum(aa, bb), equal_nan=True)
    assert close(at.minimum(aa, bb), np.minimum(aa, bb), equal_nan=True)


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "numpy"])
@_settings
@given(data=_seg_examples())
def test_fuzz_segment_sum_matches_numpy_backend(backend, data):
    x, seg, num_segments = data
    if seg.size:
        num_segments = max(num_segments, int(seg.max()) + 1)
    backend_impl = loaded_backends[backend]
    x32 = x.astype(np.float32)
    ref = at.segment_sum(x32, seg, num_segments)
    bx = backend_impl.from_numpy(x32)
    bs = backend_impl.from_numpy(seg)
    out = backend_impl.to_numpy(at.segment_sum(bx, bs, num_segments))
    assert close(out, ref, equal_nan=True)
