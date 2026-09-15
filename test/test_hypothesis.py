"""Hypothesis fuzz tests (NumPy reference; optional backend agreement)."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st

import anytensor as at

from helpers import BACKENDS, close, loaded_backends

# Keep examples modest for CI time.
_settings = settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _seg_examples():
    """(x 1-d, segment_ids, num_segments) with valid non-negative ids."""
    return st.integers(1, 8).flatmap(
        lambda n: st.tuples(
            st.lists(
                st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
                min_size=n,
                max_size=n,
            ).map(lambda xs: np.asarray(xs, dtype=np.float64)),
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
    # Ensure num_segments covers all ids used.
    num_segments = max(num_segments, int(seg.max()) + 1 if seg.size else 1)
    total = at.segment_sum(x, seg, num_segments)
    counts = at.segment_count(seg, num_segments)
    mean = at.segment_mean(x, seg, num_segments)
    assert total.shape == (num_segments,)
    assert counts.shape == (num_segments,)
    # Empty segments: sum 0, count 0, mean 0.
    for i in range(num_segments):
        mask = seg == i
        if not np.any(mask):
            assert total[i] == 0.0
            assert counts[i] == 0.0
            assert mean[i] == 0.0
        else:
            assert np.isclose(total[i], x[mask].sum())
            assert np.isclose(counts[i], mask.sum())
            assert np.isclose(mean[i], x[mask].mean())


@_settings
@given(
    xs=st.lists(
        st.floats(min_value=0.1, max_value=20, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=12,
    )
)
def test_fuzz_reductions_0d_and_positive(xs):
    x = np.asarray(xs, dtype=np.float64)
    for op in (at.sum, at.min, at.max, at.mean, at.prod):
        out = op(x)
        assert isinstance(out, np.ndarray) and out.ndim == 0
    assert close(at.sum(x), np.sum(x))
    assert close(at.mean(x), np.mean(x))


@_settings
@given(
    a=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
    b=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=1,
        max_size=10,
    ),
)
def test_fuzz_maximum_matches_numpy(a, b):
    n = min(len(a), len(b))
    assume(n >= 1)
    aa = np.asarray(a[:n], dtype=np.float64)
    bb = np.asarray(b[:n], dtype=np.float64)
    assert close(at.maximum(aa, bb), np.maximum(aa, bb))


@pytest.mark.parametrize("backend", [b for b in BACKENDS if b != "numpy"])
@_settings
@given(data=_seg_examples())
def test_fuzz_segment_sum_matches_numpy_backend(backend, data):
    x, seg, num_segments = data
    num_segments = max(num_segments, int(seg.max()) + 1 if seg.size else 1)
    backend_impl = loaded_backends[backend]
    ref = at.segment_sum(x, seg, num_segments)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    out = backend_impl.to_numpy(at.segment_sum(bx, bs, num_segments))
    assert close(out, ref)
