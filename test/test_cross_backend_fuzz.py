"""Cross-backend Hypothesis fuzz: NumPy × random other backend × @fuzz_op.

Each case registers with ``@fuzz_op(sampler)``. The sampler is a function
``sampler(draw) -> args_tuple`` and owns all input constraints (positivity,
shapes, segment coverage, …) — not just dtypes.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

import anytensor as at
from helpers import BACKENDS, close, loaded_backends

_OTHER_BACKENDS = [b for b in BACKENDS if b != "numpy"]

pytestmark = pytest.mark.skipif(
    not _OTHER_BACKENDS,
    reason="Need NumPy plus at least one other importable backend",
)

_settings = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_LO, _HI = np.float32(-20), np.float32(20)
_POS_LO, _POS_HI = np.float32(0.05), np.float32(20)

Draw = Any
Sampler = Callable[[Draw], tuple]
FUZZ_OPS: list[tuple[Callable, Sampler]] = []


def fuzz_op(sampler: Sampler):
    """Register ``fn``; ``sampler(draw) -> args`` encodes input constraints."""

    def decorator(fn: Callable):
        FUZZ_OPS.append((fn, sampler))
        return fn

    return decorator


def _f32_vec(draw: Draw, n: int, *, lo=_LO, hi=_HI) -> np.ndarray:
    return np.asarray(
        draw(
            st.lists(
                st.floats(
                    min_value=float(lo),
                    max_value=float(hi),
                    allow_nan=False,
                    allow_infinity=False,
                    width=32,
                ),
                min_size=n,
                max_size=n,
            )
        ),
        dtype=np.float32,
    )


def _i64_ids(draw: Draw, n: int, *, hi: int = 4) -> np.ndarray:
    return np.asarray(
        draw(st.lists(st.integers(0, hi), min_size=n, max_size=n)),
        dtype=np.int64,
    )


# --- samplers -------------------------------------------------------------


def sample_f32_vec(draw) -> tuple:
    return (_f32_vec(draw, draw(st.integers(1, 8))),)


def sample_pos_f32_vec(draw) -> tuple:
    """Strictly positive — log / sqrt / rsqrt."""
    return (_f32_vec(draw, draw(st.integers(1, 8)), lo=_POS_LO, hi=_POS_HI),)


def sample_f32_pair(draw) -> tuple:
    """Two vectors of equal length."""
    n = draw(st.integers(1, 8))
    return (_f32_vec(draw, n), _f32_vec(draw, n))


def sample_where(draw) -> tuple:
    n = draw(st.integers(1, 8))
    cond = np.asarray(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool)
    return (cond, _f32_vec(draw, n), _f32_vec(draw, n))


def sample_matmul(draw) -> tuple:
    """Shapes (m, k) @ (k, p)."""
    m, k, p = (draw(st.integers(1, 4)) for _ in range(3))
    return (
        _f32_vec(draw, m * k).reshape(m, k),
        _f32_vec(draw, k * p).reshape(k, p),
    )


def sample_take(draw) -> tuple:
    """Indices in ``[0, n)`` into a length-n vector."""
    n = draw(st.integers(2, 8))
    k = draw(st.integers(1, n))
    idx = np.asarray(
        draw(st.lists(st.integers(0, n - 1), min_size=k, max_size=k)),
        dtype=np.int64,
    )
    return (_f32_vec(draw, n), idx)


def sample_reshape_2x3_to_3x2(draw) -> tuple:
    return (_f32_vec(draw, 6).reshape(2, 3), (3, 2))


def sample_segment(draw) -> tuple:
    """``x``, integral ids, ``num_segments >= max(id)+1`` (may include empties)."""
    n = draw(st.integers(1, 8))
    x = _f32_vec(draw, n)
    seg = _i64_ids(draw, n)
    num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    return (x, seg, num_segments)


def sample_segment_ids_only(draw) -> tuple:
    n = draw(st.integers(1, 8))
    seg = _i64_ids(draw, n)
    num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    return (seg, num_segments)


# --- registered ops -------------------------------------------------------


@fuzz_op(sample_f32_vec)
def fuzz_sum(x):
    return at.sum(x)


@fuzz_op(sample_f32_vec)
def fuzz_min(x):
    return at.min(x)


@fuzz_op(sample_f32_vec)
def fuzz_max(x):
    return at.max(x)


@fuzz_op(sample_f32_vec)
def fuzz_mean(x):
    return at.mean(x)


@fuzz_op(sample_f32_vec)
def fuzz_prod(x):
    return at.prod(x)


@fuzz_op(sample_f32_vec)
def fuzz_cumsum(x):
    return at.cumsum(x)


@fuzz_op(sample_f32_vec)
def fuzz_exp(x):
    return at.exp(x)


@fuzz_op(sample_pos_f32_vec)
def fuzz_sqrt(x):
    return at.sqrt(x)


@fuzz_op(sample_pos_f32_vec)
def fuzz_rsqrt(x):
    return at.rsqrt(x)


@fuzz_op(sample_pos_f32_vec)
def fuzz_log(x):
    return at.log(x)


@fuzz_op(sample_f32_pair)
def fuzz_maximum(x, y):
    return at.maximum(x, y)


@fuzz_op(sample_f32_pair)
def fuzz_minimum(x, y):
    return at.minimum(x, y)


@fuzz_op(sample_matmul)
def fuzz_matmul(x, y):
    return at.matmul(x, y)


@fuzz_op(sample_where)
def fuzz_where(condition, x, y):
    return at.where(condition, x, y)


@fuzz_op(sample_take)
def fuzz_take(x, indices):
    return at.take(x, indices)


@fuzz_op(sample_reshape_2x3_to_3x2)
def fuzz_reshape(x, shape):
    return at.reshape(x, shape)


@fuzz_op(sample_segment)
def fuzz_segment_sum(x, segment_ids, num_segments):
    return at.segment_sum(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_min(x, segment_ids, num_segments):
    return at.segment_min(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_max(x, segment_ids, num_segments):
    return at.segment_max(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_mean(x, segment_ids, num_segments):
    return at.segment_mean(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_normalize(x, segment_ids, num_segments):
    return at.segment_normalize(x, segment_ids, num_segments)


@fuzz_op(sample_segment_ids_only)
def fuzz_segment_count(segment_ids, num_segments):
    return at.segment_count(segment_ids, num_segments)


# --- runner ---------------------------------------------------------------


def _run_on_backend(backend_name: str, fn: Callable, args_np: tuple):
    backend = loaded_backends[backend_name]
    converted = [
        backend.from_numpy(np.asarray(a)) if isinstance(a, np.ndarray) else a
        for a in args_np
    ]
    out = fn(*converted)
    if hasattr(out, "shape") or hasattr(out, "dtype"):
        return backend.to_numpy(out)
    return np.asarray(out)


def _agree(fn: Callable, args_np: tuple, y_np, y_other):
    name = getattr(fn, "__name__", str(fn))
    assert y_np.shape == y_other.shape, (name, y_np.shape, y_other.shape)
    y_other = y_other.astype(y_np.dtype, copy=False)
    if name in ("fuzz_segment_min", "fuzz_segment_max") and len(args_np) >= 3:
        seg = np.asarray(args_np[1])
        num = int(args_np[2])
        counts = np.bincount(seg, minlength=num)
        mask = counts > 0
        if not np.any(mask):
            return
        assert close(y_np[mask], y_other[mask]), (name, y_np, y_other)
        return
    assert close(y_np, y_other), (name, y_np, y_other)


@st.composite
def _numpy_vs_other_example(draw):
    other = draw(st.sampled_from(_OTHER_BACKENDS))
    fn, sampler = draw(st.sampled_from(FUZZ_OPS))
    args_np = sampler(draw)
    return other, fn, args_np


@_settings
@given(example=_numpy_vs_other_example())
def test_numpy_vs_random_backend_random_op(example):
    other, fn, args_np = example
    assume(other in loaded_backends)
    y_np = _run_on_backend("numpy", fn, args_np)
    y_other = _run_on_backend(other, fn, args_np)
    _agree(fn, args_np, y_np, y_other)
