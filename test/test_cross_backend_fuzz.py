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

pytestmark = [
    pytest.mark.fuzz,
    pytest.mark.skipif(
        not _OTHER_BACKENDS,
        reason="Need NumPy plus at least one other importable backend",
    ),
]

_settings = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_LO, _HI = np.float32(-20), np.float32(20)
_POS_LO, _POS_HI = np.float32(0.05), np.float32(20)
# Keep finite samples away from float32 underflow; ``inf * tiny`` is ``inf`` on
# NumPy/eager TF but ``nan`` on JAX / TF XLA when the tiny flushes to 0.
_MIN_ABS = float(np.float32(1e-3))  # exact float32

Draw = Any
Sampler = Callable[[Draw], tuple]
FUZZ_OPS: list[tuple[Callable, Sampler]] = []


def fuzz_op(sampler: Sampler):
    """Register ``fn``; ``sampler(draw) -> args`` encodes input constraints."""

    def decorator(fn: Callable):
        FUZZ_OPS.append((fn, sampler))
        return fn

    return decorator


def _finite_f32(draw: Draw, *, lo: float, hi: float) -> np.float32:
    """Finite float32 with ``0`` or ``|x| >= _MIN_ABS`` (no underflow zone)."""
    def f32(x: float) -> float:
        return float(np.float32(x))

    lo_f, hi_f = f32(lo), f32(hi)
    min_abs = f32(_MIN_ABS)
    branches = []
    if lo_f <= 0.0 <= hi_f:
        branches.append(st.just(0.0))
    if hi_f >= min_abs:
        branches.append(
            st.floats(
                min_value=f32(max(lo_f, min_abs)),
                max_value=hi_f,
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            )
        )
    if lo_f <= -min_abs:
        branches.append(
            st.floats(
                min_value=lo_f,
                max_value=f32(min(hi_f, -min_abs)),
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                width=32,
            )
        )
    return np.float32(draw(st.one_of(*branches)))


def _f32_elem(draw: Draw, *, lo=_LO, hi=_HI, allow_nan: bool = True, allow_infinity: bool = True):
    # Enrich specials so NaN / ±inf appear often at high example counts.
    roll = draw(st.integers(0, 9))
    if allow_nan and roll == 0:
        return np.float32("nan")
    if allow_infinity and roll == 1:
        return np.float32("inf")
    if allow_infinity and roll == 2:
        return np.float32("-inf")
    return _finite_f32(draw, lo=float(lo), hi=float(hi))


def _f32_vec(
    draw: Draw,
    n: int,
    *,
    lo=_LO,
    hi=_HI,
    allow_nan: bool = True,
    allow_infinity: bool = True,
) -> np.ndarray:
    """Length-n float32 vector; NaN / ±inf included by default."""
    return np.asarray(
        [
            _f32_elem(draw, lo=lo, hi=hi, allow_nan=allow_nan, allow_infinity=allow_infinity)
            for _ in range(n)
        ],
        dtype=np.float32,
    )


def _pos_f32_elem(draw: Draw, *, allow_nan: bool = True, allow_infinity: bool = True):
    """Positive finite, or specials (NaN / +inf) that backends agree on for sqrt/log."""
    # ~20% specials so NaN/inf show up often at max_examples=1000.
    roll = draw(st.integers(0, 4))
    if allow_nan and roll == 0:
        return np.float32("nan")
    if allow_infinity and roll == 1:
        return np.float32("inf")
    return _finite_f32(draw, lo=float(_POS_LO), hi=float(_POS_HI))


def _pos_f32_vec(
    draw: Draw,
    n: int,
    *,
    allow_nan: bool = True,
    allow_infinity: bool = True,
) -> np.ndarray:
    return np.asarray(
        [_pos_f32_elem(draw, allow_nan=allow_nan, allow_infinity=allow_infinity) for _ in range(n)],
        dtype=np.float32,
    )


def _i64_ids(draw: Draw, n: int, *, hi: int = 4) -> np.ndarray:
    return np.asarray(
        draw(st.lists(st.integers(0, hi), min_size=n, max_size=n)),
        dtype=np.int64,
    )


# --- samplers -------------------------------------------------------------


def sample_f32_vec(draw) -> tuple:
    """Length 0..8 (empty allowed for sum/prod/cumsum/exp/mean)."""
    return (_f32_vec(draw, draw(st.integers(0, 8)), allow_nan=True, allow_infinity=True),)


def sample_f32_vec_nonempty(draw) -> tuple:
    """Length 1..8 — min/max reductions reject empty arrays."""
    return (_f32_vec(draw, draw(st.integers(1, 8)), allow_nan=True, allow_infinity=True),)


def sample_pos_f32_vec(draw) -> tuple:
    """Positive / NaN / +inf; length 0..8 (empty ok for elementwise)."""
    return (
        _pos_f32_vec(
            draw,
            draw(st.integers(0, 8)),
            allow_nan=True,
            allow_infinity=True,
        ),
    )


def sample_f32_pair(draw) -> tuple:
    """Two vectors of equal length 0..8 (NaN / ±inf allowed)."""
    n = draw(st.integers(0, 8))
    return (
        _f32_vec(draw, n, allow_nan=True, allow_infinity=True),
        _f32_vec(draw, n, allow_nan=True, allow_infinity=True),
    )


def sample_where(draw) -> tuple:
    n = draw(st.integers(0, 8))
    cond = np.asarray(draw(st.lists(st.booleans(), min_size=n, max_size=n)), dtype=bool)
    return (
        cond,
        _f32_vec(draw, n, allow_nan=True, allow_infinity=True),
        _f32_vec(draw, n, allow_nan=True, allow_infinity=True),
    )


def sample_matmul(draw) -> tuple:
    """Shapes (m, k) @ (k, p) with m,k,p in 0..4 (zero-size matmul ok)."""
    m, k, p = (draw(st.integers(0, 4)) for _ in range(3))
    return (
        _f32_vec(draw, m * k, allow_nan=True, allow_infinity=True).reshape(m, k),
        _f32_vec(draw, k * p, allow_nan=True, allow_infinity=True).reshape(k, p),
    )


def sample_take(draw) -> tuple:
    """Source length 1..8; index length 0..n (empty take ok)."""
    n = draw(st.integers(1, 8))
    k = draw(st.integers(0, n))
    idx = np.asarray(
        draw(st.lists(st.integers(0, n - 1), min_size=k, max_size=k)),
        dtype=np.int64,
    )
    return (_f32_vec(draw, n, allow_nan=True, allow_infinity=True), idx)


def sample_reshape_2x3_to_3x2(draw) -> tuple:
    return (_f32_vec(draw, 6, allow_nan=True, allow_infinity=True).reshape(2, 3), (3, 2))


def sample_segment(draw) -> tuple:
    """Length 0..8; ``num_segments >= 1`` and covers ids when n>0."""
    n = draw(st.integers(0, 8))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    if n == 0:
        seg = np.asarray([], dtype=np.int64)
        num_segments = draw(st.integers(1, 6))
    else:
        seg = _i64_ids(draw, n)
        num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    return (x, seg, num_segments)


def sample_segment_ids_only(draw) -> tuple:
    n = draw(st.integers(0, 8))
    if n == 0:
        seg = np.asarray([], dtype=np.int64)
        num_segments = draw(st.integers(1, 6))
    else:
        seg = _i64_ids(draw, n)
        num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    return (seg, num_segments)


def sample_f32_finite_vec(draw) -> tuple:
    """Finite only — for softmax / log-domain ops where NaN paths diverge."""
    return (
        _f32_vec(
            draw,
            draw(st.integers(0, 8)),
            allow_nan=False,
            allow_infinity=False,
        ),
    )


def sample_segment_finite(draw) -> tuple:
    n = draw(st.integers(0, 8))
    x = _f32_vec(draw, n, allow_nan=False, allow_infinity=False)
    if n == 0:
        seg = np.asarray([], dtype=np.int64)
        num_segments = draw(st.integers(1, 6))
    else:
        seg = _i64_ids(draw, n)
        num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    return (x, seg, num_segments)


def sample_clip(draw) -> tuple:
    n = draw(st.integers(0, 8))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    lo = draw(st.floats(min_value=-10, max_value=0, width=32, allow_nan=False, allow_infinity=False))
    hi = draw(st.floats(min_value=0, max_value=10, width=32, allow_nan=False, allow_infinity=False))
    return (x, np.float32(lo), np.float32(hi))


def sample_fill_nan(draw) -> tuple:
    n = draw(st.integers(0, 8))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    fill = _f32_elem(draw, allow_nan=False, allow_infinity=True)
    return (x, fill)


def sample_nan_to_num(draw) -> tuple:
    n = draw(st.integers(0, 8))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    return (
        x,
        _f32_elem(draw, allow_nan=False, allow_infinity=False),
        _f32_elem(draw, allow_nan=False, allow_infinity=False),
        _f32_elem(draw, allow_nan=False, allow_infinity=False),
    )


def sample_repeat(draw) -> tuple:
    n = draw(st.integers(0, 6))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    repeats = draw(st.integers(0, 3))
    return (x, repeats)


def sample_stack_pair(draw) -> tuple:
    """Two equal-shape vectors for stack / concatenate."""
    return sample_f32_pair(draw)


def sample_transpose_2x3(draw) -> tuple:
    return (_f32_vec(draw, 6, allow_nan=True, allow_infinity=True).reshape(2, 3), (1, 0))


def sample_arange(draw) -> tuple:
    n = draw(st.integers(0, 8))
    like = _f32_vec(draw, 1, allow_nan=False, allow_infinity=False)
    return (n, like)


def sample_i32_vec_nonempty(draw) -> tuple:
    n = draw(st.integers(1, 8))
    return (
        np.asarray(
            draw(st.lists(st.integers(-20, 20), min_size=n, max_size=n)),
            dtype=np.int32,
        ),
    )


def sample_partition_softmax(draw) -> tuple:
    n_part = draw(st.integers(1, 4))
    parts = [draw(st.integers(0, 3)) for _ in range(n_part)]
    if sum(parts) == 0:
        parts[0] = 1
    n = sum(parts)
    logits = _f32_vec(draw, n, allow_nan=False, allow_infinity=False)
    # Static sum_partitions (shape-size) so jax.jit can compile jnp.repeat.
    return (logits, np.asarray(parts, dtype=np.int64), int(n))


# --- registered ops -------------------------------------------------------


@fuzz_op(sample_f32_vec)
def fuzz_sum(x):
    return at.sum(x)


@fuzz_op(sample_f32_vec_nonempty)
def fuzz_min(x):
    return at.min(x)


@fuzz_op(sample_f32_vec_nonempty)
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


@fuzz_op(sample_transpose_2x3)
def fuzz_transpose(x, axes):
    return at.transpose(x, axes)


@fuzz_op(sample_stack_pair)
def fuzz_concatenate(x, y):
    return at.concatenate([x, y], axis=0)


def sample_split(draw) -> tuple:
    """Length 1..8 with NumPy-style cut indices (may produce empty chunks)."""
    n = draw(st.integers(1, 8))
    x = _f32_vec(draw, n, allow_nan=True, allow_infinity=True)
    k = draw(st.integers(0, min(3, n)))
    cuts = sorted(
        draw(st.lists(st.integers(0, n), min_size=k, max_size=k, unique=True))
    )
    return (x, cuts)


@fuzz_op(sample_split)
def fuzz_split(x, indices):
    return at.split(x, indices, axis=0)


@fuzz_op(sample_stack_pair)
def fuzz_stack(x, y):
    return at.stack([x, y], axis=0)


@fuzz_op(sample_clip)
def fuzz_clip(x, min, max):
    return at.clip(x, min=min, max=max)


@fuzz_op(sample_f32_vec)
def fuzz_astype(x):
    return at.astype(x, at.dtype("float32", like=x))


@fuzz_op(sample_f32_vec)
def fuzz_zeros_like(x):
    return at.zeros_like(x)


@fuzz_op(sample_f32_vec)
def fuzz_ones_like(x):
    return at.ones_like(x)


@fuzz_op(sample_f32_vec)
def fuzz_full_like(x):
    return at.full_like(x, 3.0)


@fuzz_op(sample_f32_vec)
def fuzz_zeros(x):
    return at.zeros(at.shape(x), like=x)


@fuzz_op(sample_f32_vec)
def fuzz_ones(x):
    return at.ones(at.shape(x), like=x)


@fuzz_op(sample_f32_vec)
def fuzz_full(x):
    return at.full(at.shape(x), 2.0, like=x)


@fuzz_op(sample_arange)
def fuzz_arange(n, like):
    return at.arange(n, like=like)


@fuzz_op(sample_repeat)
def fuzz_repeat(x, repeats):
    return at.repeat(x, repeats)


@fuzz_op(sample_f32_vec)
def fuzz_shape(x):
    return np.asarray(at.shape(x), dtype=np.int64)


@fuzz_op(sample_f32_vec)
def fuzz_inf(x):
    return at.inf(x)


@fuzz_op(sample_f32_vec)
def fuzz_ninf(x):
    return at.ninf(x)


@fuzz_op(sample_f32_vec)
def fuzz_nan(x):
    return at.nan(x)


@fuzz_op(sample_f32_vec)
def fuzz_pi(x):
    return at.pi(x)


@fuzz_op(sample_f32_vec)
def fuzz_e(x):
    return at.e(x)


@fuzz_op(sample_f32_vec)
def fuzz_finfo(x):
    fi = at.finfo(x)
    return np.asarray([float(fi.eps), float(fi.max)], dtype=np.float64)


@fuzz_op(sample_i32_vec_nonempty)
def fuzz_iinfo(x):
    ii = at.iinfo(x)
    return np.asarray([int(ii.min), int(ii.max)], dtype=np.int64)


@fuzz_op(sample_f32_vec)
def fuzz_dtype(x):
    return at.zeros((1,), dtype=at.dtype("float32", like=x), like=x)


@fuzz_op(sample_f32_vec)
def fuzz_is_nan(x):
    return at.is_nan(x)


@fuzz_op(sample_f32_vec)
def fuzz_is_finite(x):
    return at.is_finite(x)


@fuzz_op(sample_f32_vec)
def fuzz_is_inf(x):
    return at.is_inf(x)


@fuzz_op(sample_fill_nan)
def fuzz_fill_nan(x, value):
    return at.fill_nan(x, value)


@fuzz_op(sample_fill_nan)
def fuzz_fill_nan_mask(x, value):
    return at.fill_nan_mask(x, value)


@fuzz_op(sample_nan_to_num)
def fuzz_nan_to_num(x, nan, posinf, neginf):
    return at.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@fuzz_op(sample_f32_pair)
def fuzz_equal_nan(x, y):
    return at.equal_nan(x, y)


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
def fuzz_segment_variance(x, segment_ids, num_segments):
    return at.segment_variance(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_normalize(x, segment_ids, num_segments):
    return at.segment_normalize(x, segment_ids, num_segments)


@fuzz_op(sample_segment_finite)
def fuzz_segment_softmax(x, segment_ids, num_segments):
    return at.segment_softmax(x, segment_ids, num_segments)


@fuzz_op(sample_segment)
def fuzz_segment_min_or_constant(x, segment_ids, num_segments):
    return at.segment_min_or_constant(x, segment_ids, num_segments, constant=0.0)


@fuzz_op(sample_segment)
def fuzz_segment_max_or_constant(x, segment_ids, num_segments):
    return at.segment_max_or_constant(x, segment_ids, num_segments, constant=0.0)


@fuzz_op(sample_segment_ids_only)
def fuzz_segment_count(segment_ids, num_segments):
    return at.segment_count(segment_ids, num_segments)


@fuzz_op(sample_partition_softmax)
def fuzz_partition_softmax(logits, partitions, sum_partitions):
    return at.partition_softmax(logits, partitions, sum_partitions=sum_partitions)


# Public names that are infrastructure, aliases, or non-ops — not required in FUZZ_OPS.
_NON_FUZZ_PUBLIC = frozenset(
    {
        "backends",
        "tree",
        "jraph",
        "hetero",
        "export",
        "get_backend",
        "einsum",
        "pack",
        "unpack",
        "rearrange",
        "reduce",
        "promote",
        "promote_scalars",
        "promote_options",
        "align_arrays",
        "newaxis",
        "__version__",
        "empty_segment_identity",
        "enable_torchscript",
        "enable_typecheck",
        "module_if_loaded",
        # typing helpers / jaxtyping re-exports (not runtime ops)
        "ArrayT",
        "Axes",
        "Bool",
        "DtypeLike",
        "Float",
        "FloatArray",
        "Inexact",
        "Int",
        "IntArray",
        "Integer",
        "Num",
        "Real",
        "SegmentIds",
        "SegmentOut",
        "SegmentValues",
        "ShapeLike",
        "ShapeSize",
        "Shaped",
        "ShapedArray",
        # aliases of fuzzed primaries
        "nan_fill",
        "nan_fill_mask",
        "isnan",
        "isfinite",
        "isinf",
        "cast",
    }
)


def test_all_public_ops_have_fuzz_registration():
    """Every public op/helper (minus infra/aliases) must appear as fuzz_<name>."""
    covered = {fn.__name__[len("fuzz_") :] for fn, _ in FUZZ_OPS if fn.__name__.startswith("fuzz_")}
    required = set(at.__all__) - _NON_FUZZ_PUBLIC
    missing = sorted(required - covered)
    extra = sorted(covered - required)
    assert not missing, f"public ops missing @fuzz_op: {missing}"
    assert not extra, f"fuzz regs without public name (rename or extend allowlist): {extra}"


# --- runner ---------------------------------------------------------------


def _to_numpy_result(backend_name: str, out):
    backend = loaded_backends[backend_name]
    if isinstance(out, (tuple, list)):
        return tuple(_to_numpy_result(backend_name, o) for o in out)
    if isinstance(out, np.ndarray) or np.isscalar(out):
        return np.asarray(out)
    if backend.is_appropriate_type(out):
        return backend.to_numpy(out)
    return np.asarray(out)


def _run_on_backend(backend_name: str, fn: Callable, args_np: tuple):
    backend = loaded_backends[backend_name]
    converted = [
        backend.from_numpy(np.asarray(a)) if isinstance(a, np.ndarray) else a
        for a in args_np
    ]
    return _to_numpy_result(backend_name, fn(*converted))


def _agree_arrays(name: str, y_np, y_other, args_np: tuple, fn: Callable):
    assert y_np.shape == y_other.shape, (name, y_np.shape, y_other.shape)
    if np.issubdtype(y_np.dtype, np.floating) or np.issubdtype(y_other.dtype, np.floating):
        y_other = y_other.astype(y_np.dtype, copy=False)
    elif y_np.dtype != y_other.dtype and np.can_cast(y_other.dtype, y_np.dtype):
        y_other = y_other.astype(y_np.dtype, copy=False)
    if name in ("fuzz_segment_min", "fuzz_segment_max") and len(args_np) >= 3:
        seg = np.asarray(args_np[1])
        num = int(args_np[2])
        counts = np.bincount(seg, minlength=num) if seg.size else np.zeros(num, dtype=int)
        mask = counts > 0
        if not np.any(mask):
            return
        assert close(y_np[mask], y_other[mask], equal_nan=True), (name, y_np, y_other)
        return
    if np.issubdtype(y_np.dtype, np.bool_) or y_np.dtype == bool:
        assert np.array_equal(y_np, y_other), (name, y_np, y_other)
        return
    assert close(y_np, y_other, equal_nan=True), (name, y_np, y_other)


def _agree(fn: Callable, args_np: tuple, y_np, y_other):
    name = getattr(fn, "__name__", str(fn))
    if isinstance(y_np, (tuple, list)):
        assert isinstance(y_other, (tuple, list)) and len(y_np) == len(y_other), (
            name,
            y_np,
            y_other,
        )
        for a, b in zip(y_np, y_other):
            _agree_arrays(name, a, b, args_np, fn)
        return
    _agree_arrays(name, y_np, y_other, args_np, fn)


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
