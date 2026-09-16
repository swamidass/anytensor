"""Cross-backend Hypothesis fuzz: NumPy reference × random other backend × random op."""

from __future__ import annotations

from dataclasses import dataclass
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


def _f32(seq):
    """float32 host arrays — matches JAX default and avoids f64/f32 underflow skew."""
    return np.asarray(seq, dtype=np.float32)


def _i64(seq):
    return np.asarray(seq, dtype=np.int64)


def _floats(n, lo=-20.0, hi=20.0):
    return st.lists(
        st.floats(
            min_value=lo,
            max_value=hi,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=n,
        max_size=n,
    ).map(_f32)


def _pos_floats(n):
    return _floats(n, lo=0.05, hi=20.0)


@dataclass(frozen=True)
class OpCase:
    name: str
    build: Callable[[Any], tuple[tuple, dict]]


def _unary(op, strategy):
    def build(draw):
        return (draw(strategy),), {}

    return OpCase(op.__name__, build)


@st.composite
def _segment_args(draw, need_x: bool = True):
    n = draw(st.integers(1, 8))
    seg = draw(st.lists(st.integers(0, 4), min_size=n, max_size=n).map(_i64))
    num_segments = max(int(seg.max()) + 1, draw(st.integers(1, 6)))
    if need_x:
        x = draw(_floats(n))
        return (x, seg, num_segments), {}
    return (seg, num_segments), {}


def _op_catalog() -> list[OpCase]:
    cases: list[OpCase] = []

    for op in (at.sum, at.min, at.max, at.mean, at.prod, at.cumsum, at.exp):
        cases.append(_unary(op, st.integers(1, 8).flatmap(lambda n: _floats(n))))
    for op in (at.sqrt, at.rsqrt, at.log):
        cases.append(_unary(op, st.integers(1, 8).flatmap(lambda n: _pos_floats(n))))

    def binary_build(draw, op_name: str):
        n = draw(st.integers(1, 8))
        return (draw(_floats(n)), draw(_floats(n))), {}

    for name in ("maximum", "minimum"):
        cases.append(OpCase(name, lambda draw, name=name: binary_build(draw, name)))

    def matmul_build(draw):
        m, k, n = draw(st.integers(1, 4)), draw(st.integers(1, 4)), draw(st.integers(1, 4))
        a = draw(_floats(m * k)).reshape(m, k)
        b = draw(_floats(k * n)).reshape(k, n)
        return (a, b), {}

    cases.append(OpCase("matmul", matmul_build))

    def where_build(draw):
        n = draw(st.integers(1, 8))
        cond = draw(st.lists(st.booleans(), min_size=n, max_size=n).map(np.asarray))
        return (cond, draw(_floats(n)), draw(_floats(n))), {}

    cases.append(OpCase("where", where_build))

    for name, need_x in [
        ("segment_sum", True),
        ("segment_min", True),
        ("segment_max", True),
        ("segment_mean", True),
        ("segment_normalize", True),
        ("segment_count", False),
    ]:

        def build(draw, need_x=need_x):
            return draw(_segment_args(need_x=need_x))

        cases.append(OpCase(name, build))

    def take_build(draw):
        n = draw(st.integers(2, 8))
        x = draw(_floats(n))
        idx = draw(st.lists(st.integers(0, n - 1), min_size=1, max_size=n).map(_i64))
        return (x, idx), {}

    cases.append(OpCase("take", take_build))

    def reshape_build(draw):
        return (draw(_floats(6)).reshape(2, 3), (3, 2)), {}

    cases.append(OpCase("reshape", reshape_build))
    return cases


OP_CASES = _op_catalog()


def _run_on_backend(backend_name: str, op_name: str, args_np: tuple, kwargs: dict):
    backend = loaded_backends[backend_name]
    op = getattr(at, op_name)
    converted = []
    for a in args_np:
        if isinstance(a, np.ndarray):
            converted.append(backend.from_numpy(np.asarray(a)))
        else:
            converted.append(a)
    out = op(*converted, **kwargs)
    if hasattr(out, "shape") or hasattr(out, "dtype"):
        return backend.to_numpy(out)
    return np.asarray(out)


def _agree(op_name: str, args_np: tuple, y_np, y_other):
    """Compare outputs; empty segment min/max sentinels differ by backend (±inf vs finfo)."""
    assert y_np.shape == y_other.shape, (op_name, y_np.shape, y_other.shape)
    if op_name in ("segment_min", "segment_max") and len(args_np) >= 3:
        seg = np.asarray(args_np[1])
        num = int(args_np[2])
        counts = np.bincount(seg, minlength=num)
        mask = counts > 0
        if not np.any(mask):
            return  # all empty — sentinel-only; skip value check
        assert close(y_np[mask], y_other[mask].astype(y_np.dtype, copy=False)), (
            op_name,
            y_np,
            y_other,
        )
        return
    assert close(y_np, y_other.astype(y_np.dtype, copy=False)), (op_name, y_np, y_other)


@st.composite
def _numpy_vs_other_example(draw):
    """Always NumPy as reference; randomly choose another importable backend."""
    other = draw(st.sampled_from(_OTHER_BACKENDS))
    case = draw(st.sampled_from(OP_CASES))
    args_np, kwargs = case.build(draw)
    return other, case.name, args_np, kwargs


@_settings
@given(example=_numpy_vs_other_example())
def test_numpy_vs_random_backend_random_op(example):
    """NumPy reference vs a randomly chosen other backend on a random op/inputs."""
    other, op_name, args_np, kwargs = example
    assume(other in loaded_backends)
    y_np = _run_on_backend("numpy", op_name, args_np, kwargs)
    y_other = _run_on_backend(other, op_name, args_np, kwargs)
    _agree(op_name, args_np, y_np, y_other)
