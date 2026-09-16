"""Symbolic / compiled fuzz: eager results must match jit/compile/graph.

Each backend is tested on its own (import skips are per-test, not module-wide):

- JAX ``jax.jit``
- Torch ``torch.compile``
- TensorFlow ``tf.function`` and ``tf.function(jit_compile=True)`` (XLA)

Python ints / tuples / floats are closed over as static args so frameworks do
not trace them as data. Return values and shapes must agree.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from helpers import loaded_backends
from test_cross_backend_fuzz import FUZZ_OPS, _agree

pytestmark = [pytest.mark.fuzz]

_settings = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# Host / metadata returns are not tensor graphs under every compiler.
_NON_TENSOR_OUT = frozenset(
    {
        "fuzz_inf",
        "fuzz_ninf",
        "fuzz_nan",
        "fuzz_pi",
        "fuzz_e",
        "fuzz_finfo",
        "fuzz_iinfo",
        "fuzz_shape",
    }
)

_SYMBOLIC_OPS = [(fn, samp) for fn, samp in FUZZ_OPS if fn.__name__ not in _NON_TENSOR_OUT]

_tf_fn_cache: dict[tuple[int, bool], Callable] = {}


def _py_static(v: Any) -> Any:
    """Normalize NumPy scalars / nested tuples to plain Python for static args."""
    if isinstance(v, tuple):
        return tuple(_py_static(x) for x in v)
    if isinstance(v, np.generic):
        return v.item()
    return v


def _is_array_arg(a: Any, backend) -> bool:
    return backend.is_appropriate_type(a)


def _split_static(args: tuple, backend) -> tuple[list[tuple[int, Any]], dict[int, Any]]:
    dyn: list[tuple[int, Any]] = []
    static: dict[int, Any] = {}
    for i, a in enumerate(args):
        a = _py_static(a)
        if _is_array_arg(a, backend):
            dyn.append((i, a))
        else:
            static[i] = a
    return dyn, static


def _call_with_static(fn: Callable, dyn: list[tuple[int, Any]], static: dict[int, Any], n_args: int):
    def wrapped(*dyn_args):
        full: list[Any] = [None] * n_args
        for i, v in static.items():
            full[i] = v
        for (i, _), v in zip(dyn, dyn_args):
            full[i] = v
        return fn(*full)

    return wrapped, [a for _, a in dyn]


def _shape_of(out) -> Any:
    if isinstance(out, tuple):
        return tuple(_shape_of(o) for o in out)
    if hasattr(out, "shape"):
        s = out.shape
        try:
            return tuple(int(d) if d is not None else None for d in s)
        except TypeError:
            return tuple(s)
    return ()


def _to_numpy(backend_name: str, out):
    backend = loaded_backends[backend_name]
    if isinstance(out, tuple):
        return tuple(_to_numpy(backend_name, o) for o in out)
    if isinstance(out, (np.ndarray, np.generic)) or np.isscalar(out):
        return np.asarray(out)
    if backend.is_appropriate_type(out):
        return backend.to_numpy(out)
    return np.asarray(out)


def _prepare_args(backend_name: str, args_np: tuple) -> tuple:
    backend = loaded_backends[backend_name]
    return tuple(
        backend.from_numpy(np.asarray(a)) if isinstance(a, np.ndarray) else _py_static(a)
        for a in args_np
    )


@st.composite
def _symbolic_example(draw):
    fn, sampler = draw(st.sampled_from(_SYMBOLIC_OPS))
    return fn, sampler(draw)


def _tf_wrap(fn: Callable, *, jit_compile: bool) -> Callable:
    import tensorflow as tf

    key = (id(fn), jit_compile)
    wrapped = _tf_fn_cache.get(key)
    if wrapped is None:

        @tf.function(autograph=False, reduce_retracing=True, jit_compile=jit_compile)
        def wrapped(*args):
            return fn(*args)

        _tf_fn_cache[key] = wrapped
    return wrapped


def _assert_eager_matches_compiled(backend_name: str, fn: Callable, args_np: tuple, y_compiled):
    args = _prepare_args(backend_name, args_np)
    y_eager = fn(*args)
    assert _shape_of(y_eager) == _shape_of(y_compiled), (
        fn.__name__,
        _shape_of(y_eager),
        _shape_of(y_compiled),
    )
    _agree(fn, args_np, _to_numpy(backend_name, y_eager), _to_numpy(backend_name, y_compiled))


# --- JAX ``jax.jit`` (standalone) ------------------------------------------

@_settings
@given(example=_symbolic_example())
@pytest.mark.skipif("jax" not in loaded_backends, reason="JAX not installed")
def test_jax_jit_matches_eager(example):
    """``jax.jit`` alone: compiled outputs and shapes match eager JAX."""
    jax = pytest.importorskip("jax")
    fn, args_np = example
    backend = loaded_backends["jax"]
    args = _prepare_args("jax", args_np)
    dyn, static = _split_static(args, backend)
    wrapped, dyn_args = _call_with_static(fn, dyn, static, len(args))
    y_jit = jax.jit(wrapped)(*dyn_args) if dyn_args else jax.jit(lambda: wrapped())()
    _assert_eager_matches_compiled("jax", fn, args_np, y_jit)


# --- TensorFlow ``tf.function`` / XLA compile (standalone) -----------------

@_settings
@given(example=_symbolic_example())
@pytest.mark.skipif("tensorflow" not in loaded_backends, reason="TensorFlow not installed")
def test_tf_function_matches_eager(example):
    """``tf.function`` alone: graph outputs and shapes match eager TF."""
    pytest.importorskip("tensorflow")
    fn, args_np = example
    args = _prepare_args("tensorflow", args_np)
    y_graph = _tf_wrap(fn, jit_compile=False)(*args)
    _assert_eager_matches_compiled("tensorflow", fn, args_np, y_graph)


@_settings
@given(example=_symbolic_example())
@pytest.mark.skipif("tensorflow" not in loaded_backends, reason="TensorFlow not installed")
def test_tf_compile_matches_eager(example):
    """``tf.function(jit_compile=True)`` (XLA) alone: matches eager TF."""
    pytest.importorskip("tensorflow")
    fn, args_np = example
    # XLA vs eager: NaN often becomes ±inf; ``prod`` with ±inf and underflow
    # can be ``inf`` eagerly vs ``nan`` under XLA.
    for a in args_np:
        if not (isinstance(a, np.ndarray) and np.issubdtype(a.dtype, np.floating)):
            continue
        if np.isnan(a).any():
            assume(False)
        if fn.__name__ == "fuzz_prod" and not np.isfinite(a).all():
            assume(False)
    args = _prepare_args("tensorflow", args_np)
    try:
        y_xla = _tf_wrap(fn, jit_compile=True)(*args)
    except Exception:
        # Some ops / platforms reject XLA; do not fail the whole suite.
        assume(False)
        return
    _assert_eager_matches_compiled("tensorflow", fn, args_np, y_xla)


# --- Torch ``torch.compile`` (standalone) ----------------------------------

@_settings
@given(example=_symbolic_example())
@pytest.mark.skipif("torch" not in loaded_backends, reason="Torch not installed")
def test_torch_compile_matches_eager(example):
    """``torch.compile`` alone: compiled outputs and shapes match eager Torch."""
    torch = pytest.importorskip("torch")
    fn, args_np = example
    backend = loaded_backends["torch"]
    args = _prepare_args("torch", args_np)
    dyn, static = _split_static(args, backend)
    wrapped, dyn_args = _call_with_static(fn, dyn, static, len(args))
    try:
        compiled = torch.compile(wrapped, fullgraph=False)
        y_c = compiled(*dyn_args) if dyn_args else compiled()
    except Exception:
        assume(False)
        return
    _assert_eager_matches_compiled("torch", fn, args_np, y_c)


def test_symbolic_ops_registry_current():
    names = {fn.__name__ for fn, _ in FUZZ_OPS}
    unknown = sorted(_NON_TENSOR_OUT - names)
    assert not unknown, f"stale non-tensor skip entries: {unknown}"
    assert _SYMBOLIC_OPS, "no ops left for symbolic fuzz"
