"""Fuzz: TensorFlow eager results must match ``tf.function`` (graph) results.

Samples the same ``FUZZ_OPS`` registry as cross-backend fuzz. Skips ops that
still need Python-side tensor values under tracing (``repeat`` scalar path,
``partition_softmax`` length ints).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from helpers import close, loaded_backends
from test_cross_backend_fuzz import FUZZ_OPS, _agree

tf = pytest.importorskip("tensorflow")

pytestmark = [
    pytest.mark.fuzz,
    pytest.mark.skipif(
        "tensorflow" not in loaded_backends,
        reason="TensorFlow not installed",
    ),
]

_settings = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

# Graph-incompatible today (Python ``int(tensor)`` / NumPy conversion under trace).
_TF_FUNCTION_SKIP = frozenset(
    {
        "fuzz_repeat",
        "fuzz_partition_softmax",
    }
)

_TF_FUNCTION_OPS = [(fn, samp) for fn, samp in FUZZ_OPS if fn.__name__ not in _TF_FUNCTION_SKIP]

_graph_fns: dict[int, Callable] = {}


def _tf_function_wrap(fn: Callable) -> Callable:
    key = id(fn)
    wrapped = _graph_fns.get(key)
    if wrapped is None:

        @tf.function(autograph=False, reduce_retracing=True)
        def wrapped(*args):
            return fn(*args)

        _graph_fns[key] = wrapped
    return wrapped


def _to_numpy_tf(out):
    backend = loaded_backends["tensorflow"]
    if isinstance(out, tuple):
        return tuple(_to_numpy_tf(o) for o in out)
    if hasattr(out, "numpy"):
        return np.asarray(out.numpy())
    return np.asarray(out)


def _run_eager(fn: Callable, args_np: tuple):
    backend = loaded_backends["tensorflow"]
    args = [
        backend.from_numpy(np.asarray(a)) if isinstance(a, np.ndarray) else a
        for a in args_np
    ]
    return _to_numpy_tf(fn(*args)), args


def _run_function(fn: Callable, args):
    return _to_numpy_tf(_tf_function_wrap(fn)(*args))


@st.composite
def _tf_function_example(draw):
    fn, sampler = draw(st.sampled_from(_TF_FUNCTION_OPS))
    args_np = sampler(draw)
    return fn, args_np


@_settings
@given(example=_tf_function_example())
def test_tensorflow_eager_matches_tf_function(example):
    """Symbolic / graph execution via ``tf.function`` agrees with eager TF."""
    fn, args_np = example
    assume("tensorflow" in loaded_backends)
    y_eager, args = _run_eager(fn, args_np)
    y_graph = _run_function(fn, args)
    _agree(fn, args_np, y_eager, y_graph)


def test_tf_function_skip_list_is_current():
    """Keep skip list in sync with registered fuzz ops."""
    names = {fn.__name__ for fn, _ in FUZZ_OPS}
    unknown = sorted(_TF_FUNCTION_SKIP - names)
    assert not unknown, f"stale tf.function skip entries: {unknown}"
    assert _TF_FUNCTION_OPS, "no ops left to fuzz under tf.function"
