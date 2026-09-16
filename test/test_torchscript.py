"""TorchScript / trace checks + eager-vs-scripted fuzz.

Public ``at.segment_*`` stay multi-backend when eager; after
``enable_torchscript()``, ``torch.jit.script`` can follow library code that
calls them. Fuzz compares scripted outputs to eager Torch for parity.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest
from hypothesis import given, strategies as st

torch = pytest.importorskip("torch")
th = torch

import anytensor as at  # noqa: E402
from helpers import close  # noqa: E402
from test_cross_backend_fuzz import sample_segment  # noqa: E402

assert at.enable_torchscript(), "torch must be imported before TorchScript tests"

# Script once; reuse across fuzz examples (Hypothesis max_examples can be large).
@th.jit.script
def _script_segment_sum(x: th.Tensor, s: th.Tensor, n: int) -> th.Tensor:
    return at.segment_sum(x, s, n)


@th.jit.script
def _script_segment_min(x: th.Tensor, s: th.Tensor, n: int) -> th.Tensor:
    return at.segment_min(x, s, n)


@th.jit.script
def _script_segment_max(x: th.Tensor, s: th.Tensor, n: int) -> th.Tensor:
    return at.segment_max(x, s, n)


_SCRIPT = {
    "sum": (_script_segment_sum, at.segment_sum),
    "min": (_script_segment_min, at.segment_min),
    "max": (_script_segment_max, at.segment_max),
}


def test_jit_public_api():
    """End user scripts code that calls public ``at.segment_sum``."""
    x = th.randn(10)
    s = th.randint(0, 5, (10,))
    y = _script_segment_sum(x, s, 5)
    assert y.shape[0] == 5
    assert th.allclose(y, at.segment_sum(x, s, 5))


def test_jit_through_library_function():
    """Library uses ``at.segment_sum``; user scripts a wrapper around the library."""

    def library_reduce(x: th.Tensor, s: th.Tensor) -> th.Tensor:
        return at.segment_sum(x, s, 5)

    @th.jit.script
    def user_fn(x: th.Tensor, s: th.Tensor) -> th.Tensor:
        return library_reduce(x, s)

    x = th.randn(10)
    s = th.randint(0, 5, (10,))
    y = user_fn(x, s)
    assert th.allclose(y, library_reduce(x, s))


def test_eager_still_multi_backend_with_torchscript_enabled():
    """Enabling TorchScript must not break NumPy (or other) eager dispatch."""
    x = np.arange(5.0)
    ids = np.array([0, 0, 1, 1, 2])
    y = at.segment_sum(x, ids, 3)
    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(y, np.array([1.0, 5.0, 4.0]))


def test_trace():
    x = th.randn(10)
    s = th.randint(0, 5, (10,))

    @partial(th.jit.trace, example_inputs=(x, s))
    def f(x, s):
        return at.segment_sum(x, s, 5)

    y = f(x, s)
    assert y.shape[0] == 5
    assert th.allclose(y, at.segment_sum(x, s, 5))

    x = th.randn(15)
    s = th.randint(0, 5, (15,))
    y2 = f(x, s)
    assert y2.shape[0] == 5
    assert th.allclose(y2, at.segment_sum(x, s, 5))


# --- Fuzz: scripted vs eager Torch parity ------------------------------------

@st.composite
def _segment_example(draw):
    return sample_segment(draw)


@pytest.mark.fuzz
@given(op=st.sampled_from(["sum", "min", "max"]), sample=_segment_example())
def test_fuzz_torchscript_matches_eager(op, sample):
    """``torch.jit.script`` of ``at.segment_{sum,min,max}`` matches eager Torch."""
    x_np, seg_np, num_segments = sample
    x = th.as_tensor(x_np)
    s = th.as_tensor(seg_np)
    n = int(num_segments)
    scripted, eager = _SCRIPT[op]
    y_s = scripted(x, s, n)
    y_e = eager(x, s, n)
    assert tuple(y_s.shape) == tuple(y_e.shape)
    assert y_s.dtype == y_e.dtype
    assert close(y_s.detach().cpu().numpy(), y_e.detach().cpu().numpy(), equal_nan=True)
