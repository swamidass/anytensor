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


def test_einops_rearrange_function_is_not_scriptable():
    """Einops functions are the reason they use layers (``**axes_lengths``)."""
    from einops import rearrange

    class M(th.nn.Module):
        def forward(self, x: th.Tensor) -> th.Tensor:
            return rearrange(x, "a b -> b a")

    with pytest.raises((RuntimeError, th.jit.frontend.NotSupportedError)):
        th.jit.script(M())


def test_einops_rearrange_layer_scripts():
    """Control: einops' ``nn.Module`` layer pattern is scriptable."""
    from einops import rearrange
    from einops.layers.torch import Rearrange

    class M(th.nn.Module):
        def __init__(self):
            super().__init__()
            self.rearrange = Rearrange("a b -> b a")

        def forward(self, x: th.Tensor) -> th.Tensor:
            return self.rearrange(x)

    m = th.jit.script(M())
    x = th.randn(2, 3)
    y = m(x)
    assert tuple(y.shape) == (3, 2)
    assert th.allclose(y, rearrange(x, "a b -> b a"))


def test_einops_layer_pattern_scripts_segment_sum():
    """Same split as einops layers: ``nn.Module.forward`` + static Torch kernel.

    Not a public API — proves the pattern works on our kernels, matching eager.
    """
    from anytensor import torchscript as ts

    class SegmentSum(th.nn.Module):
        def __init__(self, num_segments: int):
            super().__init__()
            self.num_segments = num_segments

        def forward(self, x: th.Tensor, segment_ids: th.Tensor) -> th.Tensor:
            return ts.segment_sum(x, segment_ids, self.num_segments)

    m = th.jit.script(SegmentSum(5))
    x = th.randn(10)
    s = th.randint(0, 5, (10,))
    y = m(x, s)
    assert y.shape[0] == 5
    assert th.allclose(y, at.segment_sum(x, s, 5))


def test_einops_static_clone_scripts_segment_softmax():
    """A Torch-only clone of ``segment_softmax`` scripts and matches eager.

    The public helper cannot (Python dispatch). This is the einops
    ``apply_for_scriptable_torch`` move applied to a composite — proof that
    the pattern works here, not a new public kernel.
    """
    from anytensor import torchscript as ts

    def segment_softmax(
        x: th.Tensor, segment_ids: th.Tensor, num_segments: int
    ) -> th.Tensor:
        maxs = ts.segment_max(x, segment_ids, num_segments)
        maxs = maxs[segment_ids]
        exps = th.exp(x - maxs)
        norms = ts.segment_sum(exps, segment_ids, num_segments)
        return exps / norms[segment_ids]

    @th.jit.script
    def f(x: th.Tensor, s: th.Tensor, n: int) -> th.Tensor:
        return segment_softmax(x, s, n)

    x = th.randn(8)
    s = th.tensor([0, 0, 0, 1, 1, 2, 2, 2])
    y = f(x, s, 3)
    assert th.allclose(y, at.segment_softmax(x, s, 3), atol=1e-6)


def test_public_segment_softmax_is_not_scriptable():
    """Public ``at.segment_softmax`` still goes through Python dispatch."""
    with pytest.raises((RuntimeError, th.jit.frontend.NotSupportedError)):

        @th.jit.script
        def f(x: th.Tensor, s: th.Tensor, n: int) -> th.Tensor:
            return at.segment_softmax(x, s, n)


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
