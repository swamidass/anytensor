"""``torch.compile`` coverage of the public API.

Portable AnyTensor is meant to run under Dynamo. These tests lock that in:

- ``fullgraph=False`` (the supported default) for every public tensor op
- ``fullgraph=True`` for the same set, except ops that are data-dependent

``backend="aot_eager"`` matches the docs examples (inductor/triton has
SIGSEGV'd on GitHub-hosted runners). Skipped when ``CI`` is set for the same
reason as the Sybil compile fences.

Runtime jaxtyping is disabled: compiled tensors can carry dynamic shapes that
fail ``SegmentIds`` even when numerics match.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import anytensor as at  # noqa: E402
from helpers import close  # noqa: E402

pytestmark = [
    pytest.mark.skipif(
        bool(os.environ.get("CI")),
        reason="torch.compile disabled on CI runners (dynamo/triton SIGSEGV)",
    ),
]

_BACKEND = "aot_eager"


@pytest.fixture(autouse=True)
def _disable_jaxtyping_for_compile():
    try:
        from jaxtyping import config
    except ImportError:  # pragma: no cover
        yield
        return
    prev = config.jaxtyping_disable
    config.update("jaxtyping_disable", True)
    try:
        yield
    finally:
        config.update("jaxtyping_disable", prev)


@dataclass(frozen=True)
class Bundle:
    x: Any
    y: Any
    c: Any
    ids: Any
    idx: Any
    x23: Any
    x2: Any
    y2: Any
    parts: Any


def _bundle() -> Bundle:
    x = torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32)
    return Bundle(
        x=x,
        y=torch.tensor([1.0, 0.0, 2.0], dtype=torch.float32),
        c=torch.tensor([True, False, True]),
        ids=torch.tensor([0, 0, 1], dtype=torch.int64),
        idx=torch.tensor([0, 2], dtype=torch.int64),
        x23=torch.arange(6, dtype=torch.float32).reshape(2, 3),
        x2=x.reshape(3, 1),
        y2=torch.tensor([[1.0], [0.5], [2.0]], dtype=torch.float32),
        parts=torch.tensor([2, 1], dtype=torch.int64),
    )


# Public tensor/metadata ops → (callable on a Bundle,). Names must match ``at.__all__``.
def _cases() -> dict[str, Callable[[Bundle], Any]]:
    return {
        "exp": lambda t: at.exp(t.x),
        "log": lambda t: at.log(t.x),
        "sqrt": lambda t: at.sqrt(t.x),
        "rsqrt": lambda t: at.rsqrt(t.x),
        "sum": lambda t: at.sum(t.x),
        "min": lambda t: at.min(t.x),
        "max": lambda t: at.max(t.x),
        "mean": lambda t: at.mean(t.x),
        "prod": lambda t: at.prod(t.x),
        "cumsum": lambda t: at.cumsum(t.x),
        "maximum": lambda t: at.maximum(t.x, t.y),
        "minimum": lambda t: at.minimum(t.x, t.y),
        "where": lambda t: at.where(t.c, t.x, t.y),
        "take": lambda t: at.take(t.x, t.idx),
        "reshape": lambda t: at.reshape(t.x, (3, 1)),
        "transpose": lambda t: at.transpose(t.x23, (1, 0)),
        "concatenate": lambda t: at.concatenate([t.x, t.y], axis=0),
        "stack": lambda t: at.stack([t.x, t.y], axis=0),
        "clip": lambda t: at.clip(t.x, 0.0, 2.0),
        "astype": lambda t: at.astype(t.x, torch.float64),
        "cast": lambda t: at.cast(t.x, torch.float64),
        "zeros_like": lambda t: at.zeros_like(t.x),
        "ones_like": lambda t: at.ones_like(t.x),
        "full_like": lambda t: at.full_like(t.x, 3.0),
        "zeros": lambda t: at.zeros((3,), like=t.x),
        "ones": lambda t: at.ones((3,), like=t.x),
        "full": lambda t: at.full((3,), 4.0, like=t.x),
        "arange": lambda t: at.arange(3, like=t.x),
        "matmul": lambda t: at.matmul(t.x2.T, t.y2),
        "repeat": lambda t: at.repeat(t.x, 2),
        "is_nan": lambda t: at.is_nan(t.x),
        "is_finite": lambda t: at.is_finite(t.x),
        "is_inf": lambda t: at.is_inf(t.x),
        "isnan": lambda t: at.isnan(t.x),
        "isfinite": lambda t: at.isfinite(t.x),
        "isinf": lambda t: at.isinf(t.x),
        "fill_nan": lambda t: at.fill_nan(t.x, 0.0),
        "nan_fill": lambda t: at.nan_fill(t.x, 0.0),
        "fill_nan_mask": lambda t: at.fill_nan_mask(t.x, 0.0),
        "nan_fill_mask": lambda t: at.nan_fill_mask(t.x, 0.0),
        "nan_to_num": lambda t: at.nan_to_num(t.x),
        "equal_nan": lambda t: at.equal_nan(t.x, t.y),
        "segment_sum": lambda t: at.segment_sum(t.x, t.ids, 2),
        "segment_min": lambda t: at.segment_min(t.x, t.ids, 2),
        "segment_max": lambda t: at.segment_max(t.x, t.ids, 2),
        "segment_mean": lambda t: at.segment_mean(t.x, t.ids, 2),
        "segment_count": lambda t: at.segment_count(t.ids, 2),
        "segment_variance": lambda t: at.segment_variance(t.x, t.ids, 2),
        "segment_normalize": lambda t: at.segment_normalize(t.x, t.ids, 2),
        "segment_softmax": lambda t: at.segment_softmax(t.x, t.ids, 2),
        "segment_min_or_constant": lambda t: at.segment_min_or_constant(t.x, t.ids, 2, 0.0),
        "segment_max_or_constant": lambda t: at.segment_max_or_constant(t.x, t.ids, 2, 0.0),
        "partition_softmax": lambda t: at.partition_softmax(t.x, t.parts, 3),
        "rearrange": lambda t: at.rearrange(t.x23, "a b -> b a"),
        "reduce": lambda t: at.reduce(t.x23, "a b -> a", "sum"),
        "einsum": lambda t: at.einsum(t.x23, t.x23.T, "i j, j k -> i k"),
        "pack": lambda t: at.pack([t.x, t.y], "i *")[0],
        "unpack": lambda t: at.unpack(t.x, [[1], [1], [1]], "*")[0],
        "shape": lambda t: at.shape(t.x),
        "inf": lambda t: at.inf(t.x),
        "ninf": lambda t: at.ninf(t.x),
        "nan": lambda t: at.nan(t.x),
        "pi": lambda t: at.pi(t.x),
        "e": lambda t: at.e(t.x),
        "dtype": lambda t: at.dtype("float32", t.x),
        "finfo": lambda t: at.finfo(t.x).max,
        "iinfo": lambda t: at.iinfo(t.ids).max,
    }


# Infrastructure / typing — not tensor graphs. Good reasons to omit.
_NON_COMPILE_PUBLIC = frozenset(
    {
        "backends",
        "get_backend",
        "promote",
        "promote_scalars",
        "promote_options",
        "align_arrays",
        "newaxis",
        "__version__",
        "empty_segment_identity",
        "enable_torchscript",
        "enable_typecheck",
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
    }
)

# Data-dependent Python control flow: compiles with graph breaks, not as one graph.
_FULLGRAPH_TRUE_XFAIL = {
    "partition_softmax": "repeat() of tensor partition lengths is data-dependent",
}


def _to_numpy(out):
    if isinstance(out, tuple):
        return tuple(_to_numpy(o) for o in out)
    if torch.is_tensor(out):
        return out.detach().cpu().numpy()
    if isinstance(out, (np.ndarray, np.generic)) or np.isscalar(out):
        return np.asarray(out)
    return np.asarray(out)


def _compile_and_match(fn: Callable[[Bundle], Any], *, fullgraph: bool) -> None:
    bundle = _bundle()
    torch._dynamo.reset()
    compiled = torch.compile(fn, fullgraph=fullgraph, backend=_BACKEND)
    y_c = compiled(bundle)
    y_e = fn(bundle)
    close(_to_numpy(y_e), _to_numpy(y_c), equal_nan=True)


_CASE_NAMES = sorted(_cases())


@pytest.mark.parametrize("name", _CASE_NAMES)
def test_compile_fullgraph_false_matches_eager(name):
    """Every public tensor op compiles with graph breaks allowed and matches eager."""
    _compile_and_match(_cases()[name], fullgraph=False)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            n,
            marks=pytest.mark.xfail(reason=_FULLGRAPH_TRUE_XFAIL[n], strict=True),
        )
        if n in _FULLGRAPH_TRUE_XFAIL
        else n
        for n in _CASE_NAMES
    ],
)
def test_compile_fullgraph_true_matches_eager(name):
    """Recent PyTorch can fuse most portable helpers; exceptions are listed."""
    _compile_and_match(_cases()[name], fullgraph=True)


def test_all_public_ops_have_compile_case():
    """Public names are either a compile case or an explicit infra skip."""
    covered = set(_cases())
    required = set(at.__all__) - _NON_COMPILE_PUBLIC
    missing = sorted(required - covered)
    extra = sorted(covered - required)
    unknown_skip = sorted(_NON_COMPILE_PUBLIC - set(at.__all__))
    assert not missing, f"public ops missing compile cases: {missing}"
    assert not extra, f"compile cases without public name: {extra}"
    assert not unknown_skip, f"stale compile skip entries: {unknown_skip}"


def test_neighbor_attention_fullgraph_true():
    """The docs GAT helper compiles as one graph on recent PyTorch."""

    def neighbor_attention(messages, scores, dst_index, num_nodes: int):
        alpha = at.where(scores > 0, scores, scores * 0.2)
        alpha = at.segment_softmax(alpha, dst_index, num_nodes)
        weighted = messages * alpha[:, None]
        return at.segment_sum(weighted, dst_index, num_nodes)

    messages = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=torch.float32
    )
    scores = torch.tensor([1.0, 1.0, 0.5, 2.0], dtype=torch.float32)
    dst = torch.tensor([0, 0, 1, 2], dtype=torch.int64)
    torch._dynamo.reset()
    compiled = torch.compile(neighbor_attention, fullgraph=True, backend=_BACKEND)
    y_c = compiled(messages, scores, dst, 3)
    y_e = neighbor_attention(messages, scores, dst, 3)
    close(_to_numpy(y_e), _to_numpy(y_c), equal_nan=True)
