"""Sybil collector for fenced Python examples in docs.

Kept separate from Hypothesis / typecheck setup so doc failures are obvious.
Registered from ``conftest.py`` (pytest only loads ``pytest_collect_file`` from
conftest / plugins, not from ordinary test modules).

``setup`` seeds the shared GAT helper and fixtures so individual examples stay
runnable under ``pytest --ff`` / node selection (Sybil does not re-run earlier
fences when collecting a single example).
"""

from __future__ import annotations

from pathlib import Path

from sybil import Sybil
from sybil.parsers.markdown import PythonCodeBlockParser, SkipParser
from sybil.sybil import SybilCollection

_DOCS = Path(__file__).resolve().parents[1] / "docs"
_PARSERS = [SkipParser(), PythonCodeBlockParser()]


def _setup(namespace: dict) -> None:
    import numpy as np
    import pytest

    import anytensor as at

    def neighbor_attention(messages, scores, dst_index, num_nodes: int):
        alpha = at.where(scores > 0, scores, scores * 0.2)
        alpha = at.segment_softmax(alpha, dst_index, num_nodes)
        weighted = messages * alpha[:, None]
        return at.segment_sum(weighted, dst_index, num_nodes)

    messages = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
        dtype=np.float32,
    )
    scores = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
    dst = np.array([0, 0, 1, 2], dtype=np.int64)
    num_nodes = 3
    out_np = neighbor_attention(messages, scores, dst, num_nodes)

    namespace.update(
        np=np,
        pytest=pytest,
        at=at,
        neighbor_attention=neighbor_attention,
        messages=messages,
        scores=scores,
        dst=dst,
        num_nodes=num_nodes,
        out_np=out_np,
    )

    try:
        import torch
    except ImportError:
        return

    namespace.update(
        torch=torch,
        messages_t=torch.as_tensor(messages),
        scores_t=torch.as_tensor(scores),
        dst_t=torch.as_tensor(dst),
    )


def _examples(path: Path, name: str) -> Sybil:
    return Sybil(
        parsers=_PARSERS,
        path=str(path),
        filenames=["examples.md"],
        setup=_setup,
        name=name,
    )


docs_sybil = SybilCollection(
    [
        _examples(_DOCS, "docs"),
        _examples(_DOCS / "tree", "docs_tree"),
        _examples(_DOCS / "jraph", "docs_jraph"),
    ]
)
