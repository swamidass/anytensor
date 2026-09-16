"""Parity: anytensor graph batch/unbatch vs DGL (Torch-only).

DGL is an optional dev dependency. Skipped when Torch or DGL is missing, or
when DGL cannot load (e.g. Graphbolt C++ lib mismatched to the installed
Torch). A Graphbolt stub is used only so core ``dgl.batch`` / ``heterograph``
still import on Torch builds without a matching Graphbolt wheel.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from anytensor import jraph as atj
from anytensor import tree
from anytensor.hetero import HeteroGraphsTuple


def _require_dgl():
    """Import ``dgl`` with Torch backend; skip if unavailable.

    Graphbolt's Torch-versioned C++ extension is stubbed before import so core
    ``batch`` / ``heterograph`` work even when the installed DGL wheel has no
    matching Graphbolt library for this Torch build.
    """
    if "dgl" in sys.modules and hasattr(sys.modules["dgl"], "batch"):
        return sys.modules["dgl"]
    sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))
    try:
        import dgl

        return dgl
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"dgl unavailable: {exc}")


dgl = _require_dgl()


def _np(x):
    return np.asarray(x.detach().cpu() if hasattr(x, "detach") else x)


def _homo_pair():
    g1 = atj.GraphsTuple(
        nodes=torch.arange(3.0).reshape(3, 1),
        edges=torch.ones(2, 1),
        senders=torch.tensor([0, 1], dtype=torch.int64),
        receivers=torch.tensor([1, 2], dtype=torch.int64),
        n_node=torch.tensor([3], dtype=torch.int64),
        n_edge=torch.tensor([2], dtype=torch.int64),
        globals=torch.tensor([[1.0]]),
    )
    g2 = atj.GraphsTuple(
        nodes=torch.arange(10.0, 12.0).reshape(2, 1),
        edges=torch.ones(1, 1) * 2,
        senders=torch.tensor([0], dtype=torch.int64),
        receivers=torch.tensor([1], dtype=torch.int64),
        n_node=torch.tensor([2], dtype=torch.int64),
        n_edge=torch.tensor([1], dtype=torch.int64),
        globals=torch.tensor([[2.0]]),
    )
    return g1, g2


def _at_to_dgl_homo(g: atj.GraphsTuple):
    dg = dgl.graph(
        (g.senders, g.receivers),
        num_nodes=int(g.n_node.sum().item()),
    )
    dg.ndata["h"] = g.nodes
    dg.edata["e"] = g.edges
    return dg


def _hetero_pair():
    et = ("author", "writes", "paper")
    g1 = HeteroGraphsTuple(
        nodes={
            "author": torch.arange(2.0).reshape(2, 1),
            "paper": torch.tensor([[10.0]]),
        },
        edges={et: torch.ones(2, 1)},
        senders={et: torch.tensor([0, 1], dtype=torch.int64)},
        receivers={et: torch.tensor([0, 0], dtype=torch.int64)},
        n_node={
            "author": torch.tensor([2], dtype=torch.int64),
            "paper": torch.tensor([1], dtype=torch.int64),
        },
        n_edge={et: torch.tensor([2], dtype=torch.int64)},
        globals=torch.tensor([[1.0]]),
    )
    g2 = HeteroGraphsTuple(
        nodes={
            "author": torch.tensor([[2.0]]),
            "paper": torch.tensor([[11.0], [12.0]]),
        },
        edges={et: torch.ones(1, 1) * 3},
        senders={et: torch.tensor([0], dtype=torch.int64)},
        receivers={et: torch.tensor([1], dtype=torch.int64)},
        n_node={
            "author": torch.tensor([1], dtype=torch.int64),
            "paper": torch.tensor([2], dtype=torch.int64),
        },
        n_edge={et: torch.tensor([1], dtype=torch.int64)},
        globals=torch.tensor([[2.0]]),
    )
    return g1, g2, et


def _at_to_dgl_hetero(g: HeteroGraphsTuple, et):
    dg = dgl.heterograph(
        {et: (g.senders[et], g.receivers[et])},
        num_nodes_dict={
            "author": int(g.n_node["author"].sum().item()),
            "paper": int(g.n_node["paper"].sum().item()),
        },
    )
    dg.nodes["author"].data["h"] = g.nodes["author"]
    dg.nodes["paper"].data["h"] = g.nodes["paper"]
    dg.edges[et[1]].data["e"] = g.edges[et]
    return dg


def test_dgl_homo_batch_unbatch_parity():
    g1, g2 = _homo_pair()
    at_b = atj.batch([g1, g2])
    dgl_b = dgl.batch([_at_to_dgl_homo(g1), _at_to_dgl_homo(g2)])

    np.testing.assert_array_equal(_np(at_b.n_node), _np(dgl_b.batch_num_nodes()))
    np.testing.assert_array_equal(_np(at_b.n_edge), _np(dgl_b.batch_num_edges()))
    src, dst = dgl_b.edges()
    np.testing.assert_array_equal(_np(at_b.senders), _np(src))
    np.testing.assert_array_equal(_np(at_b.receivers), _np(dst))
    np.testing.assert_allclose(_np(at_b.nodes), _np(dgl_b.ndata["h"]))
    np.testing.assert_allclose(_np(at_b.edges), _np(dgl_b.edata["e"]))

    at_parts = atj.unbatch(at_b)
    dgl_parts = dgl.unbatch(dgl_b)
    assert len(at_parts) == len(dgl_parts) == 2
    for ap, dp, orig in zip(at_parts, dgl_parts, (g1, g2)):
        np.testing.assert_allclose(_np(ap.nodes), _np(dp.ndata["h"]))
        np.testing.assert_allclose(_np(ap.nodes), _np(orig.nodes))
        src, dst = dp.edges()
        np.testing.assert_array_equal(_np(ap.senders), _np(src))
        np.testing.assert_array_equal(_np(ap.receivers), _np(dst))
        np.testing.assert_array_equal(_np(ap.senders), _np(orig.senders))


def test_dgl_hetero_batch_unbatch_parity():
    g1, g2, et = _hetero_pair()
    at_b = tree.batch([g1, g2])
    dgl_b = dgl.batch([_at_to_dgl_hetero(g1, et), _at_to_dgl_hetero(g2, et)])

    np.testing.assert_array_equal(
        _np(at_b.n_node["author"]), _np(dgl_b.batch_num_nodes("author"))
    )
    np.testing.assert_array_equal(
        _np(at_b.n_node["paper"]), _np(dgl_b.batch_num_nodes("paper"))
    )
    np.testing.assert_array_equal(
        _np(at_b.n_edge[et]), _np(dgl_b.batch_num_edges("writes"))
    )
    src, dst = dgl_b.edges(etype="writes")
    np.testing.assert_array_equal(_np(at_b.senders[et]), _np(src))
    np.testing.assert_array_equal(_np(at_b.receivers[et]), _np(dst))
    np.testing.assert_allclose(
        _np(at_b.nodes["author"]), _np(dgl_b.nodes["author"].data["h"])
    )
    np.testing.assert_allclose(
        _np(at_b.nodes["paper"]), _np(dgl_b.nodes["paper"].data["h"])
    )
    np.testing.assert_allclose(
        _np(at_b.edges[et]), _np(dgl_b.edges["writes"].data["e"])
    )

    at_parts = tree.unbatch(at_b)
    dgl_parts = dgl.unbatch(dgl_b)
    assert len(at_parts) == len(dgl_parts) == 2
    for ap, dp, orig in zip(at_parts, dgl_parts, (g1, g2)):
        np.testing.assert_allclose(
            _np(ap.nodes["author"]), _np(dp.nodes["author"].data["h"])
        )
        np.testing.assert_allclose(
            _np(ap.nodes["paper"]), _np(dp.nodes["paper"].data["h"])
        )
        src, dst = dp.edges(etype="writes")
        np.testing.assert_array_equal(_np(ap.senders[et]), _np(src))
        np.testing.assert_array_equal(_np(ap.receivers[et]), _np(dst))
        np.testing.assert_array_equal(_np(ap.senders[et]), _np(orig.senders[et]))
        np.testing.assert_array_equal(
            _np(ap.receivers[et]), _np(orig.receivers[et])
        )
