"""MVP tests for :mod:`anytensor.hetero`."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor.tree as tree
from anytensor.hetero import (
    HeteroGraphsTuple,
    SendRecvTuple,
    graphs_tuple_as_send_recv,
    key_schema,
    schemas_equal,
)
from anytensor.jraph import GraphsTuple


def _np_graph(
    *,
    nodes,
    edges,
    senders,
    receivers,
    n_node,
    n_edge,
    globals_=None,
):
    return HeteroGraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals_,
    )


def _author_paper_graph(
    n_author: int,
    n_paper: int,
    writes_src,
    writes_dst,
    *,
    globals_=None,
):
    et = ("author", "writes", "paper")
    return _np_graph(
        nodes={
            "author": np.arange(n_author, dtype=np.float32).reshape(n_author, 1),
            "paper": np.arange(n_paper, dtype=np.float32).reshape(n_paper, 1) + 10,
        },
        edges={et: np.ones((len(writes_src), 1), dtype=np.float32)},
        senders={et: np.asarray(writes_src, dtype=np.int32)},
        receivers={et: np.asarray(writes_dst, dtype=np.int32)},
        n_node={
            "author": np.asarray([n_author], dtype=np.int32),
            "paper": np.asarray([n_paper], dtype=np.int32),
        },
        n_edge={et: np.asarray([len(writes_src)], dtype=np.int32)},
        globals_=globals_,
    )


def test_relation_view_aliases_storage():
    g = _author_paper_graph(3, 2, [0, 1], [0, 1])
    sr = g.relation_view(("author", "writes", "paper"))
    assert isinstance(sr, SendRecvTuple)
    assert sr.nodes_send is g.nodes["author"]
    assert sr.nodes_recv is g.nodes["paper"]
    assert sr.src_ntype == "author"
    assert sr.dst_ntype == "paper"


def test_iter_relations_skip_empty_and_reverse():
    g = _author_paper_graph(2, 2, [0], [1])
    rels = list(g.iter_relations())
    assert len(rels) == 1
    rev = list(g.iter_relations(reverse=True))
    assert len(rev) == 2
    assert rev[1].src_ntype == "paper"
    assert rev[1].senders is g.receivers[("author", "writes", "paper")]


def test_update_merges_partial_dicts():
    g = _author_paper_graph(2, 1, [0, 1], [0, 0])
    g2 = g.update(nodes={"author": np.zeros((2, 1), dtype=np.float32)})
    assert np.allclose(g2.nodes["author"], 0)
    assert g2.nodes["paper"] is g.nodes["paper"]
    g3 = g.update(globals=np.asarray([[1.0]], dtype=np.float32))
    assert float(g3.globals[0, 0]) == 1.0


def test_graphs_tuple_as_send_recv_aliases():
    gt = GraphsTuple(
        nodes=np.arange(3, dtype=np.float32).reshape(3, 1),
        edges=np.ones((2, 1), dtype=np.float32),
        senders=np.asarray([0, 1], dtype=np.int32),
        receivers=np.asarray([1, 2], dtype=np.int32),
        globals=None,
        n_node=np.asarray([3], dtype=np.int32),
        n_edge=np.asarray([2], dtype=np.int32),
    )
    sr = graphs_tuple_as_send_recv(gt)
    assert sr.nodes_send is gt.nodes
    assert sr.nodes_recv is gt.nodes


def test_batch_unbatch_same_schema_roundtrip():
    g1 = _author_paper_graph(2, 1, [0, 1], [0, 0], globals_=np.asarray([[1.0]]))
    g2 = _author_paper_graph(1, 2, [0], [1], globals_=np.asarray([[2.0]]))
    batched = tree.batch([g1, g2])
    assert isinstance(batched, HeteroGraphsTuple)
    assert list(batched.n_node["author"]) == [2, 1]
    assert list(batched.n_node["paper"]) == [1, 2]
    et = ("author", "writes", "paper")
    senders = np.asarray(batched.senders[et])
    assert set(senders.tolist()) == {0, 1, 2}

    parts = tree.unbatch(batched)
    assert len(parts) == 2
    assert schemas_equal(parts[0], g1)
    assert schemas_equal(parts[1], g2)
    assert np.allclose(parts[0].nodes["author"], g1.nodes["author"])
    assert np.allclose(parts[1].nodes["paper"], g2.nodes["paper"])
    assert np.allclose(parts[0].senders[et], g1.senders[et])
    assert np.allclose(parts[1].senders[et], g2.senders[et])


def test_batch_rejects_mismatched_keys():
    et_writes = ("author", "writes", "paper")
    et_cites = ("paper", "cites", "paper")
    g1 = _author_paper_graph(2, 1, [0], [0])
    g2 = _np_graph(
        nodes={"paper": np.asarray([[3.0], [4.0]], dtype=np.float32)},
        edges={et_cites: np.asarray([[1.0]], dtype=np.float32)},
        senders={et_cites: np.asarray([0], dtype=np.int32)},
        receivers={et_cites: np.asarray([1], dtype=np.int32)},
        n_node={"paper": np.asarray([2], dtype=np.int32)},
        n_edge={et_cites: np.asarray([1], dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="identical ntype/etype keys"):
        tree.batch([g1, g2])
    # User-aligned empties batch fine
    g2_aligned = _np_graph(
        nodes={
            "author": np.zeros((0, 1), dtype=np.float32),
            "paper": np.asarray([[3.0], [4.0]], dtype=np.float32),
        },
        edges={
            et_writes: np.zeros((0, 1), dtype=np.float32),
            et_cites: np.asarray([[1.0]], dtype=np.float32),
        },
        senders={
            et_writes: np.zeros((0,), dtype=np.int32),
            et_cites: np.asarray([0], dtype=np.int32),
        },
        receivers={
            et_writes: np.zeros((0,), dtype=np.int32),
            et_cites: np.asarray([1], dtype=np.int32),
        },
        n_node={
            "author": np.asarray([0], dtype=np.int32),
            "paper": np.asarray([2], dtype=np.int32),
        },
        n_edge={
            et_writes: np.asarray([0], dtype=np.int32),
            et_cites: np.asarray([1], dtype=np.int32),
        },
    )
    g1_aligned = _np_graph(
        nodes=g1.nodes,
        edges={
            et_writes: g1.edges[et_writes],
            et_cites: np.zeros((0, 1), dtype=np.float32),
        },
        senders={
            et_writes: g1.senders[et_writes],
            et_cites: np.zeros((0,), dtype=np.int32),
        },
        receivers={
            et_writes: g1.receivers[et_writes],
            et_cites: np.zeros((0,), dtype=np.int32),
        },
        n_node=g1.n_node,
        n_edge={
            et_writes: g1.n_edge[et_writes],
            et_cites: np.asarray([0], dtype=np.int32),
        },
    )
    batched = tree.batch([g1_aligned, g2_aligned])
    parts = tree.unbatch(batched)
    assert key_schema(parts[0]) == key_schema(g1_aligned)
    assert int(parts[0].n_edge[et_cites][0]) == 0
    assert int(parts[1].n_node["author"][0]) == 0


def test_batch_rejects_nodes_keys_mismatch_n_node():
    et = ("a", "r", "a")
    g = HeteroGraphsTuple(
        nodes={},  # missing 'a'
        edges={et: None},
        senders={et: np.asarray([0], dtype=np.int32)},
        receivers={et: np.asarray([0], dtype=np.int32)},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={et: np.asarray([1], dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="nodes keys"):
        tree.batch([g, g])


def test_key_schema_and_schemas_equal():
    g1 = _author_paper_graph(1, 1, [0], [0])
    g2 = _author_paper_graph(2, 2, [0, 1], [1, 0])
    assert schemas_equal(g1, g2)
    assert key_schema(g1)[0] == ("author", "paper")


def test_iter_nodes_edges():
    g = _author_paper_graph(1, 1, [], [])
    nodes = list(g.iter_nodes())
    assert {t for t, _, _ in nodes} == {"author", "paper"}
    edges = list(g.iter_edges(skip_empty=True))
    assert edges == []


def test_batch_axis_must_be_zero():
    g = _author_paper_graph(1, 1, [0], [0])
    with pytest.raises(ValueError, match="axis=0"):
        HeteroGraphsTuple.__tree_batch__([g], axis=1)
    with pytest.raises(ValueError, match="axis=0"):
        g.__tree_unbatch__(axis=1)


def test_batch_requires_nonempty_sequence():
    with pytest.raises(ValueError, match="at least one"):
        HeteroGraphsTuple.__tree_batch__([])


def test_n_graphs_from_globals_when_no_n_node():
    g = HeteroGraphsTuple(
        nodes={},
        edges={},
        senders={},
        receivers={},
        n_node={},
        n_edge={},
        globals=np.zeros((3, 1), dtype=np.float32),
    )
    assert g.n_graphs() == 3
    g2 = HeteroGraphsTuple(
        nodes={},
        edges={},
        senders={},
        receivers={},
        n_node={},
        n_edge={},
        globals=None,
    )
    assert g2.n_graphs() == 1
    g3 = HeteroGraphsTuple(
        nodes={},
        edges={},
        senders={},
        receivers={},
        n_node={},
        n_edge={},
        globals={},
    )
    assert g3.n_graphs() == 1


def test_iter_skip_empty_nodes_and_filter_etypes():
    et = ("author", "writes", "paper")
    g = _author_paper_graph(0, 1, [], [])
    g = g.update(
        nodes={"author": np.zeros((0, 1), dtype=np.float32)},
        n_node={"author": np.asarray([0], dtype=np.int32)},
    )
    assert [t for t, _, _ in g.iter_nodes(skip_empty=True)] == ["paper"]
    assert list(g.iter_edges(skip_empty=False))
    assert list(g.iter_relations(etypes=[et], skip_empty=True)) == []
    assert list(g.iter_relations(etypes=[("x", "y", "z")])) == []


def test_direct_unbatch_and_multi_graph_input():
    et = ("a", "r", "a")
    g_multi = HeteroGraphsTuple(
        nodes={"a": np.arange(5, dtype=np.float32).reshape(5, 1)},
        edges={et: np.ones((3, 1), dtype=np.float32)},
        senders={et: np.asarray([0, 1, 3], dtype=np.int32)},
        receivers={et: np.asarray([1, 2, 4], dtype=np.int32)},
        n_node={"a": np.asarray([3, 2], dtype=np.int32)},
        n_edge={et: np.asarray([2, 1], dtype=np.int32)},
        globals=np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    parts = g_multi.__tree_unbatch__()
    assert len(parts) == 2
    assert int(parts[0].n_node["a"][0]) == 3
    assert np.allclose(parts[1].senders[et], [0])


def test_unbatch_all_zero_edge_feat_branch():
    et = ("a", "r", "b")
    g1 = HeteroGraphsTuple(
        nodes={
            "a": np.ones((1, 1), dtype=np.float32),
            "b": np.ones((1, 1), dtype=np.float32),
        },
        edges={et: np.zeros((0, 1), dtype=np.float32)},
        senders={et: np.zeros((0,), dtype=np.int32)},
        receivers={et: np.zeros((0,), dtype=np.int32)},
        n_node={
            "a": np.asarray([1], dtype=np.int32),
            "b": np.asarray([1], dtype=np.int32),
        },
        n_edge={et: np.asarray([0], dtype=np.int32)},
    )
    g2 = HeteroGraphsTuple(
        nodes={
            "a": np.ones((1, 1), dtype=np.float32) * 2,
            "b": np.ones((1, 1), dtype=np.float32) * 2,
        },
        edges={et: np.zeros((0, 1), dtype=np.float32)},
        senders={et: np.zeros((0,), dtype=np.int32)},
        receivers={et: np.zeros((0,), dtype=np.int32)},
        n_node={
            "a": np.asarray([1], dtype=np.int32),
            "b": np.asarray([1], dtype=np.int32),
        },
        n_edge={et: np.asarray([0], dtype=np.int32)},
    )
    batched = tree.batch([g1, g2])
    parts = tree.unbatch(batched)
    assert len(parts) == 2
    assert int(parts[0].n_edge[et][0]) == 0


def test_zero_n_graphs_unbatch():
    empty = HeteroGraphsTuple(
        nodes={"a": np.zeros((0, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"a": np.zeros((0,), dtype=np.int32)},
        n_edge={},
    )
    assert empty.__tree_unbatch__() == []


def test_edge_map_key_mismatch():
    et = ("a", "r", "a")
    g = HeteroGraphsTuple(
        nodes={"a": np.ones((1, 1), dtype=np.float32)},
        edges={},  # missing et
        senders={et: np.asarray([0], dtype=np.int32)},
        receivers={et: np.asarray([0], dtype=np.int32)},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={et: np.asarray([1], dtype=np.int32)},
    )
    with pytest.raises(ValueError, match="edge maps must share keys"):
        tree.batch([g, g])