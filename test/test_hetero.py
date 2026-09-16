"""MVP tests for :mod:`anytensor.hetero`."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor.tree as tree
from anytensor.hetero import (
    HeteroBatch,
    HeteroGraphsTuple,
    SendRecvTuple,
    canonicalize_schema,
    graphs_tuple_as_send_recv,
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
    assert isinstance(batched, HeteroBatch)
    assert batched.filled_ntypes == (frozenset(), frozenset())
    assert batched.filled_etypes == (frozenset(), frozenset())
    assert batched.input_n_graphs == (1, 1)
    # Stacked sizes
    assert list(batched.graph.n_node["author"]) == [2, 1]
    assert list(batched.graph.n_node["paper"]) == [1, 2]
    # Offsets applied to second graph's author sender (0 -> 2)
    et = ("author", "writes", "paper")
    senders = np.asarray(batched.graph.senders[et])
    assert set(senders.tolist()) == {0, 1, 2}

    parts = tree.unbatch(batched)
    assert len(parts) == 2
    assert schemas_equal(parts[0], g1)
    assert schemas_equal(parts[1], g2)
    assert np.allclose(parts[0].nodes["author"], g1.nodes["author"])
    assert np.allclose(parts[1].nodes["paper"], g2.nodes["paper"])
    assert np.allclose(parts[0].senders[et], g1.senders[et])
    assert np.allclose(parts[1].senders[et], g2.senders[et])


def test_batch_fills_missing_keys_policy_c_roundtrip():
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
    batched = tree.batch([g1, g2])
    assert isinstance(batched, HeteroBatch)
    assert batched.filled_ntypes[0] == frozenset()
    assert "author" in batched.filled_ntypes[1]
    assert et_cites in batched.filled_etypes[0]
    assert et_writes in batched.filled_etypes[1]
    # Union schema on the stacked graph
    assert set(batched.graph.ntypes()) == {"author", "paper"}
    assert set(batched.graph.canonical_etypes()) == {et_writes, et_cites}

    parts = tree.unbatch(batched)
    assert "author" in parts[0].n_node
    assert "author" not in parts[1].n_node
    assert et_writes in parts[0].n_edge
    assert et_writes not in parts[1].n_edge
    assert et_cites not in parts[0].n_edge
    assert et_cites in parts[1].n_edge
    assert schemas_equal(parts[0], g1)
    assert schemas_equal(parts[1], g2)


def test_canonicalize_and_schemas_equal_empty_means_absent():
    et = ("a", "r", "b")
    g = _np_graph(
        nodes={"a": np.zeros((0, 1), dtype=np.float32), "b": np.ones((1, 1), dtype=np.float32)},
        edges={et: np.zeros((0, 1), dtype=np.float32)},
        senders={et: np.zeros((0,), dtype=np.int32)},
        receivers={et: np.zeros((0,), dtype=np.int32)},
        n_node={"a": np.asarray([0], dtype=np.int32), "b": np.asarray([1], dtype=np.int32)},
        n_edge={et: np.asarray([0], dtype=np.int32)},
    )
    ntypes, etypes = canonicalize_schema(g)
    assert ntypes == ("b",)
    assert etypes == ()
    g2 = _np_graph(
        nodes={"b": np.ones((1, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"b": np.asarray([1], dtype=np.int32)},
        n_edge={},
    )
    assert schemas_equal(g, g2)


def test_iter_nodes_edges():
    g = _author_paper_graph(1, 1, [], [])
    # empty writes relation still present
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
    b = tree.batch([g])
    with pytest.raises(ValueError, match="axis=0"):
        b.__tree_unbatch__(axis=1)


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


def test_iter_skip_empty_nodes_and_filter_etypes():
    et = ("author", "writes", "paper")
    g = _author_paper_graph(0, 1, [], [])
    # Force empty author count
    g = g.update(
        nodes={"author": np.zeros((0, 1), dtype=np.float32)},
        n_node={"author": np.asarray([0], dtype=np.int32)},
    )
    assert [t for t, _, _ in g.iter_nodes(skip_empty=True)] == ["paper"]
    assert list(g.iter_edges(skip_empty=False))  # yields empty writes
    assert list(g.iter_relations(etypes=[et], skip_empty=True)) == []
    assert list(g.iter_relations(etypes=[("x", "y", "z")])) == []


def test_direct_unbatch_on_stacked_hetero():
    g1 = _author_paper_graph(2, 1, [0, 1], [0, 0])
    g2 = _author_paper_graph(1, 2, [0], [1])
    batched = tree.batch([g1, g2])
    # Unbatch the inner stacked graph (no filler strip)
    parts = batched.graph.__tree_unbatch__()
    assert len(parts) == 2
    assert "author" in parts[0].n_node


def test_batch_with_none_features_and_multi_node_chunk():
    et = ("a", "r", "a")
    # Two atomic graphs inside one HeteroGraphsTuple (n_graphs=2)
    g_multi = HeteroGraphsTuple(
        nodes={"a": np.arange(5, dtype=np.float32).reshape(5, 1)},
        edges={et: np.ones((3, 1), dtype=np.float32)},
        senders={et: np.asarray([0, 1, 3], dtype=np.int32)},
        receivers={et: np.asarray([1, 2, 4], dtype=np.int32)},
        n_node={"a": np.asarray([3, 2], dtype=np.int32)},
        n_edge={et: np.asarray([2, 1], dtype=np.int32)},
        globals=np.asarray([[1.0], [2.0]], dtype=np.float32),
    )
    g_other = HeteroGraphsTuple(
        nodes={"b": np.asarray([[9.0]], dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"b": np.asarray([1], dtype=np.int32)},
        n_edge={},
        globals=np.asarray([[3.0]], dtype=np.float32),
    )
    batched = tree.batch([g_multi, g_other])
    parts = tree.unbatch(batched)
    assert len(parts) == 3
    assert schemas_equal(parts[0], HeteroGraphsTuple(
        nodes={"a": g_multi.nodes["a"][:3]},
        edges={et: g_multi.edges[et][:2]},
        senders={et: np.asarray([0, 1], dtype=np.int32)},
        receivers={et: np.asarray([1, 2], dtype=np.int32)},
        n_node={"a": np.asarray([3], dtype=np.int32)},
        n_edge={et: np.asarray([2], dtype=np.int32)},
        globals=np.asarray([[1.0]], dtype=np.float32),
    ))


def test_offset_none_senders():
    from anytensor.hetero.graph import _offset_index

    assert _offset_index(None, 3) is None


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


def test_prototype_index_error():
    from anytensor.hetero.graph import _prototype_index

    with pytest.raises(ValueError, match="cannot infer index"):
        _prototype_index([], ("a", "r", "b"), "senders")


def test_none_node_features_in_batch():
    et = ("a", "r", "a")
    g1 = HeteroGraphsTuple(
        nodes={"a": None},
        edges={et: None},
        senders={et: np.asarray([0], dtype=np.int32)},
        receivers={et: np.asarray([0], dtype=np.int32)},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={et: np.asarray([1], dtype=np.int32)},
    )
    g2 = HeteroGraphsTuple(
        nodes={"a": None},
        edges={et: None},
        senders={et: np.asarray([0], dtype=np.int32)},
        receivers={et: np.asarray([0], dtype=np.int32)},
        n_node={"a": np.asarray([2], dtype=np.int32)},
        n_edge={et: np.asarray([1], dtype=np.int32)},
    )
    batched = tree.batch([g1, g2])
    parts = tree.unbatch(batched)
    assert parts[0].nodes["a"] is None
    assert parts[1].nodes["a"] is None


def test_empty_globals_nest_and_zero_n_graphs():
    g = HeteroGraphsTuple(
        nodes={},
        edges={},
        senders={},
        receivers={},
        n_node={},
        n_edge={},
        globals={},  # empty nest → no leaves
    )
    assert g.n_graphs() == 1
    empty = HeteroGraphsTuple(
        nodes={"a": np.zeros((0, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"a": np.zeros((0,), dtype=np.int32)},
        n_edge={},
    )
    assert empty.__tree_unbatch__() == []


def test_prototype_fallback_empty_then_index_from_n_node():
    from anytensor.hetero.graph import (
        _prototype_edge,
        _prototype_index,
        _prototype_node,
    )

    et_new = ("x", "r", "y")
    # Graph with empty x features only (sum n_node == 0) still provides prototype.
    g_empty_x = HeteroGraphsTuple(
        nodes={"x": np.zeros((0, 2), dtype=np.float32), "a": np.ones((1, 1), dtype=np.float32)},
        edges={et_new: np.zeros((0, 3), dtype=np.float32)},
        senders={},  # no senders for et_new yet
        receivers={},
        n_node={"x": np.asarray([0], dtype=np.int32), "a": np.asarray([1], dtype=np.int32)},
        n_edge={et_new: np.asarray([0], dtype=np.int32)},
    )
    g_a = HeteroGraphsTuple(
        nodes={"a": np.ones((1, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={},
    )
    assert _prototype_node([g_a, g_empty_x], "x") is not None
    assert _prototype_edge([g_a, g_empty_x], et_new) is not None
    # No sender arrays at all for et_new → fall back to n_node
    idx = _prototype_index([g_a], et_new, "senders")
    assert idx is g_a.n_node["a"]
    # Fall back via existing senders values when field/etype missing
    g_with_senders = HeteroGraphsTuple(
        nodes={"a": np.ones((1, 1), dtype=np.float32)},
        edges={},
        senders={("a", "loop", "a"): np.asarray([0], dtype=np.int32)},
        receivers={("a", "loop", "a"): np.asarray([0], dtype=np.int32)},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={},
    )
    idx2 = _prototype_index([g_with_senders], et_new, "receivers")
    assert np.asarray(idx2).shape == (1,)
    # First graph has nothing usable; continue to next
    g_blank = HeteroGraphsTuple(
        nodes={},
        edges={},
        senders={"k": None},
        receivers={},
        n_node={},
        n_edge={},
    )
    idx3 = _prototype_index([g_blank, g_a], et_new, "senders")
    assert idx3 is g_a.n_node["a"]


def test_batch_fill_from_empty_only_peer():
    """Missing type filled using a peer that only has an empty pool for that type."""
    et = ("a", "r", "b")
    g1 = HeteroGraphsTuple(
        nodes={"a": np.ones((1, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={},
    )
    g2 = HeteroGraphsTuple(
        nodes={
            "a": np.ones((1, 1), dtype=np.float32),
            "b": np.zeros((0, 1), dtype=np.float32),
        },
        edges={et: np.zeros((0, 1), dtype=np.float32)},
        senders={et: np.zeros((0,), dtype=np.int32)},
        receivers={et: np.zeros((0,), dtype=np.int32)},
        n_node={
            "a": np.asarray([1], dtype=np.int32),
            "b": np.asarray([0], dtype=np.int32),
        },
        n_edge={et: np.asarray([0], dtype=np.int32)},
    )
    batched = tree.batch([g1, g2])
    assert "b" in batched.graph.n_node
    parts = tree.unbatch(batched)
    assert "b" not in parts[0].n_node
    assert "b" in parts[1].n_node
