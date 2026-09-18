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
    multi_update_all,
    schemas_equal,
)
from anytensor.jraph import GraphsTuple
from helpers import BACKENDS, loaded_backends


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


def _to_backend_hetero(graph: HeteroGraphsTuple, backend):
    if backend.framework_name == "numpy":
        return graph

    def conv(x):
        if x is None:
            return None
        return backend.from_numpy(np.asarray(x))

    return HeteroGraphsTuple(
        nodes={k: conv(v) for k, v in graph.nodes.items()},
        edges={k: conv(v) for k, v in graph.edges.items()},
        senders={k: conv(v) for k, v in graph.senders.items()},
        receivers={k: conv(v) for k, v in graph.receivers.items()},
        n_node={k: conv(v) for k, v in graph.n_node.items()},
        n_edge={k: conv(v) for k, v in graph.n_edge.items()},
        globals=conv(graph.globals),
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


@pytest.mark.parametrize("name", BACKENDS)
def test_batch_unbatch_same_schema_all_backends(name):
    backend = loaded_backends[name]
    g1 = _author_paper_graph(2, 1, [0, 1], [0, 0], globals_=np.asarray([[1.0]]))
    g2 = _author_paper_graph(1, 2, [0], [1], globals_=np.asarray([[2.0]]))
    bg1, bg2 = _to_backend_hetero(g1, backend), _to_backend_hetero(g2, backend)
    batched = tree.batch([bg1, bg2])
    et = ("author", "writes", "paper")
    assert backend.is_appropriate_type(batched.senders[et])
    assert backend.is_appropriate_type(batched.n_node["author"])
    np.testing.assert_array_equal(
        backend.to_numpy(batched.n_node["author"]), [2, 1]
    )
    parts = tree.unbatch(batched)
    assert len(parts) == 2
    assert backend.is_appropriate_type(parts[0].nodes["author"])
    assert backend.is_appropriate_type(parts[1].senders[et])
    np.testing.assert_allclose(
        backend.to_numpy(parts[0].nodes["author"]), g1.nodes["author"]
    )
    np.testing.assert_allclose(
        backend.to_numpy(parts[1].nodes["paper"]), g2.nodes["paper"]
    )
    np.testing.assert_array_equal(
        backend.to_numpy(parts[0].senders[et]), g1.senders[et]
    )
    np.testing.assert_array_equal(
        backend.to_numpy(parts[1].senders[et]), g2.senders[et]
    )


def test_batch_already_batched_hetero_offsets():
    """Per-input ``sum(n_node)`` offsets when inputs are already multi-graph."""
    et = ("a", "r", "a")
    g = HeteroGraphsTuple(
        nodes={"a": np.arange(2, dtype=np.float32).reshape(2, 1)},
        edges={et: np.zeros((1, 1), dtype=np.float32)},
        senders={et: np.asarray([1], dtype=np.int32)},
        receivers={et: np.asarray([1], dtype=np.int32)},
        n_node={"a": np.asarray([1, 1], dtype=np.int32)},
        n_edge={et: np.asarray([0, 1], dtype=np.int32)},
    )
    batched = tree.batch([g, g])
    np.testing.assert_array_equal(batched.n_node["a"], [1, 1, 1, 1])
    np.testing.assert_array_equal(batched.senders[et], [1, 3])
    parts = tree.unbatch(batched)
    assert len(parts) == 4
    np.testing.assert_array_equal(parts[3].senders[et], [0])

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


def test_n_graphs_rejects_nonconcrete_batch_size(monkeypatch):
    import anytensor.hetero.graph as hg

    monkeypatch.setattr(hg, "_host_concrete_int", lambda _v: None)
    g = HeteroGraphsTuple(
        nodes={"a": np.ones((1, 1), dtype=np.float32)},
        edges={},
        senders={},
        receivers={},
        n_node={"a": np.asarray([1], dtype=np.int32)},
        n_edge={},
    )
    with pytest.raises(ValueError, match="concrete batch size"):
        g.n_graphs()

def _kernel_update_graph():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    return HeteroGraphsTuple(
        nodes={
            "author": np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
            "paper": np.asarray([[10.0], [20.0]], dtype=np.float32),
        },
        edges={
            writes: np.ones((3, 1), dtype=np.float32),
            cites: np.ones((1, 1), dtype=np.float32),
        },
        senders={
            writes: np.asarray([0, 1, 2], dtype=np.int32),
            cites: np.asarray([0], dtype=np.int32),
        },
        receivers={
            writes: np.asarray([0, 0, 1], dtype=np.int32),
            cites: np.asarray([1], dtype=np.int32),
        },
        n_node={
            "author": np.asarray([3], dtype=np.int32),
            "paper": np.asarray([2], dtype=np.int32),
        },
        n_edge={
            writes: np.asarray([3], dtype=np.int32),
            cites: np.asarray([1], dtype=np.int32),
        },
    )


def test_multi_update_all_copy_u_sum_values():
    g = _kernel_update_graph()
    out = multi_update_all(g, cross_reducer="sum", reduce="sum")
    # paper0 <- author0+author1 = 3; paper1 <- author2 + paper0 = 3+10 = 13
    np.testing.assert_allclose(out.nodes["paper"], [[3.0], [13.0]])
    np.testing.assert_allclose(out.nodes["author"], g.nodes["author"])


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("cross_reducer", ["sum", "max", "min", "mean"])
def test_multi_update_all_values_all_backends(name, cross_reducer):
    """Tensor values after kernel update match the NumPy reference on every backend."""
    backend = loaded_backends[name]
    g_np = _kernel_update_graph()
    ref = multi_update_all(g_np, cross_reducer=cross_reducer, reduce="sum")
    g_b = _to_backend_hetero(g_np, backend)
    out = multi_update_all(g_b, cross_reducer=cross_reducer, reduce="sum")
    assert backend.is_appropriate_type(out.nodes["paper"])
    np.testing.assert_allclose(
        backend.to_numpy(out.nodes["paper"]),
        np.asarray(ref.nodes["paper"]),
        rtol=1e-5,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        backend.to_numpy(out.nodes["author"]),
        np.asarray(ref.nodes["author"]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_multi_update_all_edge_cases_and_stack():
    import anytensor.hetero.message as hm

    g = _kernel_update_graph()
    writes = ("author", "writes", "paper")
    # single-relation path (len(parts)==1 in cross reduce)
    one = multi_update_all(g, etypes=[writes], cross_reducer="sum")
    np.testing.assert_allclose(one.nodes["paper"], [[3.0], [3.0]])
    # stack cross-reducer (DGL axis-1: n_dst × n_relations × ...)
    stacked = multi_update_all(g, cross_reducer="stack", reduce="sum")
    assert stacked.nodes["paper"].shape == (2, 2, 1)
    # etype_dict with (message_fn, reduce) tuple
    custom = multi_update_all(
        g,
        {writes: (hm.copy_u_message, "mean")},
        cross_reducer="sum",
    )
    assert custom.nodes["paper"].shape == (2, 1)
    # empty etype_dict returns graph unchanged
    assert multi_update_all(g, {}) is g
    # error paths
    with pytest.raises(KeyError):
        hm.relation_mailbox(g, ("x", "y", "z"))
    with pytest.raises(ValueError, match="unknown reduce"):
        hm.relation_mailbox(g, writes, reduce="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown cross_reducer"):
        hm._cross_reduce_leaves([np.ones(2), np.ones(2)], "nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        hm._cross_reduce_leaves([], "sum")
    with pytest.raises(ValueError, match="at least one"):
        hm._cross_reduce([], "sum")
    with pytest.raises(ValueError, match="no feature leaves"):
        hm._leading({})
    # single-part leaf cross-reduce shortcut
    np.testing.assert_array_equal(
        hm._cross_reduce_leaves([np.asarray([1.0, 2.0])], "sum"), [1.0, 2.0]
    )
    # None nodes take path
    assert hm._take_nodes(None, g.senders[writes]) is None


def _complex_float_hetero_np(*, with_empty_dsts: bool):
    """Float multi-dim u_mul_e fixture; optional paper missing an etype."""
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    if with_empty_dsts:
        # paper0 writes-only, paper1 both, paper2 cites-only
        author = [
            [0.7, -1.2, 3.5],
            [1.1, 0.25, -0.5],
            [2.4, 1.75, 0.125],
            [-0.3, 4.0, 1.5],
        ]
        w_writes = [[0.5], [1.5], [2.0], [0.25]]
        w_cites = [[1.0], [0.5], [3.0]]
        send_w, recv_w = [0, 1, 2, 3], [0, 0, 0, 1]
        send_c, recv_c = [0, 1, 2], [1, 2, 2]
        n_author, n_writes, n_cites = 4, 4, 3
    else:
        # Every paper receives both etypes; uneven degrees.
        author = [
            [0.7, -1.2, 3.5],
            [1.1, 0.25, -0.5],
            [2.4, 1.75, 0.125],
            [-0.3, 4.0, 1.5],
            [0.9, -0.8, 2.2],
        ]
        w_writes = [[0.5], [1.5], [2.0], [0.25], [1.25], [0.75]]
        w_cites = [[1.0], [0.5], [3.0], [0.8], [1.2]]
        send_w, recv_w = [0, 1, 2, 3, 4, 0], [0, 0, 0, 1, 1, 2]
        send_c, recv_c = [1, 0, 2, 0, 1], [0, 1, 1, 2, 2]
        n_author, n_writes, n_cites = 5, 6, 5
    paper = [
        [0.5, 0.5, 0.5],
        [1.25, -2.0, 0.75],
        [3.0, 0.1, -1.5],
    ]
    g = HeteroGraphsTuple(
        nodes={
            "author": np.asarray(author, dtype=np.float32),
            "paper": np.asarray(paper, dtype=np.float32),
        },
        edges={
            writes: np.asarray(w_writes, dtype=np.float32),
            cites: np.asarray(w_cites, dtype=np.float32),
        },
        senders={
            writes: np.asarray(send_w, dtype=np.int32),
            cites: np.asarray(send_c, dtype=np.int32),
        },
        receivers={
            writes: np.asarray(recv_w, dtype=np.int32),
            cites: np.asarray(recv_c, dtype=np.int32),
        },
        n_node={
            "author": np.asarray([n_author], dtype=np.int32),
            "paper": np.asarray([3], dtype=np.int32),
        },
        n_edge={
            writes: np.asarray([n_writes], dtype=np.int32),
            cites: np.asarray([n_cites], dtype=np.int32),
        },
    )
    return g, writes, cites


def _u_mul_e_message_np(src_nodes, dst_nodes, edges):
    del dst_nodes
    return src_nodes * edges


@pytest.mark.parametrize("name", BACKENDS)
@pytest.mark.parametrize("with_empty_dsts", [False, True])
@pytest.mark.parametrize("reduce", ["sum", "mean", "max", "min"])
@pytest.mark.parametrize("cross_reducer", ["sum", "mean", "max", "min"])
def test_multi_update_all_complex_float_values_all_backends(
    name, with_empty_dsts, reduce, cross_reducer
):
    """Non-trivial float u_mul_e: backends match NumPy (incl. empty dst = 0)."""
    backend = loaded_backends[name]
    g_np, writes, cites = _complex_float_hetero_np(with_empty_dsts=with_empty_dsts)
    etype_dict = {
        writes: (_u_mul_e_message_np, reduce),
        cites: (_u_mul_e_message_np, reduce),
    }
    ref = multi_update_all(g_np, etype_dict, cross_reducer=cross_reducer)
    g_b = _to_backend_hetero(g_np, backend)

    def _msg(src, dst, edges):
        del dst
        return src * edges

    out = multi_update_all(
        g_b,
        {writes: (_msg, reduce), cites: (_msg, reduce)},
        cross_reducer=cross_reducer,
    )
    np.testing.assert_allclose(
        backend.to_numpy(out.nodes["paper"]),
        np.asarray(ref.nodes["paper"]),
        rtol=1e-5,
        atol=1e-6,
    )
