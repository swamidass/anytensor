"""Tests for :mod:`anytensor.jraph` (GraphsTuple, batching, GraphNetwork)."""

from __future__ import annotations

import numpy as np
import pytest

from anytensor import jraph as atj
from helpers import BACKENDS, loaded_backends


def _nest(array):
    return {"a": array, "b": [np.ones_like(array), {"c": np.zeros_like(array)}]}


def _toy_graphs():
    g1 = atj.GraphsTuple(
        nodes=np.arange(6.0).reshape(3, 2),
        edges=np.arange(10.0).reshape(5, 2),
        senders=np.array([0, 0, 1, 1, 2]),
        receivers=np.array([1, 2, 0, 2, 1]),
        n_node=np.array([3]),
        n_edge=np.array([5]),
        globals=np.array([[1.0, 2.0]]),
    )
    g2 = atj.GraphsTuple(
        nodes=np.arange(6.0, 16.0).reshape(5, 2),
        edges=np.arange(10.0, 20.0).reshape(5, 2),
        senders=np.array([0, 1, 2, 3, 4]),
        receivers=np.array([1, 2, 0, 4, 3]),
        n_node=np.array([5]),
        n_edge=np.array([5]),
        globals=np.array([[3.0, 4.0]]),
    )
    return g1, g2


def _to_backend(graph, backend):
    if backend.framework_name == "numpy":
        return graph

    def conv(x):
        if x is None:
            return None
        return backend.from_numpy(np.asarray(x))

    def mapf(feat):
        if feat is None:
            return None
        from anytensor import tree

        return tree.map(conv, feat)

    return atj.GraphsTuple(
        nodes=mapf(graph.nodes),
        edges=mapf(graph.edges),
        senders=conv(graph.senders),
        receivers=conv(graph.receivers),
        globals=mapf(graph.globals),
        n_node=conv(graph.n_node),
        n_edge=conv(graph.n_edge),
    )


def _np(x):
    return np.asarray(x)


def test_batch_unbatch_roundtrip():
    g1, g2 = _toy_graphs()
    batched = atj.batch([g1, g2])
    assert batched.nodes.shape == (8, 2)
    assert batched.edges.shape == (10, 2)
    np.testing.assert_array_equal(_np(batched.n_node), [3, 5])
    np.testing.assert_array_equal(_np(batched.senders)[:5], g1.senders)
    np.testing.assert_array_equal(_np(batched.senders)[5:], g2.senders + 3)
    parts = atj.unbatch(batched)
    assert len(parts) == 2
    np.testing.assert_allclose(_np(parts[0].nodes), g1.nodes)
    np.testing.assert_allclose(_np(parts[1].nodes), g2.nodes)
    np.testing.assert_array_equal(_np(parts[1].senders), g2.senders)


def test_batch_already_batched_graphs_matches_jraph_offsets():
    """Offsets use per-GraphsTuple ``sum(n_node)``, not flattened segments."""
    g = atj.GraphsTuple(
        nodes=np.zeros((2, 3), dtype=np.float32),
        edges=np.zeros((1, 3), dtype=np.float32),
        receivers=np.array([1], dtype=np.int32),
        senders=np.array([1], dtype=np.int32),
        globals=np.zeros((2, 2), dtype=np.float32),
        n_node=np.array([1, 1], dtype=np.int32),
        n_edge=np.array([0, 1], dtype=np.int32),
    )
    batched = atj.batch([g, g])
    np.testing.assert_array_equal(_np(batched.n_node), [1, 1, 1, 1])
    np.testing.assert_array_equal(_np(batched.senders), [1, 3])
    np.testing.assert_array_equal(_np(batched.receivers), [1, 3])
    parts = atj.unbatch(batched)
    assert len(parts) == 4
    np.testing.assert_array_equal(_np(parts[1].senders), [0])
    np.testing.assert_array_equal(_np(parts[3].senders), [0])



@pytest.mark.parametrize("name", BACKENDS)
def test_batch_unbatch_roundtrip_all_backends(name):
    backend = loaded_backends[name]
    g1, g2 = _toy_graphs()
    bg1, bg2 = _to_backend(g1, backend), _to_backend(g2, backend)
    batched = atj.batch([bg1, bg2])
    assert backend.is_appropriate_type(batched.senders)
    assert backend.is_appropriate_type(batched.n_node)
    np.testing.assert_array_equal(backend.to_numpy(batched.n_node), [3, 5])
    np.testing.assert_array_equal(
        backend.to_numpy(batched.senders)[5:], g2.senders + 3
    )
    parts = atj.unbatch(batched)
    assert len(parts) == 2
    assert backend.is_appropriate_type(parts[0].nodes)
    assert backend.is_appropriate_type(parts[1].senders)
    np.testing.assert_allclose(backend.to_numpy(parts[0].nodes), g1.nodes)
    np.testing.assert_allclose(backend.to_numpy(parts[1].nodes), g2.nodes)
    np.testing.assert_array_equal(backend.to_numpy(parts[1].senders), g2.senders)


def test_batch_nested_features():
    g1, g2 = _toy_graphs()
    g1 = g1._replace(nodes=_nest(g1.nodes), edges=_nest(g1.edges), globals=_nest(g1.globals))
    g2 = g2._replace(nodes=_nest(g2.nodes), edges=_nest(g2.edges), globals=_nest(g2.globals))
    batched = atj.batch([g1, g2])
    assert batched.nodes["a"].shape == (8, 2)
    parts = atj.unbatch(batched)
    np.testing.assert_allclose(_np(parts[0].nodes["a"]), _np(g1.nodes["a"]))


def test_pad_unpad_and_masks():
    g1, _ = _toy_graphs()
    padded = atj.pad_with_graphs(g1, n_node=6, n_edge=8, n_graph=3)
    assert int(np.asarray(padded.n_node).sum()) == 6
    assert padded.senders.shape[0] == 8
    assert padded.n_node.shape[0] == 3
    node_mask = atj.get_node_padding_mask(padded)
    assert int(np.asarray(node_mask).sum()) == 3
    edge_mask = atj.get_edge_padding_mask(padded)
    assert int(np.asarray(edge_mask).sum()) == 5
    graph_mask = atj.get_graph_padding_mask(padded)
    assert list(np.asarray(graph_mask)) == [True, False, False]
    restored = atj.unpad_with_graphs(padded)
    np.testing.assert_allclose(_np(restored.nodes), g1.nodes)
    np.testing.assert_array_equal(_np(restored.senders), g1.senders)


def test_pad_too_small_and_n_graph():
    g1, _ = _toy_graphs()
    with pytest.raises(ValueError, match="n_graph"):
        atj.pad_with_graphs(g1, 10, 10, n_graph=1)
    with pytest.raises(RuntimeError, match="too large"):
        atj.pad_with_graphs(g1, n_node=2, n_edge=8, n_graph=2)


def test_graph_network_identity():
    g1, g2 = _toy_graphs()
    graph = atj.batch([g1, g2])
    net = atj.GraphNetwork(
        update_edge_fn=lambda e, s, r, g: e,
        update_node_fn=lambda n, s, r, g: n,
        update_global_fn=lambda n, e, g: g,
    )
    out = net(graph)
    np.testing.assert_allclose(_np(out.nodes), _np(graph.nodes))
    np.testing.assert_allclose(_np(out.edges), _np(graph.edges))
    np.testing.assert_allclose(_np(out.globals), _np(graph.globals))


def test_graph_network_none_globals_and_disabled_updates():
    g1, _ = _toy_graphs()
    graph = g1._replace(globals=None)
    net = atj.GraphNetwork(update_edge_fn=lambda e, s, r, g: e, update_node_fn=None)
    out = net(graph)
    np.testing.assert_allclose(_np(out.edges), _np(graph.edges))
    net2 = atj.GraphNetwork(update_edge_fn=None, update_node_fn=lambda n, s, r, g: n)
    out2 = net2(graph)
    np.testing.assert_allclose(_np(out2.nodes), _np(graph.nodes))


def test_graph_network_attention_requires_both():
    with pytest.raises(ValueError, match="must both be supplied"):
        atj.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=None,
            attention_logit_fn=lambda *a: 1,
            attention_reduce_fn=None,
        )
    with pytest.raises(ValueError, match="must both be supplied"):
        atj.GraphNetwork(
            update_edge_fn=None,
            update_node_fn=None,
            attention_logit_fn=None,
            attention_reduce_fn=lambda e, w: e,
        )


def test_interaction_network_and_map_features():
    g1, _ = _toy_graphs()
    inet = atj.InteractionNetwork(
        update_edge_fn=lambda e, s, r: np.concatenate([e, s, r], axis=-1),
        update_node_fn=lambda n, r: np.concatenate([n, r], axis=-1),
    )
    out = inet(g1)
    assert out.edges.shape[-1] == 6
    assert out.nodes.shape[-1] == 2 + 6
    embed = atj.GraphMapFeatures(lambda e: e * 2, lambda n: n * 3, lambda g: g * 4)
    mapped = embed(g1)
    np.testing.assert_allclose(_np(mapped.nodes), g1.nodes * 3)
    np.testing.assert_allclose(_np(mapped.edges), g1.edges * 2)
    np.testing.assert_allclose(_np(mapped.globals), g1.globals * 4)


def test_relation_network_and_deep_sets():
    g1, _ = _toy_graphs()
    rel = atj.RelationNetwork(
        update_edge_fn=lambda s, r: s + r,
        update_global_fn=lambda e: e * 2,
    )
    out = rel(g1)
    assert out.nodes is g1.nodes or np.allclose(_np(out.nodes), g1.nodes)
    assert out.globals.shape[0] == 1
    ds = atj.DeepSets(
        update_node_fn=lambda n, g: n + g,
        update_global_fn=lambda n: n * 2,
    )
    out2 = ds(g1)
    assert out2.nodes.shape == g1.nodes.shape


def test_gat_and_gcn():
    g1, _ = _toy_graphs()
    gat = atj.GAT(
        attention_query_fn=lambda n: n,
        attention_logit_fn=lambda s, r, e: (s * r).sum(axis=-1, keepdims=True),
        node_update_fn=lambda n: n,
    )
    out = gat(g1)
    assert out.nodes.shape == g1.nodes.shape
    gcn = atj.GraphConvolution(update_node_fn=lambda n: n, add_self_edges=True)
    out2 = gcn(g1)
    assert out2.nodes.shape == g1.nodes.shape
    gcn2 = atj.GraphConvolution(
        update_node_fn=lambda n: n, add_self_edges=False, symmetric_normalization=False
    )
    out3 = gcn2(g1)
    assert out3.nodes.shape == g1.nodes.shape


def test_concatenated_args_and_segment_wrappers():
    @atj.concatenated_args
    def fn(x):
        return x

    out = fn(np.ones((2, 2)), np.zeros((2, 3)))
    assert out.shape == (2, 5)

    data = np.array([1.0, 2.0, 3.0])
    ids = np.array([0, 0, 1])
    np.testing.assert_allclose(atj.segment_sum(data, ids, 2), [3.0, 3.0])
    assert atj.segment_mean(data, ids, 2).shape == (2,)
    assert atj.segment_max(data, ids, 2)[0] == 2.0
    sm = atj.segment_softmax(data, ids, 2)
    np.testing.assert_allclose(sm[:2].sum(), 1.0, atol=1e-5)
    part = atj.partition_softmax(data, np.array([2, 1]))
    np.testing.assert_allclose(part[:2].sum(), 1.0, atol=1e-5)
    atj.segment_min(data, ids, 2)
    atj.segment_variance(data, ids, 2)
    atj.segment_normalize(data, ids, 2)
    atj.segment_min_or_constant(data, ids, 2)
    atj.segment_max_or_constant(data, ids, 2)


def test_fully_connected_and_sparse_and_dynamic_batch():
    g = atj.get_fully_connected_graph(3, 2, node_features=np.arange(12.0).reshape(6, 2))
    assert g.senders.shape[0] == 2 * 9
    g2 = atj.get_fully_connected_graph(2, 1, add_self_edges=False)
    assert g2.senders.shape[0] == 2
    sparse = atj.sparse_matrix_to_graphs_tuple(
        np.array([0, 1]), np.array([1, 0]), np.array([1, 2]), np.array([2])
    )
    assert sparse.senders.shape[0] == 3
    empty = atj.sparse_matrix_to_graphs_tuple(
        np.array([]), np.array([]), np.array([]), np.array([1])
    )
    assert empty.senders.shape[0] == 0
    g1, g2 = _toy_graphs()
    batches = list(atj.dynamically_batch(iter([g1, g2]), n_node=20, n_edge=20, n_graph=4))
    assert len(batches) >= 1
    with pytest.raises(ValueError, match="equal to `2`"):
        list(atj.dynamically_batch(iter([g1]), 10, 10, 1))
    with pytest.raises(RuntimeError, match="bigger than batch"):
        list(atj.dynamically_batch(iter([g1]), n_node=3, n_edge=1, n_graph=2))


def test_zero_out_padding():
    g1, _ = _toy_graphs()
    padded = atj.pad_with_graphs(g1, n_node=6, n_edge=8, n_graph=2)
    z = atj.zero_out_padding(padded)
    mask = np.asarray(atj.get_node_padding_mask(padded))
    np.testing.assert_allclose(_np(z.nodes)[~mask], 0.0)
    wrapped = atj.with_zero_out_padding_outputs(lambda g: g)
    z2 = wrapped(padded)
    np.testing.assert_allclose(_np(z2.nodes)[~mask], 0.0)


@pytest.mark.parametrize("name", BACKENDS)
def test_graph_network_cross_backend(name):
    backend = loaded_backends[name]
    g1, g2 = _toy_graphs()
    graph = _to_backend(atj.batch([g1, g2]), backend)
    net = atj.GraphNetwork(
        update_edge_fn=lambda e, s, r, g: e,
        update_node_fn=lambda n, s, r, g: n,
        update_global_fn=lambda n, e, g: g,
    )
    out = net(graph)
    np.testing.assert_allclose(_np(out.nodes), _np(graph.nodes))


# Official jraph 0.0.6.dev0: on the module, omitted from ``jraph.__all__``.
_JRAPH_MODULE_EXTRAS = (
    "segment_mean",
    "segment_min",
    "segment_normalize",
    "segment_variance",
)
# Extra vs upstream jraph (documented; not a missing-API claim).
_ANYTENSOR_JRAPH_ONLY = ("sparse_matrix_to_graphs_tuple",)


def test_covers_jraph_public_api():
    """Full coverage of official ``jraph.__all__``, plus documented extras."""
    jraph = pytest.importorskip("jraph")
    missing = sorted(set(jraph.__all__) - set(atj.__all__))
    assert missing == [], missing
    for name in _JRAPH_MODULE_EXTRAS:
        assert hasattr(jraph, name), name
        assert hasattr(atj, name), name
        assert name in atj.__all__, name
    for name in _ANYTENSOR_JRAPH_ONLY:
        assert name in atj.__all__, name
        assert not hasattr(jraph, name), name


def test_jraph_parity_batch_and_identity():
    jraph = pytest.importorskip("jraph")
    jnp = pytest.importorskip("jax.numpy")
    g1, g2 = _toy_graphs()

    def to_jraph(g):
        return jraph.GraphsTuple(
            nodes=jnp.asarray(g.nodes),
            edges=jnp.asarray(g.edges),
            senders=jnp.asarray(g.senders),
            receivers=jnp.asarray(g.receivers),
            globals=jnp.asarray(g.globals),
            n_node=jnp.asarray(g.n_node),
            n_edge=jnp.asarray(g.n_edge),
        )

    jb = jraph.batch([to_jraph(g1), to_jraph(g2)])
    ab = atj.batch([g1, g2])
    np.testing.assert_array_equal(np.asarray(jb.senders), _np(ab.senders))
    np.testing.assert_array_equal(np.asarray(jb.receivers), _np(ab.receivers))
    np.testing.assert_allclose(np.asarray(jb.nodes), _np(ab.nodes))
    jnet = jraph.GraphNetwork(lambda e, s, r, g: e, lambda n, s, r, g: n, lambda n, e, g: g)
    anet = atj.GraphNetwork(lambda e, s, r, g: e, lambda n, s, r, g: n, lambda n, e, g: g)
    jout = jnet(jb)
    jax_backend = loaded_backends.get("jax")
    if jax_backend is None:
        pytest.skip("jax backend not loaded")
    aout = anet(_to_backend(ab, jax_backend))
    np.testing.assert_allclose(np.asarray(jout.nodes), _np(aout.nodes))
    np.testing.assert_allclose(np.asarray(jout.edges), _np(aout.edges))


def test_batch_empty_raises():
    empty = atj.GraphsTuple(
        nodes=np.zeros((0, 2)),
        edges=np.zeros((0, 2)),
        senders=np.zeros((0,), dtype=np.int32),
        receivers=np.zeros((0,), dtype=np.int32),
        globals=np.zeros((0, 2)),
        n_node=np.zeros((0,), dtype=np.int32),
        n_edge=np.zeros((0,), dtype=np.int32),
    )
    with pytest.raises(ValueError, match="at least one"):
        atj.batch([empty])


def test_batch_unbatch_zero_node_graph():
    g1, _ = _toy_graphs()
    empty = atj.GraphsTuple(
        nodes=np.zeros((0, 2)),
        edges=np.zeros((0, 2)),
        senders=np.zeros((0,), dtype=np.int32),
        receivers=np.zeros((0,), dtype=np.int32),
        globals=np.zeros((1, 2)),
        n_node=np.array([0]),
        n_edge=np.array([0]),
    )
    parts = atj.unbatch(atj.batch([g1, empty]))
    assert len(parts) == 2
    assert int(np.asarray(parts[1].n_node)[0]) == 0
    assert parts[1].nodes.shape[0] == 0
    solo = atj.unbatch(empty)
    assert len(solo) == 1
    assert solo[0].nodes.shape[0] == 0


def test_node_padding_mask_requires_nodes():
    g1, _ = _toy_graphs()
    padded = atj.pad_with_graphs(g1, 6, 8, 2)._replace(nodes=None)
    with pytest.raises(ValueError, match="node features"):
        atj.get_node_padding_mask(padded)


def test_graphnet_gat_none_attention():
    with pytest.raises(ValueError, match="attention_logit_fn"):
        atj.GraphNetGAT(lambda e, s, r, g: e, lambda n, s, r, g: n, None, lambda e, w: e)


def test_fully_connected_shape_errors():
    with pytest.raises(ValueError, match="Number of nodes"):
        atj.get_fully_connected_graph(2, 2, node_features=np.zeros((3, 1)))
    with pytest.raises(ValueError, match="globals"):
        atj.get_fully_connected_graph(2, 2, global_features=np.zeros((1, 1)))


def test_batch_np_unbatch_np():
    g1, g2 = _toy_graphs()
    b = atj.batch_np([g1, g2])
    assert isinstance(b.nodes, np.ndarray)
    parts = atj.unbatch_np(b)
    np.testing.assert_allclose(parts[0].nodes, g1.nodes)


def test_include_sent_messages():
    g1, _ = _toy_graphs()
    net = atj.InteractionNetwork(
        update_edge_fn=lambda e, s, r: e,
        update_node_fn=lambda n, s, r: n + r[:, : n.shape[-1]] * 0,
        include_sent_messages_in_node_update=True,
    )
    out = net(g1)
    assert out.nodes.shape == g1.nodes.shape


class Packed:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __eq__(self, other):
        return type(other) is Packed and np.array_equal(self.values, other.values)

    def __tree_flatten__(self):
        return (self.values,), None

    @classmethod
    def __tree_unflatten__(cls, aux, children):
        del aux
        return cls(children[0])

    @classmethod
    def __tree_batch__(cls, xs, axis=0):
        return cls(np.concatenate([x.values for x in xs], axis=axis))

    def __tree_unbatch__(self, axis=0):
        n = int(self.values.shape[axis])
        out = []
        for i in range(n):
            sl = [slice(None)] * self.values.ndim
            sl[axis] = slice(i, i + 1)
            out.append(Packed(self.values[tuple(sl)]))
        return out


def test_batch_unbatch_plain_pytree():
    a = {"temp": np.array([20.1]), "notes": None}
    b = {"temp": np.array([21.0]), "notes": None}
    joined = atj.batch([a, b])
    np.testing.assert_array_equal(joined["temp"], [20.1, 21.0])
    first, second = atj.unbatch(joined)
    np.testing.assert_array_equal(first["temp"], [20.1])
    np.testing.assert_array_equal(second["temp"], [21.0])
    assert first["notes"] is None


def test_batch_unbatch_custom_feature():
    g1, g2 = _toy_graphs()
    g1 = g1._replace(nodes=Packed(g1.nodes), edges=Packed(g1.edges))
    g2 = g2._replace(nodes=Packed(g2.nodes), edges=Packed(g2.edges))
    batched = atj.batch([g1, g2])
    assert isinstance(batched.nodes, Packed)
    assert batched.nodes.values.shape == (8, 2)
    assert batched.edges.values.shape == (10, 2)
    parts = atj.unbatch(batched)
    np.testing.assert_allclose(parts[0].nodes.values, g1.nodes.values)
    np.testing.assert_allclose(parts[1].nodes.values, g2.nodes.values)


def test_graphstuple_tree_batch_unbatch():
    from anytensor import tree as atree

    g1, g2 = _toy_graphs()
    assert atj.batch is atree.batch
    assert atj.unbatch is atree.unbatch
    batched = atree.batch([g1, g2])
    np.testing.assert_array_equal(np.asarray(batched.n_node), [3, 5])
    np.testing.assert_array_equal(np.asarray(batched.senders)[5:], g2.senders + 3)
    parts = atree.unbatch(batched)
    assert len(parts) == 2
    np.testing.assert_allclose(np.asarray(parts[0].nodes), g1.nodes)
    np.testing.assert_array_equal(np.asarray(parts[1].senders), g2.senders)
    triple = atree.unbatch(atree.batch([g1, g2, g1]))
    assert len(triple) == 3
    grouped = atree.batch(triple[:2])
    assert grouped.n_node.shape[0] == 2
    assert triple[2].n_node.shape[0] == 1
    with pytest.raises(ValueError, match="axis=0"):
        atree.batch([g1, g2], axis=1)
    with pytest.raises(ValueError, match="axis=0"):
        atj.GraphsTuple.__tree_batch__([g1, g2], axis=1)
    with pytest.raises(ValueError, match="axis=0"):
        batched.__tree_unbatch__(axis=1)


def test_graph_network_attention_and_mismatched_nodes():
    g1, _ = _toy_graphs()
    net = atj.GraphNetwork(
        update_edge_fn=lambda e, s, r, g: e,
        update_node_fn=lambda n, s, r, g: n,
        attention_logit_fn=lambda e, s, r, g: (s * r).sum(axis=-1, keepdims=True),
        attention_reduce_fn=lambda e, w: e * w,
    )
    out = net(g1)
    assert out.edges.shape == g1.edges.shape
    bad = g1._replace(nodes={"a": g1.nodes, "b": g1.nodes[:2]})
    net2 = atj.GraphNetwork(lambda e, s, r, g: e, lambda n, s, r, g: n)
    with pytest.raises(ValueError, match="same number of nodes"):
        net2(bad)


def test_graphnet_gat_and_default_gat():
    g1, _ = _toy_graphs()
    gat = atj.GraphNetGAT(
        lambda e, s, r, g: e,
        lambda n, s, r, g: n,
        lambda e, s, r, g: np.ones((e.shape[0], 1)),
        lambda e, w: e * w,
    )
    out = gat(g1)
    assert out.nodes.shape == g1.nodes.shape
    gat2 = atj.GAT(lambda n: n, lambda s, r, e: (s * r).sum(axis=-1, keepdims=True))
    out2 = gat2(g1)
    assert out2.nodes.ndim == 2
    with pytest.raises(IndexError, match="node features"):
        gat2(g1._replace(nodes=None))


def test_graph_network_none_nodes_and_map_features_identity():
    g1, _ = _toy_graphs()
    graph = g1._replace(nodes=None, senders=None, receivers=None, edges=None, n_edge=np.array([0]))
    net = atj.GraphNetwork(update_edge_fn=None, update_node_fn=None, update_global_fn=lambda n, e, g: g)
    out = net(graph)
    np.testing.assert_allclose(_np(out.globals), _np(graph.globals))
    ident = atj.GraphMapFeatures()(g1)
    np.testing.assert_allclose(_np(ident.nodes), g1.nodes)


def test_dynamically_batch_rejects_non_graphs():
    with pytest.raises(RuntimeError, match="GraphsTuple"):
        list(atj.dynamically_batch(iter([object()]), 10, 10, 2))


def _none_connectivity(g):
    return g._replace(
        senders=None,
        receivers=None,
        edges=None,
        n_edge=np.zeros_like(g.n_edge),
    )


def test_batch_unbatch_none_features_and_indices():
    g1, g2 = _toy_graphs()
    g1 = _none_connectivity(g1)._replace(nodes=None, globals=None)
    g2 = _none_connectivity(g2)._replace(nodes=None, globals=None)
    batched = atj.batch([g1, g2])
    assert batched.nodes is None
    assert batched.senders is None
    assert batched.globals is None
    parts = atj.unbatch(batched)
    assert len(parts) == 2
    assert parts[0].senders is None and parts[1].receivers is None
    host = atj.batch_np([g1, g2])
    assert host.senders is None


def test_unbatch_zero_length_n_node_vector():
    g = atj.GraphsTuple(
        nodes=None,
        edges=None,
        senders=None,
        receivers=None,
        globals=None,
        n_node=np.zeros((0,), dtype=np.int32),
        n_edge=np.zeros((0,), dtype=np.int32),
    )
    assert atj.unbatch(g) == []


def test_n_graphs_rejects_nonconcrete_batch_size(monkeypatch):
    import anytensor.jraph.utils as ju

    monkeypatch.setattr(ju, "_host_concrete_int", lambda _v: None)
    g1, _ = _toy_graphs()
    with pytest.raises(ValueError, match="concrete batch size"):
        ju._n_graphs(g1)


def test_pad_without_senders_and_zero_edge_padding():
    g1, _ = _toy_graphs()
    none_idx = _none_connectivity(g1)
    padded = atj.pad_with_graphs(none_idx, n_node=6, n_edge=2, n_graph=2)
    assert int(np.asarray(padded.n_node).sum()) == 6
    exact_edges = atj.pad_with_graphs(g1, n_node=6, n_edge=5, n_graph=2)
    restored = atj.unpad_with_graphs(exact_edges)
    np.testing.assert_array_equal(_np(restored.senders), g1.senders)
    empty = atj.GraphsTuple(
        nodes=np.zeros((0, 2)),
        edges=np.zeros((0, 2)),
        senders=np.zeros((0,), dtype=np.int32),
        receivers=np.zeros((0,), dtype=np.int32),
        globals=np.zeros((1, 2)),
        n_node=np.array([0]),
        n_edge=np.array([0]),
    )
    still_empty = atj.unpad_with_graphs(empty)
    assert int(np.asarray(still_empty.n_node).sum()) == 0
    with pytest.raises(ValueError, match="senders"):
        atj.get_edge_padding_mask(padded._replace(senders=None))


def test_graph_network_nodes_without_senders():
    g1, _ = _toy_graphs()
    graph = _none_connectivity(g1)
    net = atj.GraphNetwork(
        update_edge_fn=None,
        update_node_fn=lambda n, s, r, g: n,
        update_global_fn=lambda n, e, g: g,
    )
    out = net(graph)
    np.testing.assert_allclose(_np(out.nodes), _np(graph.nodes))


def test_concatenated_args_factory_and_fully_connected_empty_nests():
    @atj.concatenated_args(axis=-1)
    def fn(x):
        return x

    out = fn(np.ones((2, 2)), np.zeros((2, 3)))
    assert out.shape == (2, 5)
    g = atj.get_fully_connected_graph(2, 2, node_features={}, global_features={})
    assert g.nodes == {}
    assert g.globals == {}
    assert g.senders.shape[0] == 2 * 4


def test_dynamically_batch_flush_split_and_empty():
    g1, g2 = _toy_graphs()
    assert list(atj.dynamically_batch(iter([]), 10, 10, 2)) == []
    split = list(atj.dynamically_batch(iter([g1, g2]), n_node=7, n_edge=20, n_graph=4))
    assert len(split) == 2
    with pytest.raises(RuntimeError, match="bigger than batch"):
        list(atj.dynamically_batch(iter([g1, g2]), n_node=5, n_edge=20, n_graph=3))


def test_zero_out_padding_1d_features():
    g1, _ = _toy_graphs()
    flat = g1._replace(
        nodes=g1.nodes[:, 0],
        edges=g1.edges[:, 0],
        globals=g1.globals[:, 0],
    )
    padded = atj.pad_with_graphs(flat, n_node=6, n_edge=8, n_graph=2)
    z = atj.zero_out_padding(padded)
    mask = np.asarray(atj.get_node_padding_mask(padded))
    np.testing.assert_allclose(_np(z.nodes)[~mask], 0.0)


def test_flip0_namespace_fallbacks(monkeypatch):
    from anytensor.jraph import utils as ju

    x = np.array([1, 2, 3])

    class NoFlip:
        pass

    monkeypatch.setattr(ju, "array_namespace", lambda _x: NoFlip())
    np.testing.assert_array_equal(ju._flip0(x), x[::-1])

    class FlipNoAxis:
        def flip(self, arr, axis=None):
            if axis is not None:
                raise TypeError("axis not supported")
            return arr[::-1]

    monkeypatch.setattr(ju, "array_namespace", lambda _x: FlipNoAxis())
    np.testing.assert_array_equal(ju._flip0(x), x[::-1])


def test_jraph_parity_gat_with_self_edges():
    """GAT vs official jraph; self-edges are added (upstream assumes they exist)."""
    jraph = pytest.importorskip("jraph")
    jnp = pytest.importorskip("jax.numpy")
    g1, _ = _toy_graphs()
    n = int(g1.nodes.shape[0])
    self_idx = np.arange(n, dtype=np.int32)
    g1 = g1._replace(
        senders=np.concatenate([g1.senders, self_idx]),
        receivers=np.concatenate([g1.receivers, self_idx]),
        n_edge=g1.n_edge + g1.n_node,
        edges=np.concatenate([g1.edges, np.zeros((n, g1.edges.shape[-1]), dtype=g1.edges.dtype)]),
    )

    def logit(s, r, e):
        return (s * r).sum(axis=-1, keepdims=True)

    aout = atj.GAT(lambda n: n, logit, lambda n: n)(g1)
    jout = jraph.GAT(lambda n: n, logit, lambda n: n)(
        jraph.GraphsTuple(
            nodes=jnp.asarray(g1.nodes),
            edges=jnp.asarray(g1.edges),
            senders=jnp.asarray(g1.senders),
            receivers=jnp.asarray(g1.receivers),
            globals=jnp.asarray(g1.globals),
            n_node=jnp.asarray(g1.n_node),
            n_edge=jnp.asarray(g1.n_edge),
        )
    )
    np.testing.assert_allclose(np.asarray(jout.nodes), _np(aout.nodes), rtol=1e-5, atol=1e-5)


def test_jraph_parity_pad_and_masks():
    jraph = pytest.importorskip("jraph")
    jnp = pytest.importorskip("jax.numpy")
    g1, _ = _toy_graphs()
    jp = jraph.pad_with_graphs(
        jraph.GraphsTuple(
            nodes=jnp.asarray(g1.nodes),
            edges=jnp.asarray(g1.edges),
            senders=jnp.asarray(g1.senders),
            receivers=jnp.asarray(g1.receivers),
            globals=jnp.asarray(g1.globals),
            n_node=jnp.asarray(g1.n_node),
            n_edge=jnp.asarray(g1.n_edge),
        ),
        n_node=6,
        n_edge=8,
        n_graph=3,
    )
    ap = atj.pad_with_graphs(g1, n_node=6, n_edge=8, n_graph=3)
    np.testing.assert_array_equal(np.asarray(jp.n_node), _np(ap.n_node))
    np.testing.assert_array_equal(np.asarray(jp.senders), _np(ap.senders))
    np.testing.assert_array_equal(
        np.asarray(jraph.get_node_padding_mask(jp)),
        _np(atj.get_node_padding_mask(ap)),
    )

