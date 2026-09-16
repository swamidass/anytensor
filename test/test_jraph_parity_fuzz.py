"""Hypothesis parity: ``anytensor.jraph`` vs official ``jraph`` (JAX).

Skipped unless both ``jraph`` and JAX are installed. Marked ``fuzz`` so it
stays out of the coverage gate. Segment calls always pass ``num_segments``
(AnyTensor contract; jraph infers if omitted).
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from anytensor import jraph as atj
from anytensor import tree
from helpers import close

jraph = pytest.importorskip("jraph")
jnp = pytest.importorskip("jax.numpy")
import jax

# Official jraph still calls ``jax.tree_map`` / ``jax.tree_leaves`` (removed as
# module attributes in current JAX). Restore aliases for parity.
if not hasattr(jax, "tree_map"):
    jax.tree_map = jax.tree.map  # type: ignore[attr-defined]
if not hasattr(jax, "tree_leaves"):
    jax.tree_leaves = jax.tree.leaves  # type: ignore[attr-defined]

pytestmark = pytest.mark.fuzz

_settings = settings(
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.filter_too_much],
)

_finite32 = st.floats(-5, 5, allow_nan=False, allow_infinity=False, width=32)


def _nest(array):
    if array is None:
        return None
    return {"a": array, "b": [np.ones_like(array), {"c": np.zeros_like(array)}]}


@st.composite
def _graphs(draw, *, nest: bool = False, allow_none: bool = False):
    n_graph = draw(st.integers(1, 3))
    n_node = np.array([draw(st.integers(0, 5)) for _ in range(n_graph)], dtype=np.int32)
    senders: list[int] = []
    receivers: list[int] = []
    n_edge_list: list[int] = []
    offset = 0
    for nn in n_node.tolist():
        ne = 0 if nn == 0 else draw(st.integers(0, 6))
        n_edge_list.append(ne)
        if ne:
            local = draw(st.lists(st.integers(0, nn - 1), min_size=ne, max_size=ne))
            recv = draw(st.lists(st.integers(0, nn - 1), min_size=ne, max_size=ne))
            senders.extend(s + offset for s in local)
            receivers.extend(r + offset for r in recv)
        offset += nn
    n_edge = np.array(n_edge_list, dtype=np.int32)
    sum_n = int(n_node.sum())
    sum_e = int(n_edge.sum())

    def _arr(shape):
        n = int(np.prod(shape))
        if n == 0:
            return np.zeros(shape, dtype=np.float32)
        data = draw(st.lists(_finite32, min_size=n, max_size=n))
        return np.asarray(data, dtype=np.float32).reshape(shape)

    nodes = _arr((sum_n, 3))
    edges = _arr((sum_e, 3))
    globals_ = _arr((n_graph, 2))
    if allow_none and draw(st.booleans()):
        nodes = None
    if allow_none and draw(st.booleans()):
        edges = None
    if allow_none and draw(st.booleans()):
        globals_ = None
    if nest:
        nodes = _nest(nodes)
        edges = _nest(edges)
        globals_ = _nest(globals_)
    return atj.GraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=np.asarray(senders, dtype=np.int32),
        receivers=np.asarray(receivers, dtype=np.int32),
        globals=globals_,
        n_node=n_node,
        n_edge=n_edge,
    )


def _to_jraph(g: atj.GraphsTuple):
    def conv(x):
        if x is None:
            return None
        return tree.map(lambda a: jnp.asarray(np.asarray(a)), x)

    return jraph.GraphsTuple(
        nodes=conv(g.nodes),
        edges=conv(g.edges),
        senders=jnp.asarray(g.senders),
        receivers=jnp.asarray(g.receivers),
        globals=conv(g.globals),
        n_node=jnp.asarray(g.n_node),
        n_edge=jnp.asarray(g.n_edge),
    )


def _np_feat(x):
    if x is None:
        return None
    return tree.map(lambda a: np.asarray(a), x)


def _assert_graphs_close(got, ref, *, rtol=1e-5, atol=1e-5):
    for name in ("n_node", "n_edge", "senders", "receivers"):
        np.testing.assert_array_equal(
            np.asarray(getattr(got, name)),
            np.asarray(getattr(ref, name)),
            err_msg=name,
        )
    for name in ("nodes", "edges", "globals"):
        a, b = _np_feat(getattr(got, name)), _np_feat(getattr(ref, name))
        if a is None or b is None:
            assert a is None and b is None, name
            continue
        for la, lb in zip(tree.leaves(a), tree.leaves(b), strict=True):
            assert close(np.asarray(la, dtype=np.float32), np.asarray(lb, dtype=np.float32), equal_nan=True) or np.allclose(
                la, lb, rtol=rtol, atol=atol, equal_nan=True
            ), (name, la, lb)


def _add_self_edges(g: atj.GraphsTuple) -> atj.GraphsTuple:
    """Append a self-loop on every node. Official GAT assumes these exist."""
    n_node = np.asarray(g.n_node)
    total = int(n_node.sum())
    self_idx = np.arange(total, dtype=np.int32)
    senders = np.concatenate([np.asarray(g.senders), self_idx]).astype(np.int32)
    receivers = np.concatenate([np.asarray(g.receivers), self_idx]).astype(np.int32)
    n_edge = (np.asarray(g.n_edge) + n_node).astype(np.int32)
    if g.edges is None:
        edges = None
    else:
        e = np.asarray(g.edges)
        extra = np.zeros((total,) + e.shape[1:], dtype=e.dtype)
        edges = np.concatenate([e, extra], axis=0)
    return g._replace(senders=senders, receivers=receivers, n_edge=n_edge, edges=edges)


def _gat_logit(s, r, e):
    return (s * r).sum(axis=-1, keepdims=True)


def _gn_attn_logit(e, s, r, gl):
    return (s * r).sum(axis=-1, keepdims=True)


def _pad_sizes(g):
    sum_n = int(np.asarray(g.n_node).sum())
    sum_e = int(np.asarray(g.n_edge).sum())
    n_g = int(g.n_node.shape[0])
    return sum_n + 1, sum_e + 1, max(n_g + 1, 2)


@_settings
@given(g=_graphs())
def test_fuzz_batch_unbatch_parity(g):
    jb = jraph.batch([_to_jraph(g), _to_jraph(g)])
    ab = atj.batch([g, g])
    _assert_graphs_close(ab, jb)
    ap, jp = atj.unbatch(ab), jraph.unbatch(jb)
    assert len(ap) == len(jp)
    for a, b in zip(ap, jp, strict=True):
        _assert_graphs_close(a, b)


@_settings
@given(g=_graphs())
def test_fuzz_pad_unpad_mask_parity(g):
    n_node, n_edge, n_graph = _pad_sizes(g)
    jp = jraph.pad_with_graphs(_to_jraph(g), n_node, n_edge, n_graph)
    ap = atj.pad_with_graphs(g, n_node, n_edge, n_graph)
    _assert_graphs_close(ap, jp)
    np.testing.assert_array_equal(
        np.asarray(atj.get_graph_padding_mask(ap)),
        np.asarray(jraph.get_graph_padding_mask(jp)),
    )
    if g.nodes is not None:
        np.testing.assert_array_equal(
            np.asarray(atj.get_node_padding_mask(ap)),
            np.asarray(jraph.get_node_padding_mask(jp)),
        )
    np.testing.assert_array_equal(
        np.asarray(atj.get_edge_padding_mask(ap)),
        np.asarray(jraph.get_edge_padding_mask(jp)),
    )
    _assert_graphs_close(atj.unpad_with_graphs(ap), jraph.unpad_with_graphs(jp))


@_settings
@given(g=_graphs())
def test_fuzz_graph_network_parity(g):
    def scale_e(e, s, r, gl):
        return None if e is None else tree.map(lambda x: x * np.float32(2), e)

    def shift_n(n, s, r, gl):
        return None if n is None else tree.map(lambda x: x + np.float32(1), n)

    def half_g(n, e, gl):
        return None if gl is None else tree.map(lambda x: x * np.float32(0.5), gl)

    jnet = jraph.GraphNetwork(scale_e, shift_n, half_g)
    anet = atj.GraphNetwork(scale_e, shift_n, half_g)
    _assert_graphs_close(anet(g), jnet(_to_jraph(g)))
    ident_j = jraph.GraphNetwork(
        lambda e, s, r, gl: e, lambda n, s, r, gl: n, lambda n, e, gl: gl
    )
    ident_a = atj.GraphNetwork(
        lambda e, s, r, gl: e, lambda n, s, r, gl: n, lambda n, e, gl: gl
    )
    _assert_graphs_close(ident_a(g), ident_j(_to_jraph(g)))


@_settings
@given(g=_graphs(nest=True))
def test_fuzz_nested_batch_parity(g):
    _assert_graphs_close(atj.batch([g]), jraph.batch([_to_jraph(g)]))


@_settings
@given(
    data=st.lists(_finite32, min_size=1, max_size=8),
    nseg=st.integers(1, 5),
)
def test_fuzz_segment_ops_parity(data, nseg):
    x = np.asarray(data, dtype=np.float32)
    ids = np.arange(len(x), dtype=np.int32) % nseg
    jx, jids = jnp.asarray(x), jnp.asarray(ids)
    for name in ("segment_sum", "segment_mean", "segment_max", "segment_min", "segment_softmax"):
        got = np.asarray(getattr(atj, name)(x, ids, nseg))
        ref = np.asarray(getattr(jraph, name)(jx, jids, nseg))
        assert close(got, ref, equal_nan=True), name
    got = np.asarray(atj.segment_normalize(x, ids, nseg))
    ref = np.asarray(jraph.segment_normalize(jx, jids, nseg))
    assert close(got, ref, equal_nan=True)


@_settings
@given(g=_graphs())
def test_fuzz_model_zoo_parity(g):
    assume(int(np.asarray(g.n_node).sum()) > 0)
    jg = _to_jraph(g)

    mapped_a = atj.GraphMapFeatures(lambda e: e, lambda n: n, lambda gl: gl)(g)
    mapped_j = jraph.GraphMapFeatures(lambda e: e, lambda n: n, lambda gl: gl)(jg)
    _assert_graphs_close(mapped_a, mapped_j)

    ds_a = atj.DeepSets(lambda n, gl: n, lambda n: n)(g)
    ds_j = jraph.DeepSets(lambda n, gl: n, lambda n: n)(jg)
    _assert_graphs_close(ds_a, ds_j)

    inet_a = atj.InteractionNetwork(lambda e, s, r: e, lambda n, r: n)(g)
    inet_j = jraph.InteractionNetwork(lambda e, s, r: e, lambda n, r: n)(jg)
    _assert_graphs_close(inet_a, inet_j)

    rel_a = atj.RelationNetwork(lambda s, r: s + r, lambda e: e)(g)
    rel_j = jraph.RelationNetwork(lambda s, r: s + r, lambda e: e)(jg)
    _assert_graphs_close(rel_a, rel_j)

    gngat_a = atj.GraphNetGAT(
        lambda e, s, r, gl: e,
        lambda n, s, r, gl: n,
        _gn_attn_logit,
        lambda e, w: e * w,
    )(g)
    gngat_j = jraph.GraphNetGAT(
        lambda e, s, r, gl: e,
        lambda n, s, r, gl: n,
        _gn_attn_logit,
        lambda e, w: e * w,
    )(jg)
    _assert_graphs_close(gngat_a, gngat_j, rtol=1e-4, atol=1e-4)

    # Self-edges are fine: GCN adds them internally; GAT assumes they exist.
    gcn_a = atj.GraphConvolution(lambda n: n, add_self_edges=True)(g)
    gcn_j = jraph.GraphConvolution(lambda n: n, add_self_edges=True)(jg)
    _assert_graphs_close(gcn_a, gcn_j, rtol=1e-4, atol=1e-4)

    gat_g = _add_self_edges(g)
    jg_gat = _to_jraph(gat_g)
    gat_a = atj.GAT(lambda n: n, _gat_logit, lambda n: n)(gat_g)
    gat_j = jraph.GAT(lambda n: n, _gat_logit, lambda n: n)(jg_gat)
    _assert_graphs_close(gat_a, gat_j, rtol=1e-4, atol=1e-4)


@_settings
@given(n_node=st.integers(1, 4), n_graph=st.integers(1, 3), self_edges=st.booleans())
def test_fuzz_fully_connected_parity(n_node, n_graph, self_edges):
    nodes = np.arange(n_node * n_graph * 2, dtype=np.float32).reshape(n_node * n_graph, 2)
    ag = atj.get_fully_connected_graph(
        n_node, n_graph, node_features=nodes, add_self_edges=self_edges
    )
    jg = jraph.get_fully_connected_graph(
        n_node, n_graph, node_features=jnp.asarray(nodes), add_self_edges=self_edges
    )
    _assert_graphs_close(ag, jg)


@_settings
@given(g=_graphs())
def test_fuzz_zero_out_padding_parity(g):
    assume(g.nodes is not None and g.edges is not None)
    n_node, n_edge, n_graph = _pad_sizes(g)
    padded_a = atj.pad_with_graphs(g, n_node, n_edge, n_graph)
    padded_j = jraph.pad_with_graphs(_to_jraph(g), n_node, n_edge, n_graph)
    _assert_graphs_close(atj.zero_out_padding(padded_a), jraph.zero_out_padding(padded_j))
