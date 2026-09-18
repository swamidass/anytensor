"""Graph neural network models (jraph zoo, backend-agnostic)."""

from __future__ import annotations

import functools
from typing import Any, Callable, Optional, Union

import numpy as np

from anytensor import tree
from anytensor.core import concatenate, maximum, reshape, rsqrt, shape, take, where
from anytensor.core import arange as at_arange
from anytensor.core import ones as at_ones
from anytensor.segment import cache, partition_ids

from . import utils
from .graph import GraphsTuple

ArrayTree = Union[Any, list, tuple, dict]

NodeFeatures = EdgeFeatures = SenderFeatures = ReceiverFeatures = Globals = ArrayTree

AggregateEdgesToNodesFn = Callable[[EdgeFeatures, Any, int], NodeFeatures]
AggregateNodesToGlobalsFn = Callable[[NodeFeatures, Any, int], Globals]
AggregateEdgesToGlobalsFn = Callable[[EdgeFeatures, Any, int], Globals]
AttentionLogitFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], ArrayTree
]
AttentionReduceFn = Callable[[EdgeFeatures, ArrayTree], EdgeFeatures]
AttentionNormalizeFn = Callable[[EdgeFeatures, Any, int], EdgeFeatures]
GNUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures, Globals], EdgeFeatures
]
GNUpdateNodeFn = Callable[
    [NodeFeatures, SenderFeatures, ReceiverFeatures, Globals], NodeFeatures
]
GNUpdateGlobalFn = Callable[[NodeFeatures, EdgeFeatures, Globals], Globals]


def _repeat_by(values, repeats, total_length):
    """Scatter rows of ``values`` according to per-row ``repeats`` (axis 0).

    Goes through :func:`~anytensor.partition_ids` so a cache hit is shared
    with other partition helpers.
    """
    idx = partition_ids(repeats, total_length)
    return take(values, idx)


def _take_index(features, index):
    if features is None:
        return None
    if index is None:
        return features
    return tree.map(lambda n: take(n, index), features)


def GraphNetwork(
    update_edge_fn: Optional[GNUpdateEdgeFn],
    update_node_fn: Optional[GNUpdateNodeFn],
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_normalize_fn: Optional[AttentionNormalizeFn] = utils.segment_softmax,
    attention_reduce_fn: Optional[AttentionReduceFn] = None,
):
    """Returns a method that applies a configured GraphNetwork.

    Follows Algorithm 1 of https://arxiv.org/abs/1806.01261, with separate
    sender/receiver aggregations and optional softmax attention. Same call
    signature as :func:`jraph.GraphNetwork`. Apply is decorated with
    :data:`~anytensor.cache` (sticky): stacked calls reuse ``n_node`` /
    ``n_edge`` expansions via :func:`~anytensor.partition_ids`.
    """
    not_both_supplied = lambda x, y: (x != y) and ((x is None) or (y is None))
    if not_both_supplied(attention_reduce_fn, attention_logit_fn):
        raise ValueError(
            "attention_logit_fn and attention_reduce_fn must both be supplied."
        )

    @cache
    def _ApplyGraphNet(graph: GraphsTuple) -> GraphsTuple:
        nodes, edges, receivers, senders, globals_, n_node, n_edge = graph
        node_leaves = tree.leaves(nodes)
        if node_leaves:
            sum_n_node = shape(node_leaves[0])[0]
        else:
            sum_n_node = int(np_sum_n_node(n_node))
        sum_n_edge = 0 if senders is None else shape(senders)[0]
        # ``int(size)`` is rewritten by TF Autograph into a graph op, so a
        # symbolic leading dim looks "concrete". Only compare nest lengths
        # when the size is already a Python int (eager NumPy / JAX / TF).
        if (
            node_leaves
            and type(sum_n_node) is int
            and not utils._tree_all(  # noqa: SLF001
                tree.map(lambda n: n.shape[0] == sum_n_node, nodes)
            )
        ):
            raise ValueError(
                "All node arrays in nest must contain the same number of nodes."
            )

        sent_attributes = _take_index(nodes, senders)
        received_attributes = _take_index(nodes, receivers)
        if globals_ is not None and n_edge is not None and sum_n_edge is not None:
            global_edge_attributes = tree.map(
                lambda g: _repeat_by(g, n_edge, sum_n_edge), globals_
            )
        else:
            global_edge_attributes = None

        if update_edge_fn:
            edges = update_edge_fn(
                edges, sent_attributes, received_attributes, global_edge_attributes
            )

        if attention_logit_fn:
            logits = attention_logit_fn(
                edges, sent_attributes, received_attributes, global_edge_attributes
            )
            tree_calculate_weights = functools.partial(
                attention_normalize_fn, segment_ids=receivers, num_segments=sum_n_node
            )
            weights = tree.map(tree_calculate_weights, logits)
            edges = attention_reduce_fn(edges, weights)

        if update_node_fn:
            sent_attributes = (
                None
                if edges is None
                else tree.map(
                    lambda e: aggregate_edges_for_nodes_fn(e, senders, sum_n_node), edges
                )
            )
            received_attributes = (
                None
                if edges is None
                else tree.map(
                    lambda e: aggregate_edges_for_nodes_fn(e, receivers, sum_n_node),
                    edges,
                )
            )
            if globals_ is not None:
                global_attributes = tree.map(
                    lambda g: _repeat_by(g, n_node, sum_n_node), globals_
                )
            else:
                global_attributes = None
            nodes = update_node_fn(
                nodes, sent_attributes, received_attributes, global_attributes
            )

        if update_global_fn:
            n_graph = shape(n_node)[0]
            node_gr_idx = partition_ids(n_node, sum_n_node)
            edge_gr_idx = (
                None
                if senders is None
                else partition_ids(n_edge, sum_n_edge)
            )
            node_attributes = (
                None
                if nodes is None
                else tree.map(
                    lambda n: aggregate_nodes_for_globals_fn(n, node_gr_idx, n_graph),
                    nodes,
                )
            )
            edge_attributes = (
                None
                if edges is None or edge_gr_idx is None
                else tree.map(
                    lambda e: aggregate_edges_for_globals_fn(e, edge_gr_idx, n_graph),
                    edges,
                )
            )
            globals_ = update_global_fn(node_attributes, edge_attributes, globals_)
        return GraphsTuple(
            nodes=nodes,
            edges=edges,
            receivers=receivers,
            senders=senders,
            globals=globals_,
            n_node=n_node,
            n_edge=n_edge,
        )

    return _ApplyGraphNet


def np_sum_n_node(n_node):
    return int(np.asarray(n_node).sum())


InteractionUpdateNodeFn = Callable[[NodeFeatures, SenderFeatures, ReceiverFeatures], NodeFeatures]
InteractionUpdateNodeFnNoSentEdges = Callable[[NodeFeatures, ReceiverFeatures], NodeFeatures]
InteractionUpdateEdgeFn = Callable[
    [EdgeFeatures, SenderFeatures, ReceiverFeatures], EdgeFeatures
]


def InteractionNetwork(
    update_edge_fn: InteractionUpdateEdgeFn,
    update_node_fn: Union[InteractionUpdateNodeFn, InteractionUpdateNodeFnNoSentEdges],
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    include_sent_messages_in_node_update: bool = False,
):
    """Interaction network (Battaglia et al.) as a configured GraphNetwork."""
    wrapped_update_edge_fn = lambda e, s, r, g: update_edge_fn(e, s, r)
    if include_sent_messages_in_node_update:
        wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, s, r)
    else:
        wrapped_update_node_fn = lambda n, s, r, g: update_node_fn(n, r)
    return GraphNetwork(
        update_edge_fn=wrapped_update_edge_fn,
        update_node_fn=wrapped_update_node_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
    )


EmbedEdgeFn = Callable[[EdgeFeatures], EdgeFeatures]
EmbedNodeFn = Callable[[NodeFeatures], NodeFeatures]
EmbedGlobalFn = Callable[[Globals], Globals]


def GraphMapFeatures(
    embed_edge_fn: Optional[EmbedEdgeFn] = None,
    embed_node_fn: Optional[EmbedNodeFn] = None,
    embed_global_fn: Optional[EmbedGlobalFn] = None,
):
    """Embed nodes, edges, and globals independently."""
    identity = lambda x: x
    embed_edges_fn = embed_edge_fn if embed_edge_fn else identity
    embed_nodes_fn = embed_node_fn if embed_node_fn else identity
    embed_globals_fn = embed_global_fn if embed_global_fn else identity

    def Embed(graphs_tuple: GraphsTuple) -> GraphsTuple:
        return graphs_tuple._replace(
            nodes=embed_nodes_fn(graphs_tuple.nodes),
            edges=embed_edges_fn(graphs_tuple.edges),
            globals=embed_globals_fn(graphs_tuple.globals),
        )

    return Embed


def RelationNetwork(
    update_edge_fn: Callable[[SenderFeatures, ReceiverFeatures], EdgeFeatures],
    update_global_fn: Callable[[EdgeFeatures], NodeFeatures],
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
):
    """Relation network as a configured GraphNetwork."""
    return GraphNetwork(
        update_edge_fn=lambda e, s, r, g: update_edge_fn(s, r),
        update_node_fn=None,
        update_global_fn=lambda n, e, g: update_global_fn(e),
        attention_logit_fn=None,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
    )


def DeepSets(
    update_node_fn: Callable[[NodeFeatures, Globals], NodeFeatures],
    update_global_fn: Callable[[NodeFeatures], Globals],
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
):
    """DeepSets layer as a configured GraphNetwork."""
    return GraphNetwork(
        update_edge_fn=None,
        update_node_fn=lambda n, s, r, g: update_node_fn(n, g),
        update_global_fn=lambda n, e, g: update_global_fn(n),
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
    )


def GraphNetGAT(
    update_edge_fn: GNUpdateEdgeFn,
    update_node_fn: GNUpdateNodeFn,
    attention_logit_fn: AttentionLogitFn,
    attention_reduce_fn: AttentionReduceFn,
    update_global_fn: Optional[GNUpdateGlobalFn] = None,
    aggregate_edges_for_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    aggregate_nodes_for_globals_fn: AggregateNodesToGlobalsFn = utils.segment_sum,
    aggregate_edges_for_globals_fn: AggregateEdgesToGlobalsFn = utils.segment_sum,
):
    """GraphNet with required attention on edge features."""
    if (attention_logit_fn is None) or (attention_reduce_fn is None):
        raise ValueError(
            "`None` value not supported for `attention_logit_fn` or "
            "`attention_reduce_fn` in a Graph Attention network."
        )
    return GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=update_global_fn,
        attention_logit_fn=attention_logit_fn,
        attention_reduce_fn=attention_reduce_fn,
        aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
        aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
        aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn,
    )


GATAttentionQueryFn = Callable[[NodeFeatures], NodeFeatures]
GATAttentionLogitFn = Callable[[SenderFeatures, ReceiverFeatures, EdgeFeatures], EdgeFeatures]
GATNodeUpdateFn = Callable[[NodeFeatures], NodeFeatures]


def _leaky_relu(x, negative_slope: float = 0.2):
    return where(x > 0, x, x * negative_slope)


def GAT(
    attention_query_fn: GATAttentionQueryFn,
    attention_logit_fn: GATAttentionLogitFn,
    node_update_fn: Optional[GATNodeUpdateFn] = None,
):
    """Graph Attention Network layer (Veličković et al.). Expects self-edges."""
    if node_update_fn is None:

        def node_update_fn(x):
            y = _leaky_relu(x)
            return reshape(y, (shape(y)[0], -1))

    def _ApplyGAT(graph: GraphsTuple) -> GraphsTuple:
        nodes, edges, receivers, senders, _, _, _ = graph
        if nodes is None:
            raise IndexError("GAT requires node features")
        sum_n_node = shape(nodes)[0]
        nodes = attention_query_fn(nodes)
        sent_attributes = take(nodes, senders)
        received_attributes = take(nodes, receivers)
        softmax_logits = attention_logit_fn(sent_attributes, received_attributes, edges)
        weights = utils.segment_softmax(
            softmax_logits, segment_ids=receivers, num_segments=sum_n_node
        )
        messages = sent_attributes * weights
        nodes = utils.segment_sum(messages, receivers, num_segments=sum_n_node)
        nodes = node_update_fn(nodes)
        return graph._replace(nodes=nodes)

    return _ApplyGAT


def GraphConvolution(
    update_node_fn: Callable[[NodeFeatures], NodeFeatures],
    aggregate_nodes_fn: AggregateEdgesToNodesFn = utils.segment_sum,
    add_self_edges: bool = False,
    symmetric_normalization: bool = True,
):
    """GCN layer (Kipf & Welling). No activation after aggregation."""

    def _ApplyGCN(graph: GraphsTuple) -> GraphsTuple:
        nodes, _, receivers, senders, _, _, _ = graph
        nodes = update_node_fn(nodes)
        total_num_nodes = shape(tree.leaves(nodes)[0])[0]
        if add_self_edges:
            self_idx = at_arange(total_num_nodes, like=senders)
            conv_receivers = concatenate((receivers, self_idx), axis=0)
            conv_senders = concatenate((senders, self_idx), axis=0)
        else:
            conv_senders = senders
            conv_receivers = receivers

        if symmetric_normalization:
            feat_dtype = tree.leaves(nodes)[0].dtype
            one = at_ones((), dtype=feat_dtype, like=tree.leaves(nodes)[0])

            def count_edges(x):
                ones = at_ones(
                    shape(conv_senders), dtype=feat_dtype, like=conv_senders
                )
                return utils.segment_sum(ones, x, total_num_nodes)

            sender_degree = count_edges(conv_senders)
            receiver_degree = count_edges(conv_receivers)
            nodes = tree.map(
                lambda x: x
                * reshape(
                    rsqrt(maximum(sender_degree, one)),
                    (shape(sender_degree)[0],) + (1,) * (x.ndim - 1),
                ),
                nodes,
            )
            nodes = tree.map(
                lambda x: aggregate_nodes_fn(
                    take(x, conv_senders), conv_receivers, total_num_nodes
                ),
                nodes,
            )
            nodes = tree.map(
                lambda x: x
                * reshape(
                    rsqrt(maximum(receiver_degree, one)),
                    (shape(receiver_degree)[0],) + (1,) * (x.ndim - 1),
                ),
                nodes,
            )
        else:
            nodes = tree.map(
                lambda x: aggregate_nodes_fn(
                    take(x, conv_senders), conv_receivers, total_num_nodes
                ),
                nodes,
            )
        return graph._replace(nodes=nodes)

    return _ApplyGCN
