"""Utilities for :class:`~anytensor.jraph.graph.GraphsTuple`.

Batching, padding, and segment helpers follow :mod:`jraph` (Apache 2.0),
rewritten over :mod:`anytensor` so the same code runs on NumPy / JAX /
PyTorch / TensorFlow.

``num_segments`` stays **required** on segment ops (AnyTensor contract).
``unique_indices`` is accepted and ignored (JAX-only hint). ``None`` feature nests are empty pytrees (``jax.tree`` / jraph).
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Generator, Iterator, List, Optional, Sequence

import numpy as np

from anytensor import tree
from anytensor.tree import batch, unbatch
from anytensor.core import (
    arange,
    astype,
    concatenate,
    full,
    maximum,
    ones,
    reshape,
    rsqrt,
    shape,
    take,
    zeros,
)
from anytensor.core import _host_concrete_int
from anytensor.lengths import batch_ids, split_by_lengths, unbatch_ids
from anytensor.namespace import array_namespace
from anytensor.segment import (
    partition_softmax as _partition_softmax,
    segment_max as _segment_max,
    segment_max_or_constant as _segment_max_or_constant,
    segment_mean as _segment_mean,
    segment_min as _segment_min,
    segment_min_or_constant as _segment_min_or_constant,
    segment_softmax as _segment_softmax,
    segment_sum as _segment_sum,
    segment_variance as _segment_variance,
)

from .graph import ArrayTree, GraphsTuple


def _sorted_flag(indices_are_sorted: bool, sorted: bool) -> bool:
    return bool(indices_are_sorted or sorted)


def segment_sum(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Sum within segments. ``num_segments`` is required (AnyTensor)."""
    del unique_indices
    return _segment_sum(
        data, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def segment_mean(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Mean within segments. ``num_segments`` is required."""
    del unique_indices
    return _segment_mean(
        data, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def segment_variance(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Variance within segments. ``num_segments`` is required."""
    del unique_indices
    return _segment_variance(
        data, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def segment_normalize(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    eps=1e-8,
    sorted: bool = False,
):
    """Z-score normalize within segments (jraph semantics). ``num_segments`` is required."""
    del unique_indices
    flag = _sorted_flag(indices_are_sorted, sorted)
    means = take(segment_mean(data, segment_ids, num_segments, sorted=flag), segment_ids)
    variances = take(
        segment_variance(data, segment_ids, num_segments, sorted=flag), segment_ids
    )
    scale = rsqrt(maximum(variances, eps))
    return (data - means) * scale


def segment_max(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Max within segments. ``num_segments`` is required."""
    del unique_indices
    return _segment_max(
        data, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def segment_min(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Min within segments. ``num_segments`` is required."""
    del unique_indices
    return _segment_min(
        data, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def segment_min_or_constant(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    constant: float = 0.0,
    sorted: bool = False,
):
    """Segment min with a finite fill for empty segments."""
    del unique_indices
    return _segment_min_or_constant(
        data,
        segment_ids,
        num_segments,
        constant=constant,
        sorted=_sorted_flag(indices_are_sorted, sorted),
    )


def segment_max_or_constant(
    data,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    constant: float = 0.0,
    sorted: bool = False,
):
    """Segment max with a finite fill for empty segments."""
    del unique_indices
    return _segment_max_or_constant(
        data,
        segment_ids,
        num_segments,
        constant=constant,
        sorted=_sorted_flag(indices_are_sorted, sorted),
    )


def segment_softmax(
    logits,
    segment_ids,
    num_segments,
    indices_are_sorted: bool = False,
    unique_indices: bool = False,
    sorted: bool = False,
):
    """Softmax within segments. ``num_segments`` is required."""
    del unique_indices
    return _segment_softmax(
        logits, segment_ids, num_segments, sorted=_sorted_flag(indices_are_sorted, sorted)
    )


def partition_softmax(logits, partitions, sum_partitions=None):
    """Softmax within contiguous partitions of lengths ``partitions``."""
    return _partition_softmax(logits, partitions, sum_partitions=sum_partitions)


def _map_features(func, features):
    """Map ``func`` over feature leaves; ``None`` stays ``None``."""
    return tree.map(func, features)


def _tree_all(value) -> bool:
    return all(bool(np.asarray(v)) for v in tree.leaves(value))


def _np_vec(x) -> np.ndarray:
    return np.asarray(x)


def _n_graphs(graph: GraphsTuple) -> int:
    n = _host_concrete_int(shape(graph.n_node)[0])
    if n is None:
        raise ValueError("n_graphs requires a concrete batch size")
    return n


def _sum_n_node(graph: GraphsTuple) -> int:
    return int(_np_vec(graph.n_node).sum())


def _sum_n_edge(graph: GraphsTuple) -> int:
    return int(_np_vec(graph.n_edge).sum())


def _like_index(graph: GraphsTuple):
    if graph.senders is not None:
        return graph.senders
    return graph.n_node


def _zeros_like_leading(leaf, leading: int):
    rest = tuple(int(s) for s in np.asarray(leaf.shape)[1:])
    return zeros((leading,) + rest, dtype=leaf.dtype, like=leaf)


def _batch_graphs(graphs: Sequence[GraphsTuple]) -> GraphsTuple:
    """Fieldwise concat, then offset senders/receivers from ``n_node``."""
    graphs = [g for g in graphs if _n_graphs(g) > 0]
    if not graphs:
        raise ValueError("batch() requires at least one non-empty GraphsTuple")
    batched = GraphsTuple(
        n_node=batch([g.n_node for g in graphs], axis=0),
        n_edge=batch([g.n_edge for g in graphs], axis=0),
        nodes=batch([g.nodes for g in graphs], axis=0),
        edges=batch([g.edges for g in graphs], axis=0),
        globals=batch([g.globals for g in graphs], axis=0),
        senders=_concat_maybe([g.senders for g in graphs]),
        receivers=_concat_maybe([g.receivers for g in graphs]),
    )
    if batched.senders is None:
        return batched
    return batched._replace(
        senders=batch_ids(batched.senders, batched.n_node, batched.n_edge),
        receivers=batch_ids(batched.receivers, batched.n_node, batched.n_edge),
    )


def _to_numpy_leaf(x):
    if x is None:
        return None
    return np.asarray(x)


def _graph_to_numpy(graph: GraphsTuple) -> GraphsTuple:
    return GraphsTuple(
        nodes=_map_features(_to_numpy_leaf, graph.nodes),
        edges=_map_features(_to_numpy_leaf, graph.edges),
        receivers=_to_numpy_leaf(graph.receivers),
        senders=_to_numpy_leaf(graph.senders),
        globals=_map_features(_to_numpy_leaf, graph.globals),
        n_node=_to_numpy_leaf(graph.n_node),
        n_edge=_to_numpy_leaf(graph.n_edge),
    )


def batch_np(graphs: Sequence[GraphsTuple]) -> GraphsTuple:
    """NumPy implementation of :func:`batch` (host arrays)."""
    return batch([_graph_to_numpy(g) for g in graphs])


def _concat_maybe(arrays):
    if all(a is None for a in arrays):
        return None
    present = [a for a in arrays if a is not None]
    return concatenate(present, axis=0)


def _unbatch_graphs(graph: GraphsTuple) -> List[GraphsTuple]:
    """Split features by lengths, then zip into graphs."""
    n_graphs = _host_concrete_int(shape(graph.n_node)[0])
    if not n_graphs:
        return []

    ones_g = ones((n_graphs,), dtype=graph.n_node.dtype, like=graph.n_node)
    nodes = split_by_lengths(graph.nodes, graph.n_node)
    edges = split_by_lengths(graph.edges, graph.n_edge)
    if graph.senders is None:
        senders = [None] * n_graphs
        receivers = [None] * n_graphs
    else:
        senders = unbatch_ids(graph.senders, graph.n_node, graph.n_edge)
        receivers = unbatch_ids(graph.receivers, graph.n_node, graph.n_edge)
    globals_ = split_by_lengths(graph.globals, ones_g)
    n_node = split_by_lengths(graph.n_node, ones_g)
    n_edge = split_by_lengths(graph.n_edge, ones_g)

    out = []
    for i in range(n_graphs):
        out.append(
            GraphsTuple(
                nodes=nodes[i],
                edges=edges[i],
                receivers=receivers[i],
                senders=senders[i],
                globals=globals_[i],
                n_node=n_node[i],
                n_edge=n_edge[i],
            )
        )
    return out


def unbatch_np(graph: GraphsTuple) -> List[GraphsTuple]:
    """NumPy implementation of :func:`unbatch`."""
    return unbatch(_graph_to_numpy(graph))


def pad_with_graphs(
    graph: GraphsTuple, n_node: int, n_edge: int, n_graph: int = 2
) -> GraphsTuple:
    """Pad with a dummy graph (padding nodes/edges) plus empty graphs.

    Not compilable (padding sizes are data-dependent). Requires ``n_graph >= 2``.
    """
    if n_graph < 2:
        raise ValueError(
            f"n_graph is {n_graph}, which is smaller than minimum value of 2."
        )
    pad_n_node = int(n_node - _sum_n_node(graph))
    pad_n_edge = int(n_edge - _sum_n_edge(graph))
    pad_n_graph = int(n_graph - _n_graphs(graph))
    if pad_n_node <= 0 or pad_n_edge < 0 or pad_n_graph <= 0:
        raise RuntimeError(
            "Given graph is too large for the given padding. difference: "
            f"n_node {pad_n_node}, n_edge {pad_n_edge}, n_graph {pad_n_graph}"
        )
    pad_n_empty_graph = pad_n_graph - 1
    like = graph.n_node
    idx_like = _like_index(graph)

    def pad_nodes(leaf):
        return _zeros_like_leading(leaf, pad_n_node)

    def pad_edges(leaf):
        return _zeros_like_leading(leaf, pad_n_edge)

    def pad_globals(leaf):
        return _zeros_like_leading(leaf, pad_n_graph)

    pad_senders = zeros((pad_n_edge,), dtype=idx_like.dtype, like=idx_like)
    padding_graph = GraphsTuple(
        n_node=concatenate(
            [
                astype(full((1,), pad_n_node, dtype=like.dtype, like=like), like.dtype),
                zeros((pad_n_empty_graph,), dtype=like.dtype, like=like),
            ],
            axis=0,
        ),
        n_edge=concatenate(
            [
                astype(full((1,), pad_n_edge, dtype=like.dtype, like=like), like.dtype),
                zeros((pad_n_empty_graph,), dtype=like.dtype, like=like),
            ],
            axis=0,
        ),
        nodes=_map_features(pad_nodes, graph.nodes),
        edges=_map_features(pad_edges, graph.edges),
        globals=_map_features(pad_globals, graph.globals),
        senders=pad_senders,
        receivers=zeros((pad_n_edge,), dtype=idx_like.dtype, like=idx_like),
    )
    return batch([graph, padding_graph])


def _flip0(x):
    xp = array_namespace(x)
    if hasattr(xp, "flip"):
        try:
            return xp.flip(x, axis=0)
        except TypeError:
            return xp.flip(x)
    return x[::-1]


def get_number_of_padding_with_graphs_graphs(padded_graph: GraphsTuple):
    """Number of padding graphs (dummy + trailing empty). Not for unpadded graphs."""
    n_node = padded_graph.n_node
    xp = array_namespace(n_node)
    reversed_empty = _flip0(n_node) == 0
    return xp.argmin(reversed_empty) + 1


def get_number_of_padding_with_graphs_nodes(padded_graph: GraphsTuple):
    """Number of padding nodes (the dummy graph's ``n_node``)."""
    n_pad = get_number_of_padding_with_graphs_graphs(padded_graph)
    return padded_graph.n_node[-n_pad]


def get_number_of_padding_with_graphs_edges(padded_graph: GraphsTuple):
    """Number of padding edges (the dummy graph's ``n_edge``)."""
    n_pad = get_number_of_padding_with_graphs_graphs(padded_graph)
    return padded_graph.n_edge[-n_pad]


def unpad_with_graphs(padded_graph: GraphsTuple) -> GraphsTuple:
    """Remove dummy + empty padding graphs. Not compilable."""
    n_padding_graph = int(np.asarray(get_number_of_padding_with_graphs_graphs(padded_graph)))
    n_padding_node = int(np.asarray(get_number_of_padding_with_graphs_nodes(padded_graph)))
    n_padding_edge = int(np.asarray(get_number_of_padding_with_graphs_edges(padded_graph)))

    def remove_node_padding(arr):
        if n_padding_node == 0:
            return arr
        return arr[:-n_padding_node]

    def remove_edge_padding(arr):
        if n_padding_edge == 0:
            return arr
        return arr[:-n_padding_edge]

    def remove_graph_padding(arr):
        return arr[:-n_padding_graph]

    return GraphsTuple(
        n_node=remove_graph_padding(padded_graph.n_node),
        n_edge=remove_graph_padding(padded_graph.n_edge),
        nodes=_map_features(remove_node_padding, padded_graph.nodes),
        edges=_map_features(remove_edge_padding, padded_graph.edges),
        globals=_map_features(remove_graph_padding, padded_graph.globals),
        senders=remove_edge_padding(padded_graph.senders)
        if padded_graph.senders is not None
        else None,
        receivers=remove_edge_padding(padded_graph.receivers)
        if padded_graph.receivers is not None
        else None,
    )


def _get_mask(padding_length, full_length, like):
    idx = arange(full_length, dtype=getattr(like, "dtype", None), like=like)
    return idx < (full_length - padding_length)


def get_node_padding_mask(padded_graph: GraphsTuple):
    """Boolean mask, True for real nodes. Needs node features (static length)."""
    n_padding_node = get_number_of_padding_with_graphs_nodes(padded_graph)
    leaves = tree.leaves(padded_graph.nodes)
    if not leaves:
        raise ValueError("`padded_graph` must have at least one array of node features")
    total_num_nodes = leaves[0].shape[0]
    return _get_mask(n_padding_node, total_num_nodes, like=leaves[0])


def get_edge_padding_mask(padded_graph: GraphsTuple):
    """Boolean mask, True for real edges."""
    n_padding_edge = get_number_of_padding_with_graphs_edges(padded_graph)
    if padded_graph.senders is None:
        raise ValueError("`padded_graph` must have senders to infer edge count")
    total_num_edges = padded_graph.senders.shape[0]
    return _get_mask(n_padding_edge, total_num_edges, like=padded_graph.senders)


def get_graph_padding_mask(padded_graph: GraphsTuple):
    """Boolean mask, True for real graphs."""
    n_padding_graph = get_number_of_padding_with_graphs_graphs(padded_graph)
    total_num_graphs = padded_graph.n_node.shape[0]
    return _get_mask(n_padding_graph, total_num_graphs, like=padded_graph.n_node)


def concatenated_args(update: Optional[Callable] = None, *, axis: int = -1):
    """Decorator that concatenates update_fn arguments along ``axis``."""

    def _decorate(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            combined = tree.leaves(args) + tree.leaves(kwargs)
            combined = [c for c in combined if c is not None]
            return f(concatenate(combined, axis=axis))

        return wrapper

    if update:
        return _decorate(update)
    return _decorate


def get_fully_connected_graph(
    n_node_per_graph: int,
    n_graph: int,
    node_features: Optional[ArrayTree] = None,
    global_features: Optional[ArrayTree] = None,
    add_self_edges: bool = True,
) -> GraphsTuple:
    """Fully connected graphs (optionally without self-edges). ``n_graph`` is static."""
    if node_features is not None:
        leaves = tree.leaves(node_features)
        if leaves and int(np.asarray(leaves[0].shape[0])) != n_node_per_graph * n_graph:
            raise ValueError(
                "Number of nodes is not equal to num_nodes_per_graph * n_graph."
            )
    if global_features is not None:
        leaves = tree.leaves(global_features)
        if leaves and int(np.asarray(leaves[0].shape[0])) != n_graph:
            raise ValueError("The number of globals is not equal to n_graph.")

    like = None
    if node_features is not None:
        fl = tree.leaves(node_features)
        if fl:
            like = fl[0]
    tmp_senders, tmp_receivers = np.meshgrid(
        np.arange(n_node_per_graph), np.arange(n_node_per_graph)
    )
    if not add_self_edges:
        tmp_senders = np.stack(
            [np.roll(row, -i) for i, row in enumerate(tmp_senders)]
        )[:, 1:]
        tmp_receivers = tmp_receivers[:, 1:]
    tmp_senders = tmp_senders.reshape(-1)
    tmp_receivers = tmp_receivers.reshape(-1)
    senders = []
    receivers = []
    n_edge = []
    for graph_idx in range(n_graph):
        offset = graph_idx * n_node_per_graph
        senders.append(tmp_senders + offset)
        receivers.append(tmp_receivers + offset)
        n_edge.append(len(tmp_senders))

    def _as(idx):
        arr = np.concatenate(idx, axis=0) if idx else np.array([], dtype=np.int32)
        if like is None:
            return arr
        return array_namespace(like).asarray(arr)

    n_node_arr: Any = np.array([n_node_per_graph] * n_graph, dtype=np.int32)
    n_edge_arr: Any = np.array(n_edge if n_edge else [0], dtype=np.int32)
    if like is not None:
        xp = array_namespace(like)
        n_node_arr = xp.asarray(n_node_arr)
        n_edge_arr = xp.asarray(n_edge_arr)
    return GraphsTuple(
        nodes=node_features,
        edges=None,
        n_node=n_node_arr,
        n_edge=n_edge_arr,
        senders=_as(senders),
        receivers=_as(receivers),
        globals=global_features,
    )


_NUMBER_FIELDS = ("n_node", "n_edge", "n_graph")


def _get_graph_size(graphs_tuple: GraphsTuple):
    n_node = int(np.asarray(graphs_tuple.n_node).sum())
    n_edge = (
        0 if graphs_tuple.senders is None else int(np.asarray(graphs_tuple.senders).shape[0])
    )
    n_graph = _n_graphs(graphs_tuple)
    return n_node, n_edge, n_graph


def _is_over_batch_size(graph, graph_batch_size):
    return any(x > y for x, y in zip(_get_graph_size(graph), graph_batch_size))


def dynamically_batch(
    graphs_tuple_iterator: Iterator[GraphsTuple], n_node: int, n_edge: int, n_graph: int
) -> Generator[GraphsTuple, None, None]:
    """Yield padded batches from an iterator of graphs (jraph algorithm)."""
    if n_graph < 2:
        raise ValueError(
            "The number of graphs in a batch size must be greater or "
            f"equal to `2` for padding with graphs, got {n_graph}."
        )
    valid_batch_size = (n_node - 1, n_edge, n_graph - 1)
    accumulated_graphs: list[GraphsTuple] = []
    num_accumulated_nodes = 0
    num_accumulated_edges = 0
    num_accumulated_graphs = 0
    for element in graphs_tuple_iterator:
        if not isinstance(element, GraphsTuple):
            raise RuntimeError("dynamically_batch iterator must yield GraphsTuple")
        element_nodes, element_edges, element_graphs = _get_graph_size(element)
        if _is_over_batch_size(element, valid_batch_size):
            if accumulated_graphs:
                yield pad_with_graphs(batch_np(accumulated_graphs), n_node, n_edge, n_graph)
            graph_size = dict(zip(_NUMBER_FIELDS, (element_nodes, element_edges, element_graphs)))
            batch_size = dict(zip(_NUMBER_FIELDS, valid_batch_size))
            raise RuntimeError(
                "Found graph bigger than batch size. Valid Batch "
                f"Size: {batch_size}, Graph Size: {graph_size}"
            )
        if not accumulated_graphs:
            accumulated_graphs = [element]
            num_accumulated_nodes = element_nodes
            num_accumulated_edges = element_edges
            num_accumulated_graphs = element_graphs
            continue
        if (
            (num_accumulated_graphs + element_graphs > n_graph - 1)
            or (num_accumulated_nodes + element_nodes > n_node - 1)
            or (num_accumulated_edges + element_edges > n_edge)
        ):
            yield pad_with_graphs(batch_np(accumulated_graphs), n_node, n_edge, n_graph)
            accumulated_graphs = [element]
            num_accumulated_nodes = element_nodes
            num_accumulated_edges = element_edges
            num_accumulated_graphs = element_graphs
        else:
            accumulated_graphs.append(element)
            num_accumulated_nodes += element_nodes
            num_accumulated_edges += element_edges
            num_accumulated_graphs += element_graphs
    if accumulated_graphs:
        yield pad_with_graphs(batch_np(accumulated_graphs), n_node, n_edge, n_graph)


def _expand_trailing_dimensions(array, template):
    missing = int(template.ndim) - int(array.ndim)
    if missing <= 0:
        return array
    shp = tuple(int(s) for s in np.asarray(array.shape)) + (1,) * missing
    return reshape(array, shp)


def zero_out_padding(graph: GraphsTuple) -> GraphsTuple:
    """Multiply padding nodes/edges/globals by zero (overflow guard)."""
    edge_mask = get_edge_padding_mask(graph)
    node_mask = get_node_padding_mask(graph)
    global_mask = get_graph_padding_mask(graph)

    def _apply(mask):
        def fn(x):
            m = _expand_trailing_dimensions(mask, x)
            xp = array_namespace(x)
            m = xp.astype(m, x.dtype) if hasattr(xp, "astype") else astype(m, x.dtype)
            return m * x

        return fn

    return graph._replace(
        nodes=_map_features(_apply(node_mask), graph.nodes),
        edges=_map_features(_apply(edge_mask), graph.edges),
        globals=_map_features(_apply(global_mask), graph.globals),
    )


def with_zero_out_padding_outputs(graph_net: Callable[[GraphsTuple], GraphsTuple]):
    """Wrap a graph-to-graph fn so padded outputs are zeroed."""

    @functools.wraps(graph_net)
    def wrapper(graph: GraphsTuple) -> GraphsTuple:
        return zero_out_padding(graph_net(graph))

    return wrapper


def sparse_matrix_to_graphs_tuple(senders, receivers, values, n_node) -> GraphsTuple:
    """COO sparse matrix → graph (values repeat senders/receivers)."""
    values_np = np.asarray(values)
    if values_np.size == 0:
        senders_out = np.array([], dtype=np.int32)
        receivers_out = np.array([], dtype=np.int32)
        n_edge = np.array([0])
    else:
        senders_out = np.repeat(np.asarray(senders), values_np)
        receivers_out = np.repeat(np.asarray(receivers), values_np)
        n_edge = np.array([int(values_np.sum())])
    return GraphsTuple(
        nodes=None,
        edges=None,
        receivers=receivers_out,
        senders=senders_out,
        globals=None,
        n_node=np.asarray(n_node),
        n_edge=n_edge,
    )
