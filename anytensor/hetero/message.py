"""Heterogeneous message passing (DGL-style multi-relation updates).

Per relation: gather source node features along ``senders`` (``copy_u``), then
segment-reduce onto ``receivers``. Across relations that share a destination
ntype: fuse with an explicit cross-reducer (order-independent; same contract as
DGL ``multi_update_all``).
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, Optional, Sequence, Union

from anytensor import tree
from anytensor.core import maximum, minimum, shape, take
from anytensor.core import _host_concrete_int
from anytensor.segment import segment_max, segment_mean, segment_min, segment_sum

from .graph import CanonicalEtype, HeteroGraphsTuple

ReduceName = Literal["sum", "mean", "max", "min"]
CrossReduceName = Literal["sum", "mean", "max", "min", "stack"]

_SEGMENT_REDUCE = {
    "sum": segment_sum,
    "mean": segment_mean,
    "max": segment_max,
    "min": segment_min,
}

ArrayTree = Any
MessageFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]


def _leading(nodes) -> int:
    leaves = tree.leaves(nodes)
    if not leaves:
        raise ValueError("destination ntype has no feature leaves to size against")
    n = _host_concrete_int(shape(leaves[0])[0])
    if n is None:
        raise ValueError("multi_update_all requires a concrete destination size")
    return n


def _take_nodes(nodes, index):
    if nodes is None:
        return None
    return tree.map(lambda n: take(n, index), nodes)


def copy_u_message(src_nodes, dst_nodes, edges):
    """DGL ``fn.copy_u``: message is the source node feature (ignore dst/edge)."""
    del dst_nodes, edges
    return src_nodes


def _reduce_messages(messages, receivers, num_dst: int, reduce: ReduceName):
    fn = _SEGMENT_REDUCE[reduce]
    return tree.map(lambda m: fn(m, receivers, num_dst), messages)


def _cross_reduce_leaves(parts: Sequence[Any], cross_reducer: CrossReduceName):
    if not parts:
        raise ValueError("cross_reducer requires at least one relation result")
    if len(parts) == 1:
        return parts[0]
    if cross_reducer == "sum":
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out
    if cross_reducer == "mean":
        out = parts[0]
        for p in parts[1:]:
            out = out + p
        return out / len(parts)
    if cross_reducer == "max":
        out = parts[0]
        for p in parts[1:]:
            out = maximum(out, p)
        return out
    if cross_reducer == "min":
        out = parts[0]
        for p in parts[1:]:
            out = minimum(out, p)
        return out
    if cross_reducer == "stack":
        from anytensor.core import stack

        return stack(list(parts), axis=0)
    raise ValueError(f"unknown cross_reducer {cross_reducer!r}")


def _cross_reduce(parts: Sequence[ArrayTree], cross_reducer: CrossReduceName) -> ArrayTree:
    if not parts:
        raise ValueError("cross_reducer requires at least one relation result")
    if len(parts) == 1:
        return parts[0]
    return tree.map(lambda *xs: _cross_reduce_leaves(xs, cross_reducer), *parts)


def relation_mailbox(
    graph: HeteroGraphsTuple,
    etype: CanonicalEtype,
    *,
    message_fn: MessageFn = copy_u_message,
    reduce: ReduceName = "sum",
):
    """Per-relation messages reduced onto destination nodes (DGL type-wise step)."""
    if etype not in graph.n_edge:
        raise KeyError(f"etype {etype!r} not in graph")
    if reduce not in _SEGMENT_REDUCE:
        raise ValueError(f"unknown reduce {reduce!r}")
    src, _rel, dst = etype
    src_nodes = _take_nodes(graph.nodes[src], graph.senders[etype])
    dst_nodes = _take_nodes(graph.nodes[dst], graph.receivers[etype])
    edges = graph.edges.get(etype)
    messages = message_fn(src_nodes, dst_nodes, edges)
    num_dst = _leading(graph.nodes[dst])
    return _reduce_messages(messages, graph.receivers[etype], num_dst, reduce)


def multi_update_all(
    graph: HeteroGraphsTuple,
    etype_dict: Optional[
        Mapping[CanonicalEtype, Union[MessageFn, tuple]]
    ] = None,
    cross_reducer: CrossReduceName = "sum",
    *,
    reduce: ReduceName = "sum",
    etypes: Optional[Sequence[CanonicalEtype]] = None,
) -> HeteroGraphsTuple:
    """DGL-like multi-relation update: per-etype reduce, then cross-type fuse.

    Args:
        graph: Heterogeneous graph(s).
        etype_dict: Optional map ``etype -> message_fn`` or
            ``(message_fn, reduce_name)``. Default: ``copy_u`` + ``reduce`` for
            every etype in ``etypes`` / ``canonical_etypes()``.
        cross_reducer: Fuse per-relation mailboxes that share a destination
            ntype (``sum`` / ``mean`` / ``max`` / ``min`` / ``stack``).
        reduce: Default per-relation segment reduce when not set in
            ``etype_dict``.
        etypes: Subset of relations when ``etype_dict`` is omitted.

    Returns:
        A new :class:`HeteroGraphsTuple` whose destination node features are
        replaced by the cross-reduced mailboxes (same as DGL writing the
        reduced feature). Source-only ntypes are unchanged.
    """
    if etype_dict is None:
        keys = list(etypes) if etypes is not None else list(graph.canonical_etypes())
        etype_dict = {e: copy_u_message for e in keys}
    if not etype_dict:
        return graph

    mailboxes: dict[CanonicalEtype, ArrayTree] = {}
    for etype, spec in etype_dict.items():
        if isinstance(spec, tuple):
            message_fn, red = spec[0], spec[1] if len(spec) > 1 else reduce
        else:
            message_fn, red = spec, reduce
        mailboxes[etype] = relation_mailbox(
            graph, etype, message_fn=message_fn, reduce=red
        )

    new_nodes = dict(graph.nodes)
    dst_types = {e[2] for e in mailboxes}
    for ntype in dst_types:
        parts = [mailboxes[e] for e in mailboxes if e[2] == ntype]
        new_nodes[ntype] = _cross_reduce(parts, cross_reducer)
    return graph.update(nodes=new_nodes)
