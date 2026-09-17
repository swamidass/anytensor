"""Heterogeneous message passing (DGL-style multi-relation updates).

Per relation: gather source node features along ``senders`` (``copy_u``), then
segment-reduce onto ``receivers``. Optional per-edge attention (same pattern as
:func:`anytensor.jraph.GraphNetwork`) weights messages before the reduce.
Across relations that share a destination ntype: fuse with an explicit
cross-reducer (order-independent).

DGL ``multi_update_all`` alignment
---------------------------------
With ``copy_u_message`` (DGL ``fn.copy_u``) or ``u_mul_e``-style messages:

* **Per-relation** ``reduce="sum"|"mean"|"max"|"min"`` — values match DGL
  ``fn.sum`` / ``fn.mean`` / ``fn.max`` / ``fn.min``. Empty destinations are
  ``0`` (``max``/``min`` use :func:`~anytensor.segment.segment_max_or_constant`
  / :func:`~anytensor.segment.segment_min_or_constant`, not raw segment
  ``±inf`` identities).
* **Cross-reducers** ``sum`` / ``mean`` / ``max`` / ``min`` / ``stack`` —
  fuse those zero-filled mailboxes. This matches DGL ``multi_update_all`` when
  every destination receives every involved etype (and matches composing DGL
  per-etype ``update_all`` then the same cross fuse when coverage is partial).
  ``stack`` inserts a new axis at position ``1`` (shape
  ``(n_dst, n_relations, ...)``, DGL convention); relation order follows
  ``etype_dict`` insertion order.

Attention (GAT / HAN node-level / HGT)
--------------------------------------
Optional ``attention_logit_fn`` + ``attention_reduce_fn`` on a relation mirror
homo :func:`~anytensor.jraph.GraphNetwork` attention: logits →
:func:`~anytensor.segment.segment_softmax` on ``receivers`` → weight messages →
segment reduce (typically ``sum``). See :mod:`anytensor.hetero.models`.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, NamedTuple, Optional, Sequence, Union

from anytensor import tree
from anytensor.core import maximum, minimum, shape, stack, take
from anytensor.core import _host_concrete_int
from anytensor.segment import (
    segment_max_or_constant,
    segment_mean,
    segment_min_or_constant,
    segment_softmax,
    segment_sum,
)

from .graph import CanonicalEtype, HeteroGraphsTuple

ReduceName = Literal["sum", "mean", "max", "min"]
CrossReduceName = Literal["sum", "mean", "max", "min", "stack"]

_SEGMENT_REDUCE = {
    "sum": segment_sum,
    "mean": segment_mean,
    # DGL fills empty destinations with 0 for max/min; use *_or_constant.
    "max": segment_max_or_constant,
    "min": segment_min_or_constant,
}

ArrayTree = Any
MessageFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]
# (src, dst, edges) -> per-edge logits (broadcastable against messages).
AttentionLogitFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]
# (messages, softmax_weights) -> weighted messages.
AttentionReduceFn = Callable[[ArrayTree, ArrayTree], ArrayTree]


class RelationSpec(NamedTuple):
    """Per-relation update for :func:`multi_update_all`.

    Attributes:
        message_fn: ``(src, dst, edges) -> messages``.
        reduce: Segment reduce after optional attention (``sum`` with attention).
        attention_logit_fn: Optional ``(src, dst, edges) -> logits``.
        attention_reduce_fn: Optional ``(messages, weights) -> messages``.
            Defaults to element-wise multiply when only ``attention_logit_fn``
            is set.
    """

    message_fn: MessageFn
    reduce: ReduceName = "sum"
    attention_logit_fn: Optional[AttentionLogitFn] = None
    attention_reduce_fn: Optional[AttentionReduceFn] = None


def attention_weight_messages(messages: ArrayTree, weights: ArrayTree) -> ArrayTree:
    """Default attention reduce: element-wise ``messages * weights`` (GAT-style)."""
    return tree.map(lambda m, w: m * w, messages, weights)


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
        # DGL stacks relation mailboxes on axis 1 → (n_dst, n_rel, ...).
        return stack(list(parts), axis=1)
    raise ValueError(f"unknown cross_reducer {cross_reducer!r}")


def _cross_reduce(parts: Sequence[ArrayTree], cross_reducer: CrossReduceName) -> ArrayTree:
    if not parts:
        raise ValueError("cross_reducer requires at least one relation result")
    if len(parts) == 1:
        return parts[0]
    return tree.map(lambda *xs: _cross_reduce_leaves(xs, cross_reducer), *parts)


def _parse_relation_spec(spec, default_reduce: ReduceName) -> RelationSpec:
    if isinstance(spec, RelationSpec):
        return spec
    if callable(spec):
        return RelationSpec(message_fn=spec, reduce=default_reduce)
    if isinstance(spec, tuple):
        if len(spec) == 1:
            return RelationSpec(message_fn=spec[0], reduce=default_reduce)
        if len(spec) == 2:
            return RelationSpec(message_fn=spec[0], reduce=spec[1])
        if len(spec) == 4:
            return RelationSpec(
                message_fn=spec[0],
                reduce=spec[1],
                attention_logit_fn=spec[2],
                attention_reduce_fn=spec[3],
            )
        raise ValueError(
            "etype spec tuple must be (message_fn,), (message_fn, reduce), "
            "or (message_fn, reduce, attention_logit_fn, attention_reduce_fn)"
        )
    raise TypeError(
        f"etype spec must be MessageFn, RelationSpec, or tuple; got {type(spec)!r}"
    )


def relation_mailbox(
    graph: HeteroGraphsTuple,
    etype: CanonicalEtype,
    *,
    message_fn: MessageFn = copy_u_message,
    reduce: ReduceName = "sum",
    attention_logit_fn: Optional[AttentionLogitFn] = None,
    attention_reduce_fn: Optional[AttentionReduceFn] = None,
):
    """Per-relation messages reduced onto destination nodes (DGL type-wise step).

    With :func:`copy_u_message`, ``reduce`` matches DGL ``fn.copy_u`` +
    ``fn.sum``/``fn.mean``/``fn.max``/``fn.min``. Empty destinations are ``0``
    (including max/min via ``segment_*_or_constant``).

    When ``attention_logit_fn`` is set, logits are softmax-normalized per
    destination (``receivers``) and ``attention_reduce_fn`` weights messages
    before the segment reduce — same flow as
    :func:`anytensor.jraph.GraphNetwork` attention. Omit
    ``attention_reduce_fn`` to default to :func:`attention_weight_messages`.
    With attention, prefer ``reduce="sum"``.
    """
    if etype not in graph.n_edge:
        raise KeyError(f"etype {etype!r} not in graph")
    if reduce not in _SEGMENT_REDUCE:
        raise ValueError(f"unknown reduce {reduce!r}")
    if attention_reduce_fn is not None and attention_logit_fn is None:
        raise ValueError("attention_logit_fn is required when attention_reduce_fn is set")
    if attention_logit_fn is not None and attention_reduce_fn is None:
        attention_reduce_fn = attention_weight_messages

    src, _rel, dst = etype
    receivers = graph.receivers[etype]
    src_nodes = _take_nodes(graph.nodes[src], graph.senders[etype])
    dst_nodes = _take_nodes(graph.nodes[dst], receivers)
    edges = graph.edges.get(etype)
    messages = message_fn(src_nodes, dst_nodes, edges)
    num_dst = _leading(graph.nodes[dst])

    if attention_logit_fn is not None:
        logits = attention_logit_fn(src_nodes, dst_nodes, edges)
        weights = tree.map(
            lambda logit: segment_softmax(logit, receivers, num_dst),
            logits,
        )
        messages = attention_reduce_fn(messages, weights)

    return _reduce_messages(messages, receivers, num_dst, reduce)


def multi_update_all(
    graph: HeteroGraphsTuple,
    etype_dict: Optional[
        Mapping[CanonicalEtype, Union[MessageFn, RelationSpec, tuple]]
    ] = None,
    cross_reducer: CrossReduceName = "sum",
    *,
    reduce: ReduceName = "sum",
    etypes: Optional[Sequence[CanonicalEtype]] = None,
) -> HeteroGraphsTuple:
    """Multi-relation update aligned with DGL ``multi_update_all``.

    Per etype: message + optional attention + segment-reduce onto destination
    nodes. Then fuse mailboxes that share a destination ntype with
    ``cross_reducer``.

    Args:
        graph: Heterogeneous graph(s).
        etype_dict: Optional map ``etype ->`` :class:`RelationSpec`,
            ``message_fn``, ``(message_fn, reduce)``, or
            ``(message_fn, reduce, attention_logit_fn, attention_reduce_fn)``.
            Default: ``copy_u`` + ``reduce`` for every etype in ``etypes`` /
            ``canonical_etypes()``.
        cross_reducer: Fuse per-relation mailboxes for the same destination
            ntype. ``sum`` / ``mean`` / ``max`` / ``min`` / ``stack`` match DGL
            when per-relation mailboxes match (see module docstring). ``stack``
            uses axis ``1`` (DGL shape ``(n_dst, n_relations, ...)``); order is
            ``etype_dict`` insertion order.
        reduce: Default per-relation segment reduce when not set in
            ``etype_dict``. ``sum``/``mean``/``max``/``min`` match DGL
            (empty destinations ``0``).
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
    for etype, raw in etype_dict.items():
        spec = _parse_relation_spec(raw, reduce)
        mailboxes[etype] = relation_mailbox(
            graph,
            etype,
            message_fn=spec.message_fn,
            reduce=spec.reduce,
            attention_logit_fn=spec.attention_logit_fn,
            attention_reduce_fn=spec.attention_reduce_fn,
        )

    new_nodes = dict(graph.nodes)
    dst_types = {e[2] for e in mailboxes}
    for ntype in dst_types:
        parts = [mailboxes[e] for e in mailboxes if e[2] == ntype]
        new_nodes[ntype] = _cross_reduce(parts, cross_reducer)
    return graph.update(nodes=new_nodes)
