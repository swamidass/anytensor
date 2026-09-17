"""Heterogeneous message passing (DGL-style multi-relation updates).

Per relation: optionally map **source nodes** (``src_apply``, size ``N_src``),
gather along ``senders``, then segment-reduce onto ``receivers``. Optional
per-edge attention (same pattern as :func:`anytensor.jraph.GraphNetwork`)
weights messages before the reduce. Across relations that share a destination
ntype: fuse with an explicit cross-reducer (order-independent).

Efficiency
----------
Prefer ``src_apply`` + :func:`copy_u_message` for source-only linears so the
map runs on ``N_src`` before gather. Applying inside ``message_fn`` after
gather costs ``E`` rows — usually worse when ``E > N``, sometimes better on
extremely sparse graphs or when the map needs edge/destination features
(CompGCN). Keep etypes separate (different edge counts; no interleaved
multi-relation edge tensor).

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
Optional ``attention_logit_fn`` + ``attention_reduce_fn`` on a relation use
:func:`~anytensor.segment.segment_attention` (or the same pieces:
:func:`~anytensor.segment.segment_softmax` on ``receivers``, weight messages,
segment reduce). That is **vectorized** over edges — no Python loop over
messages. Schema-sized Python loops over etypes only (typically a handful of
relations) are unrolled at compile time. See :mod:`anytensor.hgraph.models`.
"""

from __future__ import annotations

from typing import Any, Callable, Literal, Mapping, NamedTuple, Optional, Sequence, Union

from anytensor import tree
from anytensor.core import maximum, minimum, shape, stack, take
from anytensor.core import _host_concrete_int
from anytensor.segment import (
    segment_attention,
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
# message_fn(src, dst, edges) -> messages  (all leading axis = E edges; see RelationSpec)
MessageFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]
# Same gathered (src, dst, edges) layout as message_fn; returns per-edge logits.
AttentionLogitFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]
# (messages, softmax_weights) -> weighted messages.
AttentionReduceFn = Callable[[ArrayTree, ArrayTree], ArrayTree]
# src_apply(nodes[src]) -> transformed source pool (leading axis = N_src).
SrcApplyFn = Callable[[ArrayTree], ArrayTree]


class RelationSpec(NamedTuple):
    """Per-relation update for :func:`multi_update_all`.

    Attributes:
        message_fn: Callable ``(src, dst, edges) -> messages``.

            All three arguments are **edge-aligned** (leading size ``E`` for
            this etype), after gather:

            * ``src`` — source node features indexed by ``senders``
              (``take(nodes[src_ntype], senders)``). If :attr:`src_apply` is
              set, this is ``take(src_apply(nodes[src_ntype]), senders)``
              instead.
            * ``dst`` — destination node features indexed by ``receivers``.
            * ``edges`` — ``graph.edges[etype]``, or ``None`` if that slot is
              empty. When present, leading size should match ``E``.

            Return value is the per-edge message tensor (or pytree of
            tensors), leading size ``E``, later segment-reduced onto
            destinations. Default helper: :func:`copy_u_message` (returns
            ``src``, ignores ``dst`` / ``edges``).

            Prefer :attr:`src_apply` + :func:`copy_u_message` for
            source-only linears (map on ``N_src`` before gather). Use
            ``message_fn`` when the map needs ``dst`` or ``edges``, or when
            you deliberately want an edge-sized map (e.g. ``E ≪ N``).

        reduce: Segment reduce after optional attention (``sum`` with attention).
        attention_logit_fn: Optional ``(src, dst, edges) -> logits`` with the
            **same gathered layout** as ``message_fn``. Logits use the **raw**
            source pool (not ``src_apply`` output) so scoring stays on
            pre-message features. Broadcastable against messages (often
            ``(E,)`` or ``(E, 1)``).
        attention_reduce_fn: Optional ``(messages, weights) -> messages``.
            Defaults to element-wise multiply when only ``attention_logit_fn``
            is set.
        src_apply: Optional ``(nodes[src_ntype]) -> …`` on the **node** pool
            (leading size ``N_src``) **before** gather. Only affects the
            ``src`` argument passed into ``message_fn`` / ``copy_u``; does not
            change ``attention_logit_fn``'s ``src``.
    """

    message_fn: MessageFn
    reduce: ReduceName = "sum"
    attention_logit_fn: Optional[AttentionLogitFn] = None
    attention_reduce_fn: Optional[AttentionReduceFn] = None
    src_apply: Optional[SrcApplyFn] = None


def attention_weight_messages(messages: ArrayTree, weights: ArrayTree) -> ArrayTree:
    """Default attention reduce: element-wise ``messages * weights``.

    Same pattern as Graph Attention Networks (GAT): after
    :func:`~anytensor.segment.segment_softmax`, multiply messages by the
    per-edge weights.
    """
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
    """DGL ``fn.copy_u``: message is the gathered source feature.

    Signature matches :data:`MessageFn`: ``(src, dst, edges) -> messages``.
    Returns ``src``; ``dst`` and ``edges`` are ignored. Pair with
    :attr:`RelationSpec.src_apply` for source linears on nodes.
    """
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
    src_apply: Optional[SrcApplyFn] = None,
):
    """Per-relation messages reduced onto destination nodes (DGL type-wise step).

    With :func:`copy_u_message`, ``reduce`` matches DGL ``fn.copy_u`` +
    ``fn.sum``/``fn.mean``/``fn.max``/``fn.min``. Empty destinations are ``0``
    (including max/min via ``segment_*_or_constant``).

    ``message_fn`` signature
        ``message_fn(src, dst, edges) -> messages`` where every argument is
        **edge-aligned** (leading size ``E``):

        * ``src`` — ``take`` of the source node pool along ``senders``
          (after :attr:`~RelationSpec.src_apply` when that is set).
        * ``dst`` — ``take`` of the destination node pool along ``receivers``.
        * ``edges`` — ``graph.edges[etype]`` or ``None``.

        Return per-edge messages (leading ``E``) for the segment reduce /
        attention path. See :class:`RelationSpec` for the full contract.

    When ``attention_logit_fn`` is set, neighborhood attention is applied with
    :func:`~anytensor.segment.segment_attention` when using the default
    weight-and-sum path (``attention_reduce_fn`` omitted or
    :func:`attention_weight_messages` with ``reduce="sum"``). Custom
    ``attention_reduce_fn`` still gets ``segment_softmax`` weights then your
    reduce. Edge ops are vectorized (``take`` / ``segment_*``); there is no
    Python loop over messages.

    ``src_apply``, if set, transforms ``graph.nodes[src]`` **before** gather.
    Use that for source-only linears (``src_apply=W`` + ``copy_u_message``)
    so the matmul runs on ``N_src`` rather than ``E``. ``attention_logit_fn``
    still receives gathered **raw** source features (``src_apply`` affects
    messages only).
    """
    if etype not in graph.n_edge:
        raise KeyError(f"etype {etype!r} not in graph")
    if reduce not in _SEGMENT_REDUCE:
        raise ValueError(f"unknown reduce {reduce!r}")
    if attention_reduce_fn is not None and attention_logit_fn is None:
        raise ValueError("attention_logit_fn is required when attention_reduce_fn is set")
    use_default_attn = attention_logit_fn is not None and (
        attention_reduce_fn is None or attention_reduce_fn is attention_weight_messages
    )
    if attention_logit_fn is not None and attention_reduce_fn is None:
        attention_reduce_fn = attention_weight_messages

    src, _rel, dst = etype
    receivers = graph.receivers[etype]
    senders = graph.senders[etype]
    src_pool = graph.nodes[src]
    # Optional node-side map before gather (cheap when E > N for src linears).
    msg_pool = src_apply(src_pool) if src_apply is not None else src_pool
    src_for_msg = _take_nodes(msg_pool, senders)
    # Attention scores stay on the raw source pool when src_apply is set.
    src_for_logit = (
        _take_nodes(src_pool, senders) if src_apply is not None else src_for_msg
    )
    dst_nodes = _take_nodes(graph.nodes[dst], receivers)
    edges = graph.edges.get(etype)
    messages = message_fn(src_for_msg, dst_nodes, edges)
    num_dst = _leading(graph.nodes[dst])

    if attention_logit_fn is not None:
        logits = attention_logit_fn(src_for_logit, dst_nodes, edges)
        if use_default_attn and reduce == "sum":
            # Preferred path: one vectorized segment_attention per leaf.
            return tree.map(
                lambda m, logit: segment_attention(m, logit, receivers, num_dst),
                messages,
                logits,
            )
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

    Per etype: optional ``src_apply``, gather, ``message_fn(src, dst, edges)``,
    optional attention, segment-reduce onto destinations. Then fuse mailboxes
    that share a destination ntype with ``cross_reducer``.

    Args:
        graph: Heterogeneous graph(s).
        etype_dict: Optional map ``etype ->`` :class:`RelationSpec`,
            ``message_fn``, ``(message_fn, reduce)``, or
            ``(message_fn, reduce, attention_logit_fn, attention_reduce_fn)``.
            A bare ``message_fn`` is ``(src, dst, edges) -> messages`` on
            **gathered** edge-sized tensors (see :class:`RelationSpec`).
            Prefer ``RelationSpec(message_fn=copy_u_message, src_apply=…)``
            for source-only linears. Default: ``copy_u`` + ``reduce`` for
            every etype in ``etypes`` / ``canonical_etypes()``.
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
            src_apply=spec.src_apply,
        )

    new_nodes = dict(graph.nodes)
    dst_types = {e[2] for e in mailboxes}
    for ntype in dst_types:
        parts = [mailboxes[e] for e in mailboxes if e[2] == ntype]
        new_nodes[ntype] = _cross_reduce(parts, cross_reducer)
    return graph.update(nodes=new_nodes)
