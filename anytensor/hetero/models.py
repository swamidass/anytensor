"""Heterogeneous GNN layers as plain functions on :class:`HeteroGraphsTuple`.

Each function takes a graph plus callables / arrays for the learnable pieces.
Weight ownership stays in your framework (Flax, Haiku, ``torch.nn``, NumPy
prototypes) — pass ``lambda x: x @ W`` or a module ``__call__`` as needed.

Citations (acronym → full name)
-------------------------------
* **R-GCN** (Relational Graph Convolutional Network) — Schlichtkrull et al.,
  “Modeling Relational Data with Graph Convolutional Networks,” ESWC 2018.
  https://arxiv.org/abs/1703.06103
* **GraphSAGE** (SAmple and aggreGatE; hetero wrap) — Hamilton et al.,
  “Inductive Representation Learning on Large Graphs,” NeurIPS 2017.
  https://arxiv.org/abs/1706.02216
* **HAN** (Heterogeneous Graph Attention Network) — Wang et al., WWW 2019.
  https://arxiv.org/abs/1903.07293
* **HGT** (Heterogeneous Graph Transformer) — Hu et al., WWW 2020.
  https://arxiv.org/abs/2003.01332
* **CompGCN** (Composition-based Multi-Relational GCN) — Vashishth et al.,
  ICLR 2020. https://arxiv.org/abs/1911.03082
* **GAT** (Graph Attention Network) edge scores — Veličković et al., ICLR 2018.
  https://arxiv.org/abs/1710.10903 (used by :func:`gat_attention_logit` / HAN)
"""

from __future__ import annotations

import math
from typing import Callable, Mapping, Optional, Sequence

from anytensor.core import (
    concatenate,
    exp,
    max as at_max,
    maximum,
    reshape,
    shape,
    sum as at_sum,
    where,
)
from anytensor.namespace import array_namespace
from anytensor.segment import cache

from .graph import ArrayTree, CanonicalEtype, HeteroGraphsTuple, Ntype
from .message import (
    RelationSpec,
    attention_weight_messages,
    multi_update_all,
)

LinearFn = Callable[[ArrayTree], ArrayTree]
LogitFn = Callable[[ArrayTree, ArrayTree, Optional[ArrayTree]], ArrayTree]
ActivationFn = Callable[[ArrayTree], ArrayTree]


def _relu(x):
    return maximum(x, 0)


def _tanh(x):
    return array_namespace(x).tanh(x)


def _softmax_axis1(x):
    """Stable softmax over axis 1 for ``(n, R)`` scores."""
    m = reshape(at_max(x, axes=1), (shape(x)[0], 1))
    e = exp(x - m)
    return e / reshape(at_sum(e, axes=1), (shape(e)[0], 1))


def _identity(x):
    return x


def _src_message(apply: LinearFn):
    def msg(src, dst, edges):
        del dst, edges
        return apply(src)

    return msg


@cache
def relational_graph_convolution(
    graph: HeteroGraphsTuple,
    relation_apply: Mapping[CanonicalEtype, LinearFn],
    self_apply: Mapping[Ntype, LinearFn],
    *,
    activation: ActivationFn = _relu,
    reducer: str = "mean",
) -> HeteroGraphsTuple:
    """R-GCN (Relational Graph Convolutional Network) layer.

    Schlichtkrull et al., ESWC 2018. For each relation ``r``, messages are
    ``relation_apply[r](h_src)``, neighborhood-aggregated with ``reducer``
    (``mean`` ≈ ``1/|N_r(i)|``), then cross-summed. Destinations update as
    ``activation(self_apply[n](h) + mailbox)``.

    Args:
        graph: Input heterograph.
        relation_apply: Per-etype ``h_src -> message`` (usually a linear).
        self_apply: Per-ntype self / root term (``W_0`` in the paper).
        activation: Pointwise nonlinearity (default ReLU).
        reducer: Per-relation segment reduce (``mean`` or ``sum``).
    """
    etype_dict = {
        etype: RelationSpec(message_fn=_src_message(fn), reduce=reducer)  # type: ignore[arg-type]
        for etype, fn in relation_apply.items()
    }
    mail = multi_update_all(graph, etype_dict, cross_reducer="sum")
    nodes = {
        ntype: activation(self_apply[ntype](graph.nodes[ntype]) + mail.nodes[ntype])
        for ntype in self_apply
    }
    merged = dict(graph.nodes)
    merged.update(nodes)
    return graph.update(nodes=merged)


@cache
def hetero_sage(
    graph: HeteroGraphsTuple,
    relation_apply: Mapping[CanonicalEtype, LinearFn],
    combine_apply: Mapping[Ntype, LinearFn],
    *,
    activation: ActivationFn = _relu,
) -> HeteroGraphsTuple:
    """Heterogeneous GraphSAGE mean layer (Hamilton et al., NeurIPS 2017).

    GraphSAGE (SAmple and aggreGatE): per-relation map on sources, ``mean``
    aggregate, cross ``sum``, then
    ``activation(combine_apply[n](concat[h_self, mailbox]))``.
    Apply is ``@cache`` (same pattern as GraphNetwork).
    """
    etype_dict = {
        etype: RelationSpec(message_fn=_src_message(fn), reduce="mean")
        for etype, fn in relation_apply.items()
    }
    mail = multi_update_all(graph, etype_dict, cross_reducer="sum")
    nodes = {
        ntype: activation(
            combine(concatenate([graph.nodes[ntype], mail.nodes[ntype]], axis=-1))
        )
        for ntype, combine in combine_apply.items()
    }
    merged = dict(graph.nodes)
    merged.update(nodes)
    return graph.update(nodes=merged)


@cache
def comp_gcn(
    graph: HeteroGraphsTuple,
    relation_apply: Mapping[CanonicalEtype, LinearFn],
    self_apply: Mapping[Ntype, LinearFn],
    *,
    composition: str = "mult",
    activation: ActivationFn = _relu,
    reducer: str = "sum",
) -> HeteroGraphsTuple:
    """CompGCN (Composition-based Multi-Relational GCN) layer.

    Vashishth et al., ICLR 2020. Requires edge features on each used etype.

    * ``mult`` — ``relation_apply[r](h_src * e)``
    * ``sum`` — ``relation_apply[r](h_src + e)``

    Then segment ``reducer``, cross ``sum``, and
    ``activation(self_apply[n](h) + mailbox)``.
    """
    if composition not in ("mult", "sum"):
        raise ValueError("composition must be 'mult' or 'sum'")

    def make_msg(apply: LinearFn):
        def msg(src, dst, edges):
            del dst
            if edges is None:
                raise ValueError("comp_gcn requires edge features on every etype")
            composed = src * edges if composition == "mult" else src + edges
            return apply(composed)

        return msg

    etype_dict = {
        etype: RelationSpec(message_fn=make_msg(fn), reduce=reducer)  # type: ignore[arg-type]
        for etype, fn in relation_apply.items()
    }
    mail = multi_update_all(graph, etype_dict, cross_reducer="sum")
    nodes = {
        ntype: activation(self_apply[ntype](graph.nodes[ntype]) + mail.nodes[ntype])
        for ntype in self_apply
    }
    merged = dict(graph.nodes)
    merged.update(nodes)
    return graph.update(nodes=merged)


@cache
def han(
    graph: HeteroGraphsTuple,
    meta_path_etypes: Sequence[CanonicalEtype],
    node_message: Mapping[CanonicalEtype, LinearFn],
    node_attention_logit: Mapping[CanonicalEtype, LogitFn],
    semantic_project: LinearFn,
    semantic_query: ArrayTree,
    *,
    node_activation: ActivationFn = _relu,
    semantic_activation: ActivationFn = _tanh,
) -> HeteroGraphsTuple:
    """HAN (Heterogeneous Graph Attention Network) layer.

    Wang et al., WWW 2019: node-level + semantic attention. Each
    ``meta_path_etypes`` entry is a meta-path hop already stored as a
    canonical etype (precompute longer paths as their own etypes).

    1. **Node-level attention** — ``node_message[e](h_src)``, logits from
       ``node_attention_logit[e](src, dst, edges)``, softmax over neighbors
       (GAT-style).
    2. Mailboxes **stacked**; **semantic attention** mixes path embeddings
       with ``semantic_query`` after ``semantic_project``.
    Apply is ``@cache`` (same pattern as GraphNetwork).
    """
    if not meta_path_etypes:
        raise ValueError("han requires at least one meta-path etype")

    etype_dict = {
        etype: RelationSpec(
            message_fn=_src_message(node_message[etype]),
            reduce="sum",
            attention_logit_fn=node_attention_logit[etype],
            attention_reduce_fn=attention_weight_messages,
        )
        for etype in meta_path_etypes
    }
    stacked = multi_update_all(graph, etype_dict, cross_reducer="stack")
    nodes = dict(graph.nodes)
    for ntype in {e[2] for e in meta_path_etypes}:
        h_stack = stacked.nodes[ntype]  # (n, R, d)
        h_act = node_activation(h_stack)
        n, r, d = shape(h_act)[:3]
        flat = reshape(h_act, (-1, d))
        proj = semantic_project(flat)
        d_s = shape(proj)[-1]
        proj = reshape(proj, (n, r, d_s))
        proj = semantic_activation(proj)
        q_vec = reshape(semantic_query, (d_s,))
        score = at_sum(proj * q_vec, axes=-1)
        alpha = reshape(_softmax_axis1(score), (n, r, 1))
        nodes[ntype] = at_sum(h_act * alpha, axes=1)
    return graph.update(nodes=nodes)


@cache
def hgt(
    graph: HeteroGraphsTuple,
    message_apply: Mapping[CanonicalEtype, LinearFn],
    attention_logit: Mapping[CanonicalEtype, LogitFn],
    target_apply: Mapping[Ntype, LinearFn],
    *,
    activation: ActivationFn = _identity,
    scale: Optional[float] = None,
) -> HeteroGraphsTuple:
    """HGT (Heterogeneous Graph Transformer) style layer.

    Hu et al., WWW 2020 — typed attention + target projection. Full HGT uses
    typed query/key/value and edge-type matrices (often multi-head). Fold
    those into the callables you pass:

    * ``message_apply[etype](h_src)`` — value / message projection.
    * ``attention_logit[etype](src, dst, edges)`` — edge logits (include
      ``1/sqrt(d)`` here, or set ``scale``).
    * Softmax over neighbors, weighted sum, cross ``sum`` across etypes.
    * ``target_apply[ntype]`` — target-type output projection.
    Apply is ``@cache`` (same pattern as GraphNetwork).
    """

    def maybe_scale(logit_fn: LogitFn) -> LogitFn:
        if scale is None:
            return logit_fn
        inv = math.sqrt(float(scale))

        def scaled(src, dst, edges):
            return logit_fn(src, dst, edges) / inv

        return scaled

    etype_dict = {
        etype: RelationSpec(
            message_fn=_src_message(message_apply[etype]),
            reduce="sum",
            attention_logit_fn=maybe_scale(attention_logit[etype]),
            attention_reduce_fn=attention_weight_messages,
        )
        for etype in message_apply
    }
    mail = multi_update_all(graph, etype_dict, cross_reducer="sum")
    nodes = {
        ntype: activation(target_apply[ntype](mail.nodes[ntype]))
        for ntype in target_apply
    }
    merged = dict(graph.nodes)
    merged.update(nodes)
    return graph.update(nodes=merged)


def gat_attention_logit(
    src: ArrayTree,
    dst: ArrayTree,
    attn_vec_apply: LinearFn,
    *,
    negative_slope: float = 0.2,
) -> ArrayTree:
    """GAT-style edge score: ``LeakyReLU(attn_vec_apply(concat(src, dst)))``.

    Graph Attention Network (GAT; Veličković et al., ICLR 2018) neighborhood
    scoring. ``attn_vec_apply`` maps concatenated features to shape ``(E, 1)``
    or ``(E,)``. Typical use inside a logit callable::

        lambda s, d, e: gat_attention_logit(s, d, my_linear)
    """
    cat = concatenate([src, dst], axis=-1)
    x = attn_vec_apply(cat)
    return where(x > 0, x, x * negative_slope)
