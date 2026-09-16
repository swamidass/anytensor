"""Heterogeneous graphs: ``SendRecvTuple``, ``HeteroGraphsTuple``, batch magic.

Separate from the jraph-mirroring package. Batching goes through
:func:`anytensor.tree.batch` / ``unbatch`` via ``__tree_batch__`` /
``__tree_unbatch__`` (same hooks as :class:`~anytensor.jraph.GraphsTuple`).

Batched inputs must share the same ntype / etype keys. Callers that need to
merge unequal schemas should insert empty (length-0) features and zero
``n_node`` / ``n_edge`` entries themselves before calling ``batch``.
"""

from __future__ import annotations

from typing import (
    Any,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from anytensor import tree
from anytensor.core import concatenate, ones, reshape, shape, sum as at_sum
from anytensor.core import _host_concrete_int
from anytensor.lengths import batch_ids, split_by_lengths, unbatch_ids

ArrayTree = Union[Any, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
CanonicalEtype = Tuple[str, str, str]
Ntype = str

_UNSET = object()


def _sum_int(x) -> int:
    return int(at_sum(x))


def _n_graphs_from_sizes(sizes: Mapping[Any, Any], globals_) -> int:
    del globals_
    n = _host_concrete_int(shape(next(iter(sizes.values())))[0])
    if n is None:
        raise ValueError("n_graphs requires a concrete batch size")
    return n


def _merge_map(base: Mapping, patch: Optional[Mapping]):
    if patch is None:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


def key_schema(g: "HeteroGraphsTuple") -> Tuple[Tuple[str, ...], Tuple[CanonicalEtype, ...]]:
    """Sorted ``(ntypes, etypes)`` from present keys (empties still count)."""
    return tuple(sorted(g.n_node.keys())), tuple(sorted(g.n_edge.keys()))


def schemas_equal(a: "HeteroGraphsTuple", b: "HeteroGraphsTuple") -> bool:
    """True if ``a`` and ``b`` have the same ntype / etype key sets."""
    return key_schema(a) == key_schema(b)


class SendRecvTuple(NamedTuple):
    """One-way send→receive incidence (homo or bipartite).

    Views alias parent storage: ``nodes_send`` / ``nodes_recv`` may be the
    same object (homo) or two ntype pools (hetero relation).
    """

    nodes_send: Optional[ArrayTree]
    nodes_recv: Optional[ArrayTree]
    senders: Any
    receivers: Any
    edges: Optional[ArrayTree]
    n_node_send: Any
    n_node_recv: Any
    n_edge: Any
    globals: Optional[ArrayTree] = None
    src_ntype: Optional[str] = None
    dst_ntype: Optional[str] = None
    etype: Optional[CanonicalEtype] = None


class HeteroGraphsTuple(NamedTuple):
    """Heterogeneous graph(s) with per-type node pools and per-relation incidence.

    Node ids in ``senders`` / ``receivers`` for etype ``(src, rel, dst)`` are
    local to ``nodes[src]`` and ``nodes[dst]`` (not a global node pool).

    ``n_node[ntype]`` and ``n_edge[etype]`` are integer vectors of length
    ``n_graphs`` (jraph-style batching within one object).

    :func:`anytensor.tree.batch` requires every input to share the same keys.
    """

    nodes: Mapping[Ntype, Optional[ArrayTree]]
    edges: Mapping[CanonicalEtype, Optional[ArrayTree]]
    senders: Mapping[CanonicalEtype, Any]
    receivers: Mapping[CanonicalEtype, Any]
    n_node: Mapping[Ntype, Any]
    n_edge: Mapping[CanonicalEtype, Any]
    globals: Optional[ArrayTree] = None

    def n_graphs(self) -> int:
        return _n_graphs_from_sizes(self.n_node, self.globals)

    def ntypes(self) -> Tuple[str, ...]:
        return tuple(sorted(self.n_node.keys()))

    def canonical_etypes(self) -> Tuple[CanonicalEtype, ...]:
        return tuple(sorted(self.n_edge.keys()))

    def update(
        self,
        nodes: Optional[Mapping[Ntype, Optional[ArrayTree]]] = None,
        edges: Optional[Mapping[CanonicalEtype, Optional[ArrayTree]]] = None,
        senders: Optional[Mapping[CanonicalEtype, Any]] = None,
        receivers: Optional[Mapping[CanonicalEtype, Any]] = None,
        n_node: Optional[Mapping[Ntype, Any]] = None,
        n_edge: Optional[Mapping[CanonicalEtype, Any]] = None,
        globals: Any = _UNSET,
    ) -> "HeteroGraphsTuple":
        """Return a new graph with shallow-merged mapping fields.

        Only keys present in the update dicts are replaced; other keys are
        kept. Pass ``globals=...`` to replace globals (including with ``None``).
        """
        return HeteroGraphsTuple(
            nodes=_merge_map(self.nodes, nodes),
            edges=_merge_map(self.edges, edges),
            senders=_merge_map(self.senders, senders),
            receivers=_merge_map(self.receivers, receivers),
            n_node=_merge_map(self.n_node, n_node),
            n_edge=_merge_map(self.n_edge, n_edge),
            globals=self.globals if globals is _UNSET else globals,
        )

    def iter_nodes(
        self, *, skip_empty: bool = False
    ) -> Iterator[Tuple[Ntype, Optional[ArrayTree], Any]]:
        for ntype in self.ntypes():
            n = self.n_node[ntype]
            if skip_empty and _sum_int(n) == 0:
                continue
            yield ntype, self.nodes.get(ntype), n

    def iter_edges(
        self, *, skip_empty: bool = False
    ) -> Iterator[Tuple[CanonicalEtype, Optional[ArrayTree], Any]]:
        for etype in self.canonical_etypes():
            n = self.n_edge[etype]
            if skip_empty and _sum_int(n) == 0:
                continue
            yield etype, self.edges.get(etype), n

    def iter_relations(
        self,
        etypes: Optional[Sequence[CanonicalEtype]] = None,
        *,
        skip_empty: bool = True,
        reverse: bool = False,
    ) -> Iterator[SendRecvTuple]:
        """Yield aliasing :class:`SendRecvTuple` views for each relation."""
        keys = list(etypes) if etypes is not None else list(self.canonical_etypes())
        for etype in keys:
            if etype not in self.n_edge:
                continue
            if skip_empty and _sum_int(self.n_edge[etype]) == 0:
                continue
            yield self.relation_view(etype)
            if reverse:
                yield self.relation_view(etype, reverse=True)

    def relation_view(
        self, etype: CanonicalEtype, *, reverse: bool = False
    ) -> SendRecvTuple:
        """Aliasing send→recv view for one canonical etype."""
        src, _rel, dst = etype
        if reverse:
            return SendRecvTuple(
                nodes_send=self.nodes.get(dst),
                nodes_recv=self.nodes.get(src),
                senders=self.receivers[etype],
                receivers=self.senders[etype],
                edges=self.edges.get(etype),
                n_node_send=self.n_node[dst],
                n_node_recv=self.n_node[src],
                n_edge=self.n_edge[etype],
                globals=self.globals,
                src_ntype=dst,
                dst_ntype=src,
                etype=(dst, _rel, src),
            )
        return SendRecvTuple(
            nodes_send=self.nodes.get(src),
            nodes_recv=self.nodes.get(dst),
            senders=self.senders[etype],
            receivers=self.receivers[etype],
            edges=self.edges.get(etype),
            n_node_send=self.n_node[src],
            n_node_recv=self.n_node[dst],
            n_edge=self.n_edge[etype],
            globals=self.globals,
            src_ntype=src,
            dst_ntype=dst,
            etype=etype,
        )

    @classmethod
    def __tree_batch__(cls, xs, axis: int = 0):
        if axis != 0:
            raise ValueError("HeteroGraphsTuple batch only supports axis=0")
        return _batch_hetero(xs)

    def __tree_unbatch__(self, axis: int = 0):
        if axis != 0:
            raise ValueError("HeteroGraphsTuple unbatch only supports axis=0")
        return _unbatch_hetero(self)


def graphs_tuple_as_send_recv(graph) -> SendRecvTuple:
    """View a jraph :class:`~anytensor.jraph.GraphsTuple` as send→recv (aliased pools)."""
    return SendRecvTuple(
        nodes_send=graph.nodes,
        nodes_recv=graph.nodes,
        senders=graph.senders,
        receivers=graph.receivers,
        edges=graph.edges,
        n_node_send=graph.n_node,
        n_node_recv=graph.n_node,
        n_edge=graph.n_edge,
        globals=graph.globals,
        src_ntype=None,
        dst_ntype=None,
        etype=None,
    )


def _require_same_keys(graphs: Sequence[HeteroGraphsTuple]) -> None:
    schema0 = key_schema(graphs[0])
    for i, g in enumerate(graphs[1:], start=1):
        if key_schema(g) != schema0:
            raise ValueError(
                "HeteroGraphsTuple batch requires identical ntype/etype keys; "
                f"graph[0] has {schema0}, graph[{i}] has {key_schema(g)}. "
                "Insert empty (length-0) features and zero n_node/n_edge for "
                "missing types before batching."
            )
    # Structural maps should agree with n_node / n_edge keys.
    g0 = graphs[0]
    node_keys = set(g0.n_node)
    edge_keys = set(g0.n_edge)
    for i, g in enumerate(graphs):
        if set(g.nodes) != node_keys:
            raise ValueError(
                f"graph[{i}].nodes keys {set(g.nodes)!r} != n_node keys {node_keys!r}"
            )
        if set(g.edges) != edge_keys or set(g.senders) != edge_keys or set(g.receivers) != edge_keys:
            raise ValueError(
                f"graph[{i}] edge maps must share keys with n_edge {edge_keys!r}"
            )


def _batch_hetero(graphs: Sequence[HeteroGraphsTuple]) -> HeteroGraphsTuple:
    """Fieldwise concat, then offset senders/receivers per input graph.

    Offsets use per-input ``sum(n_node[ntype])`` / ``sum(n_edge[etype])`` (same
    as jraph homo batch), not the flattened multi-graph length vectors.
    """
    if not graphs:
        raise ValueError("batch() requires at least one HeteroGraphsTuple")
    _require_same_keys(graphs)

    ntypes = graphs[0].ntypes()
    etypes = graphs[0].canonical_etypes()

    nodes = {t: tree.batch([g.nodes[t] for g in graphs], axis=0) for t in ntypes}
    n_node = {t: tree.batch([g.n_node[t] for g in graphs], axis=0) for t in ntypes}
    edges = {e: tree.batch([g.edges[e] for g in graphs], axis=0) for e in etypes}
    n_edge = {e: tree.batch([g.n_edge[e] for g in graphs], axis=0) for e in etypes}
    senders = {
        e: concatenate([g.senders[e] for g in graphs], axis=0) for e in etypes
    }
    receivers = {
        e: concatenate([g.receivers[e] for g in graphs], axis=0) for e in etypes
    }
    node_totals = {
        t: concatenate([reshape(at_sum(g.n_node[t]), (1,)) for g in graphs])
        for t in ntypes
    }
    edge_totals = {
        e: concatenate([reshape(at_sum(g.n_edge[e]), (1,)) for g in graphs])
        for e in etypes
    }
    # Offset ids: send uses src ntype totals, recv uses dst.
    senders = {
        e: batch_ids(senders[e], node_totals[e[0]], edge_totals[e]) for e in etypes
    }
    receivers = {
        e: batch_ids(receivers[e], node_totals[e[2]], edge_totals[e]) for e in etypes
    }

    return HeteroGraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=tree.batch([g.globals for g in graphs], axis=0),
    )


def _unbatch_hetero(graph: HeteroGraphsTuple) -> list[HeteroGraphsTuple]:
    """Split features by lengths, then zip into graphs."""
    n_graphs = graph.n_graphs()
    if n_graphs == 0:
        return []
    n_node0 = next(iter(graph.n_node.values()))
    ones_g = ones((n_graphs,), dtype=n_node0.dtype, like=n_node0)
    ntypes = graph.ntypes()
    etypes = graph.canonical_etypes()

    nodes = {t: split_by_lengths(graph.nodes[t], graph.n_node[t]) for t in ntypes}
    edges = {e: split_by_lengths(graph.edges[e], graph.n_edge[e]) for e in etypes}
    senders = {
        e: unbatch_ids(graph.senders[e], graph.n_node[e[0]], graph.n_edge[e])
        for e in etypes
    }
    receivers = {
        e: unbatch_ids(graph.receivers[e], graph.n_node[e[2]], graph.n_edge[e])
        for e in etypes
    }
    n_node = {t: split_by_lengths(graph.n_node[t], ones_g) for t in ntypes}
    n_edge = {e: split_by_lengths(graph.n_edge[e], ones_g) for e in etypes}
    globals_ = split_by_lengths(graph.globals, ones_g)

    out = []
    for i in range(n_graphs):
        out.append(
            HeteroGraphsTuple(
                nodes={t: nodes[t][i] for t in ntypes},
                edges={e: edges[e][i] for e in etypes},
                senders={e: senders[e][i] for e in etypes},
                receivers={e: receivers[e][i] for e in etypes},
                n_node={t: n_node[t][i] for t in ntypes},
                n_edge={e: n_edge[e][i] for e in etypes},
                globals=globals_[i],
            )
        )
    return out
