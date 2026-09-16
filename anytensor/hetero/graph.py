"""Heterogeneous graphs: ``SendRecvTuple``, ``HeteroGraphsTuple``, batch magic.

Separate from the jraph-mirroring package. Batching goes through
:func:`anytensor.tree.batch` / ``unbatch`` via ``__tree_batch__`` /
``__tree_unbatch__`` (same hooks as :class:`~anytensor.jraph.GraphsTuple`).

Missing keys in a merge are filled with empty arrays; fillers are recorded on
:class:`HeteroBatch` (Policy C) so ``unbatch`` can restore absent keys without
polluting the model-facing graph fields.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np

from anytensor import tree
from anytensor.core import concatenate, zeros
from anytensor.namespace import array_namespace

ArrayTree = Union[Any, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]
CanonicalEtype = Tuple[str, str, str]
Ntype = str


def _as_np_vec(x) -> np.ndarray:
    return np.asarray(x).reshape(-1)


def _sum_int(x) -> int:
    return int(np.asarray(x).sum())


def _n_graphs_from_sizes(sizes: Mapping[Any, Any], globals_) -> int:
    for v in sizes.values():
        return int(np.asarray(v).shape[0])
    if globals_ is not None:
        leaves = tree.leaves(globals_)
        if leaves:
            return int(leaves[0].shape[0])
        return 1
    return 1


def _empty_like_leading(proto, leading: int = 0):
    """Leading-axis empty (or length-``leading``) array matching ``proto``."""

    def _one(leaf):
        rest = tuple(int(s) for s in np.asarray(leaf.shape)[1:])
        return zeros((leading,) + rest, dtype=leaf.dtype, like=leaf)

    return tree.map(_one, proto)


def _zeros_n(n_graphs: int, like) -> Any:
    return zeros((n_graphs,), dtype=np.asarray(like).dtype, like=like)


def _empty_index(like) -> Any:
    return zeros((0,), dtype=np.asarray(like).dtype, like=like)


def _offset_index(index, offset: int):
    if index is None:
        return None
    xp = array_namespace(index)
    return index + xp.asarray(offset, dtype=index.dtype)


def schemas_equal(
    a: "HeteroGraphsTuple",
    b: "HeteroGraphsTuple",
    *,
    empty_means_absent: bool = True,
) -> bool:
    """Return True if ``a`` and ``b`` have the same ntypes / etypes under rules.

    With ``empty_means_absent=True`` (default), size-0 node/edge types are
    treated as missing (same as an absent key).
    """
    return canonicalize_schema(a, empty_means_absent=empty_means_absent) == canonicalize_schema(
        b, empty_means_absent=empty_means_absent
    )


def canonicalize_schema(
    g: "HeteroGraphsTuple",
    *,
    empty_means_absent: bool = True,
) -> Tuple[Tuple[str, ...], Tuple[CanonicalEtype, ...]]:
    """Sorted ``(ntypes, etypes)`` after optional empty→absent normalization."""
    ntypes = []
    for k, n in g.n_node.items():
        if empty_means_absent and _sum_int(n) == 0:
            continue
        ntypes.append(k)
    etypes = []
    for k, n in g.n_edge.items():
        if empty_means_absent and _sum_int(n) == 0:
            continue
        etypes.append(k)
    return tuple(sorted(ntypes)), tuple(sorted(etypes))


def _merge_map(base: Mapping, patch: Optional[Mapping]):
    if patch is None:
        return dict(base)
    out = dict(base)
    out.update(patch)
    return out


_UNSET = object()


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


class HeteroBatch(NamedTuple):
    """Stacked hetero graphs plus Policy-C filler metadata for round-trip.

    Models should consume ``graph`` (or its iterators). Filler sets are
    **per input** to :func:`anytensor.tree.batch` (not a union applied to every
    slice), so unbatch restores absent keys without dropping real types.
    """

    graph: HeteroGraphsTuple
    # Parallel to the batch inputs: fillers introduced while aligning each one.
    filled_ntypes: Tuple[frozenset, ...]
    filled_etypes: Tuple[frozenset, ...]
    # Atomic graph count contributed by each input (``g.n_graphs()``).
    input_n_graphs: Tuple[int, ...]

    def __tree_unbatch__(self, axis: int = 0):
        if axis != 0:
            raise ValueError("HeteroBatch unbatch only supports axis=0")
        parts = _unbatch_hetero(self.graph)
        out = []
        i = 0
        for n, fn, fe in zip(
            self.input_n_graphs, self.filled_ntypes, self.filled_etypes
        ):
            for _ in range(n):
                out.append(_strip_filled(parts[i], fn, fe))
                i += 1
        return out


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


def _prototype_node(graphs: Sequence[HeteroGraphsTuple], ntype: Ntype):
    # Prefer a non-empty pool, then any present features for trailing shape/dtype.
    nonempty = None
    any_feat = None
    for g in graphs:
        if ntype not in g.nodes or g.nodes[ntype] is None:
            continue
        any_feat = g.nodes[ntype]
        if _sum_int(g.n_node.get(ntype, [0])) > 0:
            nonempty = g.nodes[ntype]
            break
    return nonempty if nonempty is not None else any_feat


def _prototype_edge(graphs: Sequence[HeteroGraphsTuple], etype: CanonicalEtype):
    nonempty = None
    any_feat = None
    for g in graphs:
        if etype not in g.edges or g.edges[etype] is None:
            continue
        any_feat = g.edges[etype]
        if _sum_int(g.n_edge.get(etype, [0])) > 0:
            nonempty = g.edges[etype]
            break
    return nonempty if nonempty is not None else any_feat


def _prototype_index(graphs: Sequence[HeteroGraphsTuple], etype: CanonicalEtype, field: str):
    for g in graphs:
        m = getattr(g, field)
        if etype in m and m[etype] is not None:
            return m[etype]
    for g in graphs:
        for v in g.senders.values():
            if v is not None:
                return v
        for v in g.n_node.values():
            return v
    raise ValueError("cannot infer index dtype for empty filler")


def _align_graph(
    g: HeteroGraphsTuple,
    ntypes: Sequence[Ntype],
    etypes: Sequence[CanonicalEtype],
    graphs: Sequence[HeteroGraphsTuple],
) -> Tuple[HeteroGraphsTuple, frozenset, frozenset]:
    """Pad ``g`` to ``ntypes``/``etypes`` with empty arrays; return fillers used."""
    n_graphs = g.n_graphs()
    filled_n: set = set()
    filled_e: set = set()

    nodes = dict(g.nodes)
    n_node = dict(g.n_node)
    # Reference size vector for missing n_node length.
    size_like = next(iter(g.n_node.values())) if g.n_node else zeros((n_graphs,), dtype=np.int32)

    for ntype in ntypes:
        if ntype not in n_node:
            filled_n.add(ntype)
            proto = _prototype_node(graphs, ntype)
            n_node[ntype] = _zeros_n(n_graphs, size_like)
            nodes[ntype] = None if proto is None else _empty_like_leading(proto, 0)

    edges = dict(g.edges)
    senders = dict(g.senders)
    receivers = dict(g.receivers)
    n_edge = dict(g.n_edge)
    for etype in etypes:
        if etype not in n_edge:
            filled_e.add(etype)
            proto = _prototype_edge(graphs, etype)
            idx_like = _prototype_index(graphs, etype, "senders")
            n_edge[etype] = _zeros_n(n_graphs, size_like)
            edges[etype] = None if proto is None else _empty_like_leading(proto, 0)
            senders[etype] = _empty_index(idx_like)
            receivers[etype] = _empty_index(idx_like)

    aligned = HeteroGraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=g.globals,
    )
    return aligned, frozenset(filled_n), frozenset(filled_e)


def _batch_hetero(graphs: Sequence[HeteroGraphsTuple]) -> HeteroBatch:
    if not graphs:
        raise ValueError("batch() requires at least one HeteroGraphsTuple")

    ntypes = tuple(sorted({t for g in graphs for t in g.n_node}))
    etypes = tuple(sorted({t for g in graphs for t in g.n_edge}))

    aligned = []
    filled_n_list = []
    filled_e_list = []
    input_n_graphs = []
    for g in graphs:
        ag, fn, fe = _align_graph(g, ntypes, etypes, graphs)
        aligned.append(ag)
        filled_n_list.append(fn)
        filled_e_list.append(fe)
        input_n_graphs.append(ag.n_graphs())

    # Per-ntype offsets for sender/receiver rebasing.
    offsets = {t: [] for t in ntypes}
    running = {t: 0 for t in ntypes}
    for g in aligned:
        for t in ntypes:
            offsets[t].append(running[t])
            running[t] += _sum_int(g.n_node[t])

    nodes = {
        t: tree.batch([g.nodes[t] for g in aligned], axis=0)
        for t in ntypes
    }
    n_node = {
        t: tree.batch([g.n_node[t] for g in aligned], axis=0)
        for t in ntypes
    }
    edges = {
        e: tree.batch([g.edges[e] for g in aligned], axis=0)
        for e in etypes
    }
    n_edge = {
        e: tree.batch([g.n_edge[e] for g in aligned], axis=0)
        for e in etypes
    }
    senders = {}
    receivers = {}
    for e in etypes:
        src, _r, dst = e
        senders[e] = concatenate(
            [_offset_index(g.senders[e], offsets[src][i]) for i, g in enumerate(aligned)],
            axis=0,
        )
        receivers[e] = concatenate(
            [_offset_index(g.receivers[e], offsets[dst][i]) for i, g in enumerate(aligned)],
            axis=0,
        )

    globals_ = tree.batch([g.globals for g in aligned], axis=0)

    stacked = HeteroGraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=globals_,
    )
    return HeteroBatch(
        graph=stacked,
        filled_ntypes=tuple(filled_n_list),
        filled_etypes=tuple(filled_e_list),
        input_n_graphs=tuple(input_n_graphs),
    )


def _partition_sizes(sizes) -> list[int]:
    return [int(v) for v in _as_np_vec(sizes).tolist()]


def _unbatch_hetero(graph: HeteroGraphsTuple) -> list[HeteroGraphsTuple]:
    n_graphs = graph.n_graphs()
    if n_graphs == 0:
        return []

    # Per-graph node counts for each type.
    n_node_lists = {t: _partition_sizes(graph.n_node[t]) for t in graph.n_node}
    n_edge_lists = {e: _partition_sizes(graph.n_edge[e]) for e in graph.n_edge}

    # Cumulative node offsets within the stacked batch (per type).
    node_offsets = {}
    for t, counts in n_node_lists.items():
        offs = [0]
        for c in counts[:-1]:
            offs.append(offs[-1] + c)
        node_offsets[t] = offs

    # Split features via tree.unbatch units then regroup — reuse jraph approach
    # by splitting with cumsum indices manually for variable sizes.
    def split_feat(feat, sizes):
        if feat is None:
            return [None] * len(sizes)
        if all(s == 0 for s in sizes):
            return [_empty_like_leading(feat, 0) for _ in sizes]
        units = tree.unbatch(feat, axis=0)
        out = []
        i = 0
        for n in sizes:
            chunk = units[i : i + n]
            i += n
            if n == 0:
                out.append(_empty_like_leading(feat, 0))
            elif n == 1:
                out.append(chunk[0])
            else:
                out.append(tree.batch(chunk, axis=0))
        return out

    nodes_parts = {
        t: split_feat(graph.nodes[t], n_node_lists[t]) for t in graph.n_node
    }
    edges_parts = {
        e: split_feat(graph.edges[e], n_edge_lists[e]) for e in graph.n_edge
    }
    senders_parts = {
        e: split_feat(graph.senders[e], n_edge_lists[e]) for e in graph.n_edge
    }
    receivers_parts = {
        e: split_feat(graph.receivers[e], n_edge_lists[e]) for e in graph.n_edge
    }
    globals_parts = split_feat(graph.globals, [1] * n_graphs)

    out = []
    for i in range(n_graphs):
        senders_i = {}
        receivers_i = {}
        for e in graph.n_edge:
            src, _r, dst = e
            s = senders_parts[e][i]
            r = receivers_parts[e][i]
            # Undo batch offsets (offset is start of this graph's nodes in the stack).
            senders_i[e] = None if s is None else s - node_offsets[src][i]
            receivers_i[e] = None if r is None else r - node_offsets[dst][i]
        out.append(
            HeteroGraphsTuple(
                nodes={t: nodes_parts[t][i] for t in graph.n_node},
                edges={e: edges_parts[e][i] for e in graph.n_edge},
                senders=senders_i,
                receivers=receivers_i,
                n_node={t: graph.n_node[t][i : i + 1] for t in graph.n_node},
                n_edge={e: graph.n_edge[e][i : i + 1] for e in graph.n_edge},
                globals=globals_parts[i],
            )
        )
    return out


def _strip_filled(
    g: HeteroGraphsTuple,
    filled_ntypes: frozenset,
    filled_etypes: frozenset,
) -> HeteroGraphsTuple:
    nodes = {k: v for k, v in g.nodes.items() if k not in filled_ntypes}
    n_node = {k: v for k, v in g.n_node.items() if k not in filled_ntypes}
    edges = {k: v for k, v in g.edges.items() if k not in filled_etypes}
    n_edge = {k: v for k, v in g.n_edge.items() if k not in filled_etypes}
    senders = {k: v for k, v in g.senders.items() if k not in filled_etypes}
    receivers = {k: v for k, v in g.receivers.items() if k not in filled_etypes}
    return HeteroGraphsTuple(
        nodes=nodes,
        edges=edges,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=g.globals,
    )
