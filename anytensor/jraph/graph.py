"""Graph data structure (jraph ``GraphsTuple``, backend-agnostic)."""

from __future__ import annotations

from typing import Any, Iterable, Mapping, NamedTuple, Optional, Union

# Nests of array-like leaves (NumPy / JAX / Torch / TF).
ArrayTree = Union[Any, Iterable["ArrayTree"], Mapping[Any, "ArrayTree"]]


class GraphsTuple(NamedTuple):
    """An ordered collection of graphs in a sparse format.

    A port of :class:`jraph.GraphsTuple`. ``nodes``, ``edges`` and ``globals``
    may be ``None`` or an ``ArrayTree`` of features; ``senders`` / ``receivers``
    are integer index arrays (or ``None`` when there are no edges). ``n_node``
    and ``n_edge`` are integer vectors with one entry per graph in the batch.

    Sender and receiver indices are **absolute** in the batched node array
    (offset by the nodes of earlier graphs). See the jraph docs for the
    full field layout.

    ``__tree_batch__`` / ``__tree_unbatch__`` implement graph batching (not
    fieldwise array concat). :func:`anytensor.tree.batch` / ``unbatch`` are
    the same functions as :func:`~anytensor.jraph.batch` / ``unbatch``; this
    type owns the offsetting logic. Custom node/edge/global objects may
    define the same methods so feature batching uses their logic.
    """

    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[Any]
    senders: Optional[Any]
    globals: Optional[ArrayTree]
    n_node: Any
    n_edge: Any

    @classmethod
    def __tree_batch__(cls, xs, axis: int = 0):
        """Batch graphs. Senders/receivers are offset; not a fieldwise concat."""
        if axis != 0:
            raise ValueError("GraphsTuple batch only supports axis=0")
        from .utils import _batch_graphs

        return _batch_graphs(xs)

    def __tree_unbatch__(self, axis: int = 0):
        """Unbatch into one :class:`GraphsTuple` per graph (jraph ``unbatch``)."""
        if axis != 0:
            raise ValueError("GraphsTuple unbatch only supports axis=0")
        from .utils import _unbatch_graphs

        return _unbatch_graphs(self)
