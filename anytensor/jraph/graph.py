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

    ``__tree_concat__`` / ``__tree_split__`` implement graph batching (not
    fieldwise array concat). Custom node/edge/global objects may define the
    same methods so :func:`anytensor.tree.concat` / ``split`` (and therefore
    :func:`~anytensor.jraph.batch` / ``unbatch``) use their logic.
    """

    nodes: Optional[ArrayTree]
    edges: Optional[ArrayTree]
    receivers: Optional[Any]
    senders: Optional[Any]
    globals: Optional[ArrayTree]
    n_node: Any
    n_edge: Any

    @classmethod
    def __tree_concat__(cls, xs, axis: int = 0):
        """Batch graphs. Senders/receivers are offset; not a fieldwise concat."""
        if axis != 0:
            raise ValueError("GraphsTuple concatenation only supports axis=0")
        from .utils import batch

        return batch(list(xs))

    def __tree_split__(self, sizes, axis: int = 0):
        """Split a batch into chunks with the given numbers of graphs."""
        if axis != 0:
            raise ValueError("GraphsTuple split only supports axis=0")
        from .utils import batch, unbatch

        parts = unbatch(self)
        sizes = [int(s) for s in sizes]
        total = sum(sizes)
        if total != len(parts):
            raise ValueError(
                f"sizes sum to {total} but batched graph has {len(parts)} graphs"
            )
        out = []
        i = 0
        for n in sizes:
            if n <= 0:
                raise ValueError("GraphsTuple split pieces must contain at least one graph")
            chunk = parts[i : i + n]
            i += n
            out.append(chunk[0] if n == 1 else batch(chunk))
        return out
