"""Length-vector helpers for graph batch/unbatch (not part of ``anytensor.tree``).

Conversions among length vectors, split points, and segment ids, plus
offsetting concatenated indices when batching graphs. Arithmetic stays on the
caller's array backend via :mod:`anytensor.core`.
"""

from __future__ import annotations

from anytensor import tree
from anytensor.core import (
    arange,
    astype,
    concatenate,
    cumsum,
    full,
    repeat,
    reshape,
    shape,
    split as array_split,
    zeros,
)
from anytensor.core import _host_concrete_int
from anytensor.namespace import array_namespace

__all__ = [
    "batch_ids",
    "cuts_to_lengths",
    "lengths_to_ids",
    "lengths_to_splits",
    "split_by_lengths",
    "unbatch_ids",
]


def lengths_to_ids(lengths, *, total=None):
    """Length vector → segment ids: ``repeat(arange(n), lengths)``."""
    lengths = reshape(lengths, (-1,))
    n = shape(lengths)[0]
    return repeat(
        arange(n, like=lengths, dtype=lengths.dtype),
        lengths,
        total_repeat_length=total,
    )


def lengths_to_splits(lengths):
    """Length vector → cumulative split points: ``cumsum(lengths)``."""
    return cumsum(reshape(lengths, (-1,)))


def cuts_to_lengths(cuts, total):
    """Cut indices + total length → length vector."""
    if isinstance(cuts, (list, tuple)):
        import array_api_compat.numpy as xp

        cuts = xp.asarray(cuts)
    cuts = reshape(cuts, (-1,))
    zero = zeros((1,), dtype=cuts.dtype, like=cuts)
    end = full((1,), total, dtype=cuts.dtype, like=cuts)
    edges = concatenate([zero, cuts, end])
    return edges[1:] - edges[:-1]


def _exclusive_offsets(lengths):
    """Per-part start offsets: ``cumsum(lengths) - lengths``.

    Cast ``lengths`` to the cumsum dtype so backends that widen integer
    reductions (TF ``cumsum`` int32→int64) still subtract cleanly.
    """
    lengths = reshape(lengths, (-1,))
    totals = cumsum(lengths)
    return totals - astype(lengths, totals.dtype)


def _on_backend(x, like):
    """Move ``x`` onto ``like``'s backend when needed."""
    xp = array_namespace(like)
    if array_namespace(x) is xp:
        return x
    return xp.asarray(x)


def batch_ids(ids, lengths, part_lengths):
    """Offset concatenated local ids into a batched id space.

    ``ids + repeat(cumsum(lengths) - lengths, part_lengths)``.
    """
    lengths = _on_backend(lengths, ids)
    part_lengths = _on_backend(part_lengths, ids)
    offsets = astype(_exclusive_offsets(lengths), ids.dtype)
    return ids + repeat(offsets, part_lengths)


def unbatch_ids(ids, lengths, part_lengths):
    """Undo :func:`batch_ids`, then split into per-graph id arrays."""
    lengths = _on_backend(lengths, ids)
    part_lengths = _on_backend(part_lengths, ids)
    offsets = astype(_exclusive_offsets(lengths), ids.dtype)
    local = ids - repeat(offsets, part_lengths)
    cuts = lengths_to_splits(part_lengths)[:-1]
    return list(array_split(local, cuts))


def _split_structure(structure, cuts, n_parts, axis: int = 0):
    """Split every array leaf with the same cut indices; return list of pytrees."""
    if structure is None:
        return [None] * n_parts
    leaf_list, treedef = tree.flatten(structure)
    if not leaf_list:
        return [tree.unflatten(treedef, []) for _ in range(n_parts)]
    parts_per_leaf = []
    for leaf in leaf_list:
        leaf_cuts = _on_backend(cuts, leaf) if n_parts > 1 else cuts
        parts_per_leaf.append(list(array_split(leaf, leaf_cuts, axis=axis)))
    return [
        tree.unflatten(treedef, [parts[i] for parts in parts_per_leaf])
        for i in range(n_parts)
    ]


def split_by_lengths(structure, lengths, axis: int = 0):
    """Split a pytree by a length vector: ``cumsum(lengths)[:-1]`` cuts."""
    lengths = reshape(lengths, (-1,))
    n_parts = _host_concrete_int(shape(lengths)[0])
    if n_parts is None:
        raise ValueError("split_by_lengths requires a concrete batch size")
    if n_parts == 0:
        return []
    cuts = lengths_to_splits(lengths)[:-1]
    return _split_structure(structure, cuts, n_parts, axis=axis)
