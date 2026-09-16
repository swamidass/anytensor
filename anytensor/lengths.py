"""Length-vector helpers for graph batch/unbatch (not part of ``anytensor.tree``).

Conversions among length vectors, split points, and segment ids, plus
offsetting concatenated indices when batching graphs.
"""

from __future__ import annotations

import numpy as np

from anytensor import tree
from anytensor.core import arange, cumsum, repeat, split as array_split
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
    n = int(np.asarray(lengths).reshape(-1).shape[0])
    return repeat(arange(n, like=lengths), lengths, total_repeat_length=total)


def lengths_to_splits(lengths):
    """Length vector → cumulative split points: ``cumsum(lengths)``.

    Pass ``splits[:-1]`` to :func:`anytensor.split` (NumPy cut-index semantics).
    """
    return cumsum(np.asarray(lengths).reshape(-1))


def cuts_to_lengths(cuts, total) -> np.ndarray:
    """Cut indices + total length → length vector."""
    edges = [0, *[int(c) for c in np.asarray(cuts).reshape(-1).tolist()], int(total)]
    return np.asarray([edges[i + 1] - edges[i] for i in range(len(edges) - 1)], dtype=np.int64)


def _exclusive_offsets(lengths):
    """Per-part start offsets: ``cumsum(lengths) - lengths``."""
    lengths = np.asarray(lengths).reshape(-1)
    return np.cumsum(lengths) - lengths


def batch_ids(ids, lengths, part_lengths):
    """Offset concatenated local ids into a batched id space.

    ``ids + repeat(cumsum(lengths) - lengths, part_lengths)``.
    """
    offsets = _exclusive_offsets(lengths)
    xp = array_namespace(ids)
    off = xp.asarray(offsets, dtype=getattr(ids, "dtype", offsets.dtype))
    return ids + repeat(off, part_lengths)


def unbatch_ids(ids, lengths, part_lengths):
    """Undo :func:`batch_ids`, then split into per-graph id arrays."""
    offsets = _exclusive_offsets(lengths)
    xp = array_namespace(ids)
    off = xp.asarray(offsets, dtype=getattr(ids, "dtype", offsets.dtype))
    local = ids - repeat(off, part_lengths)
    splits = np.asarray(lengths_to_splits(part_lengths)).reshape(-1).tolist()
    return list(array_split(local, splits[:-1] if splits else splits))


def _split_structure(structure, cuts, axis: int = 0):
    """Split every array leaf with the same cut indices; return list of pytrees."""
    n = len(list(cuts)) + 1
    if structure is None:
        return [None] * n
    leaf_list, treedef = tree.flatten(structure)
    if not leaf_list:
        return [tree.unflatten(treedef, []) for _ in range(n)]
    parts_per_leaf = [list(array_split(leaf, cuts, axis=axis)) for leaf in leaf_list]
    return [
        tree.unflatten(treedef, [parts[i] for parts in parts_per_leaf]) for i in range(n)
    ]


def split_by_lengths(structure, lengths, axis: int = 0):
    """Split a pytree by a length vector: ``cumsum(lengths)[:-1]`` cuts."""
    splits = np.asarray(lengths_to_splits(lengths)).reshape(-1).tolist()
    return _split_structure(structure, splits[:-1] if len(splits) > 1 else [], axis=axis)
