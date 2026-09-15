"""Segment reductions and related helpers for ragged / GNN-style code."""

from __future__ import annotations

from typing import Optional

from array_api_compat import array_namespace

from .backends import get_backend
from .core import take, ones_like, where, exp, maximum, repeat, arange, cumsum


def segment_sum(x, segment_ids, num_segments: int, sorted: bool = False):
    """Sum values of ``x`` within each segment along axis 0."""
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, "sum", sorted)


def segment_max(x, segment_ids, num_segments: int, sorted: bool = False):
    """Max of values of ``x`` within each segment along axis 0."""
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, "max", sorted)


def segment_min(x, segment_ids, num_segments: int, sorted: bool = False):
    """Min of values of ``x`` within each segment along axis 0."""
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, "min", sorted)


def segment_count(segment_ids, num_segments: int, sorted: bool = False):
    """Count elements per segment id.

    Returns a float vector of shape ``(num_segments,)`` on the same backend as
    ``segment_ids``.
    """
    ones = ones_like(segment_ids)
    # Cast to float for division-friendly counts.
    xp = array_namespace(ones)
    ones = xp.astype(ones, xp.float64)
    return segment_sum(ones, segment_ids, num_segments, sorted=sorted)


def segment_mean(x, segment_ids, num_segments: int, sorted: bool = False):
    """Mean of values of ``x`` within each segment along axis 0.

    Empty segments yield 0 (count is clamped away from zero in the denominator
    only for non-contributing slots after sum is already 0).
    """
    total = segment_sum(x, segment_ids, num_segments, sorted=sorted)
    counts = segment_count(segment_ids, num_segments, sorted=sorted)
    xp = array_namespace(total, counts)
    # Broadcast counts over trailing dims of x.
    while counts.ndim < total.ndim:
        counts = xp.expand_dims(counts, axis=-1)
    denom = maximum(counts, xp.asarray(1.0, dtype=counts.dtype))
    return total / denom


def segment_variance(x, segment_ids, num_segments: int, sorted: bool = False):
    """Variance of values of ``x`` within each segment (population variance)."""
    mean = segment_mean(x, segment_ids, num_segments, sorted=sorted)
    mean_gathered = take(mean, segment_ids)
    centered = x - mean_gathered
    return segment_mean(centered * centered, segment_ids, num_segments, sorted=sorted)


def segment_normalize(x, segment_ids, num_segments: int, sorted: bool = False):
    """Divide each value by its segment sum (0/0 -> 0)."""
    sum_x = segment_sum(x, segment_ids, num_segments, sorted=sorted)
    sum_x = take(sum_x, segment_ids)
    xp = array_namespace(x, sum_x)
    out = x / sum_x
    return where(sum_x == 0, xp.zeros_like(out), out)


def segment_softmax(
    logits,
    segment_ids,
    num_segments: Optional[int] = None,
    sorted: bool = False,
):
    """Softmax within segments (numerically stable)."""
    if num_segments is None:
        xp = array_namespace(segment_ids)
        num_segments = int(xp.max(segment_ids)) + 1
    maxs = segment_max(logits, segment_ids, num_segments, sorted=sorted)
    maxs = take(maxs, segment_ids)
    centered = logits - maxs
    exps = exp(centered)
    normalizers = segment_sum(exps, segment_ids, num_segments, sorted=sorted)
    normalizers = take(normalizers, segment_ids)
    return exps / normalizers


def _replace_empty_with_constant(aggregated, segment_ids, num_segments, constant, sorted: bool = False):
    counts = segment_count(segment_ids, num_segments, sorted=sorted)
    xp = array_namespace(aggregated, counts)
    while counts.ndim < aggregated.ndim:
        counts = xp.expand_dims(counts, axis=-1)
    const = xp.asarray(constant, dtype=aggregated.dtype)
    const = xp.broadcast_to(const, aggregated.shape)
    return where(counts == 0, const, aggregated)


def segment_min_or_constant(
    x,
    segment_ids,
    num_segments: int,
    constant=0.0,
    sorted: bool = False,
):
    """Segment min; empty segments become ``constant`` instead of dtype max."""
    out = segment_min(x, segment_ids, num_segments, sorted=sorted)
    return _replace_empty_with_constant(out, segment_ids, num_segments, constant, sorted=sorted)


def segment_max_or_constant(
    x,
    segment_ids,
    num_segments: int,
    constant=0.0,
    sorted: bool = False,
):
    """Segment max; empty segments become ``constant`` instead of dtype min."""
    out = segment_max(x, segment_ids, num_segments, sorted=sorted)
    return _replace_empty_with_constant(out, segment_ids, num_segments, constant, sorted=sorted)


def partition_softmax(logits, partitions, sum_partitions: Optional[int] = None):
    """Softmax within contiguous partitions of lengths ``partitions``.

    ``partitions`` is a 1-D integer vector of partition sizes (e.g. ``n_node``).
    Array logits only (no nested ArrayTree mapping).
    """
    xp = array_namespace(logits, partitions)
    if sum_partitions is None:
        sum_partitions = int(xp.sum(partitions))
    n_part = int(partitions.shape[0])
    ids = arange(0, n_part, like=partitions)
    segment_ids = repeat(ids, partitions, total_repeat_length=sum_partitions)
    return segment_softmax(logits, segment_ids, num_segments=n_part)
