"""Segment reductions and related helpers for ragged / GNN-style code.

Empty-segment identities are standardized in :mod:`anytensor.semantics`
(float ``±inf``, integer dtype min/max, sum ``0``).

``num_segments`` is always required on **segment** ops (JAX convention).
Callers may pass a Python int, a jit/compile symbolic constant, or a 0-d
tensor scalar — never inferred from ``segment_ids`` (that would be
``max(ids)+1``, data-dependent). Partition helpers do **not** take
``num_segments``: it is ``shape(partitions)[0]``, a shape read. They do
require ``total_length`` (``shape(logits)[0]``, not a data
``sum(partitions)``). Partition helpers call :func:`partition_ids`,
which uses :meth:`cache.lookup` / :meth:`cache.store` on ``"partition"``
when a decorator (sticky across calls) / context / :meth:`cache.enable`
is active. If a cached expansion's
length does not match ``total_length`` (host Python ints), that entry
is purged, a warning is issued, and ids are recomputed; tracing skips
the check. :meth:`cache.purge` drops one tensor from one namespace.
There is no ``partition_sum`` / ``partition_min`` family.

TorchScript: :func:`enable_torchscript` wraps ``segment_sum`` / ``min`` /
``max`` with a ``torch.jit.is_scripting()`` divert. Import order does not
matter: a :func:`module_if_loaded` helper enables the divert as soon as
``torch`` is imported. Eager calls still dispatch by tensor type
(NumPy / JAX / Torch / TF); only the scripted path uses
:mod:`anytensor.torchscript`. See docs/usage.md.
"""

from __future__ import annotations

import warnings

from .backends import get_backend
from .optional import module_if_loaded
from .typing import ArrayT, ShapeSize, SegmentValues, SegmentIds, SegmentOut, ShapedArray, IntArray
from .core import (
    take,
    ones_like,
    where,
    exp,
    maximum,
    repeat,
    arange,
    shape,
    _xp,
    _asarray,
    _apply_dtype_roles,
    _host_concrete_int,
    _normalize_shape_dim,
)
from .namespace import array_namespace
from ._cache import cache as cache

_TORCHSCRIPT_ENABLED = False


def _align_segment_args(x, segment_ids):
    """Namespace-align values + ids; ids stay integral (kind=index)."""
    xp = _xp(x, segment_ids)
    converted = {
        "x": _asarray(xp, x),
        "segment_ids": _asarray(xp, segment_ids),
    }
    _apply_dtype_roles(xp, converted, {"x": "data", "segment_ids": "index"})
    return converted["x"], converted["segment_ids"]


def _require_shape_size(name: str, value):
    """Shape-sizes are required; ``None`` must not become a data-dependent length."""
    if value is None:
        raise TypeError(
            f"{name} is a required shape-size (JAX convention). "
            "Omitting it does not infer max(ids)+1 or sum(partitions)."
        )
    return _normalize_shape_dim(value)


def _segment_reduce(x, segment_ids, num_segments, reduction: str, sorted: bool = False):
    x, segment_ids = _align_segment_args(x, segment_ids)
    num_segments = _require_shape_size("num_segments", num_segments)
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, reduction, sorted)


def segment_sum(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Computes the sum within segments of an array.

    Similar to :func:`jax.ops.segment_sum` and TF ``unsorted_segment_sum``.
    Reduces ``x`` along axis 0, summing rows that share the same
    ``segment_ids`` entry.

    Args:
        x: Values to sum. The leading axis is the segment axis.
        segment_ids: Integer ids with ``segment_ids.shape[0] == x.shape[0]``.
            Values may be repeated and need not be sorted. Ids outside
            ``[0, num_segments)`` (including negatives) are dropped and do not
            contribute; they are **not** wrapped like Python negative indexing.
        num_segments: **Required** shape-size (unlike JAX, where omitting it
            defaults to ``max(segment_ids) + 1``). Accepts a Python ``int``, a
            jit/compile symbolic constant, or a 0-d integral tensor scalar —
            never inferred from ``segment_ids``. Sets the output length; empty
            slots are ``0``.
        sorted: When True, JAX/TF may use a sorted-ids fast path. **No-op on
            NumPy and Torch** (unsorted-safe scatter).

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]`` on the same backend
        as ``x``.

    Notes:
        Empty-segment identity is ``0`` for all dtypes
        (:mod:`anytensor.semantics`).

        Index **width** is backend-local: Torch casts ids to int64 at scatter;
        JAX without ``jax_enable_x64`` often keeps int32 and may warn on int64
        ids.

        Under ``torch.jit.script``, import ``torch`` in either order relative to
        AnyTensor — the divert auto-enables via :func:`anytensor.module_if_loaded`.
        Eager calls still dispatch by tensor type; only the scripted path uses
        :mod:`anytensor.torchscript`.

    Examples:
        >>> import numpy as np
        >>> import anytensor as at
        >>> x = np.arange(5.0)
        >>> ids = np.array([0, 0, 1, 1, 2])
        >>> at.segment_sum(x, ids, num_segments=3)
        array([1., 5., 4.])
    """
    return _segment_reduce(x, segment_ids, num_segments, "sum", sorted)


def segment_max(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Computes the maximum within segments of an array.

    Similar to :func:`jax.ops.segment_max`. Reduces ``x`` along axis 0 by
    ``segment_ids``.

    Args:
        x: Values to reduce. Leading axis is the segment axis.
        segment_ids: Integer ids with ``segment_ids.shape[0] == x.shape[0]``.
            May be unsorted / non-unique. Out-of-range and negative ids are
            dropped (not wrapped).
        num_segments: **Required** shape-size (Python ``int``, symbolic
            constant, or 0-d integral tensor). Empty slots get the max
            identity (``-inf`` for floats, ``iinfo(dtype).min`` for ints).
        sorted: Honored on JAX/TF; **no-op on NumPy/Torch**.

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``.

    Notes:
        Empty-slot identity is ``-inf`` / dtype min — **not** TF's native
        ``unsorted_segment_max`` finfo fill. Occupied ±inf stays ±inf.

        On TensorFlow, scatter ignores NaN updates; AnyTensor ORs NaN back in
        so a segment that saw any NaN is NaN (matches NumPy/JAX/Torch).

        Prefer :func:`segment_max_or_constant` when empty slots should be a
        finite fill instead of ``-inf``.

        Under TF XLA (``tf.function(jit_compile=True)``), NaN in min/max-like
        ops may become ±inf instead of NaN — avoid relying on NaN under XLA.

        TorchScript: :func:`enable_torchscript` (eager path stays multi-backend).
    """
    return _segment_reduce(x, segment_ids, num_segments, "max", sorted)


def segment_min(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Computes the minimum within segments of an array.

    Similar to :func:`jax.ops.segment_min`. Reduces ``x`` along axis 0 by
    ``segment_ids``.

    Args:
        x: Values to reduce. Leading axis is the segment axis.
        segment_ids: Integer ids with ``segment_ids.shape[0] == x.shape[0]``.
            May be unsorted / non-unique. Out-of-range and negative ids are
            dropped (not wrapped).
        num_segments: **Required** shape-size (Python ``int``, symbolic
            constant, or 0-d integral tensor). Empty slots get the min
            identity (``+inf`` for floats, ``iinfo(dtype).max`` for ints).
        sorted: Honored on JAX/TF; **no-op on NumPy/Torch**.

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``.

    Notes:
        Empty-slot identity is ``+inf`` / dtype max — **not** TF's native
        ``unsorted_segment_min`` finfo fill. Occupied ±inf stays ±inf.

        On TensorFlow, scatter ignores NaN updates; AnyTensor ORs NaN back in
        so a segment that saw any NaN is NaN (matches NumPy/JAX/Torch).

        Prefer :func:`segment_min_or_constant` when empty slots should be a
        finite fill instead of ``+inf``.

        Under TF XLA, NaN in min/max-like ops may become ±inf — avoid relying
        on NaN under XLA.

        TorchScript: :func:`enable_torchscript` (eager path stays multi-backend).
    """
    return _segment_reduce(x, segment_ids, num_segments, "min", sorted)


def _enable_torchscript(torch) -> bool:
    """Install the TorchScript divert. ``torch`` is the already-imported module."""
    global _TORCHSCRIPT_ENABLED, segment_sum, segment_min, segment_max, _segment_reduce
    if _TORCHSCRIPT_ENABLED:
        return True

    from . import torchscript

    docs = (segment_sum.__doc__, segment_max.__doc__, segment_min.__doc__)
    # Bind annotated originals so eager calls keep jaxtyping checks.
    torchscript.bind_eager_ops(sum=segment_sum, max=segment_max, min=segment_min)
    _segment_reduce = torch.jit.ignore(_segment_reduce)

    divert_sum = torchscript.divert_segment_sum
    divert_max = torchscript.divert_segment_max
    divert_min = torchscript.divert_segment_min
    divert_sum.__doc__ = docs[0]
    divert_max.__doc__ = docs[1]
    divert_min.__doc__ = docs[2]
    segment_sum = divert_sum
    segment_max = divert_max
    segment_min = divert_min
    _TORCHSCRIPT_ENABLED = True

    pkg = module_if_loaded("anytensor")
    if pkg is not None:  # pragma: no branch - package always loaded in normal use
        pkg.segment_sum = segment_sum
        pkg.segment_max = segment_max
        pkg.segment_min = segment_min
        pkg.enable_torchscript = enable_torchscript
    return True


def enable_torchscript() -> bool:
    """Enable ``torch.jit.script`` through public ``segment_sum`` / ``min`` / ``max``.

    Wraps those helpers with a ``torch.jit.is_scripting()`` divert to
    :mod:`anytensor.torchscript`. **Eager behavior is unchanged**: NumPy, JAX,
    Torch, and TF tensors still dispatch via backends. Only while scripting
    (or inside an already-scripted graph) do we take the pure-Torch kernels.

    That lets a third-party library call ``anytensor.segment_sum`` in ordinary
    Python, while an end user of that library can ``torch.jit.script`` their
    own code that reaches those calls.

    Does not import ``torch`` and does not require a particular import order.
    If Torch is not loaded yet, a helper is registered with
    :func:`anytensor.module_if_loaded` and the divert enables on a later
    ``import torch``. Returns ``False`` until then; safe to call more than once.
    """
    if _TORCHSCRIPT_ENABLED:
        return True
    return module_if_loaded("torch", _enable_torchscript) is not None


# Enable now if torch is already imported; otherwise when it is first imported.
enable_torchscript()


def segment_count(segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Count how many elements fall in each segment.

    Implemented as ``segment_sum`` of ones, so empty slots are ``0.0``.

    Args:
        segment_ids: Integer ids (same conventions as :func:`segment_sum`).
        num_segments: Required shape-size (Python ``int`` / symbolic / 0-d
            integral tensor).
        sorted: Forwarded to :func:`segment_sum` (no-op on NumPy/Torch).

    Returns:
        Float vector of shape ``(num_segments,)`` on the same backend as
        ``segment_ids``. Dtype follows the backend default float (often
        float32 on JAX without x64).
    """
    ones = ones_like(segment_ids)
    # Float counts for division; use default float (float32 on JAX without x64).
    xp = array_namespace(ones)
    ones = xp.astype(ones, xp.asarray(0.0).dtype)
    return segment_sum(ones, segment_ids, num_segments, sorted=sorted)


def segment_mean(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Mean of values of ``x`` within each segment along axis 0.

    Args:
        x: Values to average. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: Required shape-size.
        sorted: Forwarded to underlying segment ops (no-op on NumPy/Torch).

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``.

    Notes:
        Empty segments yield **0** (sum is already 0; the denominator is
        clamped away from zero only so division stays defined). This differs
        from a NaN-on-empty mean.
    """
    total = segment_sum(x, segment_ids, num_segments, sorted=sorted)
    counts = segment_count(segment_ids, num_segments, sorted=sorted)
    xp = array_namespace(total, counts)
    # Broadcast counts over trailing dims of x.
    while counts.ndim < total.ndim:
        counts = xp.expand_dims(counts, axis=-1)
    denom = maximum(counts, xp.asarray(1.0, dtype=counts.dtype))
    denom = xp.astype(denom, total.dtype)
    return total / denom


def segment_variance(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentOut:
    """Population variance of ``x`` within each segment along axis 0.

    Computed as the segment mean of squared deviations from the segment mean
    (divide by ``n``, not ``n-1``).

    Args:
        x: Values. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: Required shape-size.
        sorted: Forwarded to underlying segment ops (no-op on NumPy/Torch).

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``. Empty segments are
        ``0`` (same empty-mean convention as :func:`segment_mean`).
    """
    mean = segment_mean(x, segment_ids, num_segments, sorted=sorted)
    mean_gathered = take(mean, segment_ids)
    centered = x - mean_gathered
    return segment_mean(centered * centered, segment_ids, num_segments, sorted=sorted)


def segment_normalize(x: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentValues:
    """Divide each value by its segment sum (``0/0`` → ``0``).

    Args:
        x: Values. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: Required shape-size.
        sorted: Forwarded to :func:`segment_sum` (no-op on NumPy/Torch).

    Returns:
        Array with the same shape as ``x``. Rows whose segment sum is ``0``
        become ``0`` (no NaN from ``0/0``).
    """
    sum_x = segment_sum(x, segment_ids, num_segments, sorted=sorted)
    sum_x = take(sum_x, segment_ids)
    xp = array_namespace(x, sum_x)
    safe = where(sum_x == 0, xp.ones_like(sum_x), sum_x)
    out = x / safe
    return where(sum_x == 0, xp.zeros_like(out), out)


def segment_softmax(logits: SegmentValues, segment_ids: SegmentIds, num_segments: ShapeSize, sorted: bool = False) -> SegmentValues:
    """Softmax within segments (numerically stable).

    Subtracts the per-segment max before ``exp``, then normalizes by the
    per-segment sum of exps — same pattern as a stable full softmax.

    Args:
        logits: Scores. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: **Required** shape-size (Python ``int``, symbolic
            constant, or 0-d integral tensor) — not inferred from ids.
        sorted: Forwarded to underlying segment ops (no-op on NumPy/Torch).

    Returns:
        Array with the same shape as ``logits``. Within each segment, values
        sum to ``1`` along the segment grouping (empty segments contribute
        nothing useful if referenced via ids).

    Notes:
        Inherits empty-slot ``segment_max`` identity (``-inf``) and TF NaN
        OR-in behavior from :func:`segment_max` / :func:`segment_sum`.
        Not TorchScript-safe today (needs Python dispatch).
    """
    num_segments = _normalize_shape_dim(num_segments)
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
    # ``where`` broadcasts a 0-d fill; ``broadcast_to(..., aggregated.shape)``
    # fails under TF polymorphic / ONNX ``None`` leading lengths.
    return where(counts == 0, const, aggregated)


def segment_min_or_constant(
    x: SegmentValues,
    segment_ids: SegmentIds,
    num_segments: ShapeSize,
    constant: float = 0.0,
    sorted: bool = False,
) -> SegmentOut:
    """Segment min with a finite fill for empty segments.

    Like :func:`segment_min`, but empty slots become ``constant`` instead of
    ``+inf`` / dtype max.

    Args:
        x: Values. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: Required shape-size.
        constant: Fill for empty segments (default ``0.0``). Broadcast to the
            aggregated shape.
        sorted: Forwarded to :func:`segment_min` (no-op on NumPy/Torch).

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``.
    """
    num_segments = _normalize_shape_dim(num_segments)
    out = segment_min(x, segment_ids, num_segments, sorted=sorted)
    return _replace_empty_with_constant(out, segment_ids, num_segments, constant, sorted=sorted)


def segment_max_or_constant(
    x: SegmentValues,
    segment_ids: SegmentIds,
    num_segments: ShapeSize,
    constant: float = 0.0,
    sorted: bool = False,
) -> SegmentOut:
    """Segment max with a finite fill for empty segments.

    Like :func:`segment_max`, but empty slots become ``constant`` instead of
    ``-inf`` / dtype min.

    Args:
        x: Values. Leading axis is the segment axis.
        segment_ids: Integer segment ids (see :func:`segment_sum`).
        num_segments: Required shape-size.
        constant: Fill for empty segments (default ``0.0``). Broadcast to the
            aggregated shape.
        sorted: Forwarded to :func:`segment_max` (no-op on NumPy/Torch).

    Returns:
        Array of shape ``(num_segments,) + x.shape[1:]``.
    """
    num_segments = _normalize_shape_dim(num_segments)
    out = segment_max(x, segment_ids, num_segments, sorted=sorted)
    return _replace_empty_with_constant(out, segment_ids, num_segments, constant, sorted=sorted)


def partition_ids(
    partitions: IntArray,
    total_length: ShapeSize,
) -> IntArray:
    """Expand partition lengths to segment ids (``[0,0,…,1,1,…,n-1]``).

    This is the conversion other partition helpers call internally.
    ``num_segments`` is ``shape(partitions)[0]`` (not an argument; not
    data-dependent). ``total_length`` is the required flattened length
    (``shape(logits)[0]``, not a data ``sum(partitions)``). Passed to
    :func:`repeat` as ``total_repeat_length``.

    The only partition helper that talks to :data:`cache`. Uses
    :meth:`cache.lookup` / :meth:`cache.store` on ``"partition"``. Outside
    the cache, every call rebuilds ids. Inside, the same ``partitions``
    tensor returns the previous ids from ``cache["partition"]`` until
    the tensor is collected or the block exits — **one entry per
    partition vector**. The flattened total is ``shape(ids)[0]`` (not a
    separate ``sum(partitions)`` cache); on ONNX export that length is a
    ``dim_param``. If a cached expansion's length does not match
    ``total_length`` (in-place edit of a 0-d size, or a stale entry),
    that entry is purged, a warning is issued, and ids are recomputed.
    The length check uses host Python ints only; tracing skips it.
    Passing ``None`` for ``total_length`` is a ``TypeError``.
    """
    n_part = shape(partitions)[0]
    total = _require_shape_size("total_length", total_length)
    cached = cache.lookup("partition", partitions)
    if cached is not None:
        stale = _stale_cached_ids(cached, total)
        if stale is not None:
            cache.purge("partition", partitions)
            got, want = stale
            warnings.warn(
                f"cached partition ids length {got} != total_length {want}; "
                "purging and recomputing",
                stacklevel=2,
            )
        else:
            return cached
    ids = repeat(arange(n_part, like=partitions), partitions, total_repeat_length=total)
    cache.store("partition", partitions, ids)
    return ids


def _stale_cached_ids(cached, total):
    """``(got, want)`` when both lengths are Python ints and differ; else ``None``.

    ``type(...) is int`` (not truthiness / ``==`` on tensors) so TF Autograph
    does not turn this into ``tf.cond`` with mismatched branch structures.
    """
    got = _host_concrete_int(shape(cached)[0])
    want = _host_concrete_int(total)
    if type(got) is int and type(want) is int and got != want:
        return got, want
    return None


def partition_softmax(
    logits: ShapedArray,
    partitions: IntArray,
    total_length: ShapeSize,
) -> ShapedArray:
    """Softmax within contiguous partitions of lengths ``partitions``.

    Convenience: :func:`partition_ids` then :func:`segment_softmax`.
    ``num_segments`` is ``shape(partitions)[0]`` — not an argument.
    ``total_length`` is **required** (``shape(logits)[0]``, not a data
    ``sum(partitions)``). Does not talk to :data:`cache` itself —
    :func:`partition_ids` does, so a cache hit is shared with every
    partition helper. **Ids are rebuilt on every call** unless that
    cache is active. A compiler may CSE the rebuild; eager will not.
    If you already have ids, call :func:`segment_softmax`. This is not
    a pattern to grow (no ``partition_sum`` / ``partition_min``).

    Args:
        logits: Scores aligned with the flattened partitions (length
            ``total_length``).
        partitions: 1-D integer vector of partition sizes. Length is the
            number of segments.
        total_length: **Required** shape-size for the flattened length
            (``shape(logits)[0]``, not a data ``sum(partitions)``). Passed
            to :func:`repeat` as ``total_repeat_length``.

    Returns:
        Softmax of ``logits`` within each partition (same shape as ``logits``).
    """
    segment_ids = partition_ids(partitions, total_length)
    return segment_softmax(logits, segment_ids, num_segments=shape(partitions)[0])
