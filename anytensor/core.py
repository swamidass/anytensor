"""Ordinary tensor ops via array-api-compat (stable, input-adaptive)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

from array_api_compat import array_namespace

Axes = Union[int, Sequence[int], None]


def _xp(*arrays: Any):
    return array_namespace(*arrays)


def _axis(axes: Axes):
    """Map historical ``axes=`` API onto Array API ``axis=``."""
    return axes


def exp(x):
    """Element-wise exponential."""
    return _xp(x).exp(x)


def log(x):
    """Element-wise natural logarithm."""
    return _xp(x).log(x)


def sum(x, axes: Axes = None):
    """Sum of array elements over given axes (``None`` = all)."""
    return _xp(x).sum(x, axis=_axis(axes))


def min(x, axes: Axes = None):
    """Minimum of array elements over given axes."""
    return _xp(x).min(x, axis=_axis(axes))


def max(x, axes: Axes = None):
    """Maximum of array elements over given axes."""
    return _xp(x).max(x, axis=_axis(axes))


def mean(x, axes: Axes = None):
    """Mean of array elements over given axes."""
    return _xp(x).mean(x, axis=_axis(axes))


def prod(x, axes: Axes = None):
    """Product of array elements over given axes."""
    return _xp(x).prod(x, axis=_axis(axes))


def cumsum(x, axis: int = 0):
    """Cumulative sum along ``axis`` (default ``0``, never flatten)."""
    return _xp(x).cumulative_sum(x, axis=axis)


def shape(x):
    """Return the shape of ``x`` as a tuple."""
    return tuple(x.shape)


def take(x, indices, axis: int = 0):
    """Take elements from ``x`` along ``axis`` (default ``0``)."""
    xp = _xp(x, indices)
    return xp.take(x, indices, axis=axis)


def reshape(x, shape):
    """Reshape ``x`` to ``shape``."""
    return _xp(x).reshape(x, shape)


def transpose(x, axes: Optional[Sequence[int]] = None):
    """Permute axes of ``x``."""
    xp = _xp(x)
    if axes is None:
        return xp.permute_dims(x, axes=tuple(range(x.ndim - 1, -1, -1)))
    return xp.permute_dims(x, axes=tuple(axes))


def concatenate(arrays, axis: int = 0):
    """Concatenate a sequence of arrays along ``axis``."""
    return _xp(*arrays).concat(arrays, axis=axis)


def stack(arrays, axis: int = 0):
    """Stack a sequence of arrays along a new ``axis``."""
    return _xp(*arrays).stack(arrays, axis=axis)


def maximum(x, y):
    """Element-wise maximum (not a reduction)."""
    return _xp(x, y).maximum(x, y)


def minimum(x, y):
    """Element-wise minimum (not a reduction)."""
    return _xp(x, y).minimum(x, y)


def sqrt(x):
    """Element-wise square root."""
    return _xp(x).sqrt(x)


def rsqrt(x):
    """Element-wise reciprocal square root (``1 / sqrt(x)``)."""
    xp = _xp(x)
    return xp.asarray(1.0, dtype=getattr(x, "dtype", None)) / xp.sqrt(x)


def where(condition, x, y):
    """Return elements chosen from ``x`` or ``y`` depending on ``condition``."""
    return _xp(condition, x, y).where(condition, x, y)


def clip(x, min=None, max=None):
    """Clip values to the interval ``[min, max]``."""
    return _xp(x).clip(x, min=min, max=max)


def astype(x, dtype):
    """Cast ``x`` to ``dtype``."""
    return _xp(x).astype(x, dtype)


def cast(x, dtype):
    """Alias of :func:`astype`."""
    return astype(x, dtype)


def zeros_like(x, dtype=None):
    """Return an array of zeros with the same shape (and backend) as ``x``."""
    xp = _xp(x)
    return xp.zeros(x.shape, dtype=x.dtype if dtype is None else dtype)


def ones_like(x, dtype=None):
    """Return an array of ones with the same shape (and backend) as ``x``."""
    xp = _xp(x)
    return xp.ones(x.shape, dtype=x.dtype if dtype is None else dtype)


def full_like(x, fill_value, dtype=None):
    """Return an array filled with ``fill_value`` matching ``x``."""
    xp = _xp(x)
    return xp.full(x.shape, fill_value, dtype=x.dtype if dtype is None else dtype)


def zeros(shape, *, dtype=None, like=None):
    """Return an array of zeros; pass ``like=`` to select the backend."""
    if like is None:
        import numpy as np

        return np.zeros(shape, dtype=dtype)
    xp = _xp(like)
    return xp.zeros(shape, dtype=dtype if dtype is not None else like.dtype)


def ones(shape, *, dtype=None, like=None):
    """Return an array of ones; pass ``like=`` to select the backend."""
    if like is None:
        import numpy as np

        return np.ones(shape, dtype=dtype)
    xp = _xp(like)
    return xp.ones(shape, dtype=dtype if dtype is not None else like.dtype)


def full(shape, fill_value, *, dtype=None, like=None):
    """Return a filled array; pass ``like=`` to select the backend."""
    if like is None:
        import numpy as np

        return np.full(shape, fill_value, dtype=dtype)
    xp = _xp(like)
    return xp.full(shape, fill_value, dtype=dtype if dtype is not None else like.dtype)


def arange(start, /, stop=None, step=1, *, dtype=None, like=None, device=None):
    """Return evenly spaced values; pass ``like=`` to select the backend.

    ``device`` is forwarded when the Array API namespace supports it (e.g. Torch).
    """
    if stop is None:
        start, stop = 0, start
    if like is None:
        import numpy as np

        return np.arange(start, stop, step, dtype=dtype)
    xp = _xp(like)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        return xp.arange(start, stop, step, **kwargs)
    except TypeError:
        kwargs.pop("device", None)
        return xp.arange(start, stop, step, **kwargs)


def repeat(x, repeats, *, total_repeat_length: Optional[int] = None, axis: Optional[int] = None):
    """Repeat elements of ``x``.

    When ``repeats`` is an array of per-element counts and ``total_repeat_length``
    is set, matches JAX ``jnp.repeat`` semantics (important for graph batching).
    """
    xp = _xp(x)
    # NumPy / Array API: xp.repeat(x, repeats, axis=axis)
    # JAX-style total_repeat_length: use numpy path for NumPy; for others try kwargs.
    if total_repeat_length is not None:
        # Build output by concatenating repeated slices — works across backends.
        if axis is None:
            flat = xp.reshape(x, (-1,))
            reps = repeats
            if hasattr(reps, "shape") and getattr(reps, "ndim", 0) == 0:
                reps = xp.broadcast_to(reps, flat.shape)
            parts = []
            n = int(flat.shape[0])
            for i in range(n):
                count = int(reps[i])
                if count:
                    parts.append(xp.broadcast_to(flat[i : i + 1], (count,)))
            if not parts:
                return xp.zeros((0,), dtype=flat.dtype)
            out = xp.concat(parts, axis=0)
            if total_repeat_length is not None and out.shape[0] != total_repeat_length:
                # Truncate or error; JAX requires exact length when provided.
                if out.shape[0] > total_repeat_length:
                    out = out[:total_repeat_length]
                elif out.shape[0] < total_repeat_length:
                    raise ValueError(
                        f"repeat produced length {out.shape[0]}, "
                        f"expected total_repeat_length={total_repeat_length}"
                    )
            return out
        raise NotImplementedError("total_repeat_length with axis!=None is not supported yet")
    if axis is None:
        return xp.repeat(x, repeats)
    return xp.repeat(x, repeats, axis=axis)


def matmul(x, y):
    """Matrix product of two arrays."""
    return _xp(x, y).matmul(x, y)
