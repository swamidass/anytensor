"""Ordinary tensor ops via array-api-compat (stable, input-adaptive)."""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Optional, Sequence, Union

from array_api_compat import array_namespace

Axes = Union[int, Sequence[int], None]

_SCALAR_TYPES = (bool, int, float, complex)


def _is_scalar(x: Any) -> bool:
    if isinstance(x, _SCALAR_TYPES):
        return True
    try:
        import numpy as np

        return isinstance(x, np.generic)
    except ImportError:  # pragma: no cover
        return False


def _xp(*values: Any):
    """Array namespace from non-scalar values; NumPy if only scalars."""
    arrs = [v for v in values if v is not None and not _is_scalar(v)]
    if not arrs:
        import array_api_compat.numpy as xp

        return xp
    return array_namespace(*arrs)


def _asarray(xp, x):
    """Promote Python / NumPy scalars to 0-d arrays on ``xp``."""
    if _is_scalar(x):
        return xp.asarray(x)
    return x


def as_array_result(fn: Callable) -> Callable:
    """Decorator: promote scalar returns to 0-d arrays on the call's namespace.

    Reductions (and any op) may hand back ``np.float64`` / Python scalars;
    callers need a real array (``.shape``, ``.dtype``, methods).
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        out = fn(*args, **kwargs)
        if not _is_scalar(out):
            return out
        xp = _xp(*args, *kwargs.values())
        return xp.asarray(out)

    return wrapper


def promote_scalars(*names: str) -> Callable:
    """Decorator: upcast named parameters that are scalars before calling ``fn``."""

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            values = [bound.arguments[n] for n in names if n in bound.arguments]
            xp = _xp(*values)
            for n in names:
                if n in bound.arguments:
                    bound.arguments[n] = _asarray(xp, bound.arguments[n])
            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def _axis(axes: Axes):
    """Map historical ``axes=`` API onto Array API ``axis=``."""
    return axes


@as_array_result
def exp(x):
    """Element-wise exponential."""
    return array_namespace(x).exp(x)


@as_array_result
def log(x):
    """Element-wise natural logarithm."""
    return array_namespace(x).log(x)


@as_array_result
def sum(x, axes: Axes = None):
    """Sum over axes; full reduce returns a **0-d array** (not a scalar)."""
    return array_namespace(x).sum(x, axis=_axis(axes))


@as_array_result
def min(x, axes: Axes = None):
    """Minimum over axes; full reduce returns a 0-d array."""
    return array_namespace(x).min(x, axis=_axis(axes))


@as_array_result
def max(x, axes: Axes = None):
    """Maximum over axes; full reduce returns a 0-d array."""
    return array_namespace(x).max(x, axis=_axis(axes))


@as_array_result
def mean(x, axes: Axes = None):
    """Mean over axes; full reduce returns a 0-d array."""
    return array_namespace(x).mean(x, axis=_axis(axes))


@as_array_result
def prod(x, axes: Axes = None):
    """Product over axes; full reduce returns a 0-d array."""
    return array_namespace(x).prod(x, axis=_axis(axes))


@as_array_result
def cumsum(x, axis: int = 0):
    """Cumulative sum along ``axis`` (default ``0``, never flatten)."""
    return array_namespace(x).cumulative_sum(x, axis=axis)


def shape(x):
    """Return the shape of ``x`` as a tuple."""
    return tuple(x.shape)


@as_array_result
def take(x, indices, axis: int = 0):
    """Take elements from ``x`` along ``axis`` (default ``0``)."""
    return array_namespace(x, indices).take(x, indices, axis=axis)


@as_array_result
def reshape(x, shape):
    """Reshape ``x`` to ``shape``."""
    return array_namespace(x).reshape(x, shape)


@as_array_result
def transpose(x, axes: Optional[Sequence[int]] = None):
    """Permute axes of ``x``."""
    xp = array_namespace(x)
    if axes is None:
        return xp.permute_dims(x, axes=tuple(range(x.ndim - 1, -1, -1)))
    return xp.permute_dims(x, axes=tuple(axes))


@as_array_result
def concatenate(arrays, axis: int = 0):
    """Concatenate a sequence of arrays along ``axis``."""
    return array_namespace(*arrays).concat(arrays, axis=axis)


@as_array_result
def stack(arrays, axis: int = 0):
    """Stack a sequence of arrays along a new ``axis``."""
    return array_namespace(*arrays).stack(arrays, axis=axis)


@as_array_result
@promote_scalars("x", "y")
def maximum(x, y):
    """Element-wise maximum. Scalars are upcast to 0-d arrays."""
    return array_namespace(x, y).maximum(x, y)


@as_array_result
@promote_scalars("x", "y")
def minimum(x, y):
    """Element-wise minimum. Scalars are upcast to 0-d arrays."""
    return array_namespace(x, y).minimum(x, y)


@as_array_result
def sqrt(x):
    """Element-wise square root."""
    return array_namespace(x).sqrt(x)


@as_array_result
def rsqrt(x):
    """Element-wise reciprocal square root (``1 / sqrt(x)``)."""
    xp = array_namespace(x)
    return xp.asarray(1.0, dtype=getattr(x, "dtype", None)) / xp.sqrt(x)


@as_array_result
@promote_scalars("condition", "x", "y")
def where(condition, x, y):
    """Choose from ``x`` or ``y`` by ``condition``. Scalars upcast to 0-d arrays."""
    return array_namespace(condition, x, y).where(condition, x, y)


@as_array_result
def clip(x, min=None, max=None):
    """Clip values to ``[min, max]``."""
    return array_namespace(x).clip(x, min=min, max=max)


@as_array_result
def astype(x, dtype):
    """Cast ``x`` to ``dtype``."""
    return array_namespace(x).astype(x, dtype)


def cast(x, dtype):
    """Alias of :func:`astype`."""
    return astype(x, dtype)


@as_array_result
def zeros_like(x, dtype=None):
    """Return zeros with the same shape (and backend) as ``x``."""
    xp = array_namespace(x)
    return xp.zeros(x.shape, dtype=x.dtype if dtype is None else dtype)


@as_array_result
def ones_like(x, dtype=None):
    """Return ones with the same shape (and backend) as ``x``."""
    xp = array_namespace(x)
    return xp.ones(x.shape, dtype=x.dtype if dtype is None else dtype)


@as_array_result
def full_like(x, fill_value, dtype=None):
    """Return an array filled with ``fill_value`` matching ``x``."""
    xp = array_namespace(x)
    return xp.full(x.shape, fill_value, dtype=x.dtype if dtype is None else dtype)


@as_array_result
def zeros(shape, *, dtype=None, like=None):
    """Return zeros; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.zeros(shape, dtype=dtype)
    xp = array_namespace(like)
    return xp.zeros(shape, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def ones(shape, *, dtype=None, like=None):
    """Return ones; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.ones(shape, dtype=dtype)
    xp = array_namespace(like)
    return xp.ones(shape, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def full(shape, fill_value, *, dtype=None, like=None):
    """Return a filled array; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.full(shape, fill_value, dtype=dtype)
    xp = array_namespace(like)
    return xp.full(shape, fill_value, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def arange(start, /, stop=None, step=1, *, dtype=None, like=None, device=None):
    """Evenly spaced values; pass ``like=`` to select the backend.

    ``device`` is forwarded when supported (e.g. Torch).
    """
    if stop is None:
        start, stop = 0, start
    if like is None:
        import array_api_compat.numpy as xp

        return xp.arange(start, stop, step, dtype=dtype)
    xp = array_namespace(like)
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


@as_array_result
def repeat(x, repeats, *, total_repeat_length: Optional[int] = None, axis: Optional[int] = None):
    """Repeat elements of ``x``.

    When ``repeats`` is per-element counts and ``total_repeat_length`` is set,
    matches JAX ``jnp.repeat`` semantics (graph batching).
    """
    xp = array_namespace(x)
    if total_repeat_length is not None:
        if axis is None:
            flat = xp.reshape(x, (-1,))
            parts = []
            n = int(flat.shape[0])
            for i in range(n):
                count = int(repeats[i])
                if count:
                    parts.append(xp.broadcast_to(flat[i : i + 1], (count,)))
            if not parts:
                return xp.zeros((0,), dtype=flat.dtype)
            out = xp.concat(parts, axis=0)
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


@as_array_result
def matmul(x, y):
    """Matrix product of two arrays."""
    return array_namespace(x, y).matmul(x, y)
