"""Ordinary tensor ops via array-api-compat (stable, input-adaptive)."""

from __future__ import annotations

from typing import Any, Optional, Sequence, Union

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


def _xp(*arrays: Any):
    """Namespace from non-scalar operands (scalars are upcast later)."""
    arrs = [a for a in arrays if a is not None and not _is_scalar(a)]
    if not arrs:
        import array_api_compat.numpy as xp

        return xp
    return array_namespace(*arrs)


def _asarray(xp, x):
    """Promote Python / NumPy scalars to 0-d arrays on ``xp``."""
    if _is_scalar(x):
        return xp.asarray(x)
    return x


def _result(xp, out):
    """Ensure results are arrays (0-d ok), never bare Python/NumPy scalars.

    Callers can rely on array methods (``.shape``, ``.dtype``, ``.ndim``, …).
    """
    if _is_scalar(out):
        return xp.asarray(out)
    return out


def _axis(axes: Axes):
    """Map historical ``axes=`` API onto Array API ``axis=``."""
    return axes


def exp(x):
    """Element-wise exponential."""
    xp = array_namespace(x)
    return _result(xp, xp.exp(x))


def log(x):
    """Element-wise natural logarithm."""
    xp = array_namespace(x)
    return _result(xp, xp.log(x))


def sum(x, axes: Axes = None):
    """Sum over axes; full reduce returns a **0-d array** (not a scalar)."""
    xp = array_namespace(x)
    return _result(xp, xp.sum(x, axis=_axis(axes)))


def min(x, axes: Axes = None):
    """Minimum over axes; full reduce returns a 0-d array."""
    xp = array_namespace(x)
    return _result(xp, xp.min(x, axis=_axis(axes)))


def max(x, axes: Axes = None):
    """Maximum over axes; full reduce returns a 0-d array."""
    xp = array_namespace(x)
    return _result(xp, xp.max(x, axis=_axis(axes)))


def mean(x, axes: Axes = None):
    """Mean over axes; full reduce returns a 0-d array."""
    xp = array_namespace(x)
    return _result(xp, xp.mean(x, axis=_axis(axes)))


def prod(x, axes: Axes = None):
    """Product over axes; full reduce returns a 0-d array."""
    xp = array_namespace(x)
    return _result(xp, xp.prod(x, axis=_axis(axes)))


def cumsum(x, axis: int = 0):
    """Cumulative sum along ``axis`` (default ``0``, never flatten)."""
    xp = array_namespace(x)
    return _result(xp, xp.cumulative_sum(x, axis=axis))


def shape(x):
    """Return the shape of ``x`` as a tuple."""
    return tuple(x.shape)


def take(x, indices, axis: int = 0):
    """Take elements from ``x`` along ``axis`` (default ``0``)."""
    xp = array_namespace(x, indices)
    return _result(xp, xp.take(x, indices, axis=axis))


def reshape(x, shape):
    """Reshape ``x`` to ``shape``."""
    xp = array_namespace(x)
    return _result(xp, xp.reshape(x, shape))


def transpose(x, axes: Optional[Sequence[int]] = None):
    """Permute axes of ``x``."""
    xp = array_namespace(x)
    if axes is None:
        return _result(xp, xp.permute_dims(x, axes=tuple(range(x.ndim - 1, -1, -1))))
    return _result(xp, xp.permute_dims(x, axes=tuple(axes)))


def concatenate(arrays, axis: int = 0):
    """Concatenate a sequence of arrays along ``axis``."""
    xp = array_namespace(*arrays)
    return _result(xp, xp.concat(arrays, axis=axis))


def stack(arrays, axis: int = 0):
    """Stack a sequence of arrays along a new ``axis``."""
    xp = array_namespace(*arrays)
    return _result(xp, xp.stack(arrays, axis=axis))


def maximum(x, y):
    """Element-wise maximum. Scalars are upcast to 0-d arrays."""
    xp = _xp(x, y)
    return _result(xp, xp.maximum(_asarray(xp, x), _asarray(xp, y)))


def minimum(x, y):
    """Element-wise minimum. Scalars are upcast to 0-d arrays."""
    xp = _xp(x, y)
    return _result(xp, xp.minimum(_asarray(xp, x), _asarray(xp, y)))


def sqrt(x):
    """Element-wise square root."""
    xp = array_namespace(x)
    return _result(xp, xp.sqrt(x))


def rsqrt(x):
    """Element-wise reciprocal square root (``1 / sqrt(x)``)."""
    xp = array_namespace(x)
    one = xp.asarray(1.0, dtype=getattr(x, "dtype", None))
    return _result(xp, one / xp.sqrt(x))


def where(condition, x, y):
    """Choose from ``x`` or ``y`` by ``condition``. Scalars upcast to 0-d arrays."""
    xp = _xp(condition, x, y)
    return _result(
        xp,
        xp.where(_asarray(xp, condition), _asarray(xp, x), _asarray(xp, y)),
    )


def clip(x, min=None, max=None):
    """Clip values to ``[min, max]``."""
    xp = array_namespace(x)
    return _result(xp, xp.clip(x, min=min, max=max))


def astype(x, dtype):
    """Cast ``x`` to ``dtype``."""
    xp = array_namespace(x)
    return _result(xp, xp.astype(x, dtype))


def cast(x, dtype):
    """Alias of :func:`astype`."""
    return astype(x, dtype)


def zeros_like(x, dtype=None):
    """Return zeros with the same shape (and backend) as ``x``."""
    xp = array_namespace(x)
    return _result(xp, xp.zeros(x.shape, dtype=x.dtype if dtype is None else dtype))


def ones_like(x, dtype=None):
    """Return ones with the same shape (and backend) as ``x``."""
    xp = array_namespace(x)
    return _result(xp, xp.ones(x.shape, dtype=x.dtype if dtype is None else dtype))


def full_like(x, fill_value, dtype=None):
    """Return an array filled with ``fill_value`` matching ``x``."""
    xp = array_namespace(x)
    return _result(xp, xp.full(x.shape, fill_value, dtype=x.dtype if dtype is None else dtype))


def zeros(shape, *, dtype=None, like=None):
    """Return zeros; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return _result(xp, xp.zeros(shape, dtype=dtype))
    xp = array_namespace(like)
    return _result(xp, xp.zeros(shape, dtype=dtype if dtype is not None else like.dtype))


def ones(shape, *, dtype=None, like=None):
    """Return ones; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return _result(xp, xp.ones(shape, dtype=dtype))
    xp = array_namespace(like)
    return _result(xp, xp.ones(shape, dtype=dtype if dtype is not None else like.dtype))


def full(shape, fill_value, *, dtype=None, like=None):
    """Return a filled array; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return _result(xp, xp.full(shape, fill_value, dtype=dtype))
    xp = array_namespace(like)
    return _result(xp, xp.full(shape, fill_value, dtype=dtype if dtype is not None else like.dtype))


def arange(start, /, stop=None, step=1, *, dtype=None, like=None, device=None):
    """Evenly spaced values; pass ``like=`` to select the backend.

    ``device`` is forwarded when supported (e.g. Torch).
    """
    if stop is None:
        start, stop = 0, start
    if like is None:
        import array_api_compat.numpy as xp

        return _result(xp, xp.arange(start, stop, step, dtype=dtype))
    xp = array_namespace(like)
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    if device is not None:
        kwargs["device"] = device
    try:
        return _result(xp, xp.arange(start, stop, step, **kwargs))
    except TypeError:
        kwargs.pop("device", None)
        return _result(xp, xp.arange(start, stop, step, **kwargs))


def repeat(x, repeats, *, total_repeat_length: Optional[int] = None, axis: Optional[int] = None):
    """Repeat elements of ``x``.

    When ``repeats`` is per-element counts and ``total_repeat_length`` is set,
    matches JAX ``jnp.repeat`` semantics (graph batching).
    """
    xp = array_namespace(x)
    if total_repeat_length is not None:
        if axis is None:
            flat = xp.reshape(x, (-1,))
            reps = repeats
            parts = []
            n = int(flat.shape[0])
            for i in range(n):
                count = int(reps[i])
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
            return _result(xp, out)
        raise NotImplementedError("total_repeat_length with axis!=None is not supported yet")
    if axis is None:
        return _result(xp, xp.repeat(x, repeats))
    return _result(xp, xp.repeat(x, repeats, axis=axis))


def matmul(x, y):
    """Matrix product of two arrays."""
    xp = array_namespace(x, y)
    return _result(xp, xp.matmul(x, y))
