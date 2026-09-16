"""Ordinary tensor ops via array-api-compat (stable, input-adaptive)."""

from __future__ import annotations

import functools
import inspect
import math
import warnings
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator, Literal, Optional, Sequence, Union

from array_api_compat import array_namespace

Axes = Union[int, Sequence[int], None]
Fallback = Literal["copy", "error"]

_SCALAR_TYPES = (bool, int, float, complex)

# Library defaults for NumPy → framework promotion (decorators / align read these).
_promote_copy: ContextVar[bool] = ContextVar("anytensor_promote_copy", default=False)
_promote_fallback: ContextVar[Fallback] = ContextVar(
    "anytensor_promote_fallback", default="copy"
)


@contextmanager
def promote_options(*, copy: bool = False, fallback: Fallback = "copy") -> Iterator[None]:
    """Temporarily set NumPy-upcast defaults (``copy`` / non-ref ``fallback``).

    Example::

        with at.promote_options(copy=True):
            y = at.maximum(torch_x, numpy_y)  # safe if numpy_y will be mutated
    """
    t_copy = _promote_copy.set(copy)
    t_fb = _promote_fallback.set(fallback)
    try:
        yield
    finally:
        _promote_copy.reset(t_copy)
        _promote_fallback.reset(t_fb)


def _is_scalar(x: Any) -> bool:
    if isinstance(x, _SCALAR_TYPES):
        return True
    try:
        import numpy as np

        return isinstance(x, np.generic)
    except ImportError:  # pragma: no cover
        return False


def _is_numpy_ndarray(x: Any) -> bool:
    try:
        import numpy as np

        return isinstance(x, np.ndarray)
    except ImportError:  # pragma: no cover
        return False


def _xp(*values: Any):
    """Pick Array API namespace for mixed operands.

    - Scalars only → NumPy
    - NumPy arrays only → NumPy
    - Any non-NumPy framework tensor → that framework (NumPy is host data
      and is upcast; we never demote Torch/JAX/TF to NumPy)
    - Multiple distinct non-NumPy frameworks → ``array_namespace`` error
    """
    arrs = [v for v in values if v is not None and not _is_scalar(v)]
    if not arrs:
        import array_api_compat.numpy as xp

        return xp
    others = [a for a in arrs if not _is_numpy_ndarray(a)]
    if not others:
        return array_namespace(*arrs)
    return array_namespace(*others)


def _asarray(
    xp,
    x,
    *,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
):
    """Promote Python scalars and NumPy ndarrays onto ``xp``.

    Parameters
    ----------
    copy:
        If True, always copy NumPy buffers into the framework. If False
        (default), prefer a reference via ``asarray(..., copy=False)``.
        ``None`` reads :func:`promote_options` / the module default.
    fallback:
        When ``copy=False`` but zero-copy is impossible — common for some
        non-contiguous layouts, dtype conversions, or backends that refuse
        to alias:

        - ``"copy"`` (default): emit a warning and copy
        - ``"error"``: raise ``ValueError``

        ``None`` reads the current default.
    """
    if x is None:
        return x
    if _is_scalar(x):
        return xp.asarray(x)
    if not _is_numpy_ndarray(x):
        return x

    if copy is None:
        copy = _promote_copy.get()
    if fallback is None:
        fallback = _promote_fallback.get()

    if copy:
        return xp.asarray(x, copy=True)

    # Prefer reference. Non-contiguous / incompatible views may refuse.
    try:
        return xp.asarray(x, copy=False)
    except (TypeError, ValueError) as exc:
        if fallback == "error":
            raise ValueError(
                "NumPy upcast could not use a zero-copy reference "
                f"(copy=False, fallback='error'): {exc}"
            ) from exc
        warnings.warn(
            f"NumPy upcast fell back to a copy (zero-copy failed: {exc})",
            stacklevel=2,
        )
        return xp.asarray(x, copy=True)


def align_arrays(
    *arrays: Any,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
):
    """Align operands on one namespace; upcast scalars/NumPy to non-NumPy peers."""
    xp = _xp(*arrays)
    return tuple(_asarray(xp, a, copy=copy, fallback=fallback) for a in arrays)


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


OperandKind = Literal["data", "index", "mask"]


def _apply_dtype_roles(xp, converted: dict, roles: dict[str, OperandKind]) -> None:
    """In-place dtype policy after namespace alignment.

    - ``data``: shared ``result_type`` (ints widen to float when mixed with floats)
    - ``index``: must stay integral (segment ids, take indices) — never float
    - ``mask``: boolean (``where`` condition)
    """
    data_names = [n for n, k in roles.items() if k == "data" and n in converted]
    if len(data_names) >= 1:
        try:
            rt = xp.result_type(*[converted[n] for n in data_names])
        except TypeError:
            rt = None
        if rt is not None:
            for n in data_names:
                if converted[n].dtype != rt:
                    converted[n] = xp.astype(converted[n], rt)

    for n, kind in roles.items():
        if n not in converted:
            continue
        a = converted[n]
        if kind == "index":
            if not xp.isdtype(a.dtype, "integral"):
                raise TypeError(
                    f"operand {n!r} is kind='index' and must be integral, got dtype={a.dtype}"
                )
        elif kind == "mask":
            if not xp.isdtype(a.dtype, "bool"):
                converted[n] = xp.astype(a, xp.bool)


def promote(
    *,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
    **roles: OperandKind,
) -> Callable:
    """Decorator: namespace upcast + per-operand dtype policy.

    Pass keyword roles for each parameter::

        @promote(x="data", y="data")
        def maximum(x, y): ...

        @promote(x="data", indices="index")
        def take(x, indices, axis=0): ...

        @promote(condition="mask", x="data", y="data")
        def where(condition, x, y): ...

    ``data`` operands share Array API ``result_type`` (so a NumPy int beside a
    float tensor becomes float). ``index`` stays integral (width is
    backend-local: Torch may cast to int64; JAX/TF often keep int32). ``mask``
    becomes bool.

    ``copy`` / ``fallback`` control NumPy→framework buffer sharing (see
    :func:`promote_options`).
    """
    if not roles:
        raise TypeError("promote() requires at least one name=kind role")

    def decorator(fn: Callable) -> Callable:
        sig = inspect.signature(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            names = [n for n in roles if n in bound.arguments]
            values = [bound.arguments[n] for n in names]
            xp = _xp(*values)
            converted = {
                n: _asarray(xp, bound.arguments[n], copy=copy, fallback=fallback)
                for n in names
            }
            _apply_dtype_roles(xp, converted, roles)
            for n, v in converted.items():
                bound.arguments[n] = v
            return fn(*bound.args, **bound.kwargs)

        return wrapper

    return decorator


def promote_scalars(
    *names: str,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
) -> Callable:
    """Upcast named operands as ``data`` (namespace + ``result_type``).

    Prefer :func:`promote` when some args are indices/masks. Kept as a short
    form of ``@promote(x="data", y="data")``.
    """
    if not names:
        raise TypeError("promote_scalars() requires at least one parameter name")
    return promote(**{n: "data" for n in names}, copy=copy, fallback=fallback)


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
@promote(x="data", indices="index")
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
@promote(x="data", y="data")
def maximum(x, y):
    """Element-wise maximum. Scalars/NumPy upcast; dtypes via ``result_type``."""
    return array_namespace(x, y).maximum(x, y)


@as_array_result
@promote(x="data", y="data")
def minimum(x, y):
    """Element-wise minimum. Scalars/NumPy upcast; dtypes via ``result_type``."""
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
@promote(condition="mask", x="data", y="data")
def where(condition, x, y):
    """Choose from ``x`` or ``y`` by ``condition``. Scalars/NumPy upcast."""
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
@promote(x="data", repeats="index")
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
@promote(x="data", y="data")
def matmul(x, y):
    """Matrix product of two arrays. NumPy operands upcast onto peers."""
    return array_namespace(x, y).matmul(x, y)


# --- Portable constants / dtype introspection ------------------------------
# Scalars are Python floats (Array API convention) — not backend objects.
# Framework dtypes stay on the array's namespace (``like=`` / ``astype``);
# use :func:`dtype` when you need ``xp.bool`` / ``xp.float32`` for a peer array.

inf = math.inf
ninf = -math.inf
nan = math.nan
pi = math.pi
e = math.e
newaxis = None


def finfo(x):
    """Floating limits for ``x.dtype`` on ``x``'s Array API namespace."""
    return array_namespace(x).finfo(x.dtype)


def iinfo(x):
    """Integral limits for ``x.dtype`` on ``x``'s Array API namespace."""
    return array_namespace(x).iinfo(x.dtype)


def dtype(name: str, like):
    """Framework dtype ``name`` for the namespace of ``like`` (e.g. ``\"bool\"``).

    Prefer this over reaching into backends. Strings often also work directly in
    ``astype`` / ``zeros(..., dtype=)``; use this when you need the dtype object.
    """
    xp = array_namespace(like)
    if not hasattr(xp, name):
        raise AttributeError(f"{xp} has no dtype attribute {name!r}")
    return getattr(xp, name)


# --- NaN / finiteness utilities -------------------------------------------


@as_array_result
def is_nan(x):
    """Element-wise NaN test (Array API ``isnan``)."""
    return array_namespace(x).isnan(x)


@as_array_result
def is_finite(x):
    """Element-wise finite test (Array API ``isfinite``)."""
    return array_namespace(x).isfinite(x)


@as_array_result
def is_inf(x):
    """Element-wise infinity test (Array API ``isinf``)."""
    return array_namespace(x).isinf(x)


# Array API short names
isnan = is_nan
isfinite = is_finite
isinf = is_inf


@as_array_result
@promote(x="data", value="data")
def fill_nan(x, value=0.0):
    """Replace NaNs in ``x`` with ``value`` (broadcasts). Leaves ±inf unchanged."""
    xp = array_namespace(x, value)
    return xp.where(xp.isnan(x), value, x)


nan_fill = fill_nan  # alias


@promote(x="data", value="data")
def fill_nan_mask(x, value=0.0):
    """Return ``(filled, mask)``: NaNs replaced, plus a boolean NaN mask.

    ``mask`` is ``True`` where ``x`` was NaN (same polarity as :func:`is_nan` /
    NumPy masked-array invalid). Boolean, not 0/1 — cast if you need weights.
    Equivalent to ``(fill_nan(x, value), is_nan(x))``; not ``~is_finite``
    (±inf is non-NaN).
    """
    xp = array_namespace(x, value)
    mask = xp.isnan(x)
    filled = xp.where(mask, value, x)
    return filled, mask


nan_fill_mask = fill_nan_mask  # alias


@as_array_result
@promote(x="data")
def nan_to_num(x, *, nan=0.0, posinf=None, neginf=None):
    """Replace NaN and ±inf (Array API ``nan_to_num``).

    Defaults: NaN → ``nan`` (0.0); ``posinf`` / ``neginf`` ``None`` → large
    finite values from the dtype's finfo (framework-dependent).
    """
    return array_namespace(x).nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@as_array_result
@promote(x="data", y="data")
def equal_nan(x, y):
    """Element-wise equality treating NaN as equal to NaN.

    Returns a boolean array: ``(x == y) | (isnan(x) & isnan(y))``.
    Non-NaN values compare with ordinary ``==`` (so ``+inf == +inf``).
    """
    xp = array_namespace(x, y)
    both_nan = xp.logical_and(xp.isnan(x), xp.isnan(y))
    return xp.logical_or(x == y, both_nan)
