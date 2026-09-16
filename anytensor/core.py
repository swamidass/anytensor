"""Ordinary tensor ops via array-api-compat (stable, input-adaptive)."""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Literal, Optional, Union

from .namespace import array_namespace, _is_numpy_ndarray, _is_scalar
from .typing import ArrayT, Axes, DtypeLike, ShapeLike, ShapeSize, ShapedArray, IntArray

Fallback = Literal["copy", "error"]
OperandKind = Literal["data", "index", "mask", "shape"]

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


def _xp(*values: Any) -> Any:
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
    xp: Any,
    x: Any,
    *,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
) -> Any:
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
    *arrays: ShapedArray,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
) -> tuple[ShapedArray, ...]:
    """Align operands on one namespace; upcast scalars/NumPy to non-NumPy peers."""
    xp = _xp(*arrays)
    return tuple(_asarray(xp, a, copy=copy, fallback=fallback) for a in arrays)  # type: ignore[return-value]


def as_array_result(fn: Callable[..., ShapedArray]) -> Callable[..., ShapedArray]:
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


def _normalize_shape_dim(value: ShapeSize | None) -> ShapeSize | None:
    """Normalize a shape-size: Python int, symbolic size, or 0-d integral tensor.

    Unlike ``index`` / ``data``, plain Python ints stay Python (not promoted to
    0-d tensors) so ``jax.jit`` / ``tf.function`` / ``torch.compile`` can treat
    them as static sizes. Tensor scalars and backend size symbols are left as-is.
    """
    if value is None:
        return None
    from .backends import UnknownSize

    if isinstance(value, UnknownSize):
        return value
    if _is_scalar(value):
        if isinstance(value, (bool,)):
            raise TypeError("shape dim cannot be bool")
        try:
            import numpy as np

            if isinstance(value, np.bool_):
                raise TypeError("shape dim cannot be bool")
            if isinstance(value, np.generic):
                return int(value.item())
        except ImportError:  # pragma: no cover
            pass
        if isinstance(value, float):
            if not value.is_integer():
                raise TypeError(f"shape dim must be integral, got {value!r}")
            return int(value)
        return int(value)
    # Tensor / SymInt / framework size — keep; callers build shapes from it.
    return value


def _apply_dtype_roles(xp, converted: dict, roles: dict[str, OperandKind]) -> None:
    """In-place dtype policy after namespace alignment.

    - ``data``: shared ``result_type`` (ints widen to float when mixed with floats)
    - ``index``: must stay integral (segment ids, take indices) — never float
    - ``mask``: boolean (``where`` condition)
    - ``shape``: size dim — Python int, symbolic size, or 0-d integral tensor
      scalar (never a vector; Python ints are not upcast to 0-d arrays)
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
        elif kind == "shape":
            from .backends import UnknownSize

            if isinstance(a, UnknownSize) or _is_scalar(a):
                continue
            ndim = getattr(a, "ndim", None)
            if ndim is not None and ndim != 0:
                raise TypeError(
                    f"operand {n!r} is kind='shape' and must be a scalar size, "
                    f"got ndim={ndim}"
                )
            dtype = getattr(a, "dtype", None)
            if dtype is not None and not xp.isdtype(dtype, "integral"):
                raise TypeError(
                    f"operand {n!r} is kind='shape' and must be integral, got dtype={dtype}"
                )


def promote(
    *,
    copy: Optional[bool] = None,
    fallback: Optional[Fallback] = None,
    **roles: OperandKind,
) -> Callable:
    """Decorator: namespace upcast + per-operand dtype policy.

    Pass keyword roles for each parameter::

        @promote(x="data", y="data")
        def maximum(x: ArrayT, y: ArrayT) -> ArrayT: ...

        @promote(x="data", indices="index")
        def take(x, indices, axis=0): ...

        @promote(condition="mask", x="data", y="data")
        def where(condition: ArrayT, x: ArrayT, y: ArrayT) -> ArrayT: ...

        @promote(x="data", segment_ids="index", num_segments="shape")
        def segment_sum(x, segment_ids, num_segments): ...

    ``data`` operands share Array API ``result_type`` (so a NumPy int beside a
    float tensor becomes float). ``index`` stays integral (width is
    backend-local: Torch may cast to int64; JAX/TF often keep int32). ``mask``
    becomes bool. ``shape`` is a size dim (Python int / symbolic / 0-d integral
    tensor) and is **not** promoted to a 0-d array.

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
            # Namespace from data/index/mask only — pure Python shape ints stay host.
            ns_values = [
                bound.arguments[n]
                for n in names
                if roles[n] != "shape" or not _is_scalar(bound.arguments[n])
            ]
            xp = _xp(*ns_values) if ns_values else _xp()
            converted = {}
            for n in names:
                v = bound.arguments[n]
                if roles[n] == "shape":
                    if v is None:
                        continue
                    converted[n] = _normalize_shape_dim(v)
                else:
                    converted[n] = _asarray(xp, v, copy=copy, fallback=fallback)
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


def _axis(axes: Axes) -> Axes:
    """Map historical ``axes=`` API onto Array API ``axis=``."""
    return axes


@contextmanager
def _ignore_fp_invalid(xp: Any):
    """Silence NumPy invalid/overflow warnings for intentional NaN/±inf/empty ops."""
    errstate = getattr(xp, "errstate", None)
    if errstate is None:
        yield
        return
    with errstate(invalid="ignore", divide="ignore", over="ignore"):
        yield


@as_array_result
def exp(x: ShapedArray) -> ShapedArray:
    """Element-wise exponential."""
    return array_namespace(x).exp(x)


@as_array_result
def log(x: ShapedArray) -> ShapedArray:
    """Element-wise natural logarithm."""
    return array_namespace(x).log(x)


@as_array_result
def sum(x: ShapedArray, axes: Axes = None) -> ShapedArray:
    """Sum over axes; full reduce returns a **0-d array** (not a scalar)."""
    xp = array_namespace(x)
    with _ignore_fp_invalid(xp):
        return xp.sum(x, axis=_axis(axes))


@as_array_result
def min(x: ShapedArray, axes: Axes = None) -> ShapedArray:
    """Minimum over axes; full reduce returns a **0-d array** (not a scalar).

    Notes:
        Length-0 reductions are framework-defined (often error) — prefer
        nonempty. Under TF XLA (``jit_compile=True``), NaN inputs may yield
        ``±inf`` instead of NaN; eager TF / NumPy / JAX usually keep NaN.
    """
    xp = array_namespace(x)
    with _ignore_fp_invalid(xp):
        return xp.min(x, axis=_axis(axes))


@as_array_result
def max(x: ShapedArray, axes: Axes = None) -> ShapedArray:
    """Maximum over axes; full reduce returns a **0-d array** (not a scalar).

    Notes:
        Length-0 reductions are framework-defined (often error) — prefer
        nonempty. Under TF XLA, NaN inputs may yield ``±inf`` instead of NaN.
    """
    xp = array_namespace(x)
    with _ignore_fp_invalid(xp):
        return xp.max(x, axis=_axis(axes))


@as_array_result
def mean(x: ShapedArray, axes: Axes = None) -> ShapedArray:
    """Mean over axes; full reduce returns a **0-d array** (not a scalar).

    Notes:
        Empty full-reduce (``x.size == 0``) returns a 0-d NaN on the backend
        dtype to avoid NumPy's ``Mean of empty slice`` warning path.
    """
    xp = array_namespace(x)
    # NumPy emits RuntimeWarning "Mean of empty slice" (not via errstate).
    size = getattr(x, "size", None)
    if axes is None and size == 0:
        dtype = getattr(x, "dtype", None)
        return xp.asarray(float("nan"), dtype=dtype) if dtype is not None else xp.asarray(float("nan"))
    with _ignore_fp_invalid(xp):
        return xp.mean(x, axis=_axis(axes))


@as_array_result
def prod(x: ShapedArray, axes: Axes = None) -> ShapedArray:
    """Product over axes; full reduce returns a **0-d array** (not a scalar).

    Notes:
        ``inf *`` a subnormal or float32-min value may be ``inf`` (NumPy /
        eager TF) or ``nan`` (JAX / TF XLA) when the tiny flushes to 0. Not
        standardized — keep finite samples away from the underflow edge if
        you need portable results.
    """
    xp = array_namespace(x)
    with _ignore_fp_invalid(xp):
        return xp.prod(x, axis=_axis(axes))


@as_array_result
def cumsum(x: ShapedArray, axis: int = 0) -> ShapedArray:
    """Cumulative sum along ``axis`` (default ``0``, never flatten)."""
    xp = array_namespace(x)
    with _ignore_fp_invalid(xp):
        return xp.cumulative_sum(x, axis=axis)


def shape(x: ShapedArray) -> ShapeLike:
    """Return the shape of ``x`` as a tuple.

    Under tracing (``tf.function``, ``jax.jit``, ``torch.compile``), unknown
    dims are backend size tensors / symbols rather than Python ``None``, so
    callers can build matching outputs under jit/compile.

    Notes:
        Prefer this over raw ``x.shape`` when feeding ``zeros`` / ``full`` /
        ``*_like`` under polymorphic TF graphs — ``tnp.zeros`` errors on
        ``TensorShape(None,)`` after retracing.
    """
    from .backends import HashableTuple, get_backend

    s = get_backend(x).shape(x)
    if isinstance(s, HashableTuple):
        return tuple(s)
    return tuple(s)


@as_array_result
@promote(x="data", indices="index")
def take(x: ShapedArray, indices: IntArray, axis: int = 0) -> ShapedArray:
    """Take elements from ``x`` along ``axis`` (default ``0``)."""
    return array_namespace(x, indices).take(x, indices, axis=axis)


@as_array_result
def reshape(x: ShapedArray, shape: ShapeLike) -> ShapedArray:
    """Reshape ``x`` to ``shape``."""
    return array_namespace(x).reshape(x, shape)


@as_array_result
def transpose(x: ShapedArray, axes: Optional[Sequence[int]] = None) -> ShapedArray:
    """Permute axes of ``x``."""
    xp = array_namespace(x)
    if axes is None:
        return xp.permute_dims(x, axes=tuple(range(x.ndim - 1, -1, -1)))
    return xp.permute_dims(x, axes=tuple(axes))


@as_array_result
def concatenate(arrays: Sequence[ShapedArray], axis: int = 0) -> ShapedArray:
    """Concatenate a sequence of arrays along ``axis``."""
    return array_namespace(*arrays).concat(arrays, axis=axis)


@as_array_result
def stack(arrays: Sequence[ShapedArray], axis: int = 0) -> ShapedArray:
    """Stack a sequence of arrays along a new ``axis``."""
    return array_namespace(*arrays).stack(arrays, axis=axis)


@as_array_result
@promote(x="data", y="data")
def maximum(x: ShapedArray, y: ShapedArray) -> ShapedArray:
    """Element-wise maximum. Scalars/NumPy upcast; dtypes via ``result_type``.

    Notes:
        Under TF XLA (``jit_compile=True``), NaN inputs may yield ``±inf``
        instead of NaN; eager TF / NumPy / JAX usually keep NaN. Not
        standardized across backends.
    """
    return array_namespace(x, y).maximum(x, y)


@as_array_result
@promote(x="data", y="data")
def minimum(x: ShapedArray, y: ShapedArray) -> ShapedArray:
    """Element-wise minimum. Scalars/NumPy upcast; dtypes via ``result_type``.

    Notes:
        Under TF XLA, NaN inputs may yield ``±inf`` instead of NaN. Not
        standardized across backends.
    """
    return array_namespace(x, y).minimum(x, y)


@as_array_result
def sqrt(x: ShapedArray) -> ShapedArray:
    """Element-wise square root."""
    return array_namespace(x).sqrt(x)


@as_array_result
def rsqrt(x: ShapedArray) -> ShapedArray:
    """Element-wise reciprocal square root (``1 / sqrt(x)``)."""
    xp = array_namespace(x)
    return xp.asarray(1.0, dtype=getattr(x, "dtype", None)) / xp.sqrt(x)


@as_array_result
@promote(condition="mask", x="data", y="data")
def where(condition: ShapedArray, x: ShapedArray, y: ShapedArray) -> ShapedArray:
    """Choose from ``x`` or ``y`` by ``condition``. Scalars/NumPy upcast."""
    return array_namespace(condition, x, y).where(condition, x, y)


@as_array_result
def clip(x: ShapedArray, min: Any = None, max: Any = None) -> ShapedArray:
    """Clip values to ``[min, max]``."""
    return array_namespace(x).clip(x, min=min, max=max)


@as_array_result
def astype(x: ShapedArray, dtype: DtypeLike) -> ShapedArray:
    """Cast ``x`` to ``dtype``."""
    return array_namespace(x).astype(x, dtype)


def cast(x: ShapedArray, dtype: DtypeLike) -> ShapedArray:
    """Alias of :func:`astype`."""
    return astype(x, dtype)


@as_array_result
def zeros_like(x: ShapedArray, dtype: DtypeLike = None) -> ShapedArray:
    """Return zeros with the same shape (and backend) as ``x``.

    Built from :func:`shape` so polymorphic ``tf.function`` sees symbolic
    sizes instead of ``None`` dims (raw ``tnp.zeros_like`` / ``tnp.zeros`` can
    fail after retracing).
    """
    xp = array_namespace(x)
    return xp.zeros(shape(x), dtype=x.dtype if dtype is None else dtype)


@as_array_result
def ones_like(x: ShapedArray, dtype: DtypeLike = None) -> ShapedArray:
    """Return ones with the same shape (and backend) as ``x``.

    Uses :func:`shape` for graph-safe sizes under ``tf.function`` (see
    :func:`zeros_like`).
    """
    xp = array_namespace(x)
    return xp.ones(shape(x), dtype=x.dtype if dtype is None else dtype)


@as_array_result
def full_like(x: ShapedArray, fill_value: Any, dtype: DtypeLike = None) -> ShapedArray:
    """Return an array filled with ``fill_value`` matching ``x``.

    Uses :func:`shape` for graph-safe sizes under ``tf.function`` (see
    :func:`zeros_like`).
    """
    xp = array_namespace(x)
    return xp.full(shape(x), fill_value, dtype=x.dtype if dtype is None else dtype)


@as_array_result
def zeros(shape: ShapeLike, *, dtype: DtypeLike = None, like: Optional[ShapedArray] = None) -> ShapedArray:
    """Return zeros; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.zeros(shape, dtype=dtype)
    xp = array_namespace(like)
    return xp.zeros(shape, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def ones(shape: ShapeLike, *, dtype: DtypeLike = None, like: Optional[ShapedArray] = None) -> ShapedArray:
    """Return ones; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.ones(shape, dtype=dtype)
    xp = array_namespace(like)
    return xp.ones(shape, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def full(shape: ShapeLike, fill_value: Any, *, dtype: DtypeLike = None, like: Optional[ShapedArray] = None) -> ShapedArray:
    """Return a filled array; pass ``like=`` to select the backend."""
    if like is None:
        import array_api_compat.numpy as xp

        return xp.full(shape, fill_value, dtype=dtype)
    xp = array_namespace(like)
    return xp.full(shape, fill_value, dtype=dtype if dtype is not None else like.dtype)


@as_array_result
def arange(start: Any, /, stop: Any = None, step: Any = 1, *, dtype: DtypeLike = None, like: Optional[ShapedArray] = None, device: Any = None) -> ShapedArray:
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


def _leading_dim_is_concrete(size) -> bool:
    try:
        int(size)
        return True
    except (TypeError, ValueError):
        return False


def _host_concrete_int(value):
    """Return ``int(value)`` when safe on the host; else ``None`` (tracing)."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _repeats_are_host_concrete(repeats) -> bool:
    """True when every repeat count can be read as a Python int (eager only)."""
    if _is_scalar(repeats):
        return _host_concrete_int(repeats) is not None
    n = _host_concrete_int(getattr(repeats, "shape", (None,))[0])
    if n is None:
        return False
    for i in range(n):
        if _host_concrete_int(repeats[i]) is None:
            return False
    return True


def _repeat_total_length_concrete(xp, x, repeats, total_repeat_length):
    """Eager / concrete ``total_repeat_length`` path (Python lengths)."""
    flat = xp.reshape(x, (-1,))
    parts = []
    n = int(flat.shape[0])
    for i in range(n):
        count = int(repeats) if _is_scalar(repeats) else int(repeats[i])
        if count:
            parts.append(xp.broadcast_to(flat[i : i + 1], (count,)))
    if not parts:
        return xp.zeros((0,), dtype=flat.dtype)
    out = xp.concat(parts, axis=0)
    if out.shape[0] > total_repeat_length:
        return out[:total_repeat_length]
    if out.shape[0] < total_repeat_length:
        raise ValueError(
            f"repeat produced length {out.shape[0]}, "
            f"expected total_repeat_length={total_repeat_length}"
        )
    return out


def _pad_or_slice_leading(xp, out, length):
    """Match leading size to ``length`` with tensor-friendly ops (jit/compile)."""
    from .backends import get_backend

    backend = get_backend(out)
    out = out[:length]
    if backend.framework_name == "tensorflow":
        tf = backend.tf
        cur = tf.shape(out)[0]
        paddings = [[0, length - cur]] + [[0, 0]] * (len(out.shape) - 1)
        return tf.pad(out, paddings)
    if backend.framework_name == "torch":
        torch = backend.torch
        cur = out.shape[0]
        try:
            pad_n = int(length) - int(cur)
        except (TypeError, ValueError):
            pad_n = length - cur
        if isinstance(pad_n, int) and pad_n <= 0:
            return out
        z = torch.zeros((pad_n,) + tuple(out.shape[1:]), dtype=out.dtype, device=out.device)
        return torch.cat([out, z], dim=0)
    cur = out.shape[0]
    try:
        pad_n = int(length) - int(cur)
        if pad_n <= 0:
            return out
        z = xp.zeros((pad_n,) + tuple(out.shape[1:]), dtype=out.dtype)
        return xp.concat([out, z], axis=0)
    except (TypeError, ValueError):
        z = xp.zeros((length - cur,) + tuple(out.shape[1:]), dtype=out.dtype)
        return xp.concat([out, z], axis=0)


@as_array_result
@promote(x="data")
def repeat(x: ShapedArray, repeats: Any, *, total_repeat_length: Optional[ShapeSize] = None, axis: Optional[int] = None) -> ShapedArray:
    """Repeat elements of ``x``.

    Similar to NumPy / JAX ``repeat``. When ``total_repeat_length`` is set with
    per-element ``repeats``, matches JAX ``jnp.repeat(..., total_repeat_length=)``
    (pad or slice the leading axis to that length).

    Args:
        x: Input array.
        repeats: Python ``int`` (same count for every element) or an integral
            array of per-element counts. **Python scalar repeats stay Python**
            (not promoted to 0-d tensors) so ``tf.function`` / ``jax.jit`` /
            ``torch.compile`` stay happy — promoting a ``2`` to a 0-d TF tensor
            breaks ``tf.experimental.numpy.repeat``.
        total_repeat_length: Optional shape-size (Python ``int``, symbolic
            constant, or 0-d integral tensor) for the flattened output length.
            Required for a static output size under ``jax.jit`` when repeats
            are dynamic. Not supported together with ``axis is not None`` yet.
        axis: Axis to repeat along; ``None`` flattens (Array API / NumPy style).

    Returns:
        Array with repeated elements on the same backend as ``x``.

    Notes:
        Under ``jax.jit``, ``jnp.repeat`` needs static repeat counts or a
        static ``total_repeat_length``. For :func:`~anytensor.partition_softmax`,
        pass static ``sum_partitions``. Omitting ``total_repeat_length`` is
        fine eagerly and on TensorFlow.
    """
    from .backends import get_backend

    xp = array_namespace(x)
    if not _is_scalar(repeats):
        repeats = _asarray(xp, repeats)
        if not xp.isdtype(repeats.dtype, "integral"):
            raise TypeError(
                f"repeats must be integral, got dtype={getattr(repeats, 'dtype', type(repeats))}"
            )

    if total_repeat_length is not None:
        if axis is not None:
            raise NotImplementedError("total_repeat_length with axis!=None is not supported yet")
        backend = get_backend(x)
        try:
            return backend.repeat(x, repeats, total_repeat_length)
        except NotImplementedError:
            pass
        flat = xp.reshape(x, (-1,))
        # Python-loop path only when repeat *values* are host ints. Under
        # tf.function, shapes can look concrete while repeats[i] is symbolic.
        if (
            _leading_dim_is_concrete(flat.shape[0])
            and _repeats_are_host_concrete(repeats)
            and _leading_dim_is_concrete(total_repeat_length)
        ):
            return _repeat_total_length_concrete(
                xp, x, repeats, int(_normalize_shape_dim(total_repeat_length))
            )
        out = xp.repeat(flat, repeats)
        return _pad_or_slice_leading(xp, out, _normalize_shape_dim(total_repeat_length))

    if axis is None:
        return xp.repeat(x, repeats)
    return xp.repeat(x, repeats, axis=axis)


@as_array_result
@promote(x="data", y="data")
def matmul(x: ShapedArray, y: ShapedArray) -> ShapedArray:
    """Matrix product of two arrays. NumPy operands upcast onto peers."""
    return array_namespace(x, y).matmul(x, y)


# --- Portable specials / dtype introspection --------------------------------
# Resolve via the argument's backend (internal), then return that backend's
# attribute. Floating specials are Python floats (Array API style) and promote
# under ``@promote`` / ``@as_array_result`` when used in ops. ``newaxis`` is
# always ``None``.


def _backend_attr(like: ArrayT, name: str) -> Any:
    from .backends import get_backend

    return getattr(get_backend(like), name)


def inf(like: ShapedArray) -> Any:
    """``+inf`` from the backend of ``like``."""
    return _backend_attr(like, "inf")


def ninf(like: ShapedArray) -> Any:
    """``-inf`` from the backend of ``like``."""
    return _backend_attr(like, "ninf")


def nan(like: ShapedArray) -> Any:
    """``NaN`` from the backend of ``like``."""
    return _backend_attr(like, "nan")


def pi(like: ShapedArray) -> Any:
    """``π`` from the backend of ``like``."""
    return _backend_attr(like, "pi")


def e(like: ShapedArray) -> Any:
    """Euler's number from the backend of ``like``."""
    return _backend_attr(like, "e")


newaxis = None


def finfo(x: ArrayT) -> Any:
    """Floating limits for ``x.dtype`` via ``x``'s backend."""
    return _backend_attr(x, "finfo")(x.dtype)


def iinfo(x: ArrayT) -> Any:
    """Integral limits for ``x.dtype`` via ``x``'s backend."""
    return _backend_attr(x, "iinfo")(x.dtype)


def dtype(name: str, like: ShapedArray) -> DtypeLike:
    """Framework dtype ``name`` from the backend of ``like`` (e.g. ``\"bool\"``)."""
    return _backend_attr(like, name)


# --- NaN / finiteness utilities -------------------------------------------


@as_array_result
def is_nan(x: ShapedArray) -> ShapedArray:
    """Element-wise NaN test (Array API ``isnan``)."""
    return array_namespace(x).isnan(x)


@as_array_result
def is_finite(x: ShapedArray) -> ShapedArray:
    """Element-wise finite test (Array API ``isfinite``)."""
    return array_namespace(x).isfinite(x)


@as_array_result
def is_inf(x: ShapedArray) -> ShapedArray:
    """Element-wise infinity test (Array API ``isinf``)."""
    return array_namespace(x).isinf(x)


# Array API short names
isnan = is_nan
isfinite = is_finite
isinf = is_inf


@as_array_result
@promote(x="data", value="data")
def fill_nan(x: ShapedArray, value: Any = 0.0) -> ShapedArray:
    """Replace NaNs in ``x`` with ``value`` (broadcasts). Leaves ±inf unchanged."""
    xp = array_namespace(x, value)
    return xp.where(xp.isnan(x), value, x)


nan_fill = fill_nan  # alias


@promote(x="data", value="data")
def fill_nan_mask(x: ShapedArray, value: Any = 0.0) -> tuple[ShapedArray, ShapedArray]:
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
def nan_to_num(x: ShapedArray, *, nan: Any = 0.0, posinf: Any = None, neginf: Any = None) -> ShapedArray:
    """Replace NaN and ±inf (Array API ``nan_to_num``).

    Defaults: NaN → ``nan`` (0.0); ``posinf`` / ``neginf`` ``None`` → large
    finite values from the dtype's finfo (framework-dependent).
    """
    return array_namespace(x).nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)


@as_array_result
@promote(x="data", y="data")
def equal_nan(x: ShapedArray, y: ShapedArray) -> ShapedArray:
    """Element-wise equality treating NaN as equal to NaN.

    Returns a boolean array: ``(x == y) | (isnan(x) & isnan(y))``.
    Non-NaN values compare with ordinary ``==`` (so ``+inf == +inf``).
    """
    xp = array_namespace(x, y)
    both_nan = xp.logical_and(xp.isnan(x), xp.isnan(y))
    return xp.logical_or(x == y, both_nan)
