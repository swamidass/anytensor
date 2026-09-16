"""Array-namespace resolution (array-api-compat + TensorFlow shim).

``array-api-compat`` does not yet expose a TensorFlow backend in the installed
release, so EagerTensors are routed through ``tf.experimental.numpy`` with a
thin Array-API compatibility layer.
"""

from __future__ import annotations

from typing import Any, Optional


_SCALAR_TYPES = (bool, int, float, complex)


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


def _is_tensorflow_tensor(x: Any) -> bool:
    try:
        import tensorflow as tf

        return isinstance(x, (tf.Tensor, tf.Variable))
    except ImportError:
        return False


class _TensorflowNumpyNamespace:
    """``tf.experimental.numpy`` plus Array API helpers array-api-compat would normally add."""

    def __init__(self):
        import tensorflow as tf
        import tensorflow.experimental.numpy as tnp

        self._tf = tf
        self._tnp = tnp

    def __getattr__(self, name: str):
        return getattr(self._tnp, name)

    @property
    def bool(self):
        return self._tf.bool

    def asarray(self, x, dtype=None, *, copy: Optional[bool] = None, device=None):
        del device  # TF device placement is handled elsewhere / via context.
        kwargs = {}
        if dtype is not None:
            kwargs["dtype"] = dtype
        out = self._tf.convert_to_tensor(x, **kwargs)
        if copy:
            out = self._tf.identity(out)
        return out

    def astype(self, x, dtype, copy=True):
        out = self._tf.cast(x, dtype)
        if copy is False:
            return out
        return out

    def concat(self, arrays, axis=0):
        return self._tnp.concatenate(list(arrays), axis=axis)

    def permute_dims(self, x, axes=None):
        if axes is None:
            rank = len(x.shape)
            axes = tuple(range(rank - 1, -1, -1))
        return self._tnp.transpose(x, axes)

    def cumulative_sum(self, x, *, axis=0, dtype=None, include_initial=False):
        if include_initial:
            raise NotImplementedError("include_initial not supported on TensorFlow shim")
        out = self._tnp.cumsum(x, axis=axis)
        if dtype is not None:
            out = self.astype(out, dtype)
        return out

    def clip(self, x, /, min=None, max=None):
        # NumPy / tnp use a_min/a_max; Array API uses min/max.
        return self._tnp.clip(x, a_min=min, a_max=max)

    def isdtype(self, dtype, kind) -> bool:
        import numpy as np

        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        if isinstance(kind, tuple):
            return any(self.isdtype(dtype, k) for k in kind)
        if kind == "bool":
            return np.issubdtype(np_dtype, np.bool_)
        if kind in ("integral", "integer"):
            return np.issubdtype(np_dtype, np.integer)
        if kind in ("real floating", "floating"):
            return np.issubdtype(np_dtype, np.floating)
        if kind == "complex floating":
            return np.issubdtype(np_dtype, np.complexfloating)
        if kind == "numeric":
            return np.issubdtype(np_dtype, np.number)
        raise ValueError(f"unknown isdtype kind: {kind!r}")

    def nan_to_num(self, x, *, nan=0.0, posinf=None, neginf=None):
        tnp = self._tnp
        tf = self._tf
        dtype = x.dtype
        if posinf is None:
            posinf = float(tnp.finfo(dtype).max)
        if neginf is None:
            neginf = float(tnp.finfo(dtype).min)
        x = tnp.where(tnp.isnan(x), tf.cast(nan, dtype), x)
        pos = tnp.logical_and(tnp.isinf(x), x > 0)
        neg = tnp.logical_and(tnp.isinf(x), x < 0)
        x = tnp.where(pos, tf.cast(posinf, dtype), x)
        x = tnp.where(neg, tf.cast(neginf, dtype), x)
        return x

    def repeat(self, x, repeats, axis=None):
        # ``tnp.repeat`` converts repeats via NumPy and breaks under ``tf.function``.
        # ``tf.repeat`` accepts Python ints and symbolic tensor repeats.
        if axis is None:
            return self._tf.repeat(x, repeats)
        return self._tf.repeat(x, repeats, axis=axis)

    def arange(self, start, /, stop=None, step=1, dtype=None, **kwargs):
        del kwargs
        if stop is None:
            start, stop = 0, start
        # ``tf.range`` accepts symbolic ``stop`` (unlike ``tnp.arange``).
        if dtype is None:
            return self._tf.range(start, stop, step)
        return self._tf.range(start, stop, step, dtype=dtype)

    def zeros(self, shape, dtype=None):
        return self._tf.zeros(shape, dtype=dtype)

    def ones(self, shape, dtype=None):
        return self._tf.ones(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None):
        if dtype is None:
            return self._tf.fill(shape, fill_value)
        return self._tf.cast(self._tf.fill(shape, fill_value), dtype)


_TF_NS: _TensorflowNumpyNamespace | None = None


def _tensorflow_namespace() -> _TensorflowNumpyNamespace:
    global _TF_NS
    if _TF_NS is None:
        _TF_NS = _TensorflowNumpyNamespace()
    return _TF_NS


def array_namespace(*arrays: Any) -> Any:
    """Like ``array_api_compat.array_namespace``, with TensorFlow EagerTensor support."""
    from array_api_compat import array_namespace as aac_namespace

    arrs = [a for a in arrays if a is not None and not _is_scalar(a)]
    tf_arrs = [a for a in arrs if _is_tensorflow_tensor(a)]
    if tf_arrs:
        others = [a for a in arrs if not _is_tensorflow_tensor(a) and not _is_numpy_ndarray(a)]
        if others:
            raise TypeError(
                "Cannot mix TensorFlow tensors with other non-NumPy frameworks "
                f"in one op (got {type(others[0])})"
            )
        return _tensorflow_namespace()
    if not arrs:
        import array_api_compat.numpy as xp

        return xp
    return aac_namespace(*arrs)
