from __future__ import annotations
"""
Backends in `anytensor` are organized to meet the following requirements
- backends are not imported unless those are actually needed, because
    - backends may not be installed
    - importing all available backends will drive to significant memory footprint
    - backends may be present but installed with errors (but never used),
      importing may drive to crashes
- backend should be either symbolic or imperative
    - this determines which methods (from_numpy/to_numpy or create_symbol/eval_symbol) should be defined
- if backend can't provide symbols for shape dimensions, UnknownSize objects are used

This code is adapted from code written by Alex Rogozhnikov in the `einops` library. The original code can be found at 
https://github.com/arogozhnikov/einops/blob/main/einops/_backends.py

Currently supported backends are:
- numpy
- torch
- tensorflow
- tensorflow.keras
- jax

A new backend can be added by creating a subclass of `AbstractBackend` and implementing the required methods. 
"""

from typing import Literal
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version as pkg_version

from .optional import module_if_loaded

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False


def _parse_version(v: str) -> tuple[int, ...]:
    parts = []
    for p in v.split(".")[:3]:
        digits = "".join(c for c in p if c.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _require_pkg_version(distribution: str, minimum: str, *, import_name: str | None = None) -> None:
    """Raise a clear error if an installed backend is below the supported floor."""
    name = import_name or distribution
    try:
        current = pkg_version(distribution)
    except PackageNotFoundError:
        # Module may be present without metadata; fall back to __version__.
        mod = module_if_loaded(name)
        current = getattr(mod, "__version__", None)
        if current is None:
            return
    if _parse_version(str(current)) < _parse_version(minimum):
        raise RuntimeError(
            f"{distribution} >={minimum} required for anytensor "
            f"(found {current}). Upgrade or omit this backend."
        )


def get_backend(tensor: Any) -> "AbstractBackend":
    """Return the backend for ``tensor`` (e.g. NumPy for ``numpy.ndarray``).

    Optional extras (JAX, Torch, TensorFlow) are used only if already imported;
    this never imports them. NumPy is a required dependency and is the fallback
    for ndarrays.
    """
    _type = type(tensor)
    _result = _type2backend.get(_type, None)
    if _result is not None:
        return _result

    for framework_name, backend in list(_loaded_backends.items()):
        if backend.is_appropriate_type(tensor):
            _type2backend[_type] = backend
            return backend

    # Find backend subclasses recursively
    backend_subclasses = []
    backends = AbstractBackend.__subclasses__()
    while backends:
        backend = backends.pop()
        backends += backend.__subclasses__()
        backend_subclasses.append(backend)

    for BackendSubclass in backend_subclasses:
        if _debug_importing:
            print("Testing for subclass of ", BackendSubclass)
        if BackendSubclass.framework_name not in _loaded_backends:
            # Construct only if the extra is already imported; never import it here.
            if module_if_loaded(BackendSubclass.framework_name) is not None:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend

    raise RuntimeError(f"Tensor type unknown to anytensor: {type(tensor)}")


from .semantics import empty_segment_identity


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes.

    After ``__init__``, concrete backends may cache framework dtypes / specials via
    :meth:`_install_numeric_attrs` for **internal** use (segment fills, etc.).
    Public callers should use module-level :data:`anytensor.inf` / :func:`anytensor.finfo`
    / :func:`anytensor.dtype` instead of backend objects.
    """

    framework_name: str

    def _install_numeric_attrs(self, xp) -> None:
        """Internal: cache framework constants/dtypes from ``xp``.

        Not part of the public AnyTensor API — backends are an implementation
        detail. Public surface: ``anytensor.inf`` / ``ninf`` / ``nan`` (Python
        floats) and ``finfo`` / ``iinfo`` / ``dtype`` resolved from an array.
        """
        import math

        self._info_xp = xp
        self.inf = float(getattr(xp, "inf", float("inf")))
        self.ninf = -self.inf
        self.nan = float(getattr(xp, "nan", float("nan")))
        self.pi = float(getattr(xp, "pi", math.pi))
        self.e = float(getattr(xp, "e", math.e))
        self.newaxis = getattr(xp, "newaxis", None)

        bool_dt = getattr(xp, "bool", None)
        if bool_dt is None:
            bool_dt = getattr(xp, "bool_", None)
        if bool_dt is None:
            raise AttributeError(f"{type(xp)!r} has no bool / bool_ dtype")
        self.bool = bool_dt

        for name in (
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            if hasattr(xp, name):
                setattr(self, name, getattr(xp, name))

    def finfo(self, dtype):
        """Floating dtype limits (``eps``, ``max``, …) — prefer over raw attrs."""
        return self._info_xp.finfo(dtype)

    def iinfo(self, dtype):
        """Integral dtype limits."""
        return self._info_xp.iinfo(dtype)

    def device(self, x):
        """return backend specific device on which the tensor is located"""
        return x.device

    def exp(self, x):
        """exponential function"""
        return x.exp()
    
    def log(self, x):
        """natural log function"""
        return x.log()
    
    def cumsum(self, x):
        """cumulative summation"""
        return x.cumsum()

    def repeat(self, x, repeats, total_repeat_length):
        """repeat follows semantics of jax.numpy.repeat: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.repeat.html"""
        raise NotImplementedError("framework doesn't support repeat")

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        """Reduce ``x`` by ``seg_ids`` along axis 0 (``sum`` / ``min`` / ``max``).

        Empty-segment identities are standardized in :mod:`anytensor.semantics`
        (float ±inf, integer dtype min/max, sum 0). See that module for the table.
        """
        raise NotImplementedError("backend does not support segment_sum")
    
    def take(self, x, indices):
        """take follows semantics of jax.numpy.take with axis=0: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html"""
        return x[indices]

    def split(self, x, indices_or_sections, axis: int = 0):
        """Split ``x`` along ``axis`` (NumPy ``split`` semantics).

        ``indices_or_sections`` is either an ``int`` (equal parts) or a sequence
        of cut indices along ``axis``. Returns a ``list`` of chunks (possibly
        empty).
        """
        raise NotImplementedError("backend does not support split")

    def is_appropriate_type(self, tensor):
        """helper method should recognize tensors it can handle"""
        raise NotImplementedError()

    def from_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def to_numpy(self, x):
        raise NotImplementedError("framework doesn't support imperative execution")

    def create_symbol(self, shape):
        raise NotImplementedError("framework doesn't support symbolic computations")

    def eval_symbol(self, symbol, symbol_value_pairs):
        # symbol-value pairs is list[tuple[symbol, value-tensor]]
        raise NotImplementedError("framework doesn't support symbolic computations")

    def arange(self, start, stop, device=None):
        raise NotImplementedError("framework doesn't implement arange")

    def shape(self, x):
        """shape should return a tuple with integers or "shape symbols" (which will evaluate to actual size)"""
        return x.shape

    def reshape(self, x, shape):
        return x.reshape(shape)

    def transpose(self, x, axes):
        return x.transpose(axes)
    
    def reduce(self, x, operation, axes):
        return getattr(x, operation)(axis=axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        raise NotImplementedError()

    def add_axis(self, x, new_position):
        raise NotImplementedError()

    def add_axes(self, x, n_axes, pos2len):
        repeats = [1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return self.tile(x, tuple(repeats))

    def tile(self, x, repeats):
        """repeats - same lengths as x.shape"""
        raise NotImplementedError()

    def concat(self, tensors, axis: int):
        """concatenates tensors along axis.
        Assume identical across tensors: devices, dtypes and shapes except selected axis."""
        raise NotImplementedError()

    def is_float_type(self, x):
        # some backends (torch) can't compute average for non-floating types.
        # Decided to drop average for all backends if type is not floating
        raise NotImplementedError()

    def __repr__(self):
        return "<anytensor backend for {}>".format(self.framework_name)


class UnknownSize:
    """pseudo-symbol for symbolic frameworks which do not provide symbols for shape elements"""

    def __floordiv__(self, other):
        return self

    def __eq__(self, other):
        return True  # we don't know actual size

    def __mul__(self, other):
        return self

    def __rmul__(self, other):
        return self

    def __hash__(self):
        return hash(None)


class NumpyBackend(AbstractBackend):
    framework_name = "numpy"

    def __init__(self):
        import numpy

        self.np = numpy
        self._install_numeric_attrs(numpy)

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return x

    def to_numpy(self, x):
        return x

    def exp(self, x):
        return self.np.exp(x)

    def log(self, x):
        return self.np.log(x)
    
    def cumsum(self, x):
        """cumulative summation"""
        return self.np.cumsum(x)

    def arange(self, start, stop, device=None):
        return self.np.arange(start, stop)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.np.stack(tensors)

    def tile(self, x, repeats):
        return self.np.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.np.concatenate(tensors, axis=axis)

    def split(self, x, indices_or_sections, axis: int = 0):
        return list(self.np.split(x, indices_or_sections, axis=axis))

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)

    def _type_info(self, x):
        t = x.dtype
        try:
            return self.np.iinfo(t)  # type: ignore
        except ValueError:
            return self.np.finfo(t)

    def _segment_identity(self, x, reduction: str):
        return empty_segment_identity(x.dtype, reduction, xp=self.np)

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        del sorted  # NumPy path is unsorted-safe (scatter via ufunc.at).
        s = self.np.full(
            (num_segments,) + x.shape[1:],
            self._segment_identity(x, reduction),
            dtype=x.dtype,
        )

        if reduction == "sum":
            agg = self.np.add
        elif reduction == "min":
            agg = self.np.minimum
        elif reduction == "max":
            agg = self.np.maximum
        else:
            raise ValueError(f"reduction type {reduction} not supported")

        # NaN / ±inf in ``x`` are in-scope; ``ufunc.at`` otherwise spams
        # RuntimeWarning: invalid value encountered in add/….
        with self.np.errstate(invalid="ignore", over="ignore"):
            agg.at(s, seg_ids, x)
        return s
    
class JaxBackend(NumpyBackend):
    framework_name = "jax"

    def __init__(self):
        _require_pkg_version("jax", "0.4.32")
        jax = module_if_loaded("jax", raises=True)
        super(JaxBackend, self).__init__()
        self.onp = self.np

        import jax.numpy

        self.np = jax.numpy
        self._jax = jax
        self._install_numeric_attrs(self.np)

    def is_appropriate_type(self, tensor):
        # Prefer jax.Array; also accept jnp arrays / tracers used as ndarray-like.
        jax = self._jax
        if hasattr(jax, "Array") and isinstance(tensor, jax.Array):
            return True
        return isinstance(tensor, self.np.ndarray)

    def from_numpy(self, x):
        return self.np.asarray(x)

    def to_numpy(self, x):
        return self.onp.asarray(x)

    def repeat(self, x, repeats, total_repeat_length):
        return self.np.repeat(x, repeats, total_repeat_length=total_repeat_length)

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        import jax.ops

        if reduction == "sum":
            f = jax.ops.segment_sum
        elif reduction == "min":
            f = jax.ops.segment_min
        elif reduction == "max":
            f = jax.ops.segment_max
        else:
            raise ValueError(f"reduction type {reduction} not supported")

        return f(x, seg_ids, num_segments, indices_are_sorted=sorted)

    def device(self, x):
        return x.devices()


class TorchBackend(AbstractBackend):
    framework_name = "torch"

    def __init__(self):
        _require_pkg_version("torch", "2.0")
        torch = module_if_loaded("torch", raises=True)

        self.torch = torch
        self._install_numeric_attrs(torch)

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        # ``sorted`` is currently a no-op on Torch (scatter path is unsorted-safe).
        del sorted
        from . import torchscript

        # Torch scatter needs a host int length; symbolic sizes belong under compile.
        n = int(num_segments)
        if reduction == "sum":
            return torchscript.segment_sum(x, seg_ids, n)
        if reduction == "min":
            return torchscript.segment_min(x, seg_ids, n)
        if reduction == "max":
            return torchscript.segment_max(x, seg_ids, n)
        raise ValueError(f"reduction type {reduction} not supported")

    def from_numpy(self, x):
        # Do not attach autograd — conversion helpers must be side-effect free.
        return self.torch.from_numpy(x)

    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def arange(self, start, stop, device=None):
        return self.torch.arange(start, stop, dtype=self.torch.int64, device=device)

    def reduce(self, x, operation, reduced_axes):
        if operation == "min":
            return x.amin(dim=reduced_axes)
        elif operation == "max":
            return x.amax(dim=reduced_axes)
        elif operation == "sum":
            return x.sum(dim=reduced_axes)
        elif operation == "mean":
            return x.mean(dim=reduced_axes)
        elif operation in ("any", "all", "prod"):
            # pytorch supports reducing only one operation at a time
            for i in list(sorted(reduced_axes))[::-1]:
                x = getattr(x, operation)(dim=i)
            return x
        else:
            raise NotImplementedError("Unknown reduction ", operation)

    def transpose(self, x, axes):
        return x.permute(axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.torch.stack(tensors)

    def add_axes(self, x, n_axes, pos2len):
        repeats = [-1] * n_axes
        for axis_position, axis_length in pos2len.items():
            x = self.add_axis(x, axis_position)
            repeats[axis_position] = axis_length
        return x.expand(repeats)

    def tile(self, x, repeats):
        return x.repeat(repeats)

    def concat(self, tensors, axis: int):
        return self.torch.cat(tensors, dim=axis)

    def split(self, x, indices_or_sections, axis: int = 0):
        # ``tensor_split`` matches NumPy cut-index / equal-section semantics.
        return list(self.torch.tensor_split(x, indices_or_sections, dim=axis))

    def add_axis(self, x, new_position):
        return self.torch.unsqueeze(x, new_position)

    def is_float_type(self, x):
        return x.dtype in [
            self.torch.float16,
            self.torch.float32,
            self.torch.float64,
            self.torch.bfloat16,
        ]

    def cumsum(self, x):
        return self.torch.cumsum(x, 0)


class TensorflowBackend(AbstractBackend):
    framework_name = "tensorflow"

    def __init__(self):
        _require_pkg_version("tensorflow", "2.10")
        tensorflow = module_if_loaded("tensorflow", raises=True)
        import tensorflow.experimental.numpy as tnp

        self.tf = tensorflow
        # Ordinary ops use tnp (via anytensor.namespace); attrs/finfo come from there.
        self._install_numeric_attrs(tnp)
        self.bool = tensorflow.bool
        # Prefer TF dtypes when present on the root module.
        for name in (
            "float16",
            "float32",
            "float64",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
        ):
            if hasattr(tensorflow, name):
                setattr(self, name, getattr(tensorflow, name))

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, (self.tf.Tensor, self.tf.Variable))

    def take(self, x, indices):
        return self.tf.gather(x, indices, axis=0)

    def from_numpy(self, x):
        assert self.tf.executing_eagerly()
        return self.tf.convert_to_tensor(x)

    def to_numpy(self, x):
        assert self.tf.executing_eagerly()
        return x.numpy()

    def cumsum(self, x):
        return self.tf.cumsum(x)

    def arange(self, start, stop, device=None):
        with self.tf.device(device) if device else nullcontext():
            return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return tuple(UnknownSize() if d is None else int(d) for d in x.shape)
        else:
            static_shape = x.shape.as_list()
            tf_shape = self.tf.shape(x)
            # use the static shape where known, otherwise use the TF shape components
            shape = tuple([s or tf_shape[dim] for dim, s in enumerate(static_shape)])  # type: ignore
            try:
                hash(shape)
                return shape
            except BaseException:
                # unhashable symbols in shape. Wrap tuple to be hashable.
                return HashableTuple(shape)

    def exp(self, x):
        return self.tf.exp(x)

    def log(self, x):
        return self.tf.math.log(x)

    def reduce(self, x, operation, axes):
        return getattr(self.tf, "reduce_" + operation)(x, axis=axes)

    def reshape(self, x, shape):
        return self.tf.reshape(x, shape)

    def transpose(self, x, axes):
        return self.tf.transpose(x, axes)

    def stack_on_zeroth_dimension(self, tensors: list):
        return self.tf.stack(tensors)

    def tile(self, x, repeats):
        return self.tf.tile(x, repeats)

    def concat(self, tensors, axis: int):
        return self.tf.concat(tensors, axis=axis)

    def split(self, x, indices_or_sections, axis: int = 0):
        # ``tf.split`` takes section *sizes*; convert NumPy-style cut indices.
        axis = int(axis)
        if isinstance(indices_or_sections, int):
            return list(self.tf.split(x, indices_or_sections, axis=axis))
        length = int(x.shape[axis])
        cuts = [0, *[int(i) for i in indices_or_sections], length]
        sizes = [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]
        return list(self.tf.split(x, sizes, axis=axis))

    def add_axis(self, x, new_position):
        return self.tf.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def finfo(self, dtype):
        """Floating dtype limits — accept TF dtypes via NumPy bridge."""
        import numpy as np

        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        return np.finfo(np_dtype)

    def iinfo(self, dtype):
        """Integral dtype limits — accept TF dtypes via NumPy bridge."""
        import numpy as np

        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        return np.iinfo(np_dtype)

    def segment_reduce(
        self,
        x,
        seg_ids,
        num_segments,
        reduction: Literal["sum"] | Literal["min"] | Literal["max"] = "sum",
        sorted: bool = False,
    ):
        tf = self.tf
        if reduction not in ("sum", "min", "max"):
            raise ValueError(f"reduction type {reduction} not supported")
        # Sum: TF segment ops are fine (empty → 0).
        if reduction == "sum":
            if sorted:
                return tf.math.segment_sum(x, seg_ids)
            return tf.math.unsorted_segment_sum(x, seg_ids, num_segments)

        # Min/max: TF unsorted_segment_{min,max} map ±inf to finfo limits and use
        # those as empty fills. Scatter from an AnyTensor identity preserves
        # ±inf. Scatter ignores NaN updates, so OR-in segment NaN afterward.
        del sorted
        # Normalize TF dtypes for the semantics helper (expects NumPy-ish dtypes).
        import numpy as np

        np_dtype = getattr(x.dtype, "as_numpy_dtype", x.dtype)
        fill = empty_segment_identity(np_dtype, reduction, xp=np)
        shape = (num_segments,) + tuple(s for s in x.shape[1:])
        init = tf.fill(shape, value=tf.cast(fill, x.dtype))
        indices = tf.expand_dims(tf.cast(seg_ids, tf.int64), -1)
        if reduction == "min":
            out = tf.tensor_scatter_nd_min(init, indices, x)
        else:
            out = tf.tensor_scatter_nd_max(init, indices, x)

        if np.issubdtype(np_dtype, np.floating):
            # Any NaN in a segment → NaN (scatter leaves the ±inf identity).
            nan_as_one = tf.cast(tf.math.is_nan(x), x.dtype)
            has_nan = tf.math.unsorted_segment_max(nan_as_one, seg_ids, num_segments)
            # Broadcast has_nan over trailing dims of x.
            while len(has_nan.shape) < len(out.shape):
                has_nan = tf.expand_dims(has_nan, -1)
            out = tf.where(has_nan > 0, tf.cast(float("nan"), x.dtype), out)
        return out


class HashableTuple:
    """Overcomes non-hashability of symbolic elements"""

    def __init__(self, elements: tuple):
        self.elements = elements

    def __iter__(self):
        for x in self.elements:
            yield x

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    # default equality and hash is used (True only with itself, hash taken of id)

