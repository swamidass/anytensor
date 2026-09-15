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

import sys
from typing import Literal, Tuple
from contextlib import nullcontext
from importlib.metadata import PackageNotFoundError, version as pkg_version

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False


def _parse_version(v: str) -> Tuple[int, ...]:
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
        mod = sys.modules.get(name)
        current = getattr(mod, "__version__", None)
        if current is None:
            return
    if _parse_version(str(current)) < _parse_version(minimum):
        raise RuntimeError(
            f"{distribution} >={minimum} required for anytensor "
            f"(found {current}). Upgrade or omit this backend."
        )


def get_backend(tensor) -> "AbstractBackend":
    """
    Takes a correct backend (e.g. numpy backend if tensor is numpy.ndarray) for a tensor.
    If needed, imports package and creates backend
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
            # check that module was already imported. Otherwise it can't be imported
            if BackendSubclass.framework_name in sys.modules:
                if _debug_importing:
                    print("Imported backend for ", BackendSubclass.framework_name)
                backend = BackendSubclass()
                _loaded_backends[backend.framework_name] = backend
                if backend.is_appropriate_type(tensor):
                    _type2backend[_type] = backend
                    return backend

    raise RuntimeError(f"Tensor type unknown to anytensor: {type(tensor)}")


class AbstractBackend:
    """Base backend class, major part of methods are only for debugging purposes."""

    framework_name: str

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
        """segment_reduce with reduce in {sum, min, max}.

        Follows semantics of jax.ops.segment_sum:
        https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html

        Index dtypes are backend-local: ``seg_ids`` must be integral, but width
        differs (Torch scatter wants int64; JAX/TF often use int32). Callers
        should not assume NumPy int64 ids stay int64 after upcast.
        """
        raise NotImplementedError("backend does not support segment_sum")
    
    def take(self, x, indices):
        """take follows semantics of jax.numpy.take with axis=0: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.take.html"""
        return x[indices]

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

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

    def add_axis(self, x, new_position):
        return self.np.expand_dims(x, new_position)

    def _type_info(self, x):
        t = x.dtype
        try:
            return self.np.iinfo(t) # type: ignore
        except ValueError:
            return self.np.finfo(t)

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted : bool = False):
      
        s = self.np.zeros((num_segments,) + x.shape[1:], dtype=x.dtype)
        
        if reduction == "sum":
            agg = self.np.add
        elif reduction == "min":
            d = self._type_info(x).max  
            s = s + d
            agg = self.np.minimum
        elif reduction == "max":
            d = self._type_info(x).min 
            s = s + d
            agg = self.np.maximum       
        else:
            raise ValueError(f"reduction type {reduction} not supported")
        
        agg.at(s, seg_ids, x)
        return s
    
class JaxBackend(NumpyBackend):
    framework_name = "jax"

    def __init__(self):
        _require_pkg_version("jax", "0.4.32")
        super(JaxBackend, self).__init__()
        self.onp = self.np

        import jax.numpy

        self.np = jax.numpy
        self._jax = __import__("jax")

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
        import torch

        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)

    def _scatter_fill_value(self, x, reduction: str):
        """Identity for empty segments: 0 / dtype max / dtype min (not float inf on ints)."""
        if reduction == "sum":
            return 0
        is_float = x.dtype.is_floating_point
        if reduction == "min":
            if is_float:
                return float("inf")
            return self.torch.iinfo(x.dtype).max
        if reduction == "max":
            if is_float:
                return float("-inf")
            return self.torch.iinfo(x.dtype).min
        raise ValueError(f"reduction type {reduction} not supported")

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        # ``sorted`` is currently a no-op on Torch (scatter path is unsorted-safe).
        del sorted
        shape = (num_segments,) + x.shape[1:]
        ndim = len(self.shape(x))
        dim = 0

        seg_ids = seg_ids.to(dtype=self.torch.int64)
        for _ in range(1, ndim):
            seg_ids = seg_ids.unsqueeze(-1)
        seg_ids = seg_ids.expand_as(x)

        if reduction == "sum":
            out = self.torch.zeros(shape, dtype=x.dtype, device=x.device)
            return out.scatter_add(dim, seg_ids, x)
        if reduction in ("min", "max"):
            fill = self._scatter_fill_value(x, reduction)
            out = self.torch.full(shape, fill, dtype=x.dtype, device=x.device)
            reduce = "amin" if reduction == "min" else "amax"
            return out.scatter_reduce(dim, seg_ids, x, reduce=reduce, include_self=True)
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
        import tensorflow

        self.tf = tensorflow

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

    def add_axis(self, x, new_position):
        return self.tf.expand_dims(x, new_position)

    def is_float_type(self, x):
        return x.dtype in ("float16", "float32", "float64", "float128", "bfloat16")

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
        # Sorted ops are ``segment_{sum,min,max}(data, ids)`` — no num_segments.
        # Unsorted ops take ``(data, ids, num_segments)``.
        if sorted:
            op = getattr(tf.math, f"segment_{reduction}")
            return op(x, seg_ids)
        op = getattr(tf.math, f"unsorted_segment_{reduction}")
        return op(x, seg_ids, num_segments)


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

