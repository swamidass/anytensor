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
from typing import Literal

_loaded_backends: dict = {}
_type2backend: dict = {}
_debug_importing = False


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

    raise RuntimeError("Tensor type unknown to einops {}".format(type(tensor)))


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
    
    def repeat(self, x, repeats, total_repeat_length):
        """repeat follows semantics of jax.numpy.repeat: https://docs.jax.dev/en/latest/_autosummary/jax.numpy.repeat.html"""
        raise NotImplementedError("framework doesn't support repeat")

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted: bool = False):
        """segment_reduce with reduce in {sum, min, max}.
        Follows semantics of jax.ops.segment_sum: https://docs.jax.dev/en/latest/_autosummary/jax.ops.segment_sum.html"""
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
        super(JaxBackend, self).__init__()
        self.onp = self.np

        import jax.numpy

        self.np = jax.numpy

    def from_numpy(self, x):
        return self.np.asarray(x)

    def to_numpy(self, x):
        return self.onp.asarray(x)

    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted : bool = False):
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
        import torch

        self.torch = torch

    def is_appropriate_type(self, tensor):
        return isinstance(tensor, self.torch.Tensor)
    
    def segment_reduce(self, x, seg_ids, num_segments, reduction, sorted : bool = False):
        shape = (num_segments,) + x.shape[1:]
        ndim = len(self.shape(x))

        dim = 0 
        
        for i in range(1, ndim):
            seg_ids = seg_ids.unsqueeze(-1)
        seg_ids = seg_ids.expand_as(x)
        
        if reduction == "sum":
            out = self.torch.zeros(shape, dtype=x.dtype, device=x.device)
            out = out.scatter_add(dim, seg_ids, x)
            return out
        elif reduction == "min":
            out = self.torch.full(shape, float("inf"), dtype=x.dtype, device=x.device)
            out = out.scatter_reduce(dim, seg_ids, x, reduce="amin", include_self=True)
            return out
        elif reduction == "max":
            out = self.torch.full(shape, float("-inf"), dtype=x.dtype, device=x.device)
            out = out.scatter_reduce(dim, seg_ids, x, reduce="amax", include_self=True)
            return out
        else:
            raise ValueError(f"reduction type {reduction} not supported")

    def from_numpy(self, x):
        variable = self.torch.from_numpy(x)
        if self.is_float_type(variable):
            # attach grad only to floating types
            variable.requires_grad = True
        return variable

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
        return x.dtype in [self.torch.float16, self.torch.float32, self.torch.float64, self.torch.bfloat16]




class TensorflowBackend(AbstractBackend):
    framework_name = "tensorflow"

    def __init__(self):
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

    def arange(self, start, stop, device=None):
        return self.tf.range(start, stop)

    def shape(self, x):
        if self.tf.executing_eagerly():
            return tuple(UnknownSize() if d is None else int(d) for d in x.shape)
        else:
            static_shape = x.shape.as_list()
            tf_shape = self.tf.shape(x)
            # use the static shape where known, otherwise use the TF shape components
            shape = tuple([s or tf_shape[dim] for dim, s in enumerate(static_shape)]) # type: ignore
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

    def segment_reduce(self, x, seg_ids, num_segments, reduction: Literal['sum'] | Literal['min'] | Literal['max'] = 'sum', sorted : bool = False):
        tf = self.tf
        op_name = f"segment_{reduction}"
        if not sorted: op_name = "unsorted_" + op_name
        try:
          op = getattr(tf.math, op_name)
        except:
          raise ValueError(f"reduction type {reduction} not supported")
        
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

