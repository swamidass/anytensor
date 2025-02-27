from .backends import get_backend
from typing import TypeVar, TypeAlias, overload, Sequence
from .typing import Int, Float, Tensor, IntTensor, FloatTensor
import numpy as np

T = TypeVar("T", Float, Int)

#in port einops functions, which all work the same way as anytensor
from einops import einsum, pack, unpack, rearrange, reduce


def segment_sum(
  x: np.ndarray[T], 
  segment_ids : IntTensor, 
  num_segments: int, 
  sorted : bool = False
) -> np.ndarray[T]:
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'sum', sorted)

def segment_max(
  x: np.ndarray[T], 
  segment_ids : IntTensor, 
  num_segments: int, 
  sorted : bool = False
) -> np.ndarray[T]:
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'max', sorted)

def segment_min(
  x: np.ndarray[T], 
  segment_ids : IntTensor, 
  num_segments: int, 
  sorted : bool = False
) -> np.ndarray[T]:
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'min', sorted)

def segment_normalize(
  x: np.ndarray[T], 
  segment_ids : IntTensor, 
  num_segments: int, 
  sorted : bool = False
) -> FloatTensor:
    sum_x : Tensor = segment_sum(x, segment_ids, num_segments, sorted)
    sum_x = take(sum_x, segment_ids)
    return  x / sum_x

def segment_softmax(
  x: Tensor, 
  segment_ids : IntTensor, 
  num_segments: int, 
  sorted : bool = False
) -> FloatTensor:
    backend = get_backend(x)

    max_x = segment_max(x, segment_ids, num_segments, sorted)
    max_x = take(max_x, segment_ids)

    x = backend.exp(x - max_x)
    x = segment_normalize(x, segment_ids, num_segments, sorted)
    return x

def repeat(
  x: np.ndarray[T], 
  repeats: IntTensor, 
  total_repeat_length : int
) -> np.ndarray[T]:
    backend = get_backend(x)
    return backend.repeat(x, repeats, total_repeat_length)

def take(
  x: np.ndarray[T],
  indices: IntTensor
) -> np.ndarray[T]:
    backend = get_backend(x)
    return backend.take(x, indices)


def exp(x : Tensor) -> FloatTensor:
    backend = get_backend(x)
    return backend.exp(x)


def log(x : Tensor) -> FloatTensor:
    backend = get_backend(x)
    return backend.log(x)


@overload
def sum(x: np.ndarray[T], axes : None = None) -> T:
    ...

@overload
def sum(x: np.ndarray[T], axes : Sequence[int]) -> np.ndarray[T]:
    ...

def sum(x, axes = None):
    backend = get_backend(x)
    return backend.reduce(x, 'sum', axes)


@overload
def min(x: np.ndarray[T], axes : None = None) -> T:
    ...

@overload
def min(x: np.ndarray[T], axes : Sequence[int]) -> np.ndarray[T]:
    ...

def min(x, axes = None):
    backend = get_backend(x)
    return backend.reduce(x, 'min', axes)


@overload
def max(x: np.ndarray[T], axes : None = None) -> T:
    ...

@overload
def max(x: np.ndarray[T], axes : Sequence[int]) -> np.ndarray[T]:
    ...

def max(x , axes = None):
    backend = get_backend(x)
    return backend.reduce(x, 'max', axes)
