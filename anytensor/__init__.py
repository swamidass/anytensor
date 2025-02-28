from .backends import get_backend
from typing import TypeVar, TypeAlias, overload, Sequence, Optional
import numpy as np

#in port einops functions, which all work the same way as anytensor
from einops import einsum, pack, unpack, rearrange, reduce


def segment_sum(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'sum', sorted)

def segment_max(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'max', sorted)

def segment_min(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, 'min', sorted)

def segment_normalize(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    sum_x  = segment_sum(x, segment_ids, num_segments, sorted)
    sum_x = take(sum_x, segment_ids)
    return  x / sum_x

def segment_softmax(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    backend = get_backend(x)

    max_x = segment_max(x, segment_ids, num_segments, sorted)
    max_x = take(max_x, segment_ids)

    x = backend.exp(x - max_x)
    x = segment_normalize(x, segment_ids, num_segments, sorted)
    return x

def repeat(
  x, 
  repeats, 
  total_repeat_length : int
) :
    backend = get_backend(x)
    return backend.repeat(x, repeats, total_repeat_length)

def take(
  x,
  indices
):
    backend = get_backend(x)
    return backend.take(x, indices)


def exp(x):
    backend = get_backend(x)
    return backend.exp(x)


def log(x):
    backend = get_backend(x)
    return backend.log(x)


def sum(x, axes : Sequence[int] | None = None):
    backend = get_backend(x)
    return backend.reduce(x, 'sum', axes)

def min(x, axes : Sequence[int] | None = None):
    backend = get_backend(x)
    return backend.reduce(x, 'min', axes)


def max(x , axes : Sequence[int] | None = None):
    backend = get_backend(x)
    return backend.reduce(x, 'max', axes)
