from .backends import get_backend
from typing import TypeVar, TypeAlias, overload, Sequence, Optional

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

def cumsum(x):
    backend = get_backend(x)
    return backend.cumsum(x)

def shape(x):
    backend = get_backend(x)
    return backend.shape(x)