from .backends import get_backend
from .core import take

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


def segment_mean(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    backend = get_backend(x)
    sum =  backend.segment_reduce(x, segment_ids, num_segments, 'sum', sorted)

    

def segment_normalize(
  x, 
  segment_ids, 
  num_segments: int, 
  sorted : bool = False
):
    sum_x  = segment_sum(x, segment_ids, num_segments, sorted)
    sum_x = take(sum_x, segment_ids)
    return  x / sum_x
