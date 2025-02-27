from .backends import get_backend

#in port einops functions, which all work the same way as anytensor
from einops import einsum, pack, unpack, rearrange, reduce

def segment_sum(x, segment_ids, num_segments):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, reduction='sum')

def segment_max(x, segment_ids, num_segments):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, reduction='max')

def segment_min(x, segment_ids, num_segments):
    backend = get_backend(x)
    return backend.segment_reduce(x, segment_ids, num_segments, reduction='min')

def segment_normalize(x, segment_ids, num_segments):
    backend = get_backend(x)

    sum_x = backend.segment_sum(x, segment_ids, num_segments)
    sum_x = backend.take(sum_x, segment_ids)

    return  x / sum_x

def repeat(x, repeats):
    backend = get_backend(x)
    return backend.repeat(x, repeats)


def take(x, indices):
    backend = get_backend(x)
    return backend.take(x, indices)
