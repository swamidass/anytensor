import pytest
import numpy as np
import anytensor as at
from anytensor import backends
import itertools

npb = backends.NumpyBackend()

loaded_backends = {}

try:
  import jax
  b = backends.JaxBackend()
  loaded_backends[b.framework_name] = b
except ImportError:
  pass

try:
  import torch
  b = backends.TorchBackend()
  loaded_backends[b.framework_name] = b
except ImportError:
  pass

try:
  import tensorflow as tf
  b = backends.TensorflowBackend()
  loaded_backends[b.framework_name] = b
except ImportError:
  pass

segment_ops =  "segment_sum segment_max segment_min segment_normalize segment_softmax".split()

@pytest.mark.parametrize("op,ndims,backend", itertools.product(
    segment_ops, [1, 2, 3], loaded_backends
))
def test_segment_ops(backend, op, ndims):
  op = getattr(at, op)
  backend = loaded_backends[backend]

  d = [5,3,6][:ndims]
  numel = np.prod(d) #type: ignore

  x = np.arange(numel, dtype=np.float64) 
  x = x.reshape(*d)

  seg_id = np.array([0, 1, 1, 0, 1])
  num_segments = 2

  bx = backend.from_numpy(x)
  bseg_id =backend.from_numpy(seg_id)

  by = op(bx, bseg_id, num_segments)
  y = op(x, seg_id, num_segments)

  assert close(backend.to_numpy(by), y)
  assert type(bx) == type(by)

  assert len(x.shape) == ndims


@pytest.mark.parametrize("op,backend", itertools.product(
    "sum min max exp log".split(), loaded_backends
))
def test_unitary_ops(backend, op):
  op = getattr(at, op)
  backend = loaded_backends[backend]
  
  x = np.array([1, 2, 3, 4, 5], dtype=np.float64)

  bx = backend.from_numpy(x)

  by = op(bx)
  y = op(x)

  assert close(backend.to_numpy(by), y)
  assert type(bx) == type(by)


def close(x,y):
  x = np.asarray(x) # type: ignore
  y = np.asarray(y) # type: ignore
  return np.allclose(x, y) #type: ignore


