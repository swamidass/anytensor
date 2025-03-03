import torch as th
import anytensor as at
import pytest
from functools import partial


# Torchscript cannot understand the dynamic typing
@pytest.mark.xfail(reason="Not compatible with torchscript yet")
def test_jit():
  @th.jit.script
  def f(x, s):
    return at.segment_sum(x, s, 5)


# But tracing works just fine.
def test_trace():
  
  x = th.randn(10)
  s = th.randint(0, 5, (10,))

  @partial(th.jit.trace, example_inputs=(x, s))
  def f(x, s):
    return at.segment_sum(x, s, 5)

  # traced function works
  y = f(x, s)

  # traced function works with different sized inputs
  x = th.randn(15)
  s = th.randint(0, 5, (15,))
  y = f(x, s)