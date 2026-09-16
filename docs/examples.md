# Worked examples

These fenced blocks are executed by pytest (via
[Sybil](https://sybil.readthedocs.io/)) so they stay honest. Prefer editing
here when changing the story; the [home page](index.md) summarizes.

## Portable neighbor attention (GAT-style)

Same helper as on the home page: leaky scores, per-destination softmax, weighted
sum. Inspired by [Graph Attention Networks](https://arxiv.org/abs/1710.10903)
and PyTorch Geometric’s index-based
[`softmax`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_softmax.py).

```python
import anytensor as at
import numpy as np


def neighbor_attention(messages, scores, dst_index, num_nodes: int):
    alpha = at.where(scores > 0, scores, scores * 0.2)
    alpha = at.segment_softmax(alpha, dst_index, num_nodes)
    weighted = messages * alpha[:, None]
    return at.segment_sum(weighted, dst_index, num_nodes)


messages = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]],
    dtype=np.float32,
)
scores = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
dst = np.array([0, 0, 1, 2], dtype=np.int64)
num_nodes = 3

out_np = neighbor_attention(messages, scores, dst, num_nodes)
assert out_np.shape == (3, 2)
np.testing.assert_allclose(out_np, [[0.5, 0.5], [1.0, 1.0], [2.0, 0.0]])
```

### Same function on other backends

```python
torch = pytest.importorskip("torch")
out_t = neighbor_attention(
    torch.as_tensor(messages),
    torch.as_tensor(scores),
    torch.as_tensor(dst),
    num_nodes,
)
assert tuple(out_t.shape) == (3, 2)
np.testing.assert_allclose(out_t.detach().cpu().numpy(), out_np)
```

```python
jnp = pytest.importorskip("jax.numpy")
out_j = neighbor_attention(
    jnp.asarray(messages),
    jnp.asarray(scores),
    jnp.asarray(dst),
    num_nodes,
)
np.testing.assert_allclose(np.asarray(out_j), out_np)
```

```python
tf = pytest.importorskip("tensorflow")
out_f = neighbor_attention(
    tf.constant(messages),
    tf.constant(scores),
    tf.constant(dst),
    num_nodes,
)
np.testing.assert_allclose(np.asarray(out_f), out_np)
```

## Compile / graph / script (the tricky part)

Eager portability is the easy win. Compilers each need a small amount of care:
**static shape sizes**, which ops are scriptable, and how `num_nodes` is passed.

### `jax.jit` — mark `num_nodes` static

JAX will not accept a traced `num_segments`. Pass it as a static argument:

```python
jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

jitted = jax.jit(neighbor_attention, static_argnames=("num_nodes",))
out = jitted(
    jnp.asarray(messages),
    jnp.asarray(scores),
    jnp.asarray(dst),
    num_nodes,
)
np.testing.assert_allclose(np.asarray(out), out_np)
```

### `torch.compile` — full helper works

Dynamo can follow AnyTensor dispatch for this example (graph breaks may still
occur on some ops; numerics match eager):

```python
torch = pytest.importorskip("torch")

compiled = torch.compile(neighbor_attention)
out = compiled(
    torch.as_tensor(messages),
    torch.as_tensor(scores),
    torch.as_tensor(dst),
    num_nodes,
)
np.testing.assert_allclose(out.detach().cpu().numpy(), out_np)
```

### `tf.function` / XLA — keep `num_nodes` a Python `int`

```python
tf = pytest.importorskip("tensorflow")


@tf.function
def tf_neighbor(m, s, d, n):
    return neighbor_attention(m, s, d, n)


out = tf_neighbor(
    tf.constant(messages),
    tf.constant(scores),
    tf.constant(dst),
    num_nodes,
)
np.testing.assert_allclose(np.asarray(out), out_np)


@tf.function(jit_compile=True)
def tf_neighbor_xla(m, s, d, n):
    return neighbor_attention(m, s, d, n)


out_xla = tf_neighbor_xla(
    tf.constant(messages),
    tf.constant(scores),
    tf.constant(dst),
    num_nodes,
)
np.testing.assert_allclose(np.asarray(out_xla), out_np)
```

### `torch.jit.script` — only `segment_sum` / `min` / `max`

`torch.jit.script` cannot follow Array-API dispatch. After
`anytensor.enable_torchscript()`, public **`segment_sum` / `segment_min` /
`segment_max`** divert under `torch.jit.is_scripting()` to pure-Torch kernels.
Eager multi-backend behavior is unchanged.

`segment_softmax` (and thus full `neighbor_attention`) is **not** script-safe
today — use `torch.compile` / `trace`, or script only the pool step:

```python
torch = pytest.importorskip("torch")
import anytensor as at
import sys
import types

# TorchScript needs a real __module__ (Sybil exec namespaces are anonymous).
_mod = sys.modules.setdefault(
    "anytensor_doc_examples", types.ModuleType("anytensor_doc_examples")
)

at.enable_torchscript()


def library_pool(x, seg, n: int):
    """Library code — no TorchScript knowledge required."""
    return at.segment_sum(x, seg, n)


def user_pool(x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
    return library_pool(x, seg, 3)


library_pool.__module__ = _mod.__name__
user_pool.__module__ = _mod.__name__
user_pool = torch.jit.script(user_pool)

x = torch.as_tensor(messages)
seg = torch.as_tensor(dst)
y = user_pool(x, seg)
assert tuple(y.shape) == (3, 2)
```

### `torch.jit.trace` — close over Python `int` sizes

Trace only accepts tensors (and nested tensor containers). Close over
`num_nodes` instead of passing it as an example input:

```python
torch = pytest.importorskip("torch")
import sys
import types

_mod = sys.modules.setdefault(
    "anytensor_doc_examples", types.ModuleType("anytensor_doc_examples")
)


def wrapped(m, s, d):
    return neighbor_attention(m, s, d, 3)


wrapped.__module__ = _mod.__name__
traced = torch.jit.trace(
    wrapped,
    (
        torch.as_tensor(messages),
        torch.as_tensor(scores),
        torch.as_tensor(dst),
    ),
)
y = traced(
    torch.as_tensor(messages),
    torch.as_tensor(scores),
    torch.as_tensor(dst),
)
np.testing.assert_allclose(y.detach().cpu().numpy(), out_np)
```

## Gotchas checklist

| Path | What to remember |
|------|------------------|
| `jax.jit` | `static_argnames=("num_nodes",)` (or `static_argnums`) for shape-sizes |
| `tf.function` | Pass Python `int` for `num_segments` / `num_nodes`; prefer `shape(x)` over raw `.shape` under polymorphic graphs |
| `torch.compile` | Works for many AnyTensor call graphs; fuzz covers parity |
| `torch.jit.script` | Call `enable_torchscript()`; only `segment_sum` / `min` / `max` divert |
| `torch.jit.trace` | Example inputs must be tensors — close over Python ints |

See also [Usage → TorchScript](usage.md#torchscript) and
[Surprising differences](semantics.md).
