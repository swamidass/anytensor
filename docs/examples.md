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

### `torch.compile` — portable helper (`fullgraph=False`)

Dynamo can run the portable helper if you allow graph breaks
(`fullgraph=False`, the default): breaks land in `@promote` /
`array-api-compat`, those pieces run eager, and numerics still match. Reset
Dynamo first so a prior compile in the same process does not leave state
behind. Sybil setup imports ``torch`` and seeds ``messages_t`` / ``scores_t`` /
``dst_t``. Docs use ``backend="aot_eager"`` so this stays reliable after the
fuzz suite (default inductor codegen flakes in-process); apps omit ``backend``:

```python
import os

pytest.importorskip("torch")
# Dynamo/triton has SIGSEGV'd on GitHub-hosted runners; local/docs still run.
if os.environ.get("CI"):
    pytest.skip("torch.compile disabled on CI runners (dynamo/triton)")
torch._dynamo.reset()
compiled = torch.compile(
    neighbor_attention, fullgraph=False, backend="aot_eager"
)
out = compiled(messages_t, scores_t, dst_t, num_nodes)
np.testing.assert_allclose(out.detach().cpu().numpy(), out_np)
```

### `torch.compile(..., fullgraph=True)` — needs a Torch-only body

A single fused graph requires `fullgraph=True`. That fails on the portable
helper above (Dynamo graph-breaks on `@promote` / `inspect.Signature.bind` and
`array-api-compat` lookup). To get `fullgraph=True`, specialize: rewrite the
body with Torch ops only (`torch.where`, `scatter_reduce` / `scatter_add`,
etc.) and compile that function — same numerics, no AnyTensor dispatch in the
traced region. That specialization is Torch-only; it is not what the portable
helper is for.

### `torch.export` — wrap in `nn.Module.forward`

PyTorch’s replacement for deprecated `torch.jit.script` / `trace` (alongside
`torch.compile`). `torch.export.export` expects an **`nn.Module`**, not a bare
function — put the portable helper in `forward`. That path works with AnyTensor
dispatch (unlike `fullgraph=True`):

```python
import os

pytest.importorskip("torch")
import torch.nn as nn

if os.environ.get("CI"):
    pytest.skip("torch.export disabled on CI runners (dynamo/triton SIGSEGV)")


class NeighborAttention(nn.Module):
    def forward(self, messages, scores, dst_index, num_nodes: int):
        return neighbor_attention(messages, scores, dst_index, num_nodes)


exported = torch.export.export(
    NeighborAttention(),
    (messages_t, scores_t, dst_t, num_nodes),
)
out = exported.module()(messages_t, scores_t, dst_t, num_nodes)
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

## Gotchas checklist

| Path | What to remember |
|------|------------------|
| `jax.jit` | `static_argnames=("num_nodes",)` (or `static_argnums`) for shape-sizes |
| `tf.function` | Pass Python `int` for `num_segments` / `num_nodes`; prefer `shape(x)` over raw `.shape` under polymorphic graphs |
| `torch.compile` | Prefer this over deprecated `torch.jit.script` / `trace`. Portable helpers need `fullgraph=False`; docs use `backend="aot_eager"` for suite stability |
| `torch.export` | Wrap the helper in `nn.Module.forward` (bare functions are rejected) |
| ONNX | `at.shape(x)[0]` for lengths; Lightning/Torch dynamo export; Keras `tf.function`+tf2onnx; Flax rebind — [ONNX](onnx/index.md) |

See also [Usage](usage.md) and [Surprising differences](semantics.md).
GraphsTuple / GraphNetwork recipes: [Jraph examples](jraph/examples.md).
Nest helpers: [Tree examples](tree/examples.md).

