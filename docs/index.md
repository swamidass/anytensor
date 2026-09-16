# AnyTensor

Write a tensor helper **once**. Run it on **NumPy**, **JAX**, **PyTorch**, or
**TensorFlow** — whichever array the caller already has.

Ordinary math (`sum`, `exp`, `reshape`, …) goes through the
[Python Array API](https://data-apis.org/array-api/latest/) via
[`array-api-compat`](https://github.com/data-apis/array-api-compat). Segment /
GNN reductions use thin input-adaptive backends. The public surface is one
namespace: `import anytensor as at`.

---

## The problem

Scientific and ML code rarely lives in one framework forever.

- A method starts in **NumPy** for prototyping, then moves to **JAX** for `jit`,
  or **PyTorch** for training, or **TensorFlow** because a collaborator’s stack
  demands it.
- Library authors ship a useful op — and then maintain **three copies** (or
  abandon everyone who picked a different stack).
- Graph / sparse reductions are especially painful: “softmax over neighbors,”
  “sum messages per node,” “mean pool by batch” are **segment** ops. Each
  ecosystem has its own scatter/segment API, dtype quirks, and empty-slot
  conventions.

The result is either framework lock-in or a thicket of `if torch: … elif jax: …`
branches that rot.

**AnyTensor’s bet:** for the portable layer — math plus segment primitives —
you write against array *behavior*, not against a vendor. Call sites keep their
native tensors; AnyTensor dispatches.

---

## Motivation

1. **One implementation for library code.** A helper in your package can accept
   whatever the user trains with. You do not fork the algorithm per backend.
2. **Segment ops as first-class citizens.** GNN-style reductions
   (`segment_sum`, `segment_softmax`, …) are not an afterthought; they match
   the JAX / TF “required `num_segments`” discipline and document empty-slot /
   NaN semantics.
3. **NumPy as host data, not a demotion target.** Mixing a Torch tensor with a
   NumPy buffer upcasts *onto* Torch (by reference when possible). Framework
   tensors stay on their device/dtype world.
4. **Honest about differences.** When TF XLA and NumPy disagree on
   `inf * tiny`, we [document it](semantics.md) instead of papering over it.

---

## Case study: neighbor softmax from Graph Attention Networks

[Graph Attention Networks](https://arxiv.org/abs/1710.10903) (Veličković et al.,
ICLR 2018) normalize attention **per destination node** over incoming edges.
The widely used [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
stack implements that as a sparse softmax over an edge `index`
([`torch_geometric.utils.softmax`](https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/utils/_softmax.py)) —
Torch tensors only, with `scatter` / `segment` under the hood.

Stripped to the index-based path, the idea is:

```python
# Conceptual PyG / torch-scatter style (Torch only)
src_max = scatter_max(src, index, dim_size=N)
out = (src - src_max[index]).exp()
out = out / (scatter_add(out, index, dim_size=N)[index] + eps)
```

That is exactly **segment softmax**: group by `index`, stable max-subtract,
normalize. In AnyTensor it is already a primitive — or you can write the same
composition yourself for a slightly richer GAT-style step.

### A small portable attention block

Below is a minimal “attention over neighbors” helper in the spirit of GAT /
PyG `softmax`: leaky scores on edges, softmax within each destination node,
then a weighted sum of source features. **One function**, no `import torch`
inside.

```python
import anytensor as at


def neighbor_attention(messages, scores, dst_index, num_nodes: int):
    """GAT-style edge attention → node features (portable).

    messages: (E, F)   — per-edge payloads (e.g. source node features)
    scores:   (E,)     — raw attention logits per edge
    dst_index:(E,)     — destination node id for each edge
    num_nodes: int     — number of nodes (required shape-size)
    """
    # LeakyReLU(0.2) without pulling in a framework nn module
    alpha = at.where(scores > 0, scores, scores * 0.2)
    alpha = at.segment_softmax(alpha, dst_index, num_nodes)
    # weight messages, then sum into destination nodes
    weighted = messages * alpha[:, None]
    return at.segment_sum(weighted, dst_index, num_nodes)
```

### Same function, four backends

Build a tiny graph once on the host, then run the **identical**
`neighbor_attention` on NumPy, JAX, Torch, and TensorFlow arrays.

```python
import numpy as np

# 3 nodes, 4 edges into destinations [0, 0, 1, 2]
messages = np.array(
    [[1.0, 0.0],
     [0.0, 1.0],
     [1.0, 1.0],
     [2.0, 0.0]],
    dtype=np.float32,
)
scores = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
dst = np.array([0, 0, 1, 2], dtype=np.int64)
num_nodes = 3
```

=== "NumPy"

    ```python
    out = neighbor_attention(messages, scores, dst, num_nodes)
    # out.shape == (3, 2)  — one row per node
    ```

=== "JAX"

    ```python
    import jax.numpy as jnp

    out = neighbor_attention(
        jnp.asarray(messages),
        jnp.asarray(scores),
        jnp.asarray(dst),
        num_nodes,
    )
    # still a jax.Array — ready for jax.jit(neighbor_attention)(...)
    ```

=== "PyTorch"

    ```python
    import torch

    out = neighbor_attention(
        torch.as_tensor(messages),
        torch.as_tensor(scores),
        torch.as_tensor(dst),
        num_nodes,
    )
    # torch.Tensor — train with autograd as usual
    ```

=== "TensorFlow"

    ```python
    import tensorflow as tf

    out = neighbor_attention(
        tf.constant(messages),
        tf.constant(scores),
        tf.constant(dst),
        num_nodes,
    )
    # tf.Tensor — works under tf.function as well (pass static num_nodes)
    ```

You did not rewrite the algorithm. Callers keep their stack; your library keeps
one source of truth. For the battery-included form, prefer
`at.segment_softmax` directly (same numerics as the PyG index softmax path).

---

## Install

```bash
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
# or
pip install "anytensor[all]"
```

Requires Python ≥3.10. Backend floors: NumPy ≥1.24, JAX ≥0.4.32, PyTorch ≥2.0,
TensorFlow ≥2.10.

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, num_segments=2)
```

`num_segments` is required (JAX convention) — a Python `int`, jit symbolic
constant, or 0-d integral tensor scalar.

## Next

- [Usage](usage.md) — promotion, segment helpers, TorchScript, typing
- [Surprising differences](semantics.md) — NaN / ±inf / graph / GPU gotchas
- [Design](design.md) — hybrid Array API + segment backends
- [API reference](api/index.md) — generated from docstrings
