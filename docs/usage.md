# Usage

## Install

```bash
# from git (NumPy included as a core dependency)
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
# optional backends
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
# or everything
pip install "anytensor[all]"
```

From a checkout:

```bash
uv sync --extra all --group dev --group docs
```

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])

y = at.segment_sum(x, seg_ids, num_segments=2)
m = at.segment_mean(x, seg_ids, 2)
```

Same code on JAX / Torch / TF:

```python
import jax.numpy as jnp

x = jnp.arange(12.0).reshape(3, 4)
seg_ids = jnp.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, 2)
```

## Scalar / NumPy promotion

Full reductions return **0-d arrays** (not bare Python / NumPy scalars). Binary
ops and segment helpers **upcast** Python scalars and **NumPy ndarrays** onto a
peer JAX / Torch / TF tensor. NumPy is host interchange data — we never demote
a framework tensor to NumPy when mixing. Scalars alone still default to NumPy.

NumPy → framework upcast prefers **by reference** (`asarray(..., copy=False)`)
when the backend can share the buffer. A copy is used only when zero-copy is
impossible (default: warn + copy; set `fallback="error"` to raise). Use
`@promote(..., copy=True)` or `with at.promote_options(copy=True):` when the
host NumPy buffer may be mutated.

Dtype policy is per-operand via `@promote`:

```python
@promote(x="data", y="data")           # result_type — ints widen beside floats
@promote(x="data", indices="index")  # indices stay integral
@promote(condition="mask", x="data", y="data")
@promote(x="data", segment_ids="index", num_segments="shape")  # size dim
```

`num_segments` / `total_repeat_length` / `sum_partitions` are **shape** sizes:
Python `int`, jit symbolic constant, or 0-d integral tensor scalar — **required**
(JAX convention), never inferred from ids. Plain ints stay Python (not 0-d
tensors) so `jax.jit` / `tf.function` / `torch.compile` can treat them as static.

| Situation | Behavior |
|---|---|
| Only scalars | NumPy 0-d arrays |
| NumPy + framework tensor | Promote NumPy → framework (never the reverse) |
| Default upcast | Prefer **reference** (`copy=False`) |
| Zero-copy impossible | `fallback="copy"` (warn) or `"error"` |
| Mutating host buffer | `@promote(..., copy=True)` or `promote_options(copy=True)` |

## Segment helpers

| Function | Role |
|---|---|
| `segment_sum` / `min` / `max` | Reduce along axis 0 by segment id |
| `segment_count` / `mean` / `variance` | Counts and moments |
| `segment_normalize` / `segment_softmax` | Per-segment normalize / softmax |
| `segment_min_or_constant` / `segment_max_or_constant` | Empty segments → constant |
| `partition_softmax` | Softmax over contiguous partition lengths (`sum_partitions` required; `num_segments` is `shape(partitions)[0]`; rebuilds ids each call unless `partition_cache` is active) |
| `partition_ids` | Expand partition lengths to segment ids (call once, reuse) |
| `partition_cache` | Context: partition helpers reuse ids for the same tensors (weakrefs; GraphNetwork enters one per apply) |

Einops (`rearrange`, `einsum`, `reduce`, …) is re-exported for convenience.

`partition_softmax(logits, partitions, sum_partitions)` is a convenience over
`partition_ids` + `segment_softmax`. `num_segments` is not an argument — it is
`shape(partitions)[0]`. `sum_partitions` is a required shape-size
(`shape(logits)[0]`). Dropping it, or passing `None`, is a `TypeError` — not a
silent `sum(partitions)`. It rebuilds `segment_ids` on **every** call unless you
wrap the block in `partition_cache()` — then `partition_softmax` consults the
cache itself, so graph code does not thread ids through the stack. A compiler
may CSE the rebuild; eager will not. Entries are weak (GC drops them; the
context does not pin). Callbacks hold only a weakref to the cache map so a
long-lived tensor cannot keep the block alive. Use `segment_softmax` if you
already have ids. There is no process-wide `id()` cache.

```python
with at.partition_cache():
    y = at.partition_softmax(logits, partitions, sum_partitions)
    z = at.partition_softmax(other_logits, partitions, sum_partitions)
```

Under `jax.jit`, `sum_partitions` must be a static-friendly shape-size
(`shape(logits)[0]`, or a Python int). See [Surprising differences](semantics.md).

## Torch compile / export

Prefer **`torch.compile`** (training / runtime) or **`torch.export`** (AOT /
serialization). PyTorch has deprecated `torch.jit.script` / `torch.jit.trace`;
do not use them in new code.

- **`torch.compile`:** portable helpers typically need `fullgraph=False`
  (Dynamo graph-breaks on `@promote` / array-api-compat). A single fused graph
  needs a Torch-only body.
- **`torch.export`:** pass an **`nn.Module`** whose `forward` calls the portable
  helper — bare functions are rejected. See [Worked examples](examples.md).

`anytensor.enable_torchscript()` remains for legacy `torch.jit.script` call
sites that still hit `segment_sum` / `min` / `max`; it is not the recommended
path. It does not import Torch and does not care about import order: a helper
registered with `module_if_loaded("torch", …)` enables the divert as soon as
Torch is imported.

## ONNX (dynamic shapes)

**Recommend ONNX** for deployment: ONNX Runtime is well tested as a serving
engine. AnyTensor does not run ops on ORT. The recipes live in the opt-in
`anytensor.export` subpackage (not in `anytensor.__all__`, not a stable
library API). Export the **same** AnyTensor function after it is running on
Torch or TensorFlow tensors. Derive `num_segments` from `at.shape(nodes)[0]`
so `N` stays symbolic.

```python
from anytensor import export
```

| Start | Recipe |
|---|---|
| Lightning | `nn.Parameter` on the module + `to_onnx_torch(..., dynamic_shapes=)` (best named initializers) |
| Keras | `as_tensorflow_fn` / `to_onnx_tensorflow(..., params=)` so constants are created inside the trace |
| Flax | `numpy_leaves(params)` then `as_torch_module(fn, params)` (preferred) or the Keras row — not `jax2tf`, not extra inputs |

Helpers (`from anytensor import export`): [ONNX](onnx/index.md).

## Typing

Public APIs use [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations
(`Shaped` / `Integer` / helpers like `SegmentValues`) plus an `ArrayT`
`TypeVar` so operands stay on one backend type — without importing Torch /
JAX / TF for typing.

**Runtime checking is off by default.** Annotations are for editors, static
checkers, and docs. To opt in::

```python
from jaxtyping import install_import_hook
install_import_hook("anytensor", "beartype.beartype")  # before importing anytensor
import anytensor as at
# or: at.enable_typecheck() before other anytensor submodule imports
```

Install `beartype` via `anytensor[typecheck]`, `anytensor[all]`, or the `dev`
group. Set `JAXTYPING_DISABLE=1` to force runtime checks off.

## NaN helpers / specials

- `is_nan` / `is_finite` / `is_inf` (aliases `isnan` / `isfinite` / `isinf`)
- `fill_nan` (alias `nan_fill`), `fill_nan_mask` → `(filled, mask)`
- Array API `nan_to_num`, element-wise `equal_nan`
- `inf(x)` / `ninf(x)` / `nan(x)` / `pi(x)` / `e(x)`, `dtype(...)`, `finfo` / `iinfo`
- `newaxis` is `None`

Nested structures and graphs have their own sections:
[Tree](tree/index.md), [Jraph](jraph/index.md), [Hetero](hetero/index.md).

