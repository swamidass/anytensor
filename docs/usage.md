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

`num_segments` / `total_repeat_length` / `total_length` are **shape** sizes:
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

## Caller rules

Contracts for library authors writing on AnyTensor. These stay portable under
`jax.jit` / `tf.function` / `torch.compile` / ONNX. Why they exist:
[Design](design.md#5-shape-sizes-are-required-and-stay-static-friendly).
Backend surprises: [Surprising differences](semantics.md).

### Shape-sizes

| Do | Don't |
|---|---|
| Pass `num_segments` on every **segment** op | Infer `max(ids)+1`, omit, or pass `None` |
| Read lengths with `at.shape(x)[0]` | `int(x.shape[0])` / `int(at.shape(x)[0])` under tracing |
| Leave Python ints as Python ints | Wrap a size into a 0-d tensor yourself |

Kind `shape` is a Python `int`, a jit symbolic constant, or a 0-d integral
tensor. Omitting a required size, or passing `None`, is a `TypeError`.

### Partition helpers

There is no `partition_sum` / `partition_min` / `partition_max`. Partition
softmax is `partition_ids` then `segment_softmax`.

| Do | Don't |
|---|---|
| `total_length=at.shape(logits)[0]` (required) | `sum(partitions)`, omit, or `None` |
| Treat `num_segments` as `shape(partitions)[0]` — not an argument | Pass `num_segments=` to partition helpers |
| jraph: 3rd positional `sum_partitions` is that alias | Data-sum the count vector |
| GraphNetwork: `sum_n_node = shape(nodes)[0]`, `sum_n_edge = shape(senders)[0]` | `sum(n_node)` / `sum(n_edge)` |
| Single-graph counts: `at.full((1,), at.shape(x)[0], dtype=np.int32, like=x)` | Host `int(N)` fill that bakes on export |

On ONNX those totals are `dim_param`s (`Shape` of the aligned tensor), not
`ReduceSum` of the partition vector.

### Cache

Off by default. Opt in; entries are weak (GC drops them; the cache does not
pin). There is no process-wide `id()` cache: tensors are unhashable, in-place
edits would stale ids, and tracers wrap a new object every compile.

| Form | Behavior |
|---|---|
| `@cache` on a callable | **Sticky** `enable()` — later calls reuse the map (GraphNetwork / GraphConvolution apply) |
| `with cache():` | Scoped; drops on exit unless sticky |
| `cache.enable()` / `disable()` | Turn the cache on or off |
| `cache.purge("partition", tensor)` | Drop one tensor from one namespace |

`at.cache` is a dict of dicts. `partition_ids` is the only partition helper
that talks to `cache["partition"]`: **one expansion per partition vector**;
`shape(ids)[0]` *is* the flattened total (no separate `sum(partitions)` map).
Other partition helpers call `partition_ids`, so a hit is shared. If a cached
expansion's length does not match `total_length` (both host Python ints), that
entry is purged, a warning is issued, and ids are recomputed. Under tracing
the lengths are not Python ints, so the check is skipped.

#### Who sets the key

GraphNetwork does **not** invent a graph-level key. `@cache` on apply only
calls `enable()` (sticky). Keys are written by `partition_ids` when GN
expands the same `n_node` / `n_edge` vectors (`_repeat_by`, global
aggregation). Stacked `net(g)` hits because those count vectors are
unchanged; node/edge **features** are new each layer and are not the key.

| Helper | Namespace | Key | Who writes it |
|---|---|---|---|
| GraphNetwork / `partition_softmax` | `"partition"` | `(id(partitions),)` | `partition_ids` |
| GraphConvolution structure | `"gcn"` | `(id(senders), add_self_edges, symmetric_normalization)` | GCN apply (self-edges / `N` / degrees) |
| Hetero `multi_update_all` / zoo | — | — | Nothing. Dest size is `shape(dst_nodes)[0]`; incidence is already `senders` / `receivers`. No `partition_ids`. |

Follow GraphNetwork in your own helper the same way: decorate apply, then
call `partition_ids` (or GN / GCN). Do not key by the `GraphsTuple`.

```python
@at.cache
def my_apply(graph):
    total = at.shape(graph.nodes)[0]
    return at.partition_ids(graph.n_node, total)  # key is id(graph.n_node)
```

`GraphConvolution` adds a second namespace because self-edges / degrees are
**derived** (`arange` + `concat`), not a partition expansion. Per-layer
`MatMul` still appears once per apply. That lookup/store helper is
library-internal; for a custom derived tensor, subscript `cache["name"]`
while the cache is on and start the key with `id(obj)` so
`cache.purge("name", obj)` can drop it.

Hetero layers are **plain functions**, not `@cache` factories. Wrapping a
stack in `@cache` is harmless but empty unless a `message_fn` also calls
`partition_ids`. Batch/unbatch still data-sum `n_node` for offsets (eager,
like jraph pad) — that is not the apply path.

```python
import anytensor as at
import numpy as np

logits = np.array([1.0, 2.0, 3.0])
partitions = np.array([2, 1])
total_length = at.shape(logits)[0]
other_logits = logits * 2

with at.cache():
    y = at.partition_softmax(logits, partitions, total_length)
    z = at.partition_softmax(other_logits, partitions, total_length)
```

### Export

Opt-in `anytensor.export` is a recipe, not a stable library API (not in
`anytensor.__all__`). Details: [ONNX](onnx/index.md).

| Do | Don't |
|---|---|
| `from anytensor import export` | Treat export helpers as a frozen public contract |
| Keep lengths as `at.shape` so ONNX gets `dim_param`s | Python ints or data sums that bake `N` |
| Embed weights (`as_torch_module` / Lightning `nn.Parameter`, or `as_tensorflow_fn`) | Extra feeds, outer `tf.constant`, or `jax2tf` |
| Stack GraphNetwork / GraphConvolution under `@cache` | Expect a fresh partition / GCN structure subgraph per layer |

## Segment helpers

| Function | Role |
|---|---|
| `segment_sum` / `min` / `max` | Reduce along axis 0 by segment id |
| `segment_count` / `mean` / `variance` | Counts and moments |
| `segment_normalize` / `segment_softmax` | Per-segment normalize / softmax |
| `segment_min_or_constant` / `segment_max_or_constant` | Empty segments → constant |
| `partition_softmax` | Softmax over contiguous partition lengths (`total_length` required; `num_segments` is `shape(partitions)[0]`; calls `partition_ids`, which reuses ids when `cache` is active) |
| `partition_ids` | Expand partition lengths to segment ids (the cache chokepoint; other partition helpers call this) |
| `cache` | Decorator (sticky across calls) / context / `enable`+`disable`: dict of dicts; `partition_ids` stores **one expansion per partition vector** at `cache["partition"]` (`shape(ids)[0]` is the total); `purge("partition", tensor)` drops one tensor; wrong-size hit warns, purges, and recomputes |

Einops (`rearrange`, `einsum`, `reduce`, …) is re-exported for convenience.

`partition_softmax` is `partition_ids` + `segment_softmax`. Use
`segment_softmax` if you already have ids. Caller contracts (required
`total_length`, cache forms, ONNX `dim_param`s): [Caller rules](#caller-rules).

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
so `N` stays symbolic. Do/don't for lengths, weights, and stacked graphs:
[Caller rules](#export).

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

