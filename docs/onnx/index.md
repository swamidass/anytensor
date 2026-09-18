# ONNX export

!!! warning "Unstable guide"
    [`anytensor.export`](api.md) is an opt-in subpackage for downstream *model*
    builders. It is **not** in `anytensor.__all__` and **not** a stable library
    contract. Names may change.

**Recommend ONNX** as the serialization target: [ONNX Runtime](https://onnxruntime.ai/)
(ORT) is well tested and a common engine for deploying a frozen graph. AnyTensor
does not run ops *on* ORT. Export means: run the same portable function on
**Torch** or **TensorFlow** tensors, then serialize that graph so ORT (or
another ONNX runner) can serve it.

Symbolic lengths come from tensor shapes (`at.shape(x)[0]`), not from Python
ints. Learned weights must land in `graph.initializer`, not as extra feeds —
use `export.as_torch_module(fn, params)` / `export.as_tensorflow_fn`.
Library-consumer contracts (required sizes, partition totals, cache, stacked
GCN/GN): [Usage → Caller rules](../usage.md#caller-rules).

```python
from anytensor import export
```

Runnable recipes: [Examples](examples.md). Helpers: [API](api.md).

## Best pathway

| Starting stack | What to do | What not to do |
|---|---|---|
| **PyTorch Lightning** | Put AnyTensor in `LightningModule.forward` with **`nn.Parameter` weights**. `to_onnx_torch(..., dynamo=True, dynamic_shapes=...)`. A Lightning module **is** an `nn.Module`. Initializers keep the Parameter names (`W`, `Dense_0__kernel`). | `torch.jit.trace` / `script`; closing over raw tensors (not Parameters) |
| **Keras 3** | `as_tensorflow_fn(fn, params)` so named `tf.constant` values are created **inside** the trace, then `to_onnx_tensorflow`. | `model.export(format="onnx")` on a custom AnyTensor layer; closing over **outer** `tf.constant` / `Variable` (those become graph inputs) |
| **Flax** | `numpy_leaves(params)`, then the **same** function as `fn(*xs, params=tree)` on Torch (preferred) or TF. | `jax2tf` → tf2onnx (`XlaCallModule` / StableHLO does not lower); passing weights as extra ONNX inputs |

**Which rebind embeds weights best?** Torch `nn.Parameter` (via `as_torch_module` or a Lightning module). Names in the ONNX file match the pytree path. The TF helper is the CI-reliable fallback: it plants named constants *inside* the traced function (`W:0`). Outer tensors and extra arguments leak as feeds — `assert_embedded_weights` fails those graphs.

## Symbolic lengths

A Python `num_nodes: int` becomes a constant in the ONNX graph (output
`[3, feat]` even when edges are dynamic). Derive sizes from tensors:

```python
num_nodes = at.shape(nodes)[0]
at.segment_sum(messages, dst_index, num_nodes)
```

Then tell the exporter those axes are dynamic:

- Torch: `torch.export.Dim("E")` / `Dim("N")`, shared across inputs that must
  match.
- TF / Keras: `tf.TensorSpec((None, feat), …)` — `None` is the symbolic length.

`export.assert_symbolic_lengths` fails the test if those axes baked to ints.
`export.assert_embedded_weights(model, params)` fails if a weight is a feed
instead of an initializer.

### Partition totals

Caller list: [Usage → Partition helpers](../usage.md#partition-helpers).

Official jraph names the flattened length `sum_partitions` / uses
`sum(n_node)`. A data `sum(partitions)` becomes `ReduceSum` in ONNX
(and `int(sum(...))` bakes a host constant). AnyTensor takes that total as
a **shape-size**, so it stays a `dim_param`:

- `partition_ids` / `partition_softmax`: required `total_length` /
  jraph `sum_partitions` is `at.shape(logits)[0]`, not `sum(partitions)`.
- GraphNetwork apply: `sum_n_node = at.shape(nodes)[0]`,
  `sum_n_edge = at.shape(senders)[0]` — not `sum(n_node)`.
- A single-graph count vector (`n_node` / `n_edge` *values*):
  `at.full((1,), at.shape(x)[0], dtype=np.int32, like=x)`. The vector
  length is 1 (one graph); the fill is the same shape symbol.

With `cache` on, `partition_ids` stores **one expansion per partition
vector**; `shape(ids)[0]` *is* that total (no extra `sum(partitions)`
cache).

## Coverage

`test/test_onnx_export.py` exports the public tensor surface through the TF
`tf2onnx` path (runs in CI) and the GAT-style neighbor helper through Torch
dynamo ONNX (skipped on CI, same Triton SIGSEGV as `torch.compile`).
Constructors (`zeros` / `ones` / `full` / `arange` / `split`) take sizes from
`at.shape` so the new length is a graph symbol. `partition_softmax` takes
`num_segments` from `shape(partitions)[0]` and requires
`total_length=at.shape(logits)[0]` so the flattened length is not a data sum.

The same file also exports the **model zoos** as a TF/ONNX stress test:
`anytensor.hetero` (R-GCN, GraphSAGE, CompGCN, HGT, HAN) and `anytensor.jraph`
(GraphNetwork, InteractionNetwork, GraphMapFeatures, RelationNetwork, DeepSets,
GraphNetGAT, GAT, GraphConvolution). Those layers take destination sizes from
`at.shape`, not `int(shape(...))`, so node/edge axes stay `dim_param`s.
GraphNetwork apply is `@cache` (sticky), so `partition_ids` reuses the same `n_node` /
`n_edge` expansion across stacked applies and during the TF trace. Partition
flattened length is `shape(nodes)[0]` / `shape(logits)[0]`, never a data
`sum(n_node)`. Stacked `GraphConvolution` reuses self-edges, `N`, and degrees
(`cache["gcn"]`) so the ONNX graph does not duplicate `Shape` / `Range` /
`Concat` per layer.
