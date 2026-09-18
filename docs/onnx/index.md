# ONNX export

!!! warning "Unstable guide"
    [`anytensor.export`](api.md) is an opt-in subpackage for downstream *model*
    builders. It is **not** in `anytensor.__all__`, **not** a stable library
    contract, and **not** an ONNX Runtime backend. Names may change.

AnyTensor is **not** an ONNX Runtime backend. Export means: run the same
portable function on **Torch** or **TensorFlow** tensors, then serialize that
graph. Symbolic lengths come from tensor shapes (`at.shape(x)[0]`), not from
Python ints. Learned weights must land in `graph.initializer`, not as extra
feeds — use `export.as_torch_module(fn, params)` / `export.as_tensorflow_fn`.

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

## Coverage

`test/test_onnx_export.py` exports the public tensor surface through the TF
`tf2onnx` path (runs in CI) and the GAT-style neighbor helper through Torch
dynamo ONNX (skipped on CI, same Triton SIGSEGV as `torch.compile`).
`partition_softmax` is skipped (data-dependent partition lengths). Python-sized
constructors (`zeros((3,))`, `arange(3)`, …) bake ranks by API.
