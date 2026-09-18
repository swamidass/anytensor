# ONNX export

AnyTensor is **not** an ONNX Runtime backend. Export means: run the same
portable function on **Torch** or **TensorFlow** tensors, then serialize that
graph. Symbolic lengths come from tensor shapes (`at.shape(x)[0]`), not from
Python ints.

Runnable recipes: [Examples](examples.md). Helpers: [Export API](../api/export.md).

## Best pathway

| Starting stack | What to do | What not to do |
|---|---|---|
| **PyTorch Lightning** | Put AnyTensor in `LightningModule.forward`. `anytensor.export.to_onnx_torch(..., dynamo=True, dynamic_shapes=...)`. A Lightning module **is** an `nn.Module`. | `torch.jit.trace` / `script` |
| **Keras 3** | `@tf.function(input_signature=TensorSpec((None, …)))` on the AnyTensor function, then `to_onnx_tensorflow`. | `model.export(format="onnx")` on a custom AnyTensor layer (inspect/bind failures) |
| **Flax** | `numpy_leaves(params)`, call the **same** function on TF or Torch arrays, then the Keras or Lightning path. | `jax2tf` → tf2onnx (`XlaCallModule` / StableHLO does not lower) |

The Flax result is the important AnyTensor-specific trick: you do not translate
JAX primitives. You **rebind** the already-portable body onto an exportable
array type.

## Symbolic lengths

A Python `num_nodes: int` becomes a constant in the ONNX graph (output
`[3, feat]` even when edges are dynamic). Derive sizes from tensors:

```python
num_nodes = at.shape(nodes)[0]
at.segment_sum(messages, dst_index, num_nodes)
```

Then tell the exporter those axes are dynamic:

- Torch: `torch.export.Dim("E")` / `Dim("N")` via `export.torch_dim`, shared
  across inputs that must match.
- TF / Keras: `tf.TensorSpec((None, feat), …)` — `None` is the symbolic length.

`export.assert_symbolic_lengths` fails the test if those axes baked to ints.

## Coverage

`test/test_onnx_export.py` exports the public tensor surface through the TF
`tf2onnx` path (runs in CI) and the GAT-style neighbor helper through Torch
dynamo ONNX (skipped on CI, same Triton SIGSEGV as `torch.compile`).
`partition_softmax` is skipped (data-dependent partition lengths). Python-sized
constructors (`zeros((3,))`, `arange(3)`, …) bake ranks by API.
