# ONNX examples

These fenced blocks are executed by pytest (Sybil). Overview:
[ONNX export](index.md) (recommended deploy path; ORT). Helpers: [API](api.md).

The portable body uses **`at.shape(nodes)[0]`** so node count `N` is a
symbolic length, not a Python int.

```python
import anytensor as at
import numpy as np


def neighbor_from_nodes(messages, scores, dst_index, nodes):
    num_nodes = at.shape(nodes)[0]
    alpha = at.where(scores > 0, scores, scores * 0.2)
    alpha = at.segment_softmax(alpha, dst_index, num_nodes)
    weighted = messages * alpha[:, None]
    return at.segment_sum(weighted, dst_index, num_nodes)


messages = np.array(
    [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
)
scores = np.array([1.0, 1.0, 0.5, 2.0], dtype=np.float32)
dst = np.array([0, 0, 1, 2], dtype=np.int64)
nodes = np.zeros((3, 2), dtype=np.float32)
out_np = neighbor_from_nodes(messages, scores, dst, nodes)
assert out_np.shape == (3, 2)
```

## Keras / TensorFlow — `tf.function` + tf2onnx

Keras 3 `model.export(format="onnx")` is not used. Wrap the AnyTensor function.

```python
import os

tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")
from anytensor import export

signature = [
    tf.TensorSpec((None, 2), tf.float32, name="messages"),
    tf.TensorSpec((None,), tf.float32, name="scores"),
    tf.TensorSpec((None,), tf.int64, name="dst"),
    tf.TensorSpec((None, 2), tf.float32, name="nodes"),
]
proto = export.to_onnx_tensorflow(neighbor_from_nodes, signature)
dims = export.assert_symbolic_lengths(
    proto, inputs={"messages": (0,), "nodes": (0,)}
)
assert isinstance(dims["messages"][0], str)
```

## Lightning / Torch — dynamo ONNX + `Dim`

```python
import os

pytest.importorskip("torch")
if os.environ.get("CI"):
    pytest.skip("torch.onnx dynamo disabled on CI runners (dynamo/triton)")
from anytensor import export

E, N = torch.export.Dim("E"), torch.export.Dim("N")
prog = export.to_onnx_torch(
    neighbor_from_nodes,
    (
        torch.as_tensor(messages),
        torch.as_tensor(scores),
        torch.as_tensor(dst),
        torch.as_tensor(nodes),
    ),
    dynamic_shapes={
        "messages": {0: E},
        "scores": {0: E},
        "dst_index": {0: E},
        "nodes": {0: N},
    },
    input_names=["messages", "scores", "dst_index", "nodes"],
    output_names=["out"],
)
dims = export.assert_symbolic_lengths(
    prog,
    inputs={"messages": (0,), "nodes": (0,)},
    outputs={"out": (0,)},
)
assert dims["out"][0] == "N"
```

A `LightningModule` is an `nn.Module` — put weights on it as `nn.Parameter`
(not closed-over tensors) and pass the module to `to_onnx_torch`.

## Flax — rebind params as embedded weights, do not jax2tf

Preferred: Torch `nn.Parameter` via `as_torch_module(fn, params)` (named
initializers). TF: `as_tensorflow_fn` / `to_onnx_tensorflow(..., params=)` so
constants are created **inside** the trace. Do not close over outer tensors.

```python
jax = pytest.importorskip("jax")
flax = pytest.importorskip("flax")
tf = pytest.importorskip("tensorflow")
pytest.importorskip("tf2onnx")
from flax import linen as nn
from anytensor import export


class FlaxNeighbor(nn.Module):
    @nn.compact
    def __call__(self, messages, scores, dst_index, nodes):
        w = self.param("W", nn.initializers.ones, (2, 2))
        return neighbor_from_nodes(messages @ w, scores, dst_index, nodes)


mj = jax.numpy.asarray(messages)
mod = FlaxNeighbor()
variables = mod.init(
    jax.random.key(0),
    mj,
    jax.numpy.asarray(scores),
    jax.numpy.asarray(dst),
    jax.numpy.asarray(nodes),
)
params = export.numpy_leaves(variables["params"])


def apply(messages, scores, dst_index, nodes, *, params):
    return neighbor_from_nodes(messages @ params["W"], scores, dst_index, nodes)


proto = export.to_onnx_tensorflow(
    apply,
    [
        tf.TensorSpec((None, 2), tf.float32, name="messages"),
        tf.TensorSpec((None,), tf.float32, name="scores"),
        tf.TensorSpec((None,), tf.int64, name="dst"),
        tf.TensorSpec((None, 2), tf.float32, name="nodes"),
    ],
    params=params,
)
dims = export.assert_symbolic_lengths(proto, inputs={"messages": (0,)})
assert isinstance(dims["messages"][0], str)
assert export.assert_embedded_weights(proto, params)["W"].startswith("W")
```
