"""ONNX recipes for AnyTensor models (unstable guide).

This **subpackage** is **not** a stable library API, **not** in
``anytensor.__all__``, and **not** an ONNX Runtime backend. Import it
explicitly::

    from anytensor import export

It records how a downstream *model* author can serialize an AnyTensor
function that already runs on Torch or TensorFlow tensors.

Library authors should keep helpers portable (``at.shape``, segment ops) and
leave export to the application. These names may change.

**Pathway.** Keep the AnyTensor body; bind weights; serialize:

1. **Lightning / ``nn.Module``** — ``nn.Parameter`` weights +
   :func:`to_onnx_torch`. Named ONNX initializers.
2. **Keras / TensorFlow** — :func:`as_tensorflow_fn` creates named constants
   *inside* the trace, then :func:`to_onnx_tensorflow`. Outer tensors leak as
   graph inputs.
3. **Flax** — :func:`numpy_leaves` then (1) or (2). Do not ``jax2tf``.

**Symbolic lengths.** ``num_segments = at.shape(nodes)[0]``, not a Python
``int``.
"""

from ._bind import as_tensorflow_fn, as_torch_module, numpy_leaves
from ._graph import assert_embedded_weights, assert_symbolic_lengths
from ._serialize import to_onnx_tensorflow, to_onnx_torch

__all__ = [
    "as_tensorflow_fn",
    "as_torch_module",
    "assert_embedded_weights",
    "assert_symbolic_lengths",
    "numpy_leaves",
    "to_onnx_tensorflow",
    "to_onnx_torch",
]
