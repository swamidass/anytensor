"""Export AnyTensor callables to ONNX with dynamic / symbolic lengths.

ONNX Runtime is **not** a compute backend. These helpers serialize a function
that already runs on Torch or TensorFlow tensors.

**Best pathway.** Keep the AnyTensor body; pick an export-friendly array world:

1. **Lightning / ``torch.nn.Module``** — ``torch.onnx.export(..., dynamo=True,
   dynamic_shapes=...)``. A ``LightningModule`` is an ``nn.Module``.
2. **Keras / TensorFlow** — wrap the same function in
   ``tf.function(input_signature=...)`` with ``None`` dims, then
   ``tf2onnx.convert.from_function``. Keras 3 ``model.export(format="onnx")``
   is brittle for custom AnyTensor layers.
3. **Flax** — do **not** ``jax2tf`` (tf2onnx cannot lower ``XlaCallModule`` /
   StableHLO). Convert params with :func:`numpy_leaves`, call the **same**
   AnyTensor function on TF or Torch tensors, then use (1) or (2).

**Symbolic lengths.** Pass ``num_segments`` as ``at.shape(nodes)[0]`` (or
another tensor dim), not a Python ``int``. Mark those axes dynamic in the
exporter. Python ints bake constants into the graph.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Optional

from .optional import module_if_loaded

__all__ = [
    "as_torch_module",
    "assert_symbolic_lengths",
    "numpy_leaves",
    "symbolic_dims",
    "to_onnx",
    "to_onnx_tensorflow",
    "to_onnx_torch",
    "torch_dim",
]


def numpy_leaves(structure):
    """Map array leaves to NumPy so a Flax (or JAX) pytree can rebind onto TF / Torch."""
    import numpy as np

    from . import tree

    def _leaf(x):
        if x is None:
            return None
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    return tree.map(_leaf, structure)


def as_torch_module(fn: Callable) -> Any:
    """Wrap ``fn(*tensors)`` as an ``nn.Module`` for ``torch.onnx.export`` / Lightning."""
    torch = module_if_loaded("torch", raises=True)

    class _Fn(torch.nn.Module):
        def forward(self, *args, **kwargs):
            return fn(*args, **kwargs)

    _Fn.__name__ = getattr(fn, "__name__", "AnyTensorModule")
    _Fn.__qualname__ = _Fn.__name__
    return _Fn()


def torch_dim(name: str, *, min: int = 1, max: Optional[int] = None):
    """``torch.export.Dim`` for shared dynamic axes (reuse the object across inputs)."""
    torch = module_if_loaded("torch", raises=True)
    kwargs: dict[str, Any] = {"min": min}
    if max is not None:
        kwargs["max"] = max
    return torch.export.Dim(name, **kwargs)


def _model_proto(model):
    if hasattr(model, "model_proto"):
        return model.model_proto
    return model


def symbolic_dims(model) -> dict[str, tuple[str | int, ...]]:
    """Input/output dimension names: strings are symbolic, ints are static."""
    proto = _model_proto(model)
    out: dict[str, tuple[str | int, ...]] = {}
    for value in list(proto.graph.input) + list(proto.graph.output):
        dims: list[str | int] = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else int(dim.dim_value))
        out[value.name] = tuple(dims)
    return out


def assert_symbolic_lengths(
    model,
    *,
    inputs: Optional[Mapping[str, Sequence[int]]] = None,
    outputs: Optional[Mapping[str, Sequence[int]]] = None,
) -> dict[str, tuple[str | int, ...]]:
    """Require listed axes to be symbolic (``dim_param``), not a baked ``dim_value``.

    ``inputs`` / ``outputs`` map value names (or unique suffixes) to axis
    indices that must be symbolic. Omit a map to require **every** rank≥1
    input or output to have at least one symbolic axis.
    """
    dims = symbolic_dims(model)
    proto = _model_proto(model)
    if inputs is None:
        _require_any_symbolic(dims, [v.name for v in proto.graph.input], kind="input")
    else:
        _require_axes(dims, inputs, kind="input")
    if outputs is None:
        _require_any_symbolic(dims, [v.name for v in proto.graph.output], kind="output")
    else:
        _require_axes(dims, outputs, kind="output")
    return dims


def _lookup(dims: Mapping[str, tuple[str | int, ...]], name: str) -> tuple[str | int, ...]:
    if name in dims:
        return dims[name]
    matches = [k for k in dims if k.endswith(name) or k.startswith(name)]
    if len(matches) == 1:
        return dims[matches[0]]
    raise KeyError(f"ONNX value {name!r} not in {sorted(dims)}")


def _require_any_symbolic(dims, names, *, kind: str) -> None:
    for name in names:
        axes = dims[name]
        if not axes:
            continue
        if not any(isinstance(d, str) and d for d in axes):
            raise AssertionError(
                f"ONNX {kind} {name!r} has no symbolic lengths: {axes}"
            )


def _require_axes(dims, spec: Mapping[str, Sequence[int]], *, kind: str) -> None:
    for name, axes in spec.items():
        found = _lookup(dims, name)
        for i in axes:
            if i >= len(found) or not isinstance(found[i], str) or not found[i]:
                raise AssertionError(
                    f"ONNX {kind} {name!r} axis {i} is not symbolic: {found}"
                )


def _first_backend(args) -> str | None:
    torch = module_if_loaded("torch")
    tf = module_if_loaded("tensorflow")
    jax = module_if_loaded("jax")
    stack = [args]
    while stack:
        cur = stack.pop()
        if isinstance(cur, (tuple, list)):
            stack.extend(cur)
            continue
        if torch is not None and isinstance(cur, torch.Tensor):
            return "torch"
        if tf is not None and isinstance(cur, (tf.Tensor, tf.Variable)):
            return "tensorflow"
        if jax is not None and isinstance(cur, getattr(jax, "Array", ())):
            return "jax"
    return None


def to_onnx_torch(
    model,
    args,
    *,
    dynamic_shapes=None,
    input_names=None,
    output_names=None,
    **kwargs,
):
    """``torch.onnx.export`` with the dynamo / ``torch.export`` path (dynamic shapes)."""
    torch = module_if_loaded("torch", raises=True)
    if not isinstance(model, torch.nn.Module):
        model = as_torch_module(model)
    model.eval()
    kwargs.setdefault("dynamo", True)
    return torch.onnx.export(
        model,
        args,
        dynamic_shapes=dynamic_shapes,
        input_names=input_names,
        output_names=output_names,
        **kwargs,
    )


def to_onnx_tensorflow(fn: Callable, input_signature, *, opset: int = 18):
    """``tf.function`` + ``tf2onnx.convert.from_function`` (``None`` dims stay symbolic)."""
    tf = module_if_loaded("tensorflow", raises=True)
    try:
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError(
            "to_onnx_tensorflow requires tf2onnx (pip install tf2onnx)"
        ) from exc
    wrapped = tf.function(fn, input_signature=input_signature)
    proto, _storage = tf2onnx.convert.from_function(
        wrapped, input_signature=input_signature, opset=opset
    )
    return proto


def to_onnx(fn, args, *, backend: Optional[str] = None, **kwargs):
    """Export ``fn`` using Torch or TensorFlow tensors in ``args``.

    JAX arrays are rejected — convert parameters with :func:`numpy_leaves` and
    call ``fn`` on Torch or TF tensors first (the Flax recipe).
    """
    kind = backend or _first_backend(args)
    if kind == "torch":
        return to_onnx_torch(fn, args, **kwargs)
    if kind == "tensorflow":
        signature = kwargs.pop("input_signature", None)
        if signature is None:
            raise TypeError("to_onnx(..., backend='tensorflow') requires input_signature=")
        return to_onnx_tensorflow(fn, signature, **kwargs)
    if kind == "jax":
        raise RuntimeError(
            "JAX/Flax arrays cannot go through jax2tf to ONNX (XlaCallModule). "
            "Convert params with anytensor.export.numpy_leaves and call the same "
            "AnyTensor function on Torch or TensorFlow tensors."
        )
    raise RuntimeError(
        "to_onnx needs Torch or TensorFlow tensors in args "
        "(NumPy-only graphs are not an ONNX export path)."
    )
