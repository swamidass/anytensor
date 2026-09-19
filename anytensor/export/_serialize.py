"""Serialize AnyTensor callables through Torch dynamo ONNX or tf2onnx."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from anytensor.optional import module_if_loaded

from ._bind import as_tensorflow_fn, as_torch_module

__all__ = [
    "to_onnx_tensorflow",
    "to_onnx_torch",
]


def _torch_dim(name: str, *, min: int = 1, max: int | None = None):
    """``torch.export.Dim`` for shared dynamic axes (reuse the object across inputs)."""
    torch = module_if_loaded("torch", raises=True)
    kwargs: dict[str, Any] = {"min": min}
    if max is not None:
        kwargs["max"] = max
    return torch.export.Dim(name, **kwargs)


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
    params=None,
    dynamic_shapes=None,
    input_names=None,
    output_names=None,
    **kwargs,
):
    """``torch.onnx.export`` with the dynamo / ``torch.export`` path (dynamic shapes).

    Pass ``params=`` to bind a weight pytree as ``nn.Parameter`` initializers.
    """
    torch = module_if_loaded("torch", raises=True)
    if params is not None:
        if isinstance(model, torch.nn.Module):
            raise TypeError(
                "params= is for callables; register nn.Parameter on the module instead"
            )
        model = as_torch_module(model, params)
    elif not isinstance(model, torch.nn.Module):
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


def to_onnx_tensorflow(fn: Callable, input_signature, *, params=None, opset: int = 18):
    """``tf.function`` + ``tf2onnx.convert.from_function`` (``None`` dims stay symbolic).

    Pass ``params=`` to embed weights via :func:`as_tensorflow_fn` (named
    constants created inside the traced function).
    """
    tf = module_if_loaded("tensorflow", raises=True)
    try:
        import tf2onnx
    except ImportError as exc:
        raise RuntimeError(
            "to_onnx_tensorflow requires tf2onnx (pip install tf2onnx)"
        ) from exc
    if params is not None:
        fn = as_tensorflow_fn(fn, params)
    wrapped = tf.function(fn, input_signature=input_signature)
    proto, _storage = tf2onnx.convert.from_function(
        wrapped, input_signature=input_signature, opset=opset
    )
    return proto


def _to_onnx(fn, args, *, backend: str | None = None, params=None, **kwargs):
    """Export ``fn`` using Torch or TensorFlow tensors in ``args``.

    JAX arrays are rejected — convert parameters with :func:`numpy_leaves` and
    pass them as ``params=`` so they embed as ONNX initializers (the Flax
    recipe). Do not pass weights as extra graph inputs.
    """
    kind = backend or _first_backend(args)
    if kind == "torch":
        return to_onnx_torch(fn, args, params=params, **kwargs)
    if kind == "tensorflow":
        signature = kwargs.pop("input_signature", None)
        if signature is None:
            raise TypeError(
                "to_onnx(..., backend='tensorflow') requires input_signature="
            )
        return to_onnx_tensorflow(fn, signature, params=params, **kwargs)
    if kind == "jax":
        raise RuntimeError(
            "JAX/Flax arrays cannot go through jax2tf to ONNX (XlaCallModule). "
            "Convert params with anytensor.export.numpy_leaves and bind them with "
            "as_torch_module / as_tensorflow_fn (or to_onnx(..., params=))."
        )
    raise RuntimeError(
        "to_onnx needs Torch or TensorFlow tensors in args "
        "(NumPy-only graphs are not an ONNX export path)."
    )
