"""Export AnyTensor callables to ONNX with dynamic / symbolic lengths.

ONNX Runtime is **not** a compute backend. These helpers serialize a function
that already runs on Torch or TensorFlow tensors.

**Best pathway.** Keep the AnyTensor body; pick an export-friendly array world:

1. **Lightning / ``torch.nn.Module``** — put weights on the module
   (``nn.Parameter``) and call ``to_onnx_torch(..., dynamo=True,
   dynamic_shapes=...)``. A ``LightningModule`` is an ``nn.Module``. This is
   the path that embeds weights as **named ONNX initializers**.
2. **Keras / TensorFlow** — wrap with :func:`as_tensorflow_fn` so named
   ``tf.constant`` values are created **inside** the traced function, then
   ``to_onnx_tensorflow``. Closing over outer tensors or passing weights as
   extra arguments turns them into graph inputs. Keras 3
   ``model.export(format="onnx")`` is brittle for custom AnyTensor layers.
3. **Flax** — do **not** ``jax2tf`` (tf2onnx cannot lower ``XlaCallModule`` /
   StableHLO). ``numpy_leaves(params)`` then :func:`as_torch_module` /
   :func:`as_tensorflow_fn` (same AnyTensor body, ``fn(..., params=tree)``).

**Symbolic lengths.** Pass ``num_segments`` as ``at.shape(nodes)[0]`` (or
another tensor dim), not a Python ``int``. Mark those axes dynamic in the
exporter. Python ints bake constants into the graph.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .optional import module_if_loaded

__all__ = [
    "as_tensorflow_fn",
    "as_torch_module",
    "assert_embedded_weights",
    "assert_symbolic_lengths",
    "initializer_arrays",
    "numpy_leaves",
    "symbolic_dims",
    "to_onnx",
    "to_onnx_tensorflow",
    "to_onnx_torch",
    "torch_dim",
]


def _plain_pytree(x):
    """Dict-ify mapping subclasses (Flax ``FrozenDict``) so :mod:`anytensor.tree` walks them."""
    if x is None:
        return None
    if isinstance(x, Mapping) and not isinstance(x, (str, bytes)):
        return {k: _plain_pytree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain_pytree(v) for v in x]
    if isinstance(x, tuple) and not hasattr(type(x), "_fields"):
        return tuple(_plain_pytree(v) for v in x)
    return x


def numpy_leaves(structure):
    """Map array leaves to NumPy so a Flax (or JAX) pytree can rebind onto TF / Torch."""
    import numpy as np

    from . import tree

    def _leaf(x):
        if hasattr(x, "__array__"):
            return np.asarray(x)
        return x

    return tree.map(_leaf, _plain_pytree(structure))


def _is_weight(x) -> bool:
    if x is None or isinstance(x, (str, bytes, bool, int, float, complex)):
        return False
    return hasattr(x, "__array__")


def _path_name(path) -> str:
    from .tree import DictKey, GetAttrKey, SequenceKey

    parts: list[str] = []
    for p in path:
        if isinstance(p, DictKey):
            raw = str(p.key)
        elif isinstance(p, GetAttrKey):
            raw = str(p.name)
        elif isinstance(p, SequenceKey):
            raw = str(p.idx)
        else:
            raw = str(p)
        piece = "".join(c if c.isalnum() else "_" for c in raw).strip("_") or "p"
        if piece[0].isdigit():
            piece = "p_" + piece
        parts.append(piece)
    return "__".join(parts) or "param"


def _named_weight_leaves(params):
    """Return ``(treedef, slots, weights)`` for binding.

    ``slots`` is a list of ``("weight", name)`` or ``("static", value)``.
    ``weights`` maps unique names to NumPy arrays.
    """
    from . import tree

    pairs, treedef = tree.flatten_with_path(numpy_leaves(params))
    slots: list[tuple[str, Any]] = []
    weights: dict[str, Any] = {}
    used: set[str] = set()
    for path, leaf in pairs:
        if not _is_weight(leaf):
            slots.append(("static", leaf))
            continue
        name = _path_name(path)
        if name in used:
            raise ValueError(f"duplicate parameter name {name!r} from pytree path {path!r}")
        used.add(name)
        weights[name] = leaf
        slots.append(("weight", name))
    return treedef, slots, weights


def _forward_arg_names(fn: Callable) -> list[str] | None:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    names: list[str] = []
    for p in sig.parameters.values():
        if p.name == "params":
            continue
        if p.kind is inspect.Parameter.VAR_POSITIONAL:
            return None
        if p.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
            continue
        names.append(p.name)
    return names


def _patch_forward_signature(forward, names: list[str] | None) -> None:
    if not names:
        return
    params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD)]
    params.extend(
        inspect.Parameter(n, inspect.Parameter.POSITIONAL_OR_KEYWORD) for n in names
    )
    forward.__signature__ = inspect.Signature(params)


def as_torch_module(fn: Callable, params=None, *, buffers: bool = False) -> Any:
    """Wrap ``fn`` as an ``nn.Module`` for ``torch.onnx.export`` / Lightning.

    ``forward`` keeps ``fn``'s argument names so ``dynamic_shapes`` can use
    them. When ``params`` is given, array leaves are registered as named
    ``nn.Parameter`` values (or buffers) and ``fn`` is called as
    ``fn(*args, params=<rebuilt pytree>)``. Those Parameters become **named
    ONNX initializers**, not graph inputs.
    """
    import numpy as np

    torch = module_if_loaded("torch", raises=True)
    names = _forward_arg_names(fn)
    bound = params is not None
    treedef = slots = weights = None
    if bound:
        treedef, slots, weights = _named_weight_leaves(params)

    class _Fn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._treedef = treedef
            self._slots = slots
            self._weight_names = []
            if weights:
                for name, leaf in weights.items():
                    tensor = torch.from_numpy(np.array(leaf, copy=True))
                    if buffers:
                        self.register_buffer(name, tensor)
                    else:
                        self.register_parameter(name, torch.nn.Parameter(tensor))
                    self._weight_names.append(name)

        def _params_tree(self):
            from . import tree

            leaves = []
            for kind, payload in self._slots:
                if kind == "weight":
                    leaves.append(getattr(self, payload))
                else:
                    leaves.append(payload)
            return tree.unflatten(self._treedef, leaves)

        def forward(self, *args, **kwargs):
            if self._treedef is None:
                return fn(*args, **kwargs)
            return fn(*args, params=self._params_tree(), **kwargs)

    _patch_forward_signature(_Fn.forward, names)
    _Fn.__name__ = getattr(fn, "__name__", "AnyTensorModule")
    _Fn.__qualname__ = _Fn.__name__
    return _Fn()


def as_tensorflow_fn(fn: Callable, params) -> Callable:
    """Bind ``params`` as **named constants created inside** the traced function.

    Outer ``tf.constant`` / ``tf.Variable`` objects become extra ONNX inputs.
    Constants constructed during tracing embed as initializers (``W:0``).
    ``fn`` is called as ``fn(*args, params=<rebuilt pytree>, **kwargs)``.
    """
    import numpy as np

    tf = module_if_loaded("tensorflow", raises=True)
    treedef, slots, weights = _named_weight_leaves(params)
    stored = {name: np.array(leaf, copy=True) for name, leaf in weights.items()}

    def wrapped(*args, **kwargs):
        from . import tree

        leaves = []
        for kind, payload in slots:
            if kind == "weight":
                leaves.append(tf.constant(stored[payload], name=payload))
            else:
                leaves.append(payload)
        return fn(*args, params=tree.unflatten(treedef, leaves), **kwargs)

    wrapped.__name__ = getattr(fn, "__name__", "anytensor_tf_fn")
    wrapped.__qualname__ = wrapped.__name__
    return wrapped


def torch_dim(name: str, *, min: int = 1, max: int | None = None):
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


def initializer_arrays(model) -> dict[str, Any]:
    """ONNX initializers as ``name -> ndarray`` (embedded weights and constants)."""
    try:
        from onnx import numpy_helper
    except ImportError as exc:
        raise RuntimeError("initializer_arrays requires onnx (pip install onnx)") from exc
    proto = _model_proto(model)
    return {init.name: numpy_helper.to_array(init) for init in proto.graph.initializer}


def _initializer_aliases(name: str) -> list[str]:
    return [name, f"{name}:0", name.replace("__", "/"), name.replace("__", ".")]


def _required_input_names(proto) -> list[str]:
    init = {i.name for i in proto.graph.initializer}
    return [i.name for i in proto.graph.input if i.name not in init]


def _match_initializer(inits: Mapping[str, Any], name: str, value, used: set[str], *, require_names: bool):
    import numpy as np

    value = np.asarray(value)
    for alias in _initializer_aliases(name):
        arr = inits.get(alias)
        if alias in used or arr is None:
            continue
        if arr.shape == value.shape and np.allclose(arr, value, equal_nan=True, atol=1e-5):
            return alias
    if require_names:
        return None
    for key, arr in inits.items():
        if key in used:
            continue
        if arr.shape == value.shape and np.allclose(arr, value, equal_nan=True, atol=1e-5):
            return key
    return None


def assert_embedded_weights(
    model,
    params,
    *,
    require_names: bool = True,
) -> dict[str, str]:
    """Require every array leaf of ``params`` to be an ONNX initializer, not a feed.

    ``require_names`` (default True) also demands the initializer name is the
    pytree path (Torch ``W``) or TF's ``W:0`` suffix. Returns
    ``{param_name: initializer_name}``.
    """
    proto = _model_proto(model)
    inits = initializer_arrays(proto)
    feeds = set(_required_input_names(proto))
    _treedef, _slots, weights = _named_weight_leaves(params)
    if not weights:
        raise AssertionError("params has no array leaves to embed")
    used: set[str] = set()
    matched: dict[str, str] = {}
    leaked = sorted(n for n in weights if n in feeds or f"{n}:0" in feeds)
    if leaked:
        raise AssertionError(
            f"weights leaked as ONNX graph inputs (pass them via as_torch_module "
            f"/ as_tensorflow_fn, not as extra arguments): {leaked}"
        )
    for name, value in weights.items():
        key = _match_initializer(inits, name, value, used, require_names=require_names)
        if key is None:
            raise AssertionError(
                f"weight {name!r} is not an embedded ONNX initializer "
                f"(names={sorted(inits)}, require_names={require_names})"
            )
        used.add(key)
        matched[name] = key
    return matched


def assert_symbolic_lengths(
    model,
    *,
    inputs: Mapping[str, Sequence[int]] | None = None,
    outputs: Mapping[str, Sequence[int]] | None = None,
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


def to_onnx(fn, args, *, backend: str | None = None, params=None, **kwargs):
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
            raise TypeError("to_onnx(..., backend='tensorflow') requires input_signature=")
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
