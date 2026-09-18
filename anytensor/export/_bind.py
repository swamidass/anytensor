"""Bind Flax / JAX / dict weight trees onto Torch modules or TF callables."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

from anytensor import tree
from anytensor.optional import module_if_loaded
from anytensor.tree import DictKey, GetAttrKey, SequenceKey

__all__ = [
    "as_tensorflow_fn",
    "as_torch_module",
    "numpy_leaves",
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
            raise ValueError(
                f"duplicate parameter name {name!r} from pytree path {path!r}"
            )
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
