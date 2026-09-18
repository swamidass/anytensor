"""Inspect ONNX protos: symbolic axes and embedded initializers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ._bind import _named_weight_leaves

__all__ = [
    "assert_embedded_weights",
    "assert_symbolic_lengths",
]


def _model_proto(model):
    if hasattr(model, "model_proto"):
        return model.model_proto
    return model


def _symbolic_dims(model) -> dict[str, tuple[str | int, ...]]:
    """Input/output dimension names: strings are symbolic, ints are static."""
    proto = _model_proto(model)
    out: dict[str, tuple[str | int, ...]] = {}
    for value in list(proto.graph.input) + list(proto.graph.output):
        dims: list[str | int] = []
        for dim in value.type.tensor_type.shape.dim:
            dims.append(dim.dim_param if dim.dim_param else int(dim.dim_value))
        out[value.name] = tuple(dims)
    return out


def _initializer_arrays(model) -> dict[str, Any]:
    """ONNX initializers as ``name -> ndarray`` (embedded weights and constants)."""
    try:
        from onnx import numpy_helper
    except ImportError as exc:
        raise RuntimeError(
            "initializer_arrays requires onnx (pip install onnx)"
        ) from exc
    proto = _model_proto(model)
    return {init.name: numpy_helper.to_array(init) for init in proto.graph.initializer}


def _initializer_aliases(name: str) -> list[str]:
    return [name, f"{name}:0", name.replace("__", "/"), name.replace("__", ".")]


def _required_input_names(proto) -> list[str]:
    init = {i.name for i in proto.graph.initializer}
    return [i.name for i in proto.graph.input if i.name not in init]


def _match_initializer(
    inits: Mapping[str, Any], name: str, value, used: set[str], *, require_names: bool
):
    import numpy as np

    value = np.asarray(value)
    for alias in _initializer_aliases(name):
        arr = inits.get(alias)
        if alias in used or arr is None:
            continue
        if arr.shape == value.shape and np.allclose(
            arr, value, equal_nan=True, atol=1e-5
        ):
            return alias
    if require_names:
        return None
    for key, arr in inits.items():
        if key in used:
            continue
        if arr.shape == value.shape and np.allclose(
            arr, value, equal_nan=True, atol=1e-5
        ):
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
    inits = _initializer_arrays(proto)
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
    dims = _symbolic_dims(model)
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


def _lookup(
    dims: Mapping[str, tuple[str | int, ...]], name: str
) -> tuple[str | int, ...]:
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
