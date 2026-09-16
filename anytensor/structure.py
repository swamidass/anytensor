"""Structured-array unwrap / rewrap for portable ops (ragged, …).

Registered types expose ``.values`` and ``.with_values(values)``. Optional
``.same_structure(other)`` checks partition compatibility when several
structured operands appear in one call.

**Index / partition metadata is never fed to the op.** Only ``.values`` is
peeled; ``with_values`` reuses the same ``row_ids`` (no-op on the index
vector). Elementwise results keep structure when the leading length still
matches; reductions / gathers that change that length return dense.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence

_STRUCTURE_TYPES: list[type] = []


def register_structure(cls: type) -> type:
    """Register ``cls`` so :func:`as_array_result` / :func:`promote` peel it."""
    if cls not in _STRUCTURE_TYPES:
        _STRUCTURE_TYPES.append(cls)
    return cls


def unregister_structure(cls: type) -> None:
    """Drop a previously registered structure type (tests / teardown)."""
    try:
        _STRUCTURE_TYPES.remove(cls)
    except ValueError:
        pass


def is_structure(x: Any) -> bool:
    return any(isinstance(x, t) for t in _STRUCTURE_TYPES)


def same_structure(a: Any, b: Any) -> bool:
    """True when ``a`` and ``b`` share a partition (or are the same object)."""
    if a is b:
        return True
    if not (is_structure(a) and is_structure(b)):
        return False
    checker = getattr(a, "same_structure", None)
    if callable(checker):
        return bool(checker(b))
    # Fallback: same type + identical nrows + row_ids object identity.
    return type(a) is type(b) and getattr(a, "nrows", None) == getattr(b, "nrows", None) and (
        getattr(a, "row_ids", None) is getattr(b, "row_ids", None)
    )


def peel(x: Any) -> tuple[Any, Optional[Any]]:
    """Return ``(payload, template)``; ``template`` is None for plain arrays."""
    if is_structure(x):
        return x.values, x
    return x, None


def merge_templates(templates: Sequence[Any]) -> Optional[Any]:
    """Single structure template, or error if partitions disagree."""
    filtered = [t for t in templates if t is not None]
    if not filtered:
        return None
    head = filtered[0]
    for t in filtered[1:]:
        if not same_structure(head, t):
            raise ValueError(
                "structured operands must share the same partition "
                f"(got incompatible {type(head).__name__} values)"
            )
    return head


def rewrap(out: Any, template: Optional[Any]) -> Any:
    """Rewrap ``out`` when it still looks like a flat values vector."""
    if template is None or is_structure(out):
        return out
    values = template.values
    out_shape = getattr(out, "shape", None)
    val_shape = getattr(values, "shape", None)
    if out_shape is None or val_shape is None:
        return out
    if len(out_shape) == 0:
        return out
    if len(val_shape) == 0:
        return out
    # Leading-length match ⇒ elementwise-style result; else dense (reduce/gather).
    if out_shape[0] != val_shape[0]:
        return out
    return template.with_values(out)


def apply_structured(fn: Callable[..., Any], args: tuple, kwargs: dict) -> Any:
    """Peel structures in ``args``/``kwargs``, call ``fn``, rewrap if needed."""
    templates: list[Any] = []
    new_args = []
    for a in args:
        payload, tmpl = peel(a)
        templates.append(tmpl)
        new_args.append(payload)
    new_kwargs = {}
    for k, v in kwargs.items():
        payload, tmpl = peel(v)
        templates.append(tmpl)
        new_kwargs[k] = payload
    template = merge_templates(templates)
    out = fn(*new_args, **new_kwargs)
    if isinstance(out, tuple):
        return tuple(rewrap(o, template) for o in out)
    return rewrap(out, template)
