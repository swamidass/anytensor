"""Nested-structure utilities with the public API of ``jax.tree``.

Pure Python; the only binary dependency is NumPy (no JAX, no C++ pytree
extension). Walking rules follow `jax.tree` / `jax.tree_util`:

- ``None`` is an **empty pytree** (zero leaves), not a leaf.
- ``flatten(tree)`` returns ``(leaves, treedef)``.
- Dicts flatten by **sorted keys**; ``OrderedDict`` keeps insertion order.
- Arrays / tensors are leaves. ``str`` / ``bytes`` / sets / mapping views
  are leaves.

**Custom nodes**

1. Magic flatten (beta)::

       def __tree_flatten__(self):
           return children, aux

       @classmethod
       def __tree_unflatten__(cls, aux, children):
           return cls(...)

2. Concat / split (**stable**; used by :func:`concat` / :func:`split` and jraph
   ``batch`` / ``unbatch``). Checked **before** walking children::

       @classmethod
       def __tree_concat__(cls, xs, axis=0): ...

       def __tree_split__(self, sizes, axis=0): ...

   ``GraphsTuple`` implements these (offset senders/receivers).

3. Already-imported pytree registries (**beta**), looked up by **type**
   (never imported as a side effect): ``jax.tree_util``,
   ``torch.utils._pytree``, and ``optree``. Built-in containers stay on this
   module's path.

Public ``map`` / ``flatten`` / ``concat`` / ``split`` and built-in walking
rules are **stable**. Flatten-style registration (item 1 and item 3) may
change.
"""

from __future__ import annotations

import builtins
import collections
from collections import abc as collections_abc
from functools import partial, reduce as _f_reduce
import inspect
import sys
import warnings
from typing import Any, Iterable, NamedTuple

import numpy as np

__all__ = [
    "DictKey",
    "GetAttrKey",
    "PyTreeDef",
    "SequenceKey",
    "all",
    "concat",
    "flatten",
    "flatten_with_path",
    "leaves",
    "leaves_with_path",
    "map",
    "map_with_path",
    "reduce",
    "split",
    "structure",
    "tree_flatten",
    "tree_flatten_with_path",
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "tree_structure",
    "tree_unflatten",
    "unflatten",
]

_TEXT_OR_BYTES = (str, bytes)
_UNSET = object()


class SequenceKey(NamedTuple):
    """Path entry for a list/tuple child (``jax.tree`` name)."""

    idx: int


class DictKey(NamedTuple):
    """Path entry for a mapping child."""

    key: Any


class GetAttrKey(NamedTuple):
    """Path entry for a namedtuple / attrs field."""

    name: str


class PyTreeDef:
    """Tree structure leftover after flattening (``jax.tree_util.PyTreeDef``-like)."""

    __slots__ = ("kind", "metadata", "children", "_restore")

    def __init__(self, kind: str, metadata, children, restore=None):
        self.kind = kind
        self.metadata = metadata
        self.children = tuple(children)
        self._restore = restore

    @property
    def num_leaves(self) -> int:
        if self.kind == "leaf":
            return 1
        return sum(c.num_leaves for c in self.children)

    def __eq__(self, other):
        if type(other) is not PyTreeDef:
            return NotImplemented
        return (
            self.kind == other.kind
            and self.metadata == other.metadata
            and self.children == other.children
        )

    def __repr__(self) -> str:
        if self.kind == "leaf":
            return "PyTreeDef(*)"
        if self.kind == "none":
            return "PyTreeDef(None)"
        kids = ", ".join(repr(c)[len("PyTreeDef") :] if repr(c).startswith("PyTreeDef") else repr(c) for c in self.children)
        # Keep repr short and stable; exact JAX text is not required.
        return f"PyTreeDef({self.kind}[{kids}])"

    def unflatten(self, leaves):
        leaves = list(leaves)
        out, i = self._unflatten(leaves, 0)
        if i != len(leaves):
            raise ValueError(
                f"Too many leaves for PyTreeDef; expected {self.num_leaves}, got {len(leaves)}"
            )
        return out

    def _unflatten(self, leaves, i):
        if self.kind == "leaf":
            if i >= len(leaves):
                raise ValueError(
                    f"Too few leaves for PyTreeDef; expected {self.num_leaves}, got {len(leaves)}"
                )
            return leaves[i], i + 1
        if self.kind == "none":
            return None, i
        vals = []
        for child in self.children:
            v, i = child._unflatten(leaves, i)
            vals.append(v)
        return _rebuild(self, vals), i


_LEAF = PyTreeDef("leaf", None, ())
_NONE = PyTreeDef("none", None, ())


def _rebuild(td: PyTreeDef, vals):
    kind = td.kind
    if kind == "list":
        return list(vals)
    if kind == "tuple":
        return tuple(vals)
    if kind == "dict":
        return dict(zip(td.metadata, vals))
    if kind == "namedtuple":
        return td.metadata(*vals)
    if kind == "ordereddict":
        return collections.OrderedDict(zip(td.metadata, vals))
    if kind == "defaultdict":
        factory, keys = td.metadata
        return collections.defaultdict(factory, zip(keys, vals))
    if kind == "attrs":
        cls = td.metadata
        names = [a.name for a in cls.__attrs_attrs__]
        return cls(**dict(zip(names, vals)))
    if kind == "magic":
        cls, aux = td.metadata
        return cls.__tree_unflatten__(aux, vals)
    if kind == "registered":
        return td._restore(vals)
    raise ValueError(f"Unknown PyTreeDef kind {kind!r}")  # pragma: no cover


def _is_namedtuple(instance) -> bool:
    if not isinstance(instance, tuple):
        return False
    fields = getattr(type(instance), "_fields", None)
    return isinstance(fields, tuple) and all(isinstance(f, str) for f in fields)


def _has_magic(instance) -> bool:
    cls = type(instance)
    return hasattr(cls, "__tree_flatten__") and hasattr(cls, "__tree_unflatten__")


def _is_fast_array_leaf(x) -> bool:
    """Cheap leaf test so tensors never hit JAX/Torch pytree registries."""
    if x is None or isinstance(x, (*_TEXT_OR_BYTES, bool, int, float, complex, np.generic)):
        return True
    if isinstance(x, np.ndarray):
        return True
    shape = getattr(x, "shape", None)
    dtype = getattr(x, "dtype", None)
    if shape is None or dtype is None:
        return False
    if isinstance(x, (collections_abc.Mapping, collections_abc.Sequence)):
        return False
    return True


def _restore_jax(tree_unflatten, treedef, new_children):
    return tree_unflatten(treedef, list(new_children))


def _restore_torch(unflatten_fn, ctx, new_children):
    return unflatten_fn(list(new_children), ctx)


def _restore_optree(tree_unflatten, treedef, new_children):
    return tree_unflatten(list(new_children), treedef)


def _is_leaf_treedef(flat_leaves, obj) -> bool:
    return len(flat_leaves) == 1 and flat_leaves[0] is obj


def _jax_one_level(obj):
    jtu = sys.modules.get("jax.tree_util")
    tree_flatten = getattr(jtu, "tree_flatten", None) if jtu is not None else None
    tree_unflatten = getattr(jtu, "tree_unflatten", None) if jtu is not None else None
    if tree_flatten is None or tree_unflatten is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Python iterable type .* treated as a leaf",
            )
            flat_leaves, treedef = tree_flatten(obj, is_leaf=lambda x: x is not obj)
    except Exception:
        return None
    if _is_leaf_treedef(flat_leaves, obj):
        return None
    return list(flat_leaves), partial(_restore_jax, tree_unflatten, treedef)


def _torch_node_entry(pytree, cls):
    nodes = getattr(pytree, "SUPPORTED_NODES", None)
    if nodes is None:
        return None
    try:
        return nodes.get(cls)
    except (TypeError, AttributeError):
        return None


def _torch_one_level(obj):
    pytree = sys.modules.get("torch.utils._pytree")
    if pytree is None:
        return None
    spec = _torch_node_entry(pytree, type(obj))
    if spec is None:
        return None
    flatten_fn = getattr(spec, "flatten_fn", None)
    unflatten_fn = getattr(spec, "unflatten_fn", None)
    if flatten_fn is None or unflatten_fn is None:
        return None
    children, ctx = flatten_fn(obj)
    return list(children), partial(_restore_torch, unflatten_fn, ctx)


def _optree_one_level(obj):
    optree = sys.modules.get("optree")
    tree_flatten = getattr(optree, "tree_flatten", None) if optree is not None else None
    tree_unflatten = getattr(optree, "tree_unflatten", None) if optree is not None else None
    if tree_flatten is None or tree_unflatten is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Python iterable type .* treated as a leaf",
            )
            flat_leaves, treedef = tree_flatten(obj, is_leaf=lambda x: x is not obj)
    except Exception:
        return None
    if _is_leaf_treedef(flat_leaves, obj):
        return None
    return list(flat_leaves), partial(_restore_optree, tree_unflatten, treedef)


def _registry_one_level(obj):
    if _is_fast_array_leaf(obj):
        return None
    found = _jax_one_level(obj)
    if found is not None:
        return found
    found = _torch_one_level(obj)
    if found is not None:
        return found
    return _optree_one_level(obj)


def _one_level(node, is_leaf=None):
    """Return ``(kind, metadata, children, restore)`` or ``None`` if ``node`` is a leaf."""
    if is_leaf is not None and is_leaf(node):
        return None
    if node is None:
        return "none", None, (), None
    if _has_magic(node):
        children, aux = node.__tree_flatten__()
        return "magic", (type(node), aux), tuple(children), None
    if type(node) is dict:
        try:
            keys = tuple(sorted(node))
        except TypeError:
            raise TypeError("tree only supports dicts with sortable keys.") from None
        return "dict", keys, tuple(node[k] for k in keys), None
    if type(node) is collections.OrderedDict:
        keys = tuple(node.keys())
        return "ordereddict", keys, tuple(node.values()), None
    if type(node) is collections.defaultdict:
        try:
            keys = tuple(sorted(node))
        except TypeError:
            raise TypeError("tree only supports dicts with sortable keys.") from None
        return "defaultdict", (node.default_factory, keys), tuple(node[k] for k in keys), None
    if _is_namedtuple(node):
        return "namedtuple", type(node), tuple(node), None
    if type(node) is list:
        return "list", None, tuple(node), None
    if type(node) is tuple:
        return "tuple", None, tuple(node), None
    attrs = getattr(type(node), "__attrs_attrs__", None)
    if attrs is not None:
        return "attrs", type(node), tuple(getattr(node, a.name) for a in attrs), None
    if _is_fast_array_leaf(node):
        return None
    custom = _registry_one_level(node)
    if custom is not None:
        children, restore = custom
        return "registered", type(node), tuple(children), restore
    return None


def _child_path_keys(kind, metadata, n: int):
    if kind in ("list", "tuple", "magic", "registered"):
        return tuple(SequenceKey(i) for i in range(n))
    if kind == "dict":
        return tuple(DictKey(k) for k in metadata)
    if kind == "ordereddict":
        return tuple(DictKey(k) for k in metadata)
    if kind == "defaultdict":
        _, keys = metadata
        return tuple(DictKey(k) for k in keys)
    if kind == "namedtuple":
        return tuple(GetAttrKey(name) for name in metadata._fields)
    if kind == "attrs":
        return tuple(GetAttrKey(a.name) for a in metadata.__attrs_attrs__)
    return tuple(SequenceKey(i) for i in range(n))  # pragma: no cover


def _flatten_into(node, acc, is_leaf):
    entry = _one_level(node, is_leaf)
    if entry is None:
        acc.append(node)
        return _LEAF
    kind, metadata, children, restore = entry
    if kind == "none":
        return _NONE
    child_defs = [_flatten_into(c, acc, is_leaf) for c in children]
    return PyTreeDef(kind, metadata, child_defs, restore=restore)


def flatten(tree, is_leaf=None):
    """Flatten ``tree`` into ``(leaves, treedef)``.

    >>> import anytensor.tree as tree
    >>> leaves, _ = tree.flatten((1, (2, 3)))
    >>> tuple(leaves)
    (1, 2, 3)
    >>> empty, _ = tree.flatten(None)
    >>> len(empty)
    0
    """
    acc: list = []
    treedef = _flatten_into(tree, acc, is_leaf)
    return acc, treedef


def unflatten(treedef: PyTreeDef, leaves: Iterable):
    """Rebuild a tree from ``treedef`` and ``leaves``."""
    return treedef.unflatten(leaves)


def leaves(tree, is_leaf=None) -> list:
    """Return the leaves of ``tree`` (``None`` yields ``[]``)."""
    return flatten(tree, is_leaf=is_leaf)[0]


def structure(tree, is_leaf=None) -> PyTreeDef:
    """Return the ``PyTreeDef`` of ``tree``."""
    return flatten(tree, is_leaf=is_leaf)[1]


def map(f, tree, *rest, is_leaf=None):  # noqa: A001
    """Map ``f`` over the leaves of ``tree`` (and ``rest``).

    >>> import anytensor.tree as tree
    >>> tree.map(lambda v: v * 2, {"b": 1, "a": [2, 3]})
    {'a': [4, 6], 'b': 2}
    >>> tree.map(lambda x: x + 1, None) is None
    True
    """
    leaves0, treedef = flatten(tree, is_leaf=is_leaf)
    rest_leaves = []
    for other in rest:
        other_leaves, other_def = flatten(other, is_leaf=is_leaf)
        if other_def != treedef:
            raise ValueError(
                "pytree structure error: trees must have the same structure."
            )
        rest_leaves.append(other_leaves)
    if rest_leaves:
        out = [f(*xs) for xs in zip(leaves0, *rest_leaves)]
    else:
        out = [f(x) for x in leaves0]
    return unflatten(treedef, out)


def _flatten_with_path_into(node, path, acc, is_leaf):
    entry = _one_level(node, is_leaf)
    if entry is None:
        acc.append((path, node))
        return _LEAF
    kind, metadata, children, restore = entry
    if kind == "none":
        return _NONE
    keys = _child_path_keys(kind, metadata, len(children))
    child_defs = [
        _flatten_with_path_into(child, path + (key,), acc, is_leaf)
        for key, child in zip(keys, children)
    ]
    return PyTreeDef(kind, metadata, child_defs, restore=restore)


def flatten_with_path(tree, is_leaf=None):
    """Flatten into ``([(path, leaf), ...], treedef)``."""
    acc: list = []
    treedef = _flatten_with_path_into(tree, (), acc, is_leaf)
    return acc, treedef


def leaves_with_path(tree, is_leaf=None) -> list:
    """Return ``[(path, leaf), ...]``."""
    return flatten_with_path(tree, is_leaf=is_leaf)[0]


def map_with_path(f, tree, *rest, is_leaf=None):
    """Like :func:`map` but ``f`` receives ``(path, *leaves)``."""
    pairs, treedef = flatten_with_path(tree, is_leaf=is_leaf)
    paths = [p for p, _ in pairs]
    leaves0 = [v for _, v in pairs]
    rest_leaves = []
    for other in rest:
        other_pairs, other_def = flatten_with_path(other, is_leaf=is_leaf)
        if other_def != treedef:
            raise ValueError(
                "pytree structure error: trees must have the same structure."
            )
        rest_leaves.append([v for _, v in other_pairs])
    if rest_leaves:
        out = [f(p, *xs) for p, xs in zip(paths, zip(leaves0, *rest_leaves))]
    else:
        out = [f(p, x) for p, x in zip(paths, leaves0)]
    return unflatten(treedef, out)


def all(tree, *, is_leaf=None):  # noqa: A001
    """``True`` if every leaf is truthy (empty trees are ``True``)."""
    return builtins.all(leaves(tree, is_leaf=is_leaf))


def reduce(function, tree, initializer=_UNSET, is_leaf=None):  # noqa: A002
    """Reduce leaves with ``function`` (same empty-tree error as ``functools.reduce``)."""
    xs = leaves(tree, is_leaf=is_leaf)
    if initializer is _UNSET:
        return _f_reduce(function, xs)
    return _f_reduce(function, xs, initializer)


def _accepts_axis(fn) -> bool:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return True
    if "axis" in params:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())


def _call_concat(cls, xs, axis):
    fn = cls.__tree_concat__
    if _accepts_axis(fn):
        return fn(xs, axis=axis)
    return fn(xs)


def _call_split(obj, sizes, axis):
    fn = obj.__tree_split__
    if _accepts_axis(fn):
        return fn(sizes, axis=axis)
    return fn(sizes)


def concat(*structures, axis: int = 0):
    """Concatenate trees along ``axis`` (leading axis by default).

    If the type defines ``__tree_concat__(xs, axis=0)``, that method is used
    and children are **not** walked. Otherwise :func:`map` applies
    :func:`anytensor.concatenate` at the leaves. All-``None`` stays ``None``.

    >>> import numpy as np
    >>> import anytensor.tree as tree
    >>> tree.concat(np.array([1, 2]), np.array([3]))
    array([1, 2, 3])
    """
    if not structures:
        raise ValueError("Must provide at least one structure")
    return _concat_impl(structures, axis=axis)


def _concat_impl(xs, axis: int = 0):
    first = xs[0]
    if first is not None and hasattr(type(first), "__tree_concat__"):
        return _call_concat(type(first), xs, axis)
    if builtins.all(x is None for x in xs):
        return None
    entry = _one_level(first)
    if entry is None:
        for x in xs[1:]:
            if x is None or _one_level(x) is not None:
                raise ValueError(
                    "pytree structure error: trees must have the same structure."
                )
        from anytensor.core import concatenate

        return concatenate(xs, axis=axis)
    kind, metadata, children, restore = entry
    other_entries = [_one_level(x) for x in xs[1:]]
    for other_entry in other_entries:
        if other_entry is None or other_entry[0] != kind or other_entry[1] != metadata:
            raise ValueError(
                "pytree structure error: trees must have the same structure."
            )
        if len(other_entry[2]) != len(children):
            raise ValueError(
                "pytree structure error: trees must have the same structure."
            )
    packed = [
        _concat_impl([children[i], *[e[2][i] for e in other_entries]], axis=axis)
        for i in range(len(children))
    ]
    td = PyTreeDef(kind, metadata, [_LEAF] * len(children), restore=restore)
    return _rebuild(td, packed)


def split(structure, sizes, axis: int = 0):
    """Split ``structure`` along ``axis`` into pieces of leading lengths ``sizes``.

    ``__tree_split__(sizes, axis=0)`` on the object wins. Nested containers
    recurse. ``None`` yields ``[None] * len(sizes)``.

    >>> import numpy as np
    >>> import anytensor.tree as tree
    >>> head, tail = tree.split(np.arange(1, 5), (1, 3))
    >>> tuple(int(x) for x in head), tuple(int(x) for x in tail)
    ((1,), (2, 3, 4))
    """
    sizes = tuple(int(s) for s in sizes)
    return _split_impl(structure, sizes, axis=axis)


def _split_impl(structure, sizes, axis: int = 0):
    n_parts = len(sizes)
    if structure is not None and hasattr(type(structure), "__tree_split__"):
        return list(_call_split(structure, sizes, axis))
    if structure is None:
        return [None] * n_parts
    entry = _one_level(structure)
    if entry is None:
        return _split_array(structure, sizes, axis)
    kind, metadata, children, restore = entry
    child_splits = [_split_impl(child, sizes, axis=axis) for child in children]
    td = PyTreeDef(kind, metadata, [_LEAF] * len(children), restore=restore)
    return [_rebuild(td, [parts[i] for parts in child_splits]) for i in range(n_parts)]


def _split_array(array, sizes, axis: int = 0):
    ndim = int(getattr(array, "ndim", 1))
    ax = axis if axis >= 0 else axis + ndim
    out = []
    start = 0
    for n in sizes:
        if ax == 0:
            out.append(array[start : start + n])
        else:
            sl = [slice(None)] * ndim
            sl[ax] = slice(start, start + n)
            out.append(array[tuple(sl)])
        start += n
    return out


tree_flatten = flatten
tree_unflatten = unflatten
tree_leaves = leaves
tree_structure = structure
tree_map = map
tree_map_with_path = map_with_path
tree_flatten_with_path = flatten_with_path
