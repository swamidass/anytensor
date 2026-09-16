"""Nested-structure utilities with the public API of ``dm-tree``.

This is a pure-Python implementation of DeepMind `tree` (`import tree`,
package `dm-tree`) so AnyTensor does not take a C++ runtime dependency.
Functions, argument names, defaults, and nest-walking rules follow dm-tree
unless noted.

**Leaves vs nests.** A value is nested iff it is a mapping, a sequence
(not ``str`` / ``bytes``), a namedtuple, an attrs instance, or a *custom*
node (below). Everything else is a leaf — including ``None``, numbers,
NumPy arrays, and framework tensors (they are not ``Sequence``). Dicts
flatten by **sorted keys** (``OrderedDict`` insertion order is ignored).

**Custom nodes** (intentional extension; dm-tree has no user registry):

1. Magic methods on the type (highest priority)::

       def __tree_flatten__(self) -> tuple[Iterable[Any], Any]:
           return children, aux

       @classmethod
       def __tree_unflatten__(cls, aux, children):
           return cls(...)

   Same child/aux convention as JAX ``register_pytree_node``.

2. Concat / split (used by :func:`concat` / :func:`split` and by jraph
   ``batch`` / ``unbatch``). Checked **before** recursing into a nest, so an
   object can own join/partition even if it is also a namedtuple or mapping::

       @classmethod
       def __tree_concat__(cls, xs, axis=0):
           ...

       def __tree_split__(self, sizes, axis=0):
           ...

   ``GraphsTuple`` implements these (offset senders/receivers). Feature
   containers that are not plain arrays should too.

3. Already-imported pytree registries, looked up by **type** (never imported
   as a side effect): ``jax.tree_util``, ``torch.utils._pytree``, and
   ``optree``. Standard containers (list/dict/namedtuple/…) stay on the
   dm-tree path even if a backend also treats them as pytrees.

``traverse_with_path`` exists on the module like dm-tree but is not in
``__all__``.
"""

from __future__ import annotations

import collections
from collections import abc as collections_abc
from functools import partial
import inspect
import logging
import sys
import types
from typing import Any, Iterable, Mapping, Sequence, TypeVar, Union

import numpy as np

__all__ = [
    "is_nested",
    "assert_same_structure",
    "unflatten_as",
    "flatten",
    "flatten_up_to",
    "flatten_with_path",
    "flatten_with_path_up_to",
    "map_structure",
    "map_structure_up_to",
    "map_structure_with_path",
    "map_structure_with_path_up_to",
    "traverse",
    "MAP_TO_NONE",
    "concat",
    "split",
]

_TEXT_OR_BYTES = (str, bytes)

_SHALLOW_TREE_HAS_INVALID_KEYS = (
    "The shallow_tree's keys are not a subset of the input_tree's keys. The "
    "shallow_tree has the following keys that are not in the input_tree: {}."
)

_STRUCTURES_HAVE_MISMATCHING_TYPES = (
    "The two structures don't have the same sequence type. Input structure has "
    "type {input_type}, while shallow structure has type {shallow_type}."
)

_STRUCTURES_HAVE_MISMATCHING_LENGTHS = (
    "The two structures don't have the same sequence length. Input "
    "structure has length {input_length}, while shallow structure has length "
    "{shallow_length}."
)

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input has type: {}."
)

_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH = (
    "If shallow structure is a sequence, input must also be a sequence. "
    "Input at path: {path} has type: {input_type}."
)

K = TypeVar("K")
V = TypeVar("V")

StructureKV = Union[
    Sequence["StructureKV[K, V]"],
    Mapping[K, "StructureKV[K, V]"],
    V,
]
Structure = StructureKV[str, V]


def _sorted(dictionary):
    """Sorted dict keys; dm-tree error if keys are not sortable."""
    try:
        return sorted(dictionary)
    except TypeError:
        raise TypeError("tree only supports dicts with sortable keys.") from None


def _is_namedtuple(instance) -> bool:
    """True iff ``instance`` is a namedtuple (same rules as dm-tree)."""
    if not isinstance(instance, tuple):
        return False
    fields = getattr(type(instance), "_fields", None)
    return isinstance(fields, tuple) and all(isinstance(f, str) for f in fields)


def _is_attrs(instance) -> bool:
    return getattr(type(instance), "__attrs_attrs__", None) is not None


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
    # Containers with shape/dtype (rare) must not look like tensors.
    if isinstance(x, (collections_abc.Mapping, collections_abc.Sequence)):
        return False
    return True


def _get_attrs_items(obj):
    return [(attr.name, getattr(obj, attr.name)) for attr in obj.__class__.__attrs_attrs__]


def _same_namedtuples(a, b) -> bool:
    return type(a).__name__ == type(b).__name__ and getattr(a, "_fields", None) == getattr(
        b, "_fields", None
    )


def _magic_one_level(obj):
    """Return ``(children, restore)`` for magic-method nodes, else ``None``."""
    if not _has_magic(obj):
        return None
    children, aux = obj.__tree_flatten__()
    children = list(children)
    return children, partial(_restore_magic, type(obj), aux)


def _registry_one_level(obj):
    """One-level flatten via jax / torch / optree if those modules are loaded."""
    if _is_fast_array_leaf(obj):
        return None
    found = _jax_one_level(obj)
    if found is not None:
        return found
    found = _torch_one_level(obj)
    if found is not None:
        return found
    return _optree_one_level(obj)


def _restore_jax(tree_unflatten, treedef, new_children):
    return tree_unflatten(treedef, list(new_children))


def _restore_torch(unflatten_fn, ctx, new_children):
    return unflatten_fn(list(new_children), ctx)


def _restore_optree(tree_unflatten, treedef, new_children):
    return tree_unflatten(list(new_children), treedef)


def _restore_magic(cls, aux, new_children):
    return cls.__tree_unflatten__(aux, list(new_children))


def _is_leaf_treedef(leaves, obj) -> bool:
    return len(leaves) == 1 and leaves[0] is obj


def _jax_one_level(obj):
    jtu = sys.modules.get("jax.tree_util")
    if jtu is None:
        return None
    tree_flatten = getattr(jtu, "tree_flatten", None)
    tree_unflatten = getattr(jtu, "tree_unflatten", None)
    if tree_flatten is None or tree_unflatten is None:
        return None
    try:
        leaves, treedef = tree_flatten(obj, is_leaf=lambda x: x is not obj)
    except Exception:
        return None
    if _is_leaf_treedef(leaves, obj):
        return None
    return list(leaves), partial(_restore_jax, tree_unflatten, treedef)


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
    if optree is None:
        return None
    tree_flatten = getattr(optree, "tree_flatten", None)
    tree_unflatten = getattr(optree, "tree_unflatten", None)
    if tree_flatten is None or tree_unflatten is None:
        return None
    try:
        leaves, treedef = tree_flatten(obj, is_leaf=lambda x: x is not obj)
    except Exception:
        return None
    if _is_leaf_treedef(leaves, obj):
        return None
    return list(leaves), partial(_restore_optree, tree_unflatten, treedef)


def _custom_one_level(obj):
    found = _magic_one_level(obj)
    if found is not None:
        return found
    return _registry_one_level(obj)


def is_nested(structure) -> bool:
    """Checks if a given structure is nested.

    >>> import anytensor.tree as tree
    >>> tree.is_nested(42)
    False
    >>> tree.is_nested({"foo": 42})
    True

    Args:
        structure: A structure to check.

    Returns:
        True if a given structure is nested (sequence, mapping, namedtuple,
        attrs, magic-method node, or registered pytree node) and False
        otherwise.
    """
    if isinstance(structure, _TEXT_OR_BYTES):
        return False
    if _has_magic(structure):
        return True
    if isinstance(structure, collections_abc.Mapping):
        return True
    if isinstance(structure, collections_abc.MappingView):
        return True
    if _is_attrs(structure) or _is_namedtuple(structure):
        return True
    if isinstance(structure, collections_abc.Sequence):
        return True
    if _is_fast_array_leaf(structure):
        return False
    return _custom_one_level(structure) is not None


def _yield_sorted_items(iterable):
    """Yield ``(key, value)`` pairs in dm-tree order."""
    custom = _custom_one_level(iterable) if not _is_standard_container(iterable) else None
    if custom is not None:
        children, _ = custom
        for i, child in enumerate(children):
            yield i, child
        return
    if isinstance(iterable, collections_abc.Mapping):
        for key in _sorted(iterable):
            yield key, iterable[key]
    elif _is_attrs(iterable):
        yield from _get_attrs_items(iterable)
    elif _is_namedtuple(iterable):
        for field in iterable._fields:
            yield field, getattr(iterable, field)
    else:
        yield from enumerate(iterable)


def _is_standard_container(obj) -> bool:
    if _has_magic(obj):
        return False
    if isinstance(obj, _TEXT_OR_BYTES):
        return False
    if isinstance(obj, collections_abc.Mapping):
        return True
    if isinstance(obj, collections_abc.MappingView):
        return True
    if _is_attrs(obj) or _is_namedtuple(obj):
        return True
    return isinstance(obj, collections_abc.Sequence)


def _yield_value(iterable):
    for _, v in _yield_sorted_items(iterable):
        yield v


def _num_elements(structure) -> int:
    if _is_attrs(structure):
        return len(getattr(structure.__class__, "__attrs_attrs__"))
    custom = _custom_one_level(structure) if not _is_standard_container(structure) else None
    if custom is not None:
        return len(custom[0])
    return len(structure)


def _sequence_like(instance, args):
    """Pack ``args`` into the same type as ``instance`` (dm-tree + custom)."""
    args = list(args)
    custom = _custom_one_level(instance) if not _is_standard_container(instance) else None
    if custom is not None:
        _, restore = custom
        return restore(args)
    if isinstance(instance, (dict, collections_abc.Mapping)):
        result = dict(zip(_sorted(instance), args))
        keys_and_values = ((key, result[key]) for key in instance)
        if isinstance(instance, collections.defaultdict):
            return type(instance)(instance.default_factory, keys_and_values)
        if isinstance(instance, types.MappingProxyType):
            return type(instance)(dict(keys_and_values))
        return type(instance)(keys_and_values)
    if isinstance(instance, collections_abc.MappingView):
        return list(args)
    if _is_namedtuple(instance) or _is_attrs(instance):
        instance_type = type(instance)
        try:
            if _is_attrs(instance):
                return instance_type(
                    **{attr.name: arg for attr, arg in zip(instance_type.__attrs_attrs__, args)}
                )
            return instance_type(*args)
        except Exception as exc:
            raise TypeError(f"Couldn't traverse {instance!r} with arguments {args}") from exc
    return type(instance)(args)


def flatten(structure) -> list:
    """Flattens a possibly nested structure into a list.

    >>> import anytensor.tree as tree
    >>> tree.flatten([[1, 2, 3], [4, [5], [[6]]]])
    [1, 2, 3, 4, 5, 6]
    >>> tree.flatten(None)
    [None]
    >>> tree.flatten(1)
    [1]
    >>> tree.flatten({100: 'world!', 6: 'Hello'})
    ['Hello', 'world!']

    Args:
        structure: An arbitrarily nested structure.

    Returns:
        A list, the flattened version of the input ``structure``.

    Raises:
        TypeError: If ``structure`` is or contains a mapping with non-sortable keys.
    """
    if not is_nested(structure):
        return [structure]
    leaves: list = []
    for v in _yield_value(structure):
        leaves.extend(flatten(v))
    return leaves


class _DotString:
    def __str__(self):
        return "."

    def __repr__(self):
        return "."


_DOT = _DotString()


def _structures_match(a, b, check_types: bool) -> None:
    if is_nested(a) != is_nested(b):
        raise ValueError(
            "The two structures don't have the same nested structure."
        )
    if not is_nested(a):
        return
    if check_types and type(a) is not type(b):
        a_nt = _is_namedtuple(a)
        b_nt = _is_namedtuple(b)
        if a_nt and b_nt:
            if not _same_namedtuples(a, b):
                raise TypeError(
                    _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                        input_type=type(b), shallow_type=type(a)
                    )
                )
        elif not (
            isinstance(a, collections_abc.Mapping) and isinstance(b, collections_abc.Mapping)
        ):
            raise TypeError(
                _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                    input_type=type(b), shallow_type=type(a)
                )
            )
    if _num_elements(a) != _num_elements(b):
        raise ValueError(
            "The two structures don't have the same nested structure."
        )
    items_a = list(_yield_sorted_items(a))
    items_b = list(_yield_sorted_items(b))
    keys_a = [k for k, _ in items_a]
    keys_b = [k for k, _ in items_b]
    if keys_a != keys_b:
        raise ValueError(
            "The two structures don't have the same nested structure."
        )
    for (_, va), (_, vb) in zip(items_a, items_b):
        _structures_match(va, vb, check_types)


def assert_same_structure(a, b, check_types: bool = True) -> None:
    """Asserts that two structures are nested in the same way.

    >>> import anytensor.tree as tree
    >>> tree.assert_same_structure([(0, 1)], [(2, 3)])

    Namedtuples with identical name and fields match even with
    ``check_types=True``. Different namedtuple names do not.

    Args:
        a: an arbitrarily nested structure.
        b: an arbitrarily nested structure.
        check_types: if True (default) types of sequences are checked as well,
            including the keys of dictionaries. If False, a list and a tuple of
            the same size look the same.

    Raises:
        ValueError: If the two structures do not have the same nested layout.
        TypeError: If ``check_types`` is True and sequence types differ.
    """
    try:
        _structures_match(a, b, check_types)
    except (ValueError, TypeError) as exc:
        str1 = str(map_structure(lambda _: _DOT, a))
        str2 = str(map_structure(lambda _: _DOT, b))
        raise type(exc)(
            "%s\nEntire first structure:\n%s\nEntire second structure:\n%s" % (exc, str1, str2)
        ) from None


def _packed_nest_with_indices(structure, flat, index):
    packed = []
    for s in _yield_value(structure):
        if is_nested(s):
            new_index, child = _packed_nest_with_indices(s, flat, index)
            packed.append(_sequence_like(s, child))
            index = new_index
        else:
            packed.append(flat[index])
            index += 1
    return index, packed


def unflatten_as(structure, flat_sequence):
    """Unflattens a sequence into a given structure.

    >>> import anytensor.tree as tree
    >>> tree.unflatten_as([[1, 2], [[3], [4]]], [5, 6, 7, 8])
    [[5, 6], [[7], [8]]]
    >>> tree.unflatten_as(None, [1])
    1
    >>> tree.unflatten_as({1: None, 2: None}, ['Hello', 'world!'])
    {1: 'Hello', 2: 'world!'}

    Args:
        structure: Arbitrarily nested structure (the template).
        flat_sequence: Sequence to unflatten.

    Returns:
        ``flat_sequence`` packed into ``structure``.

    Raises:
        ValueError: If ``flat_sequence`` and ``structure`` have different
            element counts.
        TypeError: If ``flat_sequence`` is not a sequence, or ``structure``
            contains a mapping with non-sortable keys.
    """
    if not is_nested(flat_sequence):
        raise TypeError(
            "flat_sequence must be a sequence not a {}:\n{}".format(
                type(flat_sequence), flat_sequence
            )
        )
    if not is_nested(structure):
        if len(flat_sequence) != 1:
            raise ValueError(
                "Structure is a scalar but len(flat_sequence) == %d > 1" % len(flat_sequence)
            )
        return flat_sequence[0]
    flat_structure = flatten(structure)
    if len(flat_structure) != len(flat_sequence):
        raise ValueError(
            "Could not pack sequence. Structure had %d elements, but "
            "flat_sequence had %d elements. Structure: %s, flat_sequence: %s."
            % (len(flat_structure), len(flat_sequence), structure, flat_sequence)
        )
    _, packed = _packed_nest_with_indices(structure, list(flat_sequence), 0)
    return _sequence_like(structure, packed)


def map_structure(func, *structures, **kwargs):
    """Maps ``func`` through given structures.

    >>> import anytensor.tree as tree
    >>> structure = [[1], [2], [3]]
    >>> tree.map_structure(lambda v: v**2, structure)
    [[1], [4], [9]]
    >>> tree.map_structure(lambda x, y: x * y, structure, structure)
    [[1], [4], [9]]

    Args:
        func: A callable that accepts as many arguments as there are structures.
        *structures: Arbitrarily nested structures of the same layout.
        **kwargs: The only valid keyword argument is ``check_types``. If True
            (default) component types must match (list vs tuple is an error).
            Namedtuples with identical name and fields are the same type.

    Returns:
        A new structure with the same layout as ``structures[0]``.

    Raises:
        TypeError: If ``func`` is not callable or layouts / types mismatch.
        ValueError: If no structures were given or an unknown keyword is passed.
    """
    if not callable(func):
        raise TypeError("func must be callable, got: %s" % func)
    if not structures:
        raise ValueError("Must provide at least one structure")
    check_types = kwargs.pop("check_types", True)
    if kwargs:
        raise ValueError(
            "Only valid keyword arguments are `check_types` not: `%s`"
            % ("`, `".join(kwargs.keys()))
        )
    for other in structures[1:]:
        assert_same_structure(structures[0], other, check_types=check_types)
    return unflatten_as(
        structures[0], [func(*args) for args in zip(*map(flatten, structures))]
    )


def map_structure_with_path(func, *structures, **kwargs):
    """Maps ``func`` through structures, passing a path as the first argument.

    >>> import anytensor.tree as tree
    >>> tree.map_structure_with_path(
    ...     lambda path, v: (path, v**2), [{"foo": 42}])
    [{'foo': ((0, 'foo'), 1764)}]

    Args:
        func: Callable ``(path, *leaves)``. ``path`` is a tuple of keys/indices.
        *structures: Nested structures of the same layout.
        **kwargs: Only ``check_types`` (default True).

    Returns:
        A new structure with the layout of ``structures[0]``.
    """
    if not structures:
        raise ValueError("Must provide at least one structure")
    return map_structure_with_path_up_to(structures[0], func, *structures, **kwargs)


def _yield_flat_up_to(shallow_tree, input_tree, path=()):
    if not is_nested(shallow_tree):
        yield (path, input_tree)
        return
    input_tree = dict(_yield_sorted_items(input_tree))
    for shallow_key, shallow_subtree in _yield_sorted_items(shallow_tree):
        subpath = path + (shallow_key,)
        input_subtree = input_tree[shallow_key]
        yield from _yield_flat_up_to(shallow_subtree, input_subtree, path=subpath)


def _multiyield_flat_up_to(shallow_tree, *input_trees):
    zipped_iterators = zip(
        *[_yield_flat_up_to(shallow_tree, input_tree) for input_tree in input_trees]
    )
    try:
        for paths_and_values in zipped_iterators:
            paths, values = zip(*paths_and_values)
            yield paths[:1] + values
    except KeyError as exc:
        paths = locals().get("paths", ((),))
        raise ValueError(
            f"Could not find key '{exc.args[0]}' in some `input_trees`. "
            "Please ensure the structure of all `input_trees` are "
            "compatible with `shallow_tree`. The last valid path "
            f"yielded was {paths[0]}."
        ) from exc


def _assert_shallow_structure(shallow_tree, input_tree, path=None, check_types=True):
    if not is_nested(shallow_tree):
        return
    if not is_nested(input_tree):
        if path is not None:
            raise TypeError(
                _IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ_WITH_PATH.format(
                    path=list(path), input_type=type(input_tree)
                )
            )
        raise TypeError(_IF_SHALLOW_IS_SEQ_INPUT_MUST_BE_SEQ.format(type(input_tree)))

    shallow_type = type(shallow_tree)
    if check_types and not isinstance(input_tree, shallow_type):
        shallow_is_namedtuple = _is_namedtuple(shallow_tree)
        input_is_namedtuple = _is_namedtuple(input_tree)
        if shallow_is_namedtuple and input_is_namedtuple:
            if not _same_namedtuples(shallow_tree, input_tree):
                raise TypeError(
                    _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                        input_type=type(input_tree), shallow_type=shallow_type
                    )
                )
        elif not (
            isinstance(shallow_tree, collections_abc.Mapping)
            and isinstance(input_tree, collections_abc.Mapping)
        ):
            raise TypeError(
                _STRUCTURES_HAVE_MISMATCHING_TYPES.format(
                    input_type=type(input_tree), shallow_type=shallow_type
                )
            )

    if _num_elements(input_tree) != _num_elements(shallow_tree):
        raise ValueError(
            _STRUCTURES_HAVE_MISMATCHING_LENGTHS.format(
                input_length=_num_elements(input_tree),
                shallow_length=_num_elements(shallow_tree),
            )
        )

    shallow_iter = _yield_sorted_items(shallow_tree)
    input_items = dict(_yield_sorted_items(input_tree))

    for shallow_key, shallow_branch in shallow_iter:
        if shallow_key not in input_items:
            raise ValueError(_SHALLOW_TREE_HAS_INVALID_KEYS.format([shallow_key]))
        _assert_shallow_structure(
            shallow_branch,
            input_items[shallow_key],
            path + (shallow_key,) if path is not None else None,
            check_types=check_types,
        )


def flatten_up_to(shallow_structure, input_structure, check_types: bool = True) -> list:
    """Flattens ``input_structure`` up to ``shallow_structure``.

    >>> import anytensor.tree as tree
    >>> structure = [[1, 1], [2, 2]]
    >>> tree.flatten_up_to([None, None], structure)
    [[1, 1], [2, 2]]
    >>> tree.flatten_up_to([None, [None, None]], structure)
    [[1, 1], 2, 2]
    >>> tree.flatten_up_to(42, 1)
    [1]
    >>> tree.flatten_up_to(42, [1, 2, 3])
    [[1, 2, 3]]

    Args:
        shallow_structure: Layout to stop flattening at (leaves of this tree
            keep the corresponding ``input_structure`` subtree intact).
        input_structure: An arbitrarily nested structure.
        check_types: If True, corresponding nodes must have the same type.

    Returns:
        A list, the partially flattened version of ``input_structure``.
    """
    _assert_shallow_structure(
        shallow_structure, input_structure, path=None, check_types=check_types
    )
    return [v for _, v in _yield_flat_up_to(shallow_structure, input_structure)]


def flatten_with_path_up_to(shallow_structure, input_structure, check_types: bool = True):
    """``flatten_up_to`` plus paths: a list of ``(path, item)`` pairs."""
    _assert_shallow_structure(
        shallow_structure, input_structure, path=(), check_types=check_types
    )
    return list(_yield_flat_up_to(shallow_structure, input_structure))


def map_structure_up_to(shallow_structure, func, *structures, **kwargs):
    """Maps ``func`` through ``structures`` only as deep as ``shallow_structure``.

    >>> import anytensor.tree as tree
    >>> structure = [[1, 1], [2, 2]]
    >>> tree.map_structure_up_to([None, None], len, structure)
    [2, 2]
    >>> tree.map_structure_up_to([None, [None, None]], str, structure)
    ['[1, 1]', ['2', '2']]
    """
    return map_structure_with_path_up_to(
        shallow_structure, lambda _, *args: func(*args), *structures, **kwargs
    )


def map_structure_with_path_up_to(shallow_structure, func, *structures, **kwargs):
    """``map_structure_up_to`` with a path as ``func``'s first argument."""
    if "check_types" in kwargs:
        logging.warning(
            "The use of `check_types` is deprecated and does not have any effect."
        )
    del kwargs
    results = []
    for path_and_values in _multiyield_flat_up_to(shallow_structure, *structures):
        results.append(func(*path_and_values))
    return unflatten_as(shallow_structure, results)


def flatten_with_path(structure) -> list:
    """Flattens into ``(path, item)`` pairs.

    >>> import anytensor.tree as tree
    >>> tree.flatten_with_path([{"foo": 42}])
    [((0, 'foo'), 42)]
    """
    return list(_yield_flat_up_to(structure, structure))


MAP_TO_NONE = object()
"""Special value for use with :func:`traverse` (replace a subtree by ``None``)."""


def traverse(fn, structure, top_down: bool = True):
    """Traverses the nested structure, applying ``fn`` depth-first.

    If ``top_down`` is True (default), parents are visited before children.
    Returning ``None`` from ``fn`` continues into the subtree; any other value
    replaces that subtree (use :data:`MAP_TO_NONE` to replace with ``None``).

    >>> import anytensor.tree as tree
    >>> visited = []
    >>> tree.traverse(visited.append, [(1, 2), [3], {"a": 4}], top_down=True)
    [(1, 2), [3], {'a': 4}]
    >>> visited
    [[(1, 2), [3], {'a': 4}], (1, 2), 1, 2, [3], 3, {'a': 4}, 4]
    """
    return traverse_with_path(lambda _, x: fn(x), structure, top_down=top_down)


def traverse_with_path(fn, structure, top_down: bool = True):
    """Like :func:`traverse` but ``fn`` receives ``(path, subtree)``."""

    def traverse_impl(path, node):
        def subtree_fn(item):
            subtree_path, subtree = item
            return traverse_impl(path + (subtree_path,), subtree)

        def traverse_subtrees():
            if is_nested(node):
                return _sequence_like(
                    node, map(subtree_fn, _yield_sorted_items(node))
                )
            return node

        if top_down:
            ret = fn(path, node)
            if ret is None:
                return traverse_subtrees()
            if ret is MAP_TO_NONE:
                return None
            return ret
        traversed_structure = traverse_subtrees()
        ret = fn(path, traversed_structure)
        if ret is None:
            return traversed_structure
        if ret is MAP_TO_NONE:
            return None
        return ret

    return traverse_impl((), structure)


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
    """Concatenate structures along ``axis`` (leading axis by default).

    Extension beyond dm-tree. At each node, if the type defines
    ``__tree_concat__(xs, axis=0)``, that method is used and children are
    **not** walked. Otherwise mappings/sequences/namedtuples recurse, and
    array leaves go through :func:`anytensor.concatenate`. All-``None``
    (jraph empty features) stays ``None``; ``None`` siblings are dropped.

    >>> import numpy as np
    >>> import anytensor.tree as tree
    >>> tree.concat(np.array([1, 2]), np.array([3]))
    array([1, 2, 3])
    """
    if not structures:
        raise ValueError("Must provide at least one structure")
    return _concat_impl(structures, axis=axis)


def _concat_impl(xs, axis: int = 0):
    if all(x is None for x in xs):
        return None
    present = [x for x in xs if x is not None]
    first = present[0]
    if hasattr(type(first), "__tree_concat__"):
        return _call_concat(type(first), present, axis)
    if is_nested(first):
        for other in present[1:]:
            assert_same_structure(first, other, check_types=True)
        child_lists = [list(_yield_value(x)) for x in present]
        packed = [
            _concat_impl([children[i] for children in child_lists], axis=axis)
            for i in range(len(child_lists[0]))
        ]
        return _sequence_like(first, packed)
    from anytensor.core import concatenate

    return concatenate(present, axis=axis)


def split(structure, sizes, axis: int = 0):
    """Split ``structure`` along ``axis`` into pieces of leading lengths ``sizes``.

    Extension beyond dm-tree. ``__tree_split__(sizes, axis=0)`` on the object
    wins (no child walk). Nested containers recurse and are rebuilt per piece.
    ``None`` yields ``[None] * len(sizes)``.

    >>> import numpy as np
    >>> import anytensor.tree as tree
    >>> [list(x) for x in tree.split(np.array([1, 2, 3, 4]), [1, 3])]
    [[1], [2, 3, 4]]
    """
    sizes = tuple(int(s) for s in sizes)
    return _split_impl(structure, sizes, axis=axis)


def _split_impl(structure, sizes, axis: int = 0):
    n_parts = len(sizes)
    if structure is None:
        return [None] * n_parts
    if hasattr(type(structure), "__tree_split__"):
        parts = _call_split(structure, sizes, axis)
        return list(parts)
    if is_nested(structure):
        child_splits = [
            _split_impl(child, sizes, axis=axis) for child in _yield_value(structure)
        ]
        return [
            _sequence_like(structure, [parts[i] for parts in child_splits])
            for i in range(n_parts)
        ]
    return _split_array(structure, sizes, axis)


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

