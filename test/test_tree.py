"""Tests for :mod:`anytensor.tree` (dm-tree API + custom nodes)."""

from __future__ import annotations

import collections
import sys
import types
from collections import namedtuple
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import anytensor.tree as tree


Foo = namedtuple("Foo", ["a", "b"])
AlsoFoo = namedtuple("Foo", ["a", "b"])
Bar = namedtuple("Bar", ["a", "b"])


class Pair:
    """Custom nest via magic methods."""

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return type(other) is Pair and self.x == other.x and self.y == other.y

    def __tree_flatten__(self):
        return (self.x, self.y), None

    @classmethod
    def __tree_unflatten__(cls, aux, children):
        del aux
        return cls(*children)


class _FakeAttr:
    def __init__(self, name):
        self.name = name


class AttrPair:
    __attrs_attrs__ = (_FakeAttr("x"), _FakeAttr("y"))

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return type(other) is AttrPair and self.x == other.x and self.y == other.y


class RegistryPair:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return type(other) is type(self) and self.x == other.x and self.y == other.y


def test_is_nested_scalars_and_containers():
    assert tree.is_nested(42) is False
    assert tree.is_nested(None) is False
    assert tree.is_nested("hi") is False
    assert tree.is_nested(b"hi") is False
    assert tree.is_nested({"foo": 42}) is True
    assert tree.is_nested([42]) is True
    assert tree.is_nested((42,)) is True
    assert tree.is_nested([]) is True
    assert tree.is_nested({1, 2}) is False


def test_numpy_and_tensor_like_are_leaves():
    arr = np.arange(3)
    assert tree.is_nested(arr) is False
    out = tree.flatten(arr)
    assert len(out) == 1 and out[0] is arr
    mapped = tree.map_structure(lambda v: v * 2, arr)
    np.testing.assert_array_equal(mapped, np.arange(3) * 2)

    class FakeTensor:
        shape = (2, 3)
        dtype = np.float32

    assert tree.is_nested(FakeTensor()) is False
    assert tree._is_fast_array_leaf(FakeTensor()) is True

    class SeqWithShape(list):
        shape = (2,)
        dtype = np.float32

    class MapWithShape(dict):
        shape = (2,)
        dtype = np.float32

    assert tree._is_fast_array_leaf(SeqWithShape([1, 2])) is False
    assert tree._is_fast_array_leaf(MapWithShape(a=1)) is False
    assert tree._registry_one_level(arr) is None
    assert tree._registry_one_level(FakeTensor()) is None


def test_flatten_unflatten_docs():
    assert tree.flatten([[1, 2, 3], [4, [5], [[6]]]]) == [1, 2, 3, 4, 5, 6]
    assert tree.flatten(None) == [None]
    assert tree.flatten(1) == [1]
    assert tree.flatten({100: "world!", 6: "Hello"}) == ["Hello", "world!"]
    assert tree.unflatten_as([[1, 2], [[3], [4]]], [5, 6, 7, 8]) == [[5, 6], [[7], [8]]]
    assert tree.unflatten_as(None, [1]) == 1
    assert tree.unflatten_as({1: None, 2: None}, ["Hello", "world!"]) == {
        1: "Hello",
        2: "world!",
    }


def test_map_structure_docs():
    structure = [[1], [2], [3]]
    assert tree.map_structure(lambda v: v**2, structure) == [[1], [4], [9]]
    assert tree.map_structure(lambda x, y: x * y, structure, structure) == [[1], [4], [9]]
    nt = Foo(a=1, b=2)
    assert tree.map_structure(lambda v: v * 2, nt) == Foo(a=2, b=4)


def test_flatten_with_path_and_up_to():
    assert tree.flatten_with_path([{"foo": 42}]) == [((0, "foo"), 42)]
    structure = [[1, 1], [2, 2]]
    assert tree.flatten_up_to([None, None], structure) == [[1, 1], [2, 2]]
    assert tree.flatten_up_to([None, [None, None]], structure) == [[1, 1], 2, 2]
    assert tree.flatten_up_to(42, 1) == [1]
    assert tree.flatten_up_to(42, [1, 2, 3]) == [[1, 2, 3]]
    assert tree.map_structure_up_to([None, None], len, structure) == [2, 2]
    assert tree.map_structure_up_to([None, [None, None]], str, structure) == [
        "[1, 1]",
        ["2", "2"],
    ]
    pairs = tree.flatten_with_path_up_to([None, None], structure)
    assert pairs == [((0,), [1, 1]), ((1,), [2, 2])]


def test_map_structure_with_path():
    out = tree.map_structure_with_path(lambda path, v: (path, v**2), [{"foo": 42}])
    assert out == [{"foo": ((0, "foo"), 1764)}]


def test_namedtuple_same_name_and_mismatch():
    tree.assert_same_structure(Foo(0, 1), AlsoFoo(2, 3))
    with pytest.raises(TypeError, match="same nested structure|sequence type"):
        tree.assert_same_structure(Foo(0, 1), Bar(2, 3))


def test_assert_same_structure_ok_and_list_tuple():
    tree.assert_same_structure([(0, 1)], [(2, 3)])
    tree.assert_same_structure([1, 2], (1, 2), check_types=False)
    with pytest.raises(TypeError):
        tree.assert_same_structure([1, 2], (1, 2), check_types=True)
    with pytest.raises(ValueError):
        tree.assert_same_structure([1], [1, 2])
    with pytest.raises(ValueError):
        tree.assert_same_structure({"a": 1}, {"b": 1})
    with pytest.raises(ValueError):
        tree.assert_same_structure([1], 1)


def test_unflatten_errors():
    with pytest.raises(TypeError, match="flat_sequence must be a sequence"):
        tree.unflatten_as([1], 1)
    with pytest.raises(ValueError, match="scalar"):
        tree.unflatten_as(None, [1, 2])
    with pytest.raises(ValueError, match="Could not pack"):
        tree.unflatten_as([1, 2], [1])


def test_map_structure_errors():
    with pytest.raises(TypeError, match="func must be callable"):
        tree.map_structure(1, [1])
    with pytest.raises(ValueError, match="at least one"):
        tree.map_structure(lambda x: x)
    with pytest.raises(ValueError, match="Only valid keyword"):
        tree.map_structure(lambda x: x, [1], bogus=True)
    with pytest.raises(ValueError, match="at least one"):
        tree.map_structure_with_path(lambda p, x: x)


def test_unsortable_dict_keys():
    class Unsortable:
        pass

    with pytest.raises(TypeError, match="sortable keys"):
        tree.flatten({Unsortable(): 1, Unsortable(): 2})


def test_defaultdict_and_mapping_proxy():
    d = collections.defaultdict(int, {"b": 2, "a": 1})
    out = tree.map_structure(lambda v: v + 1, d)
    assert isinstance(out, collections.defaultdict)
    assert out.default_factory is int
    assert dict(out) == {"b": 3, "a": 2}

    proxy = MappingProxyType({"z": 1, "a": 2})
    mapped = tree.map_structure(lambda v: v * 10, proxy)
    assert isinstance(mapped, MappingProxyType)
    assert dict(mapped) == {"z": 10, "a": 20}


def test_mapping_view_flatten():
    keys = {"b": 1, "a": 2}.keys()
    assert tree.is_nested(keys) is True
    # Views reconstruct as lists.
    assert tree.map_structure(lambda k: k, keys) == list(keys)
    assert tree._is_standard_container(keys) is True
    assert tree._is_standard_container({"a": 1}.values()) is True
    assert tree._is_standard_container({"a": 1}.items()) is True
    assert tree._is_standard_container("hi") is False
    assert tree._is_standard_container(b"hi") is False


def test_attrs_instance():
    p = AttrPair(1, {"k": 2})
    assert tree.is_nested(p) is True
    assert tree.flatten(p) == [1, 2]
    out = tree.map_structure(lambda v: v + 1 if isinstance(v, int) else v, AttrPair(1, 2))
    assert out == AttrPair(2, 3)
    tree.assert_same_structure(AttrPair(1, 2), AttrPair(3, 4))


def test_magic_methods_nested_children():
    p = Pair([1, 2], 3)
    assert tree.is_nested(p) is True
    assert tree.flatten(p) == [1, 2, 3]
    out = tree.unflatten_as(p, [4, 5, 6])
    assert out == Pair([4, 5], 6)
    doubled = tree.map_structure(lambda v: v * 2, Pair(1, 2))
    assert doubled == Pair(2, 4)
    tree.assert_same_structure(Pair(1, 2), Pair(3, 4))


def test_traverse_top_down_and_bottom_up():
    visited = []
    out = tree.traverse(visited.append, [(1, 2), [3], {"a": 4}], top_down=True)
    assert out == [(1, 2), [3], {"a": 4}]
    assert visited == [[(1, 2), [3], {"a": 4}], (1, 2), 1, 2, [3], 3, {"a": 4}, 4]

    visited = []
    tree.traverse(visited.append, [(1, 2), [3], {"a": 4}], top_down=False)
    assert visited == [1, 2, (1, 2), 3, [3], 4, {"a": 4}, [(1, 2), [3], {"a": 4}]]


def test_traverse_replace_and_map_to_none():
    def drop_ints(x):
        if isinstance(x, int):
            return tree.MAP_TO_NONE
        return None

    assert tree.traverse(drop_ints, [1, "a"]) == [None, "a"]
    assert tree.traverse(lambda x: 0 if isinstance(x, int) else None, [1, [2]]) == [0, [0]]

    def bottom_none(x):
        if isinstance(x, int):
            return tree.MAP_TO_NONE
        return None

    assert tree.traverse(bottom_none, [1, "a"], top_down=False) == [None, "a"]

    def bottom(x):
        if isinstance(x, list):
            return tuple(x)
        return None

    assert tree.traverse(bottom, [1, [2]], top_down=False) == (1, (2,))


def test_traverse_with_path():
    visited = []
    tree.traverse_with_path(
        lambda path, subtree: visited.append((path, subtree)),
        [(1, 2), [3], {"a": 4}],
        top_down=True,
    )
    assert visited[0] == ((), [(1, 2), [3], {"a": 4}])
    assert ((0, 0), 1) in visited


def test_shallow_structure_errors():
    with pytest.raises(TypeError, match="must also be a sequence"):
        tree.flatten_up_to([None], 1)
    with pytest.raises(TypeError, match="path"):
        tree.flatten_with_path_up_to([None], 1)
    with pytest.raises(ValueError, match="sequence length"):
        tree.flatten_up_to([None, None], [1])
    with pytest.raises(TypeError, match="sequence type"):
        tree.flatten_up_to([None], (1,))
    with pytest.raises(ValueError, match="keys are not a subset"):
        tree.flatten_up_to({"a": None}, {"b": 1})
    with pytest.raises(TypeError, match="sequence type"):
        tree.flatten_up_to(Foo(None, None), Bar(1, 2))


def test_map_structure_up_to_kwargs_and_callable(caplog):
    import logging

    # dm-tree ignores unknown kwargs here (unlike map_structure).
    assert tree.map_structure_up_to([None], lambda x: x, [1], extra=1) == [1]
    with pytest.raises(TypeError, match="not callable"):
        tree.map_structure_up_to([None], 1, [1])
    with pytest.raises(ValueError, match="Could not pack"):
        tree.map_structure_up_to([None], lambda x: x)
    with caplog.at_level(logging.WARNING):
        out = tree.map_structure_up_to([None], lambda x: x, [1], check_types=False)
    assert out == [1]
    assert any("check_types" in r.message for r in caplog.records)


def test_multiyield_missing_key():
    with pytest.raises(ValueError, match="Could not find key"):
        tree.map_structure_up_to({"a": None}, lambda x, y: x, {"a": 1}, {"b": 2})


def test_namedtuple_rebuild_error():
    class Strict(namedtuple("Strict", "a")):
        def __new__(cls, a):
            if not isinstance(a, int):
                raise TypeError("need int")
            return super().__new__(cls, a)

    with pytest.raises(TypeError, match="Couldn't traverse"):
        tree.unflatten_as(Strict(1), ["nope"])


def test_check_types_false_mappings():
    tree.assert_same_structure({"a": 1}, collections.OrderedDict(a=2), check_types=False)
    tree.assert_same_structure({"a": 1}, collections.OrderedDict(a=2), check_types=True)
    out = tree.map_structure(lambda x, y: x + y, {"a": 1}, collections.OrderedDict(a=2), check_types=False)
    assert out == {"a": 3}
    assert tree.flatten_up_to({"a": None}, collections.OrderedDict(a=1), check_types=True) == [1]
    assert tree.flatten_up_to(Foo(None, None), AlsoFoo(1, 2)) == [1, 2]
    dd = collections.defaultdict(int, {"a": None})
    od = collections.OrderedDict(a=1)
    assert tree.flatten_up_to(dd, od, check_types=True) == [1]


def test_jax_registry_custom_type():
    jax = pytest.importorskip("jax")
    jtu = jax.tree_util

    class JaxPair:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return type(other) is JaxPair and self.x == other.x and self.y == other.y

    jtu.register_pytree_node(
        JaxPair,
        lambda o: ((o.x, o.y), None),
        lambda aux, children: JaxPair(*children),
    )
    p = JaxPair([1, 2], 3)
    assert tree.is_nested(p) is True
    assert tree.flatten(p) == [1, 2, 3]
    assert tree.map_structure(lambda v: v + 1, JaxPair(1, 2)) == JaxPair(2, 3)


def test_torch_registry_custom_type():
    torch = pytest.importorskip("torch")
    pytree = pytest.importorskip("torch.utils._pytree")

    class TorchPair:
        def __init__(self, x, y):
            self.x = x
            self.y = y

        def __eq__(self, other):
            return type(other) is TorchPair and self.x == other.x and self.y == other.y

    pytree.register_pytree_node(
        TorchPair,
        lambda o: ([o.x, o.y], None),
        lambda children, ctx: TorchPair(*children),
    )
    t = torch.tensor([1.0, 2.0])
    assert tree.is_nested(t) is False
    assert tree.flatten(t)[0] is t
    p = TorchPair(1, 2)
    assert tree.flatten(p) == [1, 2]
    assert tree.map_structure(lambda v: v * 2, p) == TorchPair(2, 4)


def test_dm_tree_parity_if_installed():
    dm = pytest.importorskip("tree")
    samples = [
        None,
        1,
        [1, 2, 3],
        (1, {"b": 2, "a": [3, 4]}),
        Foo(1, [2, 3]),
        {"z": (1, 2), "a": None},
        [[1, 1], [2, 2]],
    ]
    for s in samples:
        assert tree.is_nested(s) == dm.is_nested(s)
        assert tree.flatten(s) == dm.flatten(s)
        assert tree.flatten_with_path(s) == dm.flatten_with_path(s)
        if tree.is_nested(s):
            flat = tree.flatten(s)
            assert tree.unflatten_as(s, flat) == dm.unflatten_as(s, flat)
            assert tree.map_structure(lambda v: v, s) == dm.map_structure(lambda v: v, s)
    structure = [[1, 1], [2, 2]]
    assert tree.flatten_up_to([None, None], structure) == dm.flatten_up_to(
        [None, None], structure
    )


def test_registry_hooks_with_fake_modules(monkeypatch):
    class FakeTreeDef:
        pass

    class FakeJtu:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            if type(obj) is RegistryPair:
                return [obj.x, obj.y], FakeTreeDef()
            return [obj], object()

        @staticmethod
        def tree_unflatten(treedef, children):
            del treedef
            return RegistryPair(*children)

    monkeypatch.setitem(sys.modules, "jax.tree_util", FakeJtu)
    p = RegistryPair(1, {"k": 2})
    assert tree.is_nested(p) is True
    assert tree.flatten(p) == [1, 2]
    assert tree.map_structure(lambda v: v, RegistryPair(3, 4)) == RegistryPair(3, 4)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)

    class Spec:
        flatten_fn = staticmethod(lambda o: ([o.x, o.y], None))
        unflatten_fn = staticmethod(lambda children, ctx: RegistryPair(*children))

    class FakePytree:
        SUPPORTED_NODES = {RegistryPair: Spec()}

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", FakePytree)
    assert tree.flatten(RegistryPair(5, 6)) == [5, 6]
    assert tree.map_structure(lambda v: v * 2, RegistryPair(1, 2)) == RegistryPair(2, 4)
    monkeypatch.delitem(sys.modules, "torch.utils._pytree", raising=False)

    class FakeOptree:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            if type(obj) is RegistryPair:
                return [obj.x, obj.y], "spec"
            return [obj], "leaf"

        @staticmethod
        def tree_unflatten(children, treedef):
            del treedef
            return RegistryPair(*children)

    monkeypatch.setitem(sys.modules, "optree", FakeOptree)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)
    monkeypatch.delitem(sys.modules, "torch.utils._pytree", raising=False)
    assert tree.map_structure(lambda v: v + 1, RegistryPair(1, 2)) == RegistryPair(2, 3)
    assert tree.is_nested(object()) is False


def test_registry_leaf_and_failure(monkeypatch):
    class Boom:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            raise RuntimeError("nope")

        tree_unflatten = staticmethod(lambda *a: None)

    monkeypatch.setitem(sys.modules, "jax.tree_util", Boom)
    assert tree.is_nested(object()) is False

    class NoFns:
        pass

    monkeypatch.setitem(sys.modules, "jax.tree_util", NoFns)
    assert tree.is_nested(object()) is False

    class BadNodes:
        SUPPORTED_NODES = []  # .get will fail

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", BadNodes)
    assert tree.is_nested(object()) is False

    class NoNodes:
        SUPPORTED_NODES = None

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", NoNodes)
    assert tree.is_nested(object()) is False

    class IncompleteSpec:
        class S:
            flatten_fn = None
            unflatten_fn = None

        SUPPORTED_NODES = {RegistryPair: S()}

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", IncompleteSpec)
    assert tree.is_nested(RegistryPair(1, 2)) is False

    class OptBoom:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            raise RuntimeError("nope")

        tree_unflatten = staticmethod(lambda *a: None)

    monkeypatch.setitem(sys.modules, "optree", OptBoom)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)
    monkeypatch.delitem(sys.modules, "torch.utils._pytree", raising=False)
    assert tree.is_nested(object()) is False

    class OptNoFns:
        pass

    monkeypatch.setitem(sys.modules, "optree", OptNoFns)
    assert tree.is_nested(object()) is False


def test_tuple_without_fields_is_still_tuple():
    assert tree.is_nested((1, 2)) is True
    assert tree.flatten((1, 2)) == [1, 2]


def test_bytes_and_empty_dict():
    assert tree.flatten({}) == []
    assert tree.unflatten_as({}, []) == {}
    assert tree.map_structure(lambda x: x, {}) == {}


def test_shallow_mapping_vs_list_without_types():
    out = tree.flatten_up_to({0: None}, ["foo"], check_types=False)
    assert out == ["foo"]


class Packed:
    """Leaf feature object that owns concat/split."""

    def __init__(self, values, tag="ok"):
        self.values = np.asarray(values)
        self.tag = tag

    def __eq__(self, other):
        return (
            type(other) is Packed
            and other.tag == self.tag
            and np.array_equal(self.values, other.values)
        )

    @classmethod
    def __tree_concat__(cls, xs, axis=0):
        tags = {x.tag for x in xs}
        if len(tags) != 1:
            raise ValueError("Packed.tag mismatch")
        return cls(np.concatenate([x.values for x in xs], axis=axis), tag=xs[0].tag)

    def __tree_split__(self, sizes, axis=0):
        if axis != 0:
            raise ValueError("Packed split only supports axis=0")
        start = 0
        out = []
        for n in sizes:
            out.append(Packed(self.values[start : start + n], tag=self.tag))
            start += n
        return out


class PackedNoAxis:
    def __init__(self, values):
        self.values = np.asarray(values)

    def __eq__(self, other):
        return type(other) is PackedNoAxis and np.array_equal(self.values, other.values)

    @classmethod
    def __tree_concat__(cls, xs):
        return cls(np.concatenate([x.values for x in xs], axis=0))

    def __tree_split__(self, sizes):
        start = 0
        out = []
        for n in sizes:
            out.append(PackedNoAxis(self.values[start : start + n]))
            start += n
        return out


def test_concat_split_arrays_and_nests():
    a = np.array([1, 2])
    b = np.array([3])
    np.testing.assert_array_equal(tree.concat(a, b), np.array([1, 2, 3]))
    parts = tree.split(np.array([1, 2, 3, 4]), [1, 3])
    np.testing.assert_array_equal(parts[0], [1])
    np.testing.assert_array_equal(parts[1], [2, 3, 4])
    nested = tree.concat({"a": a}, {"a": b})
    np.testing.assert_array_equal(nested["a"], [1, 2, 3])
    split_n = tree.split({"a": np.array([1, 2, 3])}, [2, 1])
    np.testing.assert_array_equal(split_n[0]["a"], [1, 2])
    np.testing.assert_array_equal(split_n[1]["a"], [3])
    assert tree.concat(None, None) is None
    assert tree.split(None, [1, 2]) == [None, None]
    with pytest.raises(ValueError, match="at least one"):
        tree.concat()
    stacked = tree.concat(np.arange(4).reshape(2, 2), np.arange(4, 6).reshape(1, 2), axis=0)
    assert stacked.shape == (3, 2)
    col = tree.concat(np.arange(2).reshape(2, 1), np.arange(2, 4).reshape(2, 1), axis=1)
    assert col.shape == (2, 2)
    parts_ax = tree.split(np.arange(6).reshape(2, 3), [1, 2], axis=1)
    assert parts_ax[0].shape == (2, 1)
    assert parts_ax[1].shape == (2, 2)


def test_concat_split_magic_methods():
    p = tree.concat(Packed([1, 2], tag="t"), Packed([3], tag="t"))
    assert p == Packed([1, 2, 3], tag="t")
    parts = tree.split(p, [2, 1])
    assert parts[0] == Packed([1, 2], tag="t")
    assert parts[1] == Packed([3], tag="t")
    nested = tree.concat({"h": Packed([1], tag="t")}, {"h": Packed([2, 3], tag="t")})
    assert nested["h"] == Packed([1, 2, 3], tag="t")
    no_axis = tree.concat(PackedNoAxis([1]), PackedNoAxis([2]))
    assert no_axis == PackedNoAxis([1, 2])
    assert tree.split(no_axis, [1, 1]) == [PackedNoAxis([1]), PackedNoAxis([2])]
    with pytest.raises(ValueError, match="tag mismatch"):
        tree.concat(Packed([1], tag="a"), Packed([2], tag="b"))
    mixed = tree.concat(None, Packed([1], tag="t"), None)
    assert mixed == Packed([1], tag="t")
    mixed_nest = tree.concat({"a": None}, {"a": Packed([1, 2], tag="t")})
    assert mixed_nest["a"] == Packed([1, 2], tag="t")
    neg = tree.split(np.arange(6).reshape(2, 3), [1, 2], axis=-1)
    assert neg[0].shape == (2, 1)
    assert neg[1].shape == (2, 2)


def test_concat_accepts_axis_without_signature(monkeypatch):
    import inspect as inspect_mod

    def boom(_fn):
        raise ValueError("no signature")

    monkeypatch.setattr(inspect_mod, "signature", boom)
    p = tree.concat(Packed([1], tag="t"), Packed([2], tag="t"))
    assert p == Packed([1, 2], tag="t")
    parts = tree.split(p, [1, 1])
    assert parts == [Packed([1], tag="t"), Packed([2], tag="t")]


def test_concat_kwargs_axis_accepted():
    class Kw:
        def __init__(self, values):
            self.values = np.asarray(values)

        def __eq__(self, other):
            return type(other) is Kw and np.array_equal(self.values, other.values)

        @classmethod
        def __tree_concat__(cls, xs, **kwargs):
            axis = kwargs.get("axis", 0)
            return cls(np.concatenate([x.values for x in xs], axis=axis))

        def __tree_split__(self, sizes, **kwargs):
            axis = kwargs.get("axis", 0)
            start = 0
            out = []
            for n in sizes:
                sl = [slice(None)] * self.values.ndim
                sl[axis] = slice(start, start + n)
                out.append(Kw(self.values[tuple(sl)]))
                start += n
            return out

    out = tree.concat(Kw([1, 2]), Kw([3]))
    assert out == Kw([1, 2, 3])
    assert tree.split(out, [2, 1]) == [Kw([1, 2]), Kw([3])]


def test_concat_namedtuple_of_arrays():
    out = tree.concat(Foo(np.array([1]), np.array([2])), Foo(np.array([3]), np.array([4])))
    assert isinstance(out, Foo)
    np.testing.assert_array_equal(out.a, [1, 3])
    np.testing.assert_array_equal(out.b, [2, 4])

