"""Tests for :mod:`anytensor.tree` (jax.tree API + custom concat/split)."""

from __future__ import annotations

import collections
import inspect as inspect_mod
import sys
from collections import namedtuple
from types import MappingProxyType

import numpy as np
import pytest

import anytensor.tree as tree


Foo = namedtuple("Foo", ["a", "b"])
Bar = namedtuple("Bar", ["a", "b"])


class Pair:
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


class Packed:
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


def test_none_is_empty_pytree():
    leaves, treedef = tree.flatten(None)
    assert leaves == []
    assert treedef == tree.structure(None)
    assert tree.unflatten(treedef, []) is None
    assert tree.leaves(None) == []
    assert tree.map(lambda x: x + 1, None) is None
    assert tree.all(None) is True


def test_scalars_and_arrays_are_leaves():
    assert tree.leaves(42) == [42]
    assert tree.leaves("hi") == ["hi"]
    assert tree.leaves(b"hi") == [b"hi"]
    assert tree.leaves({1, 2})[0] == {1, 2}
    arr = np.arange(3)
    assert tree.leaves(arr)[0] is arr
    mapped = tree.map(lambda v: v * 2, arr)
    np.testing.assert_array_equal(mapped, np.arange(3) * 2)

    class FakeTensor:
        shape = (2, 3)
        dtype = np.float32

    assert tree.leaves(FakeTensor())[0].shape == (2, 3)
    assert tree._is_fast_array_leaf(FakeTensor()) is True
    assert tree._registry_one_level(arr) is None
    assert tree._registry_one_level(FakeTensor()) is None

    class SeqWithShape(list):
        shape = (2,)
        dtype = np.float32

    class MapWithShape(dict):
        shape = (2,)
        dtype = np.float32

    assert tree._is_fast_array_leaf(SeqWithShape([1, 2])) is False
    assert tree._is_fast_array_leaf(MapWithShape(a=1)) is False


def test_dict_sorted_keys_and_containers():
    assert tree.leaves({"b": 1, "a": 2}) == [2, 1]
    assert tree.map(lambda v: v * 10, {"b": 1, "a": 2}) == {"a": 20, "b": 10}
    assert tree.leaves([1, [2, 3]]) == [1, 2, 3]
    assert tree.leaves((1, 2)) == [1, 2]
    assert tree.leaves({}) == []
    assert tree.unflatten(tree.structure({}), []) == {}
    assert tree.map(lambda x: x, {}) == {}
    proxy = MappingProxyType({"z": 1, "a": 2})
    assert tree.leaves(proxy)[0] is proxy


def test_ordereddict_and_defaultdict():
    od = collections.OrderedDict([("b", 1), ("a", 2)])
    assert tree.leaves(od) == [1, 2]
    out = tree.map(lambda v: v + 1, od)
    assert isinstance(out, collections.OrderedDict)
    assert list(out.items()) == [("b", 2), ("a", 3)]

    dd = collections.defaultdict(int, {"b": 2, "a": 1})
    out = tree.map(lambda v: v + 1, dd)
    assert isinstance(out, collections.defaultdict)
    assert out.default_factory is int
    assert dict(out) == {"a": 2, "b": 3}


def test_namedtuple_and_attrs():
    assert tree.leaves(Foo(1, [2, 3])) == [1, 2, 3]
    assert tree.map(lambda v: v * 2, Foo(1, 2)) == Foo(2, 4)
    p = AttrPair(1, {"k": 2})
    assert tree.leaves(p) == [1, 2]
    assert tree.map(lambda v: v + 1 if isinstance(v, int) else v, AttrPair(1, 2)) == AttrPair(
        2, 3
    )


def test_magic_flatten():
    p = Pair([1, 2], 3)
    assert tree.leaves(p) == [1, 2, 3]
    doubled = tree.map(lambda v: v * 2, Pair(1, 2))
    assert doubled == Pair(2, 4)
    leaves, treedef = tree.flatten(p)
    assert tree.unflatten(treedef, [4, 5, 6]) == Pair([4, 5], 6)


def test_is_leaf():
    leaves, td = tree.flatten([1, 2], is_leaf=lambda x: type(x) is list)
    assert leaves == [[1, 2]]
    assert tree.unflatten(td, leaves) == [1, 2]
    leaves, _ = tree.flatten(
        [1, [2, 3]], is_leaf=lambda x: type(x) is list and all(type(i) is int for i in x)
    )
    assert leaves == [1, [2, 3]]
    leaves, _ = tree.flatten({"a": None, "b": 1}, is_leaf=lambda x: x is None)
    assert None in leaves


def test_map_multiple_and_mismatch():
    assert tree.map(lambda x, y: x + y, {"a": 1}, {"a": 2}) == {"a": 3}
    with pytest.raises(ValueError, match="same structure"):
        tree.map(lambda x, y: x, [1], (1,))
    with pytest.raises(ValueError, match="same structure"):
        tree.map(lambda x, y: x, None, 1)
    with pytest.raises(ValueError, match="same structure"):
        tree.map_with_path(lambda p, x, y: x, [1], (1,))


def test_unflatten_leaf_count():
    _, td = tree.flatten([1, 2])
    with pytest.raises(ValueError, match="Too few"):
        tree.unflatten(td, [1])
    with pytest.raises(ValueError, match="Too many"):
        tree.unflatten(td, [1, 2, 3])
    assert tree.unflatten(td, [8, 9]) == [8, 9]


def test_paths_and_map_with_path():
    pairs, _ = tree.flatten_with_path({"b": 1, "a": [2]})
    assert pairs[0][0] == (tree.DictKey("a"), tree.SequenceKey(0))
    assert pairs[0][1] == 2
    assert pairs[1] == ((tree.DictKey("b"),), 1)
    assert tree.leaves_with_path(None) == []
    out = tree.map_with_path(lambda path, v: (path, v * 2), Foo(1, 2))
    assert out == Foo(a=((tree.GetAttrKey("a"),), 2), b=((tree.GetAttrKey("b"),), 4))
    attr_out = tree.map_with_path(lambda path, v: path[-1].name, AttrPair(1, 2))
    assert attr_out == AttrPair("x", "y")
    pairs, _ = tree.flatten_with_path(collections.OrderedDict([("b", 1), ("a", 2)]))
    assert pairs[0][0] == (tree.DictKey("b"),)
    pairs, _ = tree.flatten_with_path(collections.defaultdict(int, {"b": 2, "a": 1}))
    assert pairs[0][0] == (tree.DictKey("a"),)
    tup = tree.concat((np.array([1]), np.array([2])), (np.array([3]), np.array([4])))
    assert isinstance(tup, tuple)
    np.testing.assert_array_equal(tup[0], [1, 3])
    with pytest.raises(ValueError, match="same structure"):
        tree.concat([np.array([1])], [np.array([1]), np.array([2])])
    assert tree.map_with_path(lambda p, x, y: x + y, {"a": 1}, {"a": 2}) == {"a": 3}
    with pytest.raises(ValueError, match="same structure"):
        tree.map_with_path(lambda p, x, y: x, {"a": 1}, {"b": 1})


def test_all_and_reduce():
    assert tree.all({"a": True, "b": [True, True]}) is True
    assert tree.all({"a": True, "b": False}) is False
    assert tree.all({}) is True
    assert tree.reduce(lambda a, b: a + b, {"a": 1, "b": 2}) == 3
    assert tree.reduce(lambda a, b: a + b, None, 0) == 0
    with pytest.raises(TypeError):
        tree.reduce(lambda a, b: a + b, None)


def test_pytreedef_eq_and_num_leaves():
    td = tree.structure({"a": None, "b": [1, 2]})
    assert td.num_leaves == 2
    assert td == tree.structure({"b": [1, 2], "a": None})
    assert td != tree.structure({"a": None, "b": (1, 2)})
    assert (td == object()) is False
    assert "PyTreeDef" in repr(td)
    assert repr(tree.structure(3)) == "PyTreeDef(*)"
    assert repr(tree.structure(None)) == "PyTreeDef(None)"
    assert tree.tree_structure([1]) == tree.structure([1])


def test_unsortable_dict_keys():
    class Unsortable:
        pass

    with pytest.raises(TypeError, match="sortable keys"):
        tree.flatten({Unsortable(): 1, Unsortable(): 2})
    with pytest.raises(TypeError, match="sortable keys"):
        tree.flatten(collections.defaultdict(int, {Unsortable(): 1, Unsortable(): 2}))


def test_aliases():
    leaves, td = tree.tree_flatten([1, 2])
    assert tree.tree_leaves([1, 2]) == [1, 2]
    assert tree.tree_unflatten(td, leaves) == [1, 2]
    assert tree.tree_map(lambda x: x + 1, [1]) == [2]
    pairs, _ = tree.tree_flatten_with_path([1])
    assert pairs[0][0] == (tree.SequenceKey(0),)
    assert tree.tree_map_with_path(lambda p, x: x, [1]) == [1]


def test_jax_tree_parity_if_installed():
    jt = pytest.importorskip("jax").tree
    samples = [
        None,
        1,
        [1, 2, 3],
        (1, {"b": 2, "a": [3, 4]}),
        Foo(1, [2, 3]),
        {"z": (1, 2), "a": None},
        [[1, 1], [2, 2]],
        {},
        [],
        collections.OrderedDict([("b", 1), ("a", 2)]),
        collections.defaultdict(int, {"b": 2, "a": 1}),
    ]
    for s in samples:
        assert tree.leaves(s) == jt.leaves(s)
        assert tree.map(lambda v: v, s) == jt.map(lambda v: v, s)
        our_leaves, our_td = tree.flatten(s)
        jax_leaves, jax_td = jt.flatten(s)
        assert our_leaves == jax_leaves
        assert tree.unflatten(our_td, our_leaves) == jt.unflatten(jax_td, jax_leaves)


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
    assert tree.leaves(p) == [1, 2, 3]
    assert tree.map(lambda v: v + 1, JaxPair(1, 2)) == JaxPair(2, 3)


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
    assert tree.leaves(t)[0] is t
    assert tree.leaves(TorchPair(1, 2)) == [1, 2]
    assert tree.map(lambda v: v * 2, TorchPair(1, 2)) == TorchPair(2, 4)


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
    assert tree.leaves(p) == [1, 2]
    assert tree.map(lambda v: v, RegistryPair(3, 4)) == RegistryPair(3, 4)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)

    class Spec:
        flatten_fn = staticmethod(lambda o: ([o.x, o.y], None))
        unflatten_fn = staticmethod(lambda children, ctx: RegistryPair(*children))

    class FakePytree:
        SUPPORTED_NODES = {RegistryPair: Spec()}

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", FakePytree)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)
    assert tree.leaves(RegistryPair(5, 6)) == [5, 6]
    assert tree.map(lambda v: v * 2, RegistryPair(1, 2)) == RegistryPair(2, 4)
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
    assert tree.map(lambda v: v + 1, RegistryPair(1, 2)) == RegistryPair(2, 3)
    assert len(tree.leaves(object())) == 1


def test_registry_leaf_and_failure(monkeypatch):
    class Boom:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            raise RuntimeError("nope")

        tree_unflatten = staticmethod(lambda *a: None)

    monkeypatch.setitem(sys.modules, "jax.tree_util", Boom)
    assert tree.leaves(object())[0].__class__ is object

    class NoFns:
        pass

    monkeypatch.setitem(sys.modules, "jax.tree_util", NoFns)
    assert len(tree.leaves(object())) == 1

    class BadNodes:
        SUPPORTED_NODES = []

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", BadNodes)
    assert len(tree.leaves(object())) == 1

    class NoNodes:
        SUPPORTED_NODES = None

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", NoNodes)
    assert len(tree.leaves(object())) == 1

    class IncompleteSpec:
        class S:
            flatten_fn = None
            unflatten_fn = None

        SUPPORTED_NODES = {RegistryPair: S()}

    monkeypatch.setitem(sys.modules, "torch.utils._pytree", IncompleteSpec)
    assert tree.leaves(RegistryPair(1, 2)) == [RegistryPair(1, 2)]

    class OptBoom:
        @staticmethod
        def tree_flatten(obj, is_leaf=None):
            raise RuntimeError("nope")

        tree_unflatten = staticmethod(lambda *a: None)

    monkeypatch.setitem(sys.modules, "optree", OptBoom)
    monkeypatch.delitem(sys.modules, "jax.tree_util", raising=False)
    monkeypatch.delitem(sys.modules, "torch.utils._pytree", raising=False)
    assert len(tree.leaves(object())) == 1

    class OptNoFns:
        pass

    monkeypatch.setitem(sys.modules, "optree", OptNoFns)
    assert len(tree.leaves(object())) == 1


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
    neg = tree.split(np.arange(6).reshape(2, 3), [1, 2], axis=-1)
    assert neg[0].shape == (2, 1)
    with pytest.raises(ValueError, match="same structure"):
        tree.concat(np.array([1]), None)
    with pytest.raises(ValueError, match="same structure"):
        tree.concat([np.array([1])], (np.array([2]),))
    with pytest.raises(ValueError, match="same structure"):
        tree.concat({"a": np.array([1])}, {"a": np.array([1]), "b": np.array([2])})
    empty = tree.concat([], [])
    assert empty == []


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


def test_concat_accepts_axis_without_signature(monkeypatch):
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
    parts = tree.split(out, [1, 1])
    np.testing.assert_array_equal(parts[0].a, [1])
    od = tree.concat(
        collections.OrderedDict(a=np.array([1])),
        collections.OrderedDict(a=np.array([2])),
    )
    np.testing.assert_array_equal(od["a"], [1, 2])
    dd = tree.concat(
        collections.defaultdict(int, {"a": np.array([1])}),
        collections.defaultdict(int, {"a": np.array([2])}),
    )
    np.testing.assert_array_equal(dd["a"], [1, 2])
    assert isinstance(dd, collections.defaultdict)
    ap = tree.concat(AttrPair(np.array([1]), np.array([2])), AttrPair(np.array([3]), np.array([4])))
    np.testing.assert_array_equal(ap.x, [1, 3])
    pair = tree.concat(Pair(np.array([1]), np.array([2])), Pair(np.array([3]), np.array([4])))
    np.testing.assert_array_equal(pair.x, [1, 3])
    split_pair = tree.split(pair, [1, 1])
    np.testing.assert_array_equal(split_pair[1].y, [4])
