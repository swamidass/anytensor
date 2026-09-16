"""Contract tests for structured-array peel / rewrap (ragged-ready)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

import anytensor as at
from anytensor.structure import (
    apply_structured,
    register_structure,
    unregister_structure,
)


@register_structure
@dataclass
class _Toy:
    values: object
    tag: str = "a"

    def with_values(self, values):
        return _Toy(values=values, tag=self.tag)

    def same_structure(self, other):
        return isinstance(other, _Toy) and self.tag == other.tag


@pytest.fixture(autouse=True)
def _toy_registered():
    # Decorator already registered; ensure cleanup does not leave orphans if
    # re-imported. Re-register each test module load is fine.
    register_structure(_Toy)
    yield
    unregister_structure(_Toy)


def test_at_exp_peels_values_keeps_tag_and_ids_object():
    values = np.array([0.0, 1.0], dtype=np.float64)
    toy = _Toy(values, tag="part")
    out = at.exp(toy)
    assert isinstance(out, _Toy)
    assert out.tag == "part"
    np.testing.assert_allclose(out.values, np.exp(values))
    # with_values must not invent a new partition identity for the toy tag
    assert out.tag is toy.tag or out.tag == toy.tag


def test_at_maximum_two_structures_same_partition():
    a = _Toy(np.array([1.0, 3.0]), tag="x")
    b = _Toy(np.array([2.0, 0.0]), tag="x")
    out = at.maximum(a, b)
    assert isinstance(out, _Toy)
    np.testing.assert_allclose(out.values, [2.0, 3.0])


def test_promote_only_fill_nan_mask_peels_structure():
    """``fill_nan_mask`` uses ``@promote`` without ``@as_array_result``."""
    x = _Toy(np.array([1.0, np.nan, 3.0]), tag="m")
    filled, mask = at.fill_nan_mask(x, 0.0)
    assert isinstance(filled, _Toy)
    assert filled.tag == "m"
    np.testing.assert_allclose(filled.values, [1.0, 0.0, 3.0])
    # mask leading length matches → also rewrapped
    assert isinstance(mask, _Toy)
    np.testing.assert_array_equal(mask.values, [False, True, False])


def test_at_maximum_mismatched_partition_raises():
    a = _Toy(np.array([1.0]), tag="a")
    b = _Toy(np.array([2.0]), tag="b")
    with pytest.raises(ValueError, match="same partition"):
        at.maximum(a, b)


def test_reduction_returns_dense():
    toy = _Toy(np.array([1.0, 2.0, 3.0]), tag="x")
    out = at.sum(toy)
    assert not isinstance(out, _Toy)
    np.testing.assert_allclose(out, 6.0)


def test_apply_structured_index_vector_untouched_pattern():
    """Simulate ragged: only values enter fn; aux id vector is never an arg."""
    values = np.array([1.0, 2.0])
    row_ids = np.array([0, 1])
    ids_before = row_ids.copy()

    class R:
        def __init__(self, values, row_ids):
            self.values = values
            self.row_ids = row_ids

        def with_values(self, values):
            # reuse same row_ids object
            return R(values, self.row_ids)

    register_structure(R)
    try:
        r = R(values, row_ids)
        out = apply_structured(np.exp, (r,), {})
        assert out.row_ids is row_ids
        np.testing.assert_array_equal(row_ids, ids_before)
    finally:
        unregister_structure(R)


def test_unregister_missing_is_noop():
    class _Gone:
        pass

    unregister_structure(_Gone)  # no raise


def test_same_structure_fallback_without_method():
    @register_structure
    class _Bare:
        def __init__(self, values, row_ids, nrows):
            self.values = values
            self.row_ids = row_ids
            self.nrows = nrows

        def with_values(self, values):
            return _Bare(values, self.row_ids, self.nrows)

    try:
        ids = np.array([0, 1])
        a = _Bare(np.array([1.0, 2.0]), ids, 2)
        b = _Bare(np.array([3.0, 4.0]), ids, 2)  # same row_ids object
        c = _Bare(np.array([3.0, 4.0]), np.array([0, 1]), 2)  # equal but new object
        from anytensor.structure import same_structure

        assert same_structure(a, a)
        assert same_structure(a, b)
        assert not same_structure(a, c)  # fallback is identity of row_ids
        assert not same_structure(a, np.array([1.0]))
        out = at.exp(a)
        assert isinstance(out, _Bare)
        assert out.row_ids is ids
    finally:
        unregister_structure(_Bare)


def test_rewrap_leading_length_mismatch_stays_dense():
    from anytensor.structure import rewrap

    toy = _Toy(np.array([1.0, 2.0, 3.0]), tag="z")
    gathered = np.array([1.0, 2.0])  # leading len 2 != 3
    out = rewrap(gathered, toy)
    assert not isinstance(out, _Toy)
    np.testing.assert_array_equal(out, gathered)


def test_rewrap_edge_cases():
    from anytensor.structure import peel, rewrap

    toy = _Toy(np.array([1.0, 2.0]), tag="z")
    assert rewrap(toy, toy) is toy
    assert rewrap(np.asarray(1.0), toy).shape == ()
    assert not isinstance(rewrap(object(), toy), _Toy)
    scalar_toy = _Toy(np.asarray(1.0), tag="s")
    assert not isinstance(rewrap(np.array([1.0, 2.0]), scalar_toy), _Toy)
    assert peel(np.array([1.0]))[1] is None
    assert rewrap(np.array([1.0]), None) is not None



def test_apply_structured_tuple_and_kwargs():
    def both(x, *, y):
        return x + 1, y + 2

    a = _Toy(np.array([1.0, 2.0]), tag="t")
    b = _Toy(np.array([3.0, 4.0]), tag="t")
    o1, o2 = apply_structured(both, (a,), {"y": b})
    assert isinstance(o1, _Toy) and isinstance(o2, _Toy)
    np.testing.assert_allclose(o1.values, [2.0, 3.0])
    np.testing.assert_allclose(o2.values, [5.0, 6.0])
