"""Tests for :mod:`anytensor.lengths` (graph length/id helpers)."""

from __future__ import annotations

import numpy as np

from anytensor import lengths


def test_lengths_to_ids_splits_and_cuts():
    assert lengths.lengths_to_splits(np.array([2, 1, 3])).tolist() == [2, 3, 6]
    assert lengths.lengths_to_splits(np.array([5])).tolist() == [5]
    np.testing.assert_array_equal(lengths.cuts_to_lengths([2, 3], 6), [2, 1, 3])
    np.testing.assert_array_equal(lengths.lengths_to_ids(np.array([2, 1])), [0, 0, 1])


def test_batch_unbatch_ids_roundtrip():
    local = [np.array([0, 1], dtype=np.int32), np.array([0], dtype=np.int32)]
    n_node = np.array([3, 2], dtype=np.int32)
    n_edge = np.array([2, 1], dtype=np.int32)
    concat = np.concatenate(local)
    batched = lengths.batch_ids(concat, n_node, n_edge)
    np.testing.assert_array_equal(batched, [0, 1, 3])
    parts = lengths.unbatch_ids(batched, n_node, n_edge)
    assert len(parts) == 2
    np.testing.assert_array_equal(parts[0], [0, 1])
    np.testing.assert_array_equal(parts[1], [0])


def test_split_by_lengths_array_and_nest():
    node_parts = lengths.split_by_lengths(np.arange(3), np.array([2, 1]))
    np.testing.assert_array_equal(node_parts[0], [0, 1])
    np.testing.assert_array_equal(node_parts[1], [2])
    nested = {"h": np.arange(4).reshape(2, 2)}
    nested_parts = lengths.split_by_lengths(nested, np.array([1, 1]))
    assert nested_parts[0]["h"].shape == (1, 2)
    assert lengths.split_by_lengths(None, np.array([1, 1])) == [None, None]
    assert lengths.split_by_lengths({}, np.array([1, 1])) == [{}, {}]
