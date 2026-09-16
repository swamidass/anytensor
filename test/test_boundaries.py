"""Explicit boundary / edge-case behavior for core and segment ops."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at

from helpers import BACKENDS, close, loaded_backends


@pytest.mark.parametrize("backend", BACKENDS)
def test_take_shape_cast(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    idx = np.array([2, 0], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bi = backend_impl.from_numpy(idx)
    assert close(backend_impl.to_numpy(at.take(bx, bi)), x[[2, 0]])
    assert at.shape(bx) == (3, 2)
    assert at.shape(x) == (3, 2)
    # cast is alias of astype
    out = at.cast(bx, getattr(bx, "dtype", x.dtype))
    assert close(backend_impl.to_numpy(out), x)


@pytest.mark.parametrize("backend", BACKENDS)
def test_reduction_single_element_is_0d(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([7.0])
    bx = backend_impl.from_numpy(x)
    s = at.sum(bx)
    assert s.ndim == 0
    assert close(backend_impl.to_numpy(s), np.array(7.0))


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_vector_ops(backend):
    """Length-0 inputs where the op is defined."""
    backend_impl = loaded_backends[backend]
    x = np.array([], dtype=np.float32)
    seg = np.array([], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    assert close(backend_impl.to_numpy(at.sum(bx)), np.array(0.0, dtype=np.float32))
    assert close(backend_impl.to_numpy(at.prod(bx)), np.array(1.0, dtype=np.float32))
    assert close(
        backend_impl.to_numpy(at.segment_sum(bx, bs, 3)),
        np.zeros(3, dtype=np.float32),
    )
    assert close(
        backend_impl.to_numpy(at.segment_count(bs, 3)),
        np.zeros(3, dtype=np.float32),
    )
    empty_idx = backend_impl.from_numpy(np.array([], dtype=np.int64))
    src = backend_impl.from_numpy(np.array([1.0, 2.0], dtype=np.float32))
    assert backend_impl.to_numpy(at.take(src, empty_idx)).shape == (0,)


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_all_same_id(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, 2.0, 3.0])
    seg = np.array([0, 0, 0], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    assert close(backend_impl.to_numpy(at.segment_sum(bx, bs, 1)), np.array([6.0]))
    assert close(backend_impl.to_numpy(at.segment_mean(bx, bs, 1)), np.array([2.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_trailing_empty_slots(backend):
    """num_segments > max(id)+1 leaves empty trailing segments."""
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, 2.0, 3.0])
    seg = np.array([0, 0, 1], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    # slot 2 empty
    assert close(backend_impl.to_numpy(at.segment_sum(bx, bs, 3)), np.array([3.0, 3.0, 0.0]))
    assert close(backend_impl.to_numpy(at.segment_count(bs, 3)), np.array([2.0, 1.0, 0.0]))
    # raw max empty → -inf (float identity); or_constant → -1
    raw = backend_impl.to_numpy(at.segment_max(bx, bs, 3))
    assert raw[0] == 2.0 and raw[1] == 3.0
    assert np.isneginf(raw[2])
    filled = backend_impl.to_numpy(at.segment_max_or_constant(bx, bs, 3, constant=-1.0))
    assert close(filled, np.array([2.0, 3.0, -1.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_normalize_zero_sum_segment(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, -1.0, 2.0])
    seg = np.array([0, 0, 1], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    out = backend_impl.to_numpy(at.segment_normalize(bx, bs, 2))
    # segment 0 sums to 0 → entries become 0; segment 1 → 1
    assert close(out, np.array([0.0, 0.0, 1.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_softmax_singleton_and_uniform(backend):
    backend_impl = loaded_backends[backend]
    logits = np.array([5.0, 1.0, 1.0], dtype=np.float64)
    seg = np.array([0, 1, 1], dtype=np.int64)
    bx = backend_impl.from_numpy(logits)
    bs = backend_impl.from_numpy(seg)
    out = backend_impl.to_numpy(at.segment_softmax(bx, bs, 2))
    assert close(out[0:1], np.array([1.0]))
    assert close(out[1:], np.array([0.5, 0.5]))
    assert close(out.sum(keepdims=False), 2.0) or abs(out[0] + out[1] + out[2] - 2.0) < 1e-6


@pytest.mark.parametrize("backend", BACKENDS)
def test_segment_unsorted_ids_match_sorted_permutation(backend):
    backend_impl = loaded_backends[backend]
    x = np.array([1.0, 2.0, 3.0, 4.0])
    seg = np.array([1, 0, 1, 0], dtype=np.int64)
    bx = backend_impl.from_numpy(x)
    bs = backend_impl.from_numpy(seg)
    assert close(backend_impl.to_numpy(at.segment_sum(bx, bs, 2)), np.array([6.0, 4.0]))


@pytest.mark.parametrize("backend", BACKENDS)
def test_partition_softmax_single_partition(backend):
    backend_impl = loaded_backends[backend]
    logits = np.array([1.0, 2.0, 3.0])
    parts = np.array([3], dtype=np.int64)
    blog = backend_impl.from_numpy(logits)
    bpart = backend_impl.from_numpy(parts)
    out = backend_impl.to_numpy(at.partition_softmax(blog, bpart, sum_partitions=3))
    ref = np.exp(logits - logits.max())
    ref = ref / ref.sum()
    assert close(out, ref)
