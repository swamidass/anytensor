"""Runtime jaxtyping checks are enabled in tests via ``conftest`` import hook."""

from __future__ import annotations

import numpy as np
import pytest

import anytensor as at


def test_segment_sum_accepts_matching_shapes():
    x = np.arange(5.0)
    ids = np.array([0, 0, 1, 1, 2])
    y = at.segment_sum(x, ids, 3)
    np.testing.assert_allclose(y, [1.0, 5.0, 4.0])


def test_segment_sum_rejects_mismatched_leading_length():
    """Ids length must match values' axis-0 (annotation ``n``)."""
    x = np.arange(5.0)
    ids = np.array([0, 0, 1])  # length 3 ≠ 5
    with pytest.raises(Exception):
        at.segment_sum(x, ids, 2)


def test_typecheck_hook_is_active():
    """Ensure the test suite actually installed the runtime hook."""
    # Wrapped functions expose __jaxtyped__ / __wrapped__ depending on version.
    fn = at.segment_sum
    assert getattr(fn, "__jaxtyped__", None) is not None or hasattr(fn, "__wrapped__")
