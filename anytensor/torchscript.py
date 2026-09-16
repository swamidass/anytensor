"""Pure-Torch segment kernels used under ``torch.jit.script``.

Loaded by :func:`anytensor.enable_torchscript`. Public ``segment_sum`` / ``min``
/ ``max`` keep multi-backend eager dispatch; only the ``is_scripting()`` branch
calls into this module — so libraries can use ``anytensor.segment_sum`` and end
users can still script through those call sites.

Importing this module requires PyTorch. ``num_segments`` must be a Python
``int`` under script; segment ids are cast to ``int64``. Empty-slot identities
match :mod:`anytensor.semantics`. Integer fills avoid ``torch.iinfo`` (not
scriptable) and cover common integer dtypes.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _expand_ids(x: Tensor, segment_ids: Tensor) -> Tensor:
    ids = segment_ids.to(dtype=torch.int64)
    for _ in range(1, x.dim()):
        ids = ids.unsqueeze(-1)
    return ids.expand_as(x)


def _min_fill(x: Tensor) -> Tensor:
    """Empty-segment identity for min (+inf / dtype max)."""
    if x.is_floating_point():
        return torch.tensor(float("inf"), dtype=x.dtype, device=x.device)
    # torch.iinfo is not TorchScript-able; enumerate common integer dtypes.
    if x.dtype == torch.int64:
        return torch.tensor(9223372036854775807, dtype=torch.int64, device=x.device)
    if x.dtype == torch.int32:
        return torch.tensor(2147483647, dtype=torch.int32, device=x.device)
    if x.dtype == torch.int16:
        return torch.tensor(32767, dtype=torch.int16, device=x.device)
    if x.dtype == torch.int8:
        return torch.tensor(127, dtype=torch.int8, device=x.device)
    if x.dtype == torch.uint8:
        return torch.tensor(255, dtype=torch.uint8, device=x.device)
    # Fallback: treat as float-like max via conversion (should be rare under script).
    return torch.tensor(2147483647, dtype=x.dtype, device=x.device)


def _max_fill(x: Tensor) -> Tensor:
    """Empty-segment identity for max (-inf / dtype min)."""
    if x.is_floating_point():
        return torch.tensor(float("-inf"), dtype=x.dtype, device=x.device)
    if x.dtype == torch.int64:
        # Literal INT64_MIN is rejected by the TorchScript parser; build it.
        return (~torch.tensor(9223372036854775807, dtype=torch.int64, device=x.device))
    if x.dtype == torch.int32:
        return torch.tensor(-2147483648, dtype=torch.int32, device=x.device)
    if x.dtype == torch.int16:
        return torch.tensor(-32768, dtype=torch.int16, device=x.device)
    if x.dtype == torch.int8:
        return torch.tensor(-128, dtype=torch.int8, device=x.device)
    if x.dtype == torch.uint8:
        return torch.tensor(0, dtype=torch.uint8, device=x.device)
    return torch.tensor(-2147483648, dtype=x.dtype, device=x.device)


def segment_sum(x: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """Sum ``x`` into ``num_segments`` bins along axis 0 (scatter-add)."""
    shape = (num_segments,) + x.shape[1:]
    out = torch.zeros(shape, dtype=x.dtype, device=x.device)
    return out.scatter_add(0, _expand_ids(x, segment_ids), x)


def segment_min(x: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """Min of ``x`` per segment; empty slots are ``+inf`` / dtype max."""
    shape = (num_segments,) + x.shape[1:]
    out = torch.full(shape, _min_fill(x), dtype=x.dtype, device=x.device)
    return out.scatter_reduce(0, _expand_ids(x, segment_ids), x, reduce="amin", include_self=True)


def segment_max(x: Tensor, segment_ids: Tensor, num_segments: int) -> Tensor:
    """Max of ``x`` per segment; empty slots are ``-inf`` / dtype min."""
    shape = (num_segments,) + x.shape[1:]
    out = torch.full(shape, _max_fill(x), dtype=x.dtype, device=x.device)
    return out.scatter_reduce(0, _expand_ids(x, segment_ids), x, reduce="amax", include_self=True)
