"""Portable semantics for segment reductions and other cross-backend contracts.

This module is the **source of truth** for behaviors AnyTensor standardizes so
NumPy / JAX / Torch / TF agree. The user-facing write-up lives in
``docs/semantics.md`` (MkDocs “Surprising differences”).

Empty-segment identities
------------------------

=============  ==============  =========================
reduction      floating        integral
=============  ==============  =========================
``sum``        ``0``           ``0``
``min``        ``+inf``        ``iinfo(dtype).max``
``max``        ``-inf``        ``iinfo(dtype).min``
=============  ==============  =========================

Occupied slots always reflect the true reduction (including when the only
values are ±inf / NaN). Use ``segment_min_or_constant`` /
``segment_max_or_constant`` when empty slots should be a finite fill instead
of the identity sentinel.

Surprises we paper over (standardized)
--------------------------------------

* **TF ``unsorted_segment_{min,max}``** — empty slots and ±inf-only segments
  become **finfo** limits. AnyTensor uses scatter from a ±inf / iinfo identity
  so empties and occupied ±inf match NumPy/JAX/Torch.
* **TF ``tensor_scatter_nd_{min,max}``** — NaN updates are ignored (segment
  stays at the identity). After scatter we OR in per-segment NaN via
  ``unsorted_segment_max(is_nan(x))`` so NaN-containing segments become NaN.
* **``num_segments``** — required shape-size (Python int / symbolic / 0-d
  integral tensor); never inferred from ``segment_ids`` (JAX convention).
* **TF graph ``repeat`` / ``arange`` / ``*_like``** — Python scalar repeats
  stay Python; shim uses ``tf.repeat`` / ``tf.range``; ``*_like`` builds from
  symbolic ``shape(x)`` so polymorphic ``tf.function`` does not see
  ``TensorShape(None,)``.

Surprises we document only (backend-local)
-----------------------------------------

* **Index width** — Torch scatter casts to int64; JAX/TF often keep int32
  (especially without JAX x64).
* **``sorted=``** — honored on JAX/TF; no-op on NumPy/Torch (unsorted-safe).
* **Default float width** — JAX may truncate float64→float32 without x64.
  ``inf *`` subnormal / float32-min may be ``inf`` (NumPy, eager TF) vs ``nan``
  (JAX, TF XLA) when the tiny flushes to 0.
* **``jax.jit`` + ``repeat``** — needs static repeats or
  ``total_repeat_length`` / ``sum_partitions``.
* **TF XLA vs eager with NaN** — eager → NaN; ``jit_compile=True`` may → ±inf
  for min/max/maximum/minimum and similar.
* **Empty axis ``min``/``max``** — length-0 is framework-defined (often error).

GPU notes (no GPU CI)
---------------------

* Torch CUDA scatter still wants int64 ids (cast on Torch path).
* Outputs should stay on the input device (fills / arange / empty identities).
* Prefer float32 compares; GPU TF32 and float64 support vary.
* Equal-value tie order for min/max is not portable under atomics.
* Empty CUDA buffers and GPU XLA / ``torch.compile`` are stricter than CPU;
  MPS ≠ CUDA.
"""

from __future__ import annotations

from typing import Any


def empty_segment_identity(dtype: Any, reduction: str, *, xp: Any) -> Any:
    """Return the AnyTensor empty-segment identity for ``reduction`` on ``dtype``.

    This is the portable fill used for empty slots in ``segment_sum`` /
    ``segment_min`` / ``segment_max`` (and backends that implement them).

    Args:
        dtype: Target dtype (NumPy / Array API / TF dtype with
            ``as_numpy_dtype``).
        reduction: One of ``"sum"``, ``"min"``, ``"max"``.
        xp: Array API-ish namespace with ``iinfo``, ``asarray``, and floating
            constants (``inf``) — e.g. ``numpy`` or an ``array_api_compat`` ns.

    Returns:
        ``0`` for ``sum``; ``+inf`` / ``iinfo.max`` for ``min``; ``-inf`` /
        ``iinfo.min`` for ``max``.

    Notes:
        Differs from TF ``unsorted_segment_{min,max}``, which fill empties
        (and ±inf-only segments) with **finfo** limits. AnyTensor uses these
        identities so empties and occupied ±inf match NumPy/JAX/Torch.
    """
    if reduction == "sum":
        return 0
    if reduction not in ("min", "max"):
        raise ValueError(f"reduction type {reduction} not supported")

    # Prefer Array API / NumPy dtype checks; accept TF dtypes via as_numpy_dtype.
    is_float = False
    try:
        import numpy as np

        np_dtype = getattr(dtype, "as_numpy_dtype", dtype)
        is_float = np.issubdtype(np_dtype, np.floating)
    except Exception:
        kind = getattr(dtype, "kind", None)
        is_float = kind == "f"

    if is_float:
        return xp.inf if reduction == "min" else -xp.inf

    info = xp.iinfo(dtype)
    return info.max if reduction == "min" else info.min
