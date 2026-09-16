"""Portable semantics for segment reductions and other cross-backend contracts.

This module is the **source of truth** for behaviors AnyTensor standardizes so
NumPy / JAX / Torch / TF agree. See also the “Portable differences” section in
``README.md``.

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

Backend-local (not unified)
---------------------------

* **Index width** — must be integral; Torch scatter casts to int64; JAX/TF
  often keep int32 (especially without JAX x64).
* **``sorted=``** — honored on JAX/TF; no-op on NumPy/Torch (unsorted-safe).
* **Default float width** — JAX may truncate float64→float32 without x64.
"""

from __future__ import annotations

from typing import Any


def empty_segment_identity(dtype, reduction: str, *, xp: Any) -> Any:
    """Return the AnyTensor empty-segment identity for ``reduction`` on ``dtype``.

    ``xp`` is an Array API-ish namespace with ``iinfo``, ``asarray``, and
    floating constants (``inf``) — e.g. ``numpy`` or ``array_api_compat`` ns.
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
