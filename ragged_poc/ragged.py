"""Composable ragged / partitioned tensor on top of AnyTensor.

Canonical storage is flat ``values`` + ``row_ids`` + ``nrows`` (segment
discipline). Lengths and row-splits are construction helpers / derived views.

Arithmetic / matmul use Python ``__`` methods on ``values`` only; the index
vector is unchanged (``with_values`` reuses the same ``row_ids`` object).
Prefer ``at.*`` for portable library code; ``np`` / ``torch`` dispatch is a
convenience (and an antipattern for new code).

This is a prototype: APIs may change before any move into ``anytensor``.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

import anytensor as at
from anytensor.structure import apply_structured, register_structure


def lengths_to_row_ids(
    lengths: Any,
    *,
    total_length: Optional[at.ShapeSize] = None,
) -> Any:
    """Expand partition lengths into a flat row-id vector.

    Example: ``lengths=[2, 0, 3]`` → ``[0, 0, 2, 2, 2]``.

    Under ``jax.jit``, pass a static ``total_length`` (``sum(lengths)``) so
    ``at.repeat`` can compile.
    """
    n_rows = at.shape(lengths)[0]
    ids = at.arange(n_rows, like=lengths)
    if total_length is None:
        return at.repeat(ids, lengths)
    return at.repeat(ids, lengths, total_repeat_length=total_length)


def row_ids_to_lengths(row_ids: Any, nrows: at.ShapeSize) -> Any:
    """Count elements per row (integral; empty rows are ``0``)."""
    ones = at.ones_like(row_ids)
    return at.segment_sum(ones, row_ids, num_segments=nrows)


def lengths_to_row_splits(lengths: Any) -> Any:
    """Exclusive row splits: ``[0, l0, l0+l1, ...]``."""
    zero = at.zeros((1,), like=lengths)
    prefixed = at.concatenate([zero, lengths], axis=0)
    return at.cumsum(prefixed, axis=0)


def row_splits_to_lengths(row_splits: Any) -> Any:
    """``lengths[i] = row_splits[i+1] - row_splits[i]``."""
    return row_splits[1:] - row_splits[:-1]


def concatenate(raggeds: Sequence[Ragged], axis: int = 0) -> Ragged:
    """Concatenate ragged tensors. See :meth:`Ragged.concatenate`."""
    return Ragged.concatenate(raggeds, axis=axis)


def _all_equal(a: Any, b: Any) -> bool:
    """Host equality for partition ids. Under tracing, only ``is`` is safe."""
    if a is b:
        return True
    eq = a == b
    try:
        all_fn = getattr(eq, "all", None)
        if callable(all_fn):
            return bool(all_fn())
        return bool(eq)
    except Exception as exc:
        # jax TracerBoolConversionError / TF symbolic bool, etc.
        raise ValueError(
            "cannot compare ragged row_ids under tracing; reuse the same "
            "row_ids object (e.g. with_values / shared partition metadata)"
        ) from exc


def _host_concrete_int(value: Any) -> Optional[int]:
    """``int(value)`` when host-readable; else ``None`` (tracing / symbolic)."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
    except ImportError:  # pragma: no cover
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _require_matching_partition(a: "Ragged", b: "Ragged", *, what: str) -> None:
    """Require exact partition match.

    Tracing-safe path: shared ``row_ids`` object (+ concrete or identical
    ``nrows``). Value equality of distinct id vectors is only checked when
    host-concrete; otherwise raise so callers share partition metadata.
    """
    if a is b:
        return
    if a.row_ids is b.row_ids:
        if a.nrows is b.nrows:
            return
        na, nb = _host_concrete_int(a.nrows), _host_concrete_int(b.nrows)
        if na is not None and nb is not None:
            if na != nb:
                raise ValueError(
                    f"{what}: nrows mismatch ({na} vs {nb}) with shared row_ids"
                )
            return
        raise ValueError(
            f"{what}: cannot verify nrows under tracing; use the same Python "
            "int (or identical) nrows object when sharing row_ids"
        )

    na, nb = _host_concrete_int(a.nrows), _host_concrete_int(b.nrows)
    if na is None or nb is None:
        raise ValueError(
            f"{what}: under tracing, feature-axis concatenate requires "
            "shared row_ids object identity (not merely equal values)"
        )
    if na != nb:
        raise ValueError(
            f"{what}: exact matching ragged partition required "
            f"(nrows {na} vs {nb})"
        )
    if _all_equal(a.row_ids, b.row_ids):
        return
    raise ValueError(
        f"{what}: exact matching ragged partition required "
        "(same nrows and row_ids)"
    )


def _is_full_slice(key: Any) -> bool:
    return (
        isinstance(key, slice)
        and key.start is None
        and key.stop is None
        and key.step is None
    )


def _expand_ellipsis(keys: tuple, ndim: int) -> tuple:
    """Expand a single ``...`` to fill missing logical axes."""
    ellipsis_at = [i for i, k in enumerate(keys) if k is Ellipsis]
    if not ellipsis_at:
        return keys
    if len(ellipsis_at) > 1:
        raise IndexError("only one ellipsis is allowed")
    i = ellipsis_at[0]
    before, after = keys[:i], keys[i + 1 :]
    n_fill = ndim - len(before) - len(after)
    if n_fill < 0:
        raise IndexError("too many indices for ragged tensor")
    return before + (slice(None),) * n_fill + after


def _leading_len(x: Any) -> int:
    shape = getattr(x, "shape", ())
    if not shape:
        raise TypeError(f"expected an array index, got {type(x)!r}")
    return int(shape[0])


def _as_index_array(key: Any):
    """Host numpy view of an index array (bool or integer)."""
    import numpy as np

    return np.asarray(key)


def _is_static_one(dim: Any) -> bool:
    """True when ``dim`` is known to be 1 (Python/NumPy), not a traced size."""
    if isinstance(dim, bool):
        return False
    if isinstance(dim, int):
        return dim == 1
    try:
        import numpy as np

        if isinstance(dim, np.integer):
            return int(dim) == 1
    except ImportError:  # pragma: no cover
        pass
    # Traced / symbolic sizes: not known to be one
    return False


def _rank(x: Any) -> int:
    ndim = getattr(x, "ndim", None)
    if ndim is None:
        raise TypeError(f"expected an array, got {type(x)!r}")
    return int(ndim)


def _is_scalar_like(x: Any) -> bool:
    if isinstance(x, (bool, int, float, complex)):
        return True
    try:
        import numpy as np

        if isinstance(x, np.generic):
            return True
    except ImportError:  # pragma: no cover
        pass
    shape = getattr(x, "shape", None)
    return shape is not None and len(shape) == 0


@register_structure
@dataclass(eq=False)
class Ragged:
    """Single-axis ragged tensor backed by AnyTensor segment ops.

    Logical layout is ``(batch, ragged, *F)`` stored as flat ``values`` of shape
    ``(N, *F)`` plus ``row_ids`` / ``nrows``.

    Attributes:
        values: Flat payload. Leading axis length ``N`` matches ``row_ids``.
        row_ids: Integer id in ``[0, nrows)`` for each leading row of ``values``.
            May be unsorted / non-unique (GNN-style). Out-of-range ids follow
            AnyTensor segment drop semantics. **Never mutated by elementwise
            / library ops** — those touch ``values`` only.
        nrows: Required shape-size (Python ``int`` preferred under jit).

    Operators (``+``, ``*``, ``@``, ``**``, ``^``, …) run on ``values`` with
    normal array broadcasting; ``row_ids`` are reused. Two :class:`Ragged`
    operands must share a partition. ``eq=False`` so ``==`` is elementwise.

    Indexing always returns a :class:`Ragged` (never silently densifies).
    """

    values: Any
    row_ids: Any
    nrows: at.ShapeSize

    # --- constructors -----------------------------------------------------

    @classmethod
    def from_row_ids(cls, values: Any, row_ids: Any, nrows: at.ShapeSize) -> Ragged:
        """Build from unsorted / sparse row ids (canonical form)."""
        return cls(values=values, row_ids=row_ids, nrows=nrows)

    @classmethod
    def from_lengths(
        cls,
        values: Any,
        lengths: Any,
        *,
        total_length: Optional[at.ShapeSize] = None,
    ) -> Ragged:
        """Build from contiguous partition lengths (jraph-style)."""
        nrows = at.shape(lengths)[0]
        row_ids = lengths_to_row_ids(lengths, total_length=total_length)
        return cls(values=values, row_ids=row_ids, nrows=nrows)

    @classmethod
    def from_row_splits(cls, values: Any, row_splits: Any) -> Ragged:
        """Build from TF-style exclusive row splits."""
        lengths = row_splits_to_lengths(row_splits)
        return cls.from_lengths(values, lengths)

    @classmethod
    def concatenate(cls, raggeds: Sequence[Ragged], axis: int = 0) -> Ragged:
        """Concatenate along a logical axis of ``(batch, ragged, *F)``.

        - ``axis=0`` (batch): offset ``row_ids`` by running ``nrows``, concat
          packed ``values``; partitions need not match. Prefer Python-int
          ``nrows`` under ``jax.jit`` / ``tf.function`` / ``torch.compile``.
        - ``axis >= 2`` (feature): requires an **exact** matching ragged
          partition. Under tracing, share the same ``row_ids`` object
          (``with_values``); host value-equality is used eagerly only.
        - ``axis=1`` (ragged packing): not supported.
        """
        seq = list(raggeds)
        if not seq:
            raise ValueError("need at least one Ragged to concatenate")
        if not all(isinstance(r, Ragged) for r in seq):
            raise TypeError("concatenate expects only Ragged operands")
        head = seq[0]
        ndim = head.logical_ndim
        ax = operator.index(axis)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for logical ndim={ndim}"
            )
        for r in seq[1:]:
            if int(getattr(r.values, "ndim", -1)) != int(
                getattr(head.values, "ndim", -1)
            ):
                raise ValueError(
                    "Ragged concatenate requires matching values.ndim "
                    f"({getattr(head.values, 'ndim', None)} vs "
                    f"{getattr(r.values, 'ndim', None)})"
                )

        if ax == 0:
            return cls._concatenate_batch(seq)
        if ax == 1:
            raise ValueError(
                "concatenate along the ragged axis (axis=1) is not supported; "
                "use axis=0 (batch) or axis>=2 when partitions match exactly"
            )
        what = "concatenate along a feature axis"
        for r in seq[1:]:
            _require_matching_partition(head, r, what=what)
        values = at.concatenate([r.values for r in seq], axis=ax - 1)
        return head.with_values(values)

    @classmethod
    def _concatenate_batch(cls, seq: Sequence[Ragged]) -> Ragged:
        """Batch-axis concat; tracing-safe when ``nrows`` are Python ints."""
        values = at.concatenate([r.values for r in seq], axis=0)
        id_parts = []
        offset: Any = 0
        for r in seq:
            # Array + Python int (or tensor offset) — no host bool / int cast.
            id_parts.append(r.row_ids + offset)
            n = r.nrows
            n_c = _host_concrete_int(n)
            o_c = _host_concrete_int(offset)
            if n_c is not None and o_c is not None:
                offset = o_c + n_c
            else:
                offset = offset + n
        row_ids = at.concatenate(id_parts, axis=0)
        total = _host_concrete_int(offset)
        return cls(
            values=values,
            row_ids=row_ids,
            nrows=total if total is not None else offset,
        )

    # --- derived partitions -----------------------------------------------

    @property
    def logical_ndim(self) -> int:
        """``1 + values.ndim`` → batch + ragged packing axis + trailing ``F``."""
        return 1 + int(getattr(self.values, "ndim", 0))

    def lengths(self) -> Any:
        """Per-row counts (integral vector of shape ``(nrows,)``)."""
        return row_ids_to_lengths(self.row_ids, self.nrows)

    def row_splits(self) -> Any:
        """Exclusive splits of shape ``(nrows + 1,)`` from :meth:`lengths`.

        Note: if ``row_ids`` are unsorted, lengths still count correctly but
        the splits describe a *reordered* contiguous layout, not the current
        storage order of ``values``.
        """
        return lengths_to_row_splits(self.lengths())

    def with_values(self, values: Any) -> Ragged:
        """Same partition, new payload — **reuses** ``row_ids`` (no-op on ids)."""
        return Ragged(values=values, row_ids=self.row_ids, nrows=self.nrows)

    def same_structure(self, other: Any) -> bool:
        """True when ``other`` is a :class:`Ragged` with the same partition."""
        if other is self:
            return True
        if not isinstance(other, Ragged):
            return False
        return self.nrows == other.nrows and _all_equal(self.row_ids, other.row_ids)

    # --- indexing (logical batch / ragged / F; always Ragged) -------------

    def __getitem__(self, key: Any) -> Ragged:
        """Index as logical ``(batch, ragged, *F)``.

        Always returns a :class:`Ragged`. There is no silent densify path —
        a single batch row is ``nrows=1``, a single token per row stays ragged
        with length-1 rows.

        Examples::

            r[0]          # first batch row → Ragged with nrows=1
            r[0:2]        # batch slice
            r[[0, 2]]     # fancy batch gather (order preserved)
            r[batch_mask] # bool length ``nrows``
            r[value_mask] # bool length ``N`` (flat values)
            r[:, 0]       # token 0 within each row (length-1 rows)
            r[:, :, 1]    # feature 1; all batch & tokens
            r[..., 1]     # same as trailing feature select
        """
        keys = key if isinstance(key, tuple) else (key,)
        keys = _expand_ellipsis(keys, self.logical_ndim)
        if len(keys) > self.logical_ndim:
            raise IndexError(
                f"too many indices for ragged tensor of logical ndim "
                f"{self.logical_ndim}"
            )
        batch_key = keys[0] if keys else slice(None)
        ragged_key = keys[1] if len(keys) > 1 else slice(None)
        feat_keys = keys[2:] if len(keys) > 2 else ()

        out = self._index_batch(batch_key)
        out = out._index_ragged(ragged_key)
        if feat_keys:
            out = out.with_values(out.values[(slice(None),) + tuple(feat_keys)])
        if not isinstance(out, Ragged):  # pragma: no cover - invariant
            raise TypeError("ragged indexing must return Ragged, not dense")
        return out

    def _index_batch(self, key: Any) -> Ragged:
        if _is_full_slice(key):
            return self

        import numpy as np

        nrows = int(self.nrows)

        # Bool mask: length nrows → rows; length N → flat values.
        if hasattr(key, "dtype") and np.asarray(key).dtype == bool:
            mask = _as_index_array(key)
            if mask.ndim != 1:
                raise IndexError("boolean index must be 1-D")
            n = int(mask.shape[0])
            n_vals = int(self.values.shape[0])
            if n == nrows:
                return self._select_rows(list(np.flatnonzero(mask)))
            if n == n_vals:
                return self._mask_values(mask)
            raise IndexError(
                f"boolean index length {n} matches neither nrows={nrows} "
                f"nor n_values={n_vals}"
            )

        # Integer scalar batch row → nrows=1 Ragged (not dense).
        if isinstance(key, (int, np.integer)) or (
            hasattr(key, "shape") and tuple(getattr(key, "shape", ())) == ()
        ):
            i = int(np.asarray(key).item())
            if i < 0:
                i += nrows
            if i < 0 or i >= nrows:
                raise IndexError(f"batch index {int(np.asarray(key).item())} out of range for nrows={nrows}")
            return self._select_rows([i])

        if isinstance(key, slice):
            return self._select_rows(list(range(*key.indices(nrows))))

        # Fancy integer batch indices.
        idx = _as_index_array(key)
        if idx.ndim != 1:
            raise IndexError("batch fancy index must be 1-D")
        if idx.dtype == bool:
            raise IndexError("boolean batch index handled above")
        rows = []
        for raw in idx.tolist():
            i = int(raw)
            if i < 0:
                i += nrows
            if i < 0 or i >= nrows:
                raise IndexError(f"batch index {int(raw)} out of range for nrows={nrows}")
            rows.append(i)
        return self._select_rows(rows)

    def _select_rows(self, row_indices: list[int]) -> Ragged:
        """Gather batch rows in order; remaps ids to ``0..len-1``."""
        import numpy as np

        values = np.asarray(self.values)
        row_ids = np.asarray(self.row_ids)
        if not row_indices:
            return Ragged(values[:0], row_ids[:0], nrows=0)
        parts_v = []
        parts_i = []
        for new_i, old_i in enumerate(row_indices):
            part = values[row_ids == old_i]
            parts_v.append(part)
            parts_i.append(np.full((part.shape[0],), new_i, dtype=row_ids.dtype))
        return Ragged(
            np.concatenate(parts_v, axis=0),
            np.concatenate(parts_i, axis=0),
            nrows=len(row_indices),
        )

    def _mask_values(self, mask: Any) -> Ragged:
        """Keep flat entries where ``mask``; ``nrows`` unchanged (empty rows ok)."""
        import numpy as np

        mask = _as_index_array(mask)
        values = np.asarray(self.values)[mask]
        row_ids = np.asarray(self.row_ids)[mask]
        return Ragged(values, row_ids, nrows=self.nrows)

    def _index_ragged(self, key: Any) -> Ragged:
        if _is_full_slice(key):
            return self

        import numpy as np

        values = np.asarray(self.values)
        row_ids = np.asarray(self.row_ids)
        nrows = int(self.nrows)
        lengths = np.asarray(row_ids_to_lengths(row_ids, nrows))

        # Integer: one token per batch row → still Ragged (all lengths 1).
        if isinstance(key, (int, np.integer)) or (
            hasattr(key, "shape") and tuple(getattr(key, "shape", ())) == ()
        ):
            k = int(np.asarray(key).item())
            parts_v = []
            parts_i = []
            for b in range(nrows):
                L = int(lengths[b])
                kk = k + L if k < 0 else k
                if kk < 0 or kk >= L:
                    raise IndexError(
                        f"ragged index {k} out of range for batch {b} with length {L}"
                    )
                row_vals = values[row_ids == b]
                parts_v.append(row_vals[kk : kk + 1])
                parts_i.append(np.array([b], dtype=row_ids.dtype))
            return Ragged(
                np.concatenate(parts_v, axis=0),
                np.concatenate(parts_i, axis=0),
                nrows=nrows,
            )

        if isinstance(key, slice):
            parts_v = []
            parts_i = []
            for b in range(nrows):
                taken = values[row_ids == b][key]
                parts_v.append(taken)
                parts_i.append(np.full((taken.shape[0],), b, dtype=row_ids.dtype))
            if nrows == 0:
                return Ragged(values[:0], row_ids[:0], nrows=0)
            return Ragged(
                np.concatenate(parts_v, axis=0),
                np.concatenate(parts_i, axis=0),
                nrows=nrows,
            )

        raise TypeError(
            f"unsupported ragged-axis index type {type(key)!r}; "
            "use int or slice (no silent densify)"
        )

    def map(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Ragged:
        """Apply ``fn`` to flat ``values`` only; ``row_ids`` unchanged.

        Prefer operators / ``at.*``; ``map`` is the escape hatch for callables
        that do not go through either.
        """
        return self.with_values(fn(self.values, *args, **kwargs))

    # --- arithmetic / bitwise / compare / matmul (values only) ------------

    def broadcast_to_packed(self, dense: Any) -> Any:
        """Map ``dense`` onto packed ``values`` using *logical* dense broadcast rules.

        Logical shape is ``(batch, ragged, *F)`` — the same as a materialized
        padded tensor — but this **never materializes** padding. Only the packed
        encoding is produced / consumed.

        Dense may broadcast over the ragged axis only with size ``1`` there
        (after left-padding ones to logical rank). A non-1 ragged axis is an
        error (use another :class:`Ragged` with the same partition instead).

        Examples (logical ``(B, R, F)``)::

            scalar / (F,) / (1, 1, F)  → packed feature broadcast
            (B, 1, F)                 → take by row_ids, still packed
            (B, F)                    → error (does not broadcast with (B, R, F))
        """
        if _is_scalar_like(dense):
            return dense

        values = self.values
        logical_ndim = 1 + _rank(values)  # batch + ragged + trailing of values
        d = dense
        d_rank = _rank(d)
        if d_rank > logical_ndim:
            raise ValueError(
                f"dense rank {d_rank} exceeds logical ragged rank {logical_ndim}"
            )
        # Left-pad ones to logical rank (same as densifying shape for broadcast)
        for _ in range(logical_ndim - d_rank):
            d = d[None, ...]

        # d shape ~ (batch, ragged, *feat)
        ragged_extent = d.shape[1]
        if not _is_static_one(ragged_extent):
            raise ValueError(
                "dense operand must have size 1 on the ragged axis to broadcast "
                "over variable-length rows (pass shape (..., 1, ...) in that "
                "position); packed non-encoded slots stay non-encoded — this "
                "never materializes a dense pad"
            )
        # Drop ragged axis of size 1 → (batch, *feat)
        d = d[:, 0, ...]

        batch_extent = d.shape[0]
        if _is_static_one(batch_extent):
            # (1, *feat) broadcasts over packed leading axis under dense rules
            return d
        # Per-batch: gather to packed length N (still no B×Rmax materialization)
        return at.take(d, self.row_ids, axis=0)

    def _align_binary_operands(self, other: Any) -> tuple[Any, Any, Ragged]:
        """Align for elementwise ops: logical dense rules, packed execution.

        - Ragged + Ragged: same partition, then ``op`` on both packed ``values``
        - Ragged + dense/scalar: :meth:`broadcast_to_packed` then ``op`` on
          packed ``values`` only (never allocates a padded dense ragged)
        """
        if isinstance(other, Ragged):
            if not self.same_structure(other):
                raise ValueError(
                    "ragged operands must share the same partition "
                    f"(nrows={self.nrows} vs {other.nrows})"
                )
            return self.values, other.values, self
        return self.values, self.broadcast_to_packed(other), self

    def _binary(self, other: Any, op: Callable[..., Any]) -> Ragged:
        left, right, template = self._align_binary_operands(other)
        return template.with_values(op(left, right))

    def _binary_rev(self, other: Any, op: Callable[..., Any]) -> Ragged:
        if isinstance(other, Ragged):
            left, right, template = self._align_binary_operands(other)
            return template.with_values(op(right, left))
        return self.with_values(op(self.broadcast_to_packed(other), self.values))

    # arithmetic
    def __add__(self, other: Any) -> Any:
        return self._binary(other, operator.add)

    def __radd__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.add)

    def __sub__(self, other: Any) -> Any:
        return self._binary(other, operator.sub)

    def __rsub__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.sub)

    def __mul__(self, other: Any) -> Any:
        return self._binary(other, operator.mul)

    def __rmul__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.mul)

    def __matmul__(self, other: Any) -> Any:
        """``values @ other`` (e.g. ``r @ W``); leading axis / partition kept."""
        if isinstance(other, Ragged):
            raise ValueError("matmul between two Ragged tensors is not supported")
        return self.with_values(operator.matmul(self.values, other))

    def __rmatmul__(self, other: Any) -> Any:
        if isinstance(other, Ragged):
            raise ValueError("matmul between two Ragged tensors is not supported")
        return self.with_values(operator.matmul(other, self.values))

    def __truediv__(self, other: Any) -> Any:
        return self._binary(other, operator.truediv)

    def __rtruediv__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.truediv)

    def __floordiv__(self, other: Any) -> Any:
        return self._binary(other, operator.floordiv)

    def __rfloordiv__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.floordiv)

    def __mod__(self, other: Any) -> Any:
        return self._binary(other, operator.mod)

    def __rmod__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.mod)

    def __divmod__(self, other: Any) -> Any:
        left, right, template = self._align_binary_operands(other)
        q, r = divmod(left, right)
        return template.with_values(q), template.with_values(r)

    def __rdivmod__(self, other: Any) -> Any:
        if isinstance(other, Ragged):
            left, right, template = self._align_binary_operands(other)
            q, r = divmod(right, left)
            return template.with_values(q), template.with_values(r)
        other_p = self.broadcast_to_packed(other)
        q, r = divmod(other_p, self.values)
        return self.with_values(q), self.with_values(r)

    def __pow__(self, other: Any) -> Any:
        return self._binary(other, operator.pow)

    def __rpow__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.pow)

    # bitwise
    def __and__(self, other: Any) -> Any:
        return self._binary(other, operator.and_)

    def __rand__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.and_)

    def __or__(self, other: Any) -> Any:
        return self._binary(other, operator.or_)

    def __ror__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.or_)

    def __xor__(self, other: Any) -> Any:
        return self._binary(other, operator.xor)

    def __rxor__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.xor)

    def __lshift__(self, other: Any) -> Any:
        return self._binary(other, operator.lshift)

    def __rlshift__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.lshift)

    def __rshift__(self, other: Any) -> Any:
        return self._binary(other, operator.rshift)

    def __rrshift__(self, other: Any) -> Any:
        return self._binary_rev(other, operator.rshift)

    # comparisons (elementwise; dataclass eq disabled)
    def __eq__(self, other: Any) -> Any:
        return self._binary(other, operator.eq)

    def __ne__(self, other: Any) -> Any:
        return self._binary(other, operator.ne)

    def __lt__(self, other: Any) -> Any:
        return self._binary(other, operator.lt)

    def __le__(self, other: Any) -> Any:
        return self._binary(other, operator.le)

    def __gt__(self, other: Any) -> Any:
        return self._binary(other, operator.gt)

    def __ge__(self, other: Any) -> Any:
        return self._binary(other, operator.ge)

    # unary
    def __neg__(self) -> Ragged:
        return self.with_values(operator.neg(self.values))

    def __pos__(self) -> Ragged:
        return self.with_values(operator.pos(self.values))

    def __abs__(self) -> Ragged:
        return self.with_values(operator.abs(self.values))

    def __invert__(self) -> Ragged:
        return self.with_values(operator.invert(self.values))

    # in-place: return a new Ragged (caller rebinds); ids still shared
    def __iadd__(self, other: Any) -> Any:
        return self.__add__(other)

    def __isub__(self, other: Any) -> Any:
        return self.__sub__(other)

    def __imul__(self, other: Any) -> Any:
        return self.__mul__(other)

    def __imatmul__(self, other: Any) -> Any:
        return self.__matmul__(other)

    def __itruediv__(self, other: Any) -> Any:
        return self.__truediv__(other)

    def __ifloordiv__(self, other: Any) -> Any:
        return self.__floordiv__(other)

    def __imod__(self, other: Any) -> Any:
        return self.__mod__(other)

    def __ipow__(self, other: Any) -> Any:
        return self.__pow__(other)

    def __iand__(self, other: Any) -> Any:
        return self.__and__(other)

    def __ior__(self, other: Any) -> Any:
        return self.__or__(other)

    def __ixor__(self, other: Any) -> Any:
        return self.__xor__(other)

    def __ilshift__(self, other: Any) -> Any:
        return self.__lshift__(other)

    def __irshift__(self, other: Any) -> Any:
        return self.__rshift__(other)

    # --- NumPy / Torch dispatch (convenience; prefer at.* + operators) ----

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented
        try:
            return apply_structured(ufunc, inputs, kwargs)
        except ValueError:
            return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        import numpy as np

        if func is np.concatenate:
            return concatenate(args[0], **kwargs)
        try:
            return apply_structured(func, args, kwargs)
        except Exception:
            return NotImplemented

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        # torch.cat / concatenate: args[0] is the sequence
        name = getattr(func, "__name__", "")
        if name in ("cat", "concatenate") and args:
            return concatenate(args[0], **kwargs)
        try:
            return apply_structured(func, args, kwargs)
        except Exception:
            return NotImplemented

    # --- reductions -------------------------------------------------------

    def _normalize_values_axis(self, axis: int) -> int:
        ndim = int(getattr(self.values, "ndim", 0))
        ax = operator.index(axis)
        if ax < 0:
            ax += ndim
        if ax < 0 or ax >= ndim:
            raise ValueError(
                f"axis {axis} is out of bounds for values.ndim={ndim}"
            )
        return ax

    def _reduce(
        self,
        *,
        axis: Optional[int],
        sorted: bool,
        segment_fn: Callable[..., Any],
        dense_fn: Callable[..., Any],
    ) -> Any:
        """Reduce along ``axis``.

        - ``axis=0`` (ragged / segment axis): ``segment_*`` → dense
          ``(nrows,) + values.shape[1:]``
        - ``axis`` on an inner dim: ordinary ``at`` reduce on ``values`` →
          same partition (:class:`Ragged`)
        - ``axis=None``: full reduce over flat ``values`` → dense (ignores
          segmentation; same as dense ``at.sum(values)`` for sums)
        """
        if axis is None:
            return dense_fn(self.values, axes=None)
        ax = self._normalize_values_axis(axis)
        if ax == 0:
            return segment_fn(
                self.values, self.row_ids, self.nrows, sorted=sorted
            )
        return self.with_values(dense_fn(self.values, axes=ax))

    def sum(self, axis: Optional[int] = 0, *, sorted: bool = False) -> Any:
        """Sum. Default ``axis=0`` is per-row ``segment_sum`` (dense over rows)."""
        return self._reduce(
            axis=axis,
            sorted=sorted,
            segment_fn=at.segment_sum,
            dense_fn=at.sum,
        )

    def mean(self, axis: Optional[int] = 0, *, sorted: bool = False) -> Any:
        """Mean. Default ``axis=0`` is per-row ``segment_mean``."""
        return self._reduce(
            axis=axis,
            sorted=sorted,
            segment_fn=at.segment_mean,
            dense_fn=at.mean,
        )

    def min(self, axis: Optional[int] = 0, *, sorted: bool = False) -> Any:
        """Min. Default ``axis=0`` is per-row ``segment_min``."""
        return self._reduce(
            axis=axis,
            sorted=sorted,
            segment_fn=at.segment_min,
            dense_fn=at.min,
        )

    def max(self, axis: Optional[int] = 0, *, sorted: bool = False) -> Any:
        """Max. Default ``axis=0`` is per-row ``segment_max``."""
        return self._reduce(
            axis=axis,
            sorted=sorted,
            segment_fn=at.segment_max,
            dense_fn=at.max,
        )

    def count(self, *, sorted: bool = False) -> Any:
        """Float counts per row (AnyTensor ``segment_count`` convention)."""
        return at.segment_count(self.row_ids, self.nrows, sorted=sorted)

    # --- elementwise within rows (same ragged shape) ----------------------

    def softmax(self, *, sorted: bool = False) -> Ragged:
        """Stable softmax within each row; returns a new :class:`Ragged`."""
        out = at.segment_softmax(
            self.values, self.row_ids, self.nrows, sorted=sorted
        )
        return self.with_values(out)

    def normalize(self, *, sorted: bool = False) -> Ragged:
        """Divide by per-row sum (``0/0`` → ``0``)."""
        out = at.segment_normalize(
            self.values, self.row_ids, self.nrows, sorted=sorted
        )
        return self.with_values(out)


def _register_jax_pytree() -> None:
    """Values-only leaf so ``jax.tree.map`` never touches ``row_ids``."""
    try:
        from jax.tree_util import register_pytree_node
    except ImportError:
        return

    def flatten(r: Ragged):
        return ((r.values,), (r.row_ids, r.nrows))

    def unflatten(aux, children):
        row_ids, nrows = aux
        return Ragged(values=children[0], row_ids=row_ids, nrows=nrows)

    register_pytree_node(Ragged, flatten, unflatten)


_register_jax_pytree()
