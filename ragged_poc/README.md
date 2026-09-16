# ragged_poc

Prototype **ragged / partitioned tensors** on top of AnyTensor segment
primitives. Not part of the public `anytensor` package yet — intended to move
in (or become a sibling module) once the shape feels right.

## Mental model

Canonical form: **`values` + `row_ids` + `nrows`**.

Ops touch **`values` only**. The index vector is a no-op —
`with_values` reuses the same `row_ids` object.

## Preferred API

| Prefer | Why |
|---|---|
| `r + x`, `r * x`, `r @ W`, `r ** 2`, `r ^ 1`, `-r`, … | Magic methods on `values` |
| `at.exp(r)`, `at.matmul(r, W)`, … | Portable library path (`anytensor.structure`) |
| `r.sum()` / `r.softmax()` | Segment / row reductions |

`np.exp(r)` / `torch.exp(r)` still work via dispatch hooks, but treat that as a
**convenience antipattern** — new code should use `at.*` or operators.

| Not the goal | Notes |
|---|---|
| `jnp.exp(r)` | No JAX custom-type hook; use `at` / `jax.tree.map` / operators on values |
| Guaranteeing every TF op | Use `at` or convert when needed |

## Indexing

Logical layout ``(batch, ragged, *F)``. **`__getitem__` always returns
`Ragged`** — never silently materializes dense (a single batch row is
`nrows=1`; a single token per row is length-1 rows).

| Key | Meaning |
|---|---|
| `r[i]` / `r[i:j]` / `r[[i, k]]` | batch select |
| `r[bool]` length `nrows` | batch mask |
| `r[bool]` length `N` | flat value mask (`nrows` kept) |
| `r[:, t]` | token `t` within each row |
| `r[:, :, f]` / `r[..., f]` | feature select on `values` |

## Broadcast arithmetic

**Semantics** = dense broadcast against logical ``(batch, ragged, *F)``  
**Execution** = packed ``values`` only — padding is never materialized
(``r + 1`` is ``values + 1``, not a dense ``(B, Rmax, F)`` fill).

Consequence of ordinary dense rules: a **size-1** axis aligned to
``ragged`` broadcasts across every row’s variable length (same as ``1``
broadcasting over ``R`` in a materialized ``(B, R, F)``). Non-encoded
slots stay non-encoded — we only apply the op to packed entries (via
``values`` and, when needed, ``take`` by ``row_ids``).

| Dense operand (after left-pad to logical rank) | Result |
|---|---|
| scalar / ``(F,)`` → ``(1, 1, F)`` | packed feature / scalar broadcast |
| ``(B, 1, F)`` (ragged axis = 1) | ``take`` by ``row_ids`` → packed ``(N, F)`` |
| ragged-axis size ≠ ``1`` (e.g. ``(B, F)``, ``(N, F)``) | ``ValueError`` |
| other ``Ragged`` | same partition; ``op`` on both packed ``values`` |

Axis roles come from rank and where size-``1`` sits, not from matching
``nrows`` vs ``N``.

## Concatenate

Logical ``(batch, ragged, *F)``:

| Axis | Rule |
|---|---|
| ``0`` (batch) | Always: offset ``row_ids``, concat packed ``values``, sum ``nrows`` |
| ``1`` (ragged) | Not supported |
| ``>= 2`` (feature) | Only if partitions match **exactly**; concat on ``values`` |

```python
at.concatenate([r1, r2])              # batch
at.concatenate([r_a, r_b], axis=2)    # feature; same row_ids
```

Under ``jax.jit`` / ``tf.function`` / ``torch.compile``: keep ``nrows`` as
Python ints; for feature-axis concat share the same ``row_ids`` object
(``with_values``). Distinct-but-equal id vectors cannot be checked while
tracing — that path raises.

## Reductions

`r.sum` / `mean` / `min` / `max` take `axis` (default `0` = ragged axis):

| `axis` | Behavior | Result |
|---|---|---|
| `0` (default) | `segment_*` over `row_ids` | dense `(nrows,) + …` |
| inner (`1`, `-1`, …) | ordinary `at` reduce on `values` | same partition (`Ragged`) |
| `None` | full reduce of flat `values` | dense (segmentation ignored) |

## Scope (v0)

- Single ragged dimension; ids / lengths / row_splits constructors
- Segment reduce / softmax / normalize
- Operators + `at` structure peel (index untouched)
- NumPy smoke tests (+ optional Torch)

## Run

```bash
uv run pytest ragged_poc -q
uv run pytest test/test_structure.py -q
```
