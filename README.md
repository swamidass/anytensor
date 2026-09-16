# AnyTensor

Portable tensor ops across **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, with a focus on **segment / GNN** primitives.

Ordinary math (`sum`, `exp`, `reshape`, …) delegates to the [Python Array API](https://data-apis.org/array-api/latest/) via [`array-api-compat`](https://github.com/data-apis/array-api-compat). Segment reductions stay on thin **input-adaptive** backends (einops-style `get_backend(x)`), so you write one function and run it on whatever tensor the caller already has.

## Install

```bash
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
# optional backends
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
# or
pip install "anytensor[all]"
```

Requires Python ≥3.10. Backend floors: NumPy ≥1.24, JAX ≥0.4.32, PyTorch ≥2.0, TensorFlow ≥2.10.

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])

y = at.segment_sum(x, seg_ids, num_segments=2)
m = at.segment_mean(x, seg_ids, 2)
```

The same calls work on JAX / Torch / TF tensors without changing the code:

```python
import jax.numpy as jnp

x = jnp.arange(12.0).reshape(3, 4)
seg_ids = jnp.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, 2)
```

### Scalar / NumPy promotion

Full reductions return **0-d arrays** (not bare Python / NumPy scalars). Binary ops and segment helpers **upcast** Python scalars and **NumPy ndarrays** onto a peer JAX / Torch / TF tensor. NumPy is host interchange data — we never demote a framework tensor to NumPy when mixing. Scalars alone still default to NumPy.

NumPy → framework upcast prefers **by reference** (`asarray(..., copy=False)`) when the backend can share the buffer (Torch does; JAX/TF may still materialize). A copy is used only when zero-copy is impossible (default: warn + copy; set `fallback="error"` to raise). Use `@promote(..., copy=True)` or `with at.promote_options(copy=True):` when the host NumPy buffer may be mutated.

Dtype policy is per-operand via `@promote`:

```python
@promote(x="data", y="data")           # result_type — ints widen beside floats
@promote(x="data", indices="index")  # indices stay integral
@promote(condition="mask", x="data", y="data")
```

### Portable differences (read this)

These are the behaviors AnyTensor **standardizes** or **documents as backend-local**. Full tables live in [`anytensor/semantics.py`](anytensor/semantics.py).

#### Empty segment slots (standardized)

When `num_segments` is larger than the set of ids present (or an id never appears), empty slots keep the reduction **identity**:

| reduction | floating | integral |
|---|---|---|
| `segment_sum` | `0` | `0` |
| `segment_min` | `+inf` | `iinfo(dtype).max` |
| `segment_max` | `-inf` | `iinfo(dtype).min` |

- Occupied slots always do a real reduce (e.g. a segment whose only value is `+inf` stays `+inf` for min — not a finfo stand-in).
- Prefer `segment_min_or_constant` / `segment_max_or_constant` when empties should be a finite fill.
- `sorted=` is honored on JAX/TF; on NumPy/Torch the path is unsorted-safe and `sorted` is currently a no-op.

#### Index integer width (backend-local)

`@promote(..., "index")` only requires an **integral** dtype. Width is not unified:

| Backend | Typical index dtype | Notes |
|---|---|---|
| NumPy | `int64` (or platform `int_`) | Host ids often start here |
| PyTorch | cast to **`int64`** at scatter | Required by `scatter_*` |
| JAX | often **`int32`** without `jax_enable_x64` | `int64` may truncate with a warning |
| TensorFlow | often **`int32`** | Mixed width can friction in graphs |

Do not assume NumPy `int64` segment ids stay `int64` after upcast to JAX.

#### Float width / NaN / ±inf

- Cross-backend fuzz compares in **float32** where JAX defaults truncate `float64`.
- NaN and ±inf are in-scope for portable math and segment ops; comparisons use `equal_nan=True` in tests.
- Helpers: `is_nan` / `is_finite` / `is_inf` (aliases `isnan` / `isfinite` / `isinf`), `fill_nan` (alias `nan_fill`), `fill_nan_mask` → `(filled, mask)` with mask True where NaN was, Array API `nan_to_num`, and element-wise `equal_nan`.
- Portable scalars on the package: `inf` / `ninf` / `nan` / `pi` / `e` / `newaxis` (Python floats). Framework dtypes via `dtype("bool", like=x)` or strings in `astype` / `zeros`; limits via `finfo(x)` / `iinfo(x)`. Backends stay internal.
- Empty **axis reductions** (`min`/`max` on length-0) remain framework-defined (often error); prefer nonempty for those.

#### NumPy promotion / copy

| Situation | Behavior |
|---|---|
| Only scalars | NumPy 0-d arrays |
| NumPy + framework tensor | Promote NumPy → framework (never the reverse) |
| Default upcast | Prefer **reference** (`copy=False`) |
| Zero-copy impossible | `fallback="copy"` (warn) or `"error"` |
| Mutating host buffer | `@promote(..., copy=True)` or `promote_options(copy=True)` |

### Segment helpers

| Function | Role |
|---|---|
| `segment_sum` / `min` / `max` | Reduce along axis 0 by segment id |
| `segment_count` / `mean` / `variance` | Counts and moments |
| `segment_normalize` / `segment_softmax` | Per-segment normalize / softmax |
| `segment_min_or_constant` / `segment_max_or_constant` | Empty segments → constant |
| `partition_softmax` | Softmax over contiguous partition lengths (array logits only) |

Einops (`rearrange`, `einsum`, `reduce`, …) is re-exported for convenience.

## Design

- **Ordinary ops** → `array_api_compat.array_namespace(x)` (+ `@as_array_result` / `@promote_scalars`).
- **Segment ops** → `backends.get_backend(x).segment_reduce(...)`.
- Backends are imported lazily; missing optional deps are fine until you pass that framework’s tensors.

## Roadmap

- [x] Ordinary ops via Array API compat
- [x] Segment helpers for GNN / ragged workloads
- [ ] jraph-style `GraphsTuple` API
- [ ] RaggedTensor / more partition helpers
- [ ] Broader contract tests in CI

## Contributing

PRs welcome. Prefer adding portable helpers in `anytensor/core.py` or `segment.py`; only extend `backends.py` when the Array API cannot express the op (e.g. `segment_reduce`). Add coverage in `test/test_ops.py`, boundaries in `test/test_boundaries.py`, fuzz in `test/test_hypothesis.py`, and backend contracts in `test/test_backend_contracts.py`.

```bash
uv sync --extra jax --extra torch --group dev
uv run pytest --cov=anytensor --cov-report=term-missing
```

## License

MIT. Backend dispatch patterns adapted from [einops](https://github.com/arogozhnikov/einops); see `NOTICE`.
