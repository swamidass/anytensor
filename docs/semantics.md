# Surprising differences

Cross-backend and symbolic fuzz turned up several places where frameworks
disagree, or where a “native” op looks right until you hit NaN, ±inf, or
`jit` / `tf.function`. AnyTensor **standardizes** some of these; others stay
**backend-local**. The machine-readable contract lives in
[`anytensor.semantics`](api/semantics_api.md).

## What we standardize

| Topic | Raw framework surprise | AnyTensor behavior |
|---|---|---|
| Empty `segment_min` / `max` | TF `unsorted_segment_{min,max}` fills empties with **finfo** limits; maps a segment whose only value is `+inf`/`-inf` to finfo too | Empty → **±inf** (floats) or **iinfo** (ints); occupied ±inf stays ±inf |
| NaN in TF `segment_min` / `max` | `tensor_scatter_nd_{min,max}` **ignores** NaN updates, so a NaN-only segment keeps the ±inf identity | After scatter, any segment that saw a NaN becomes **NaN** |
| `num_segments` | Easy to infer as `max(ids)+1` | **Required** (JAX convention). Kind `shape`: Python `int`, jit symbolic constant, or 0-d integral tensor — never inferred |
| Scalar `repeats` under TF graph | Promoting a Python `2` to a 0-d TF tensor breaks `tf.experimental.numpy.repeat` | Python scalar repeats stay Python; TF shim uses `tf.repeat` / `tf.range` |
| `zeros_like` / `full_like` under `tf.function` | After retracing, `x.shape` is `(None,)` and `tnp.zeros` errors | Use symbolic `shape(x)` (static dim or `tf.shape` component) |

## Backend-local (we document, do not unify)

| Topic | What happens |
|---|---|
| **Index width** | Torch scatter needs **int64** (we cast). JAX without `jax_enable_x64` often keeps **int32** and may warn/truncate on int64 ids. TF often int32. Do not assume NumPy int64 ids stay int64 after upcast. |
| **Float width / underflow** | JAX may truncate float64→float32 without x64. ``inf *`` subnormal or float32-min may be ``inf`` (NumPy / eager TF) vs ``nan`` (JAX / TF XLA) when the tiny flushes to 0. Fuzz keeps finite samples at ``|x| >= 1e-3`` or exact 0. |
| **`sorted=`** | Honored on JAX/TF; **no-op** on NumPy/Torch (unsorted-safe path). |
| **`jax.jit` + `repeat` / `partition_softmax`** | `jnp.repeat` needs a **static** repeat count or `total_repeat_length`. `partition_softmax` requires `total_length` (`shape(logits)[0]`); `num_segments` is `shape(partitions)[0]`, not an argument. |
| **TF XLA vs eager with NaN** | Eager often yields NaN; `tf.function(jit_compile=True)` may yield **±inf** for `min`/`max`/`maximum`/`minimum` and similar. Not portable — avoid relying on NaN under XLA. |
| **Empty axis `min`/`max`** | Length-0 reductions are framework-defined (often error). Prefer nonempty. |
| **GPU (no GPU CI)** | Torch CUDA still wants int64 ids; keep outputs on the input device; compare float32; equal-value tie order is not portable under atomics; empty CUDA / GPU XLA are stricter than CPU; MPS ≠ CUDA. |

## Empty segment slots

When `num_segments` is larger than the set of ids present (or an id never
appears), empty slots keep the reduction **identity**:

| reduction | floating | integral |
|---|---|---|
| `segment_sum` | `0` | `0` |
| `segment_min` | `+inf` | `iinfo(dtype).max` |
| `segment_max` | `-inf` | `iinfo(dtype).min` |

Occupied slots always do a real reduce (including NaN / ±inf). Prefer
`segment_min_or_constant` / `segment_max_or_constant` when empties should be a
finite fill.

## How TF segment min/max is implemented

1. Scatter from an AnyTensor empty identity (±inf / iinfo) so empties and
   occupied ±inf match other backends.
2. OR in per-segment NaN via `unsorted_segment_max(is_nan(x))`, because scatter
   ignores NaN updates.

Ordinary TF math goes through a `tf.experimental.numpy` shim (array-api-compat
has no TF backend yet).
