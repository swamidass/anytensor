# Usage

## Install

```bash
# from git (NumPy included as a core dependency)
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
# optional backends
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
# or everything
pip install "anytensor[all]"
```

From a checkout:

```bash
uv sync --extra all --group dev --group docs
```

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])

y = at.segment_sum(x, seg_ids, num_segments=2)
m = at.segment_mean(x, seg_ids, 2)
```

Same code on JAX / Torch / TF:

```python
import jax.numpy as jnp

x = jnp.arange(12.0).reshape(3, 4)
seg_ids = jnp.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, 2)
```

## Scalar / NumPy promotion

Full reductions return **0-d arrays** (not bare Python / NumPy scalars). Binary
ops and segment helpers **upcast** Python scalars and **NumPy ndarrays** onto a
peer JAX / Torch / TF tensor. NumPy is host interchange data — we never demote
a framework tensor to NumPy when mixing. Scalars alone still default to NumPy.

NumPy → framework upcast prefers **by reference** (`asarray(..., copy=False)`)
when the backend can share the buffer. A copy is used only when zero-copy is
impossible (default: warn + copy; set `fallback="error"` to raise). Use
`@promote(..., copy=True)` or `with at.promote_options(copy=True):` when the
host NumPy buffer may be mutated.

Dtype policy is per-operand via `@promote`:

```python
@promote(x="data", y="data")           # result_type — ints widen beside floats
@promote(x="data", indices="index")  # indices stay integral
@promote(condition="mask", x="data", y="data")
@promote(x="data", segment_ids="index", num_segments="shape")  # size dim
```

`num_segments` / `total_repeat_length` / `sum_partitions` are **shape** sizes:
Python `int`, jit symbolic constant, or 0-d integral tensor scalar — **required**
(JAX convention), never inferred from ids. Plain ints stay Python (not 0-d
tensors) so `jax.jit` / `tf.function` / `torch.compile` can treat them as static.

| Situation | Behavior |
|---|---|
| Only scalars | NumPy 0-d arrays |
| NumPy + framework tensor | Promote NumPy → framework (never the reverse) |
| Default upcast | Prefer **reference** (`copy=False`) |
| Zero-copy impossible | `fallback="copy"` (warn) or `"error"` |
| Mutating host buffer | `@promote(..., copy=True)` or `promote_options(copy=True)` |

## Segment helpers

| Function | Role |
|---|---|
| `segment_sum` / `min` / `max` | Reduce along axis 0 by segment id |
| `segment_count` / `mean` / `variance` | Counts and moments |
| `segment_normalize` / `segment_softmax` | Per-segment normalize / softmax |
| `segment_min_or_constant` / `segment_max_or_constant` | Empty segments → constant |
| `partition_softmax` | Softmax over contiguous partition lengths |

Einops (`rearrange`, `einsum`, `reduce`, …) is re-exported for convenience.

Under `jax.jit`, pass a static `sum_partitions` to `partition_softmax` so
`jnp.repeat` can compile. See [Surprising differences](semantics.md).

## TorchScript

`torch.jit.script` cannot follow AAC / backend dispatch. After
`anytensor.enable_torchscript()` (automatic if `torch` was imported before
`anytensor`, or when the Torch backend first loads), public
`segment_sum` / `min` / `max` gain a `torch.jit.is_scripting()` divert to
`anytensor.torchscript`.

**Eager behavior stays multi-backend.** NumPy, JAX, Torch, and TF tensors still
dispatch normally when not scripting — enabling TorchScript does not replace
those paths with Torch-only functions.

That is the library-friendly pattern: a library writes portable AnyTensor
calls; an end user who scripts their own code can still compile through those
calls.

```python
# library.py — no TorchScript knowledge required
import anytensor as at

def message_pass(x, edge_index, n_node: int):
    return at.segment_sum(x, edge_index, n_node)

# user code
import torch
import anytensor as at
import library

at.enable_torchscript()  # no-op if torch was imported before anytensor

@torch.jit.script
def f(x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return library.message_pass(x, s, 5)
```

You can also call `anytensor.torchscript.segment_sum` directly inside a
scripted function. `torch.jit.trace` / `torch.compile` can use `at.segment_*`
without the divert. Under script, `num_segments` must be a Python `int`.

Parity is fuzzed: scripted `segment_sum` / `min` / `max` vs eager Torch
(`pytest -m fuzz`, `test_fuzz_torchscript_matches_eager`).

## Typing

Public APIs use [jaxtyping](https://docs.kidger.site/jaxtyping/) annotations
(`Shaped` / `Integer` / helpers like `SegmentValues`) plus an `ArrayT`
`TypeVar` so operands stay on one backend type — without importing Torch /
JAX / TF for typing.

**Runtime checking is off by default.** Annotations are for editors, static
checkers, and docs. To opt in::

```python
from jaxtyping import install_import_hook
install_import_hook("anytensor", "beartype.beartype")  # before importing anytensor
import anytensor as at
# or: at.enable_typecheck() before other anytensor submodule imports
```

Install `beartype` via `anytensor[typecheck]`, `anytensor[all]`, or the `dev`
group. Set `JAXTYPING_DISABLE=1` to force runtime checks off.

## NaN helpers / specials

- `is_nan` / `is_finite` / `is_inf` (aliases `isnan` / `isfinite` / `isinf`)
- `fill_nan` (alias `nan_fill`), `fill_nan_mask` → `(filled, mask)`
- Array API `nan_to_num`, element-wise `equal_nan`
- `inf(x)` / `ninf(x)` / `nan(x)` / `pi(x)` / `e(x)`, `dtype(...)`, `finfo` / `iinfo`
- `newaxis` is `None`
