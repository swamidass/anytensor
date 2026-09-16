# Design

This page explains **why** AnyTensor is shaped the way it is, and what that
implies when you hit an edge case. For the concrete matrix of “framework A
does X, we do Y,” see [Surprising differences](semantics.md). For runnable
compile/script recipes, see [Worked examples](examples.md).

--

## Mental model (short)

1. Pass the tensors you already have.
2. Tag operands with the right promote kind (`data` / `index` / `mask` / `shape`).
3. Always pass shape-sizes (`num_segments`, …) as static-friendly values.
4. Trust empty-segment identities and TF NaN OR-in for segment min/max.
5. Treat index width, XLA NaN, and GPU ties as non-portable.
6. For TorchScript, only rely on the `segment_sum` / `min` / `max` divert.

That is the design: a small set of hard contracts, and clear warnings everywhere
else.

---

## Design goals

1. **Write once, run on the caller’s tensors.** Library code should not fork
   into NumPy / JAX / Torch / TF copies of the same algorithm.
2. **Segment / GNN ops are first-class.** Scatter-style reductions are where
   frameworks diverge most; we own that surface instead of waving at “use the
   Array API.”
3. **Prefer standards, patch where necessary.** Ordinary math rides the
   [Python Array API](https://data-apis.org/array-api/latest/) via
   [`array-api-compat`](https://github.com/data-apis/array-api-compat). Where
   AAC cannot express an op (or TF is missing), we add a thin shim — not a
   parallel math library.
4. **Standardize contracts we can defend; document the rest.** Empty-segment
   identities and TF NaN-in-scatter are portable. Index width, XLA-vs-eager
   NaN, and GPU atomics are not — we tell you so.
5. **Keep compile paths honest.** `num_segments` is a shape-size (JAX
   discipline). TorchScript gets an explicit divert, not a pretend-portable
   script of AAC dispatch.

Non-goals (for now): a full GraphsTuple / RaggedTensor API, ONNX Runtime as a
backend, or papering over every XLA vs eager disagreement.

---

## Hybrid architecture

```text
                    ┌─────────────────────────┐
   at.sum(x)  ───►  │ array-api-compat (AAC)  │  ordinary math
   at.exp(x)        │ + TF tnp shim           │
                    └─────────────────────────┘
                    ┌─────────────────────────┐
   at.segment_* ─►  │ backends.get_backend(x) │  NumPy / JAX / Torch / TF
                    │   .segment_reduce(...)  │
                    └─────────────────────────┘
```

**Why not “everything through AAC”?** Segment reductions are not in the Array
API. Each framework’s scatter / unsorted-segment / `jax.ops.segment_*` has
different empty-slot fills, NaN rules, and index dtypes. A single
`get_backend(x).segment_reduce(...)` keeps that complexity in one place.

**Why not “everything through custom backends”?** Reimplementing `matmul`,
`where`, `reshape`, … would duplicate AAC and drift from the Array API. Ordinary
ops stay thin wrappers (`@as_array_result`, `@promote`).

**TensorFlow ordinary ops.** AAC does not ship a TF backend yet. EagerTensors
go through `anytensor.namespace`: `tf.experimental.numpy` plus graph-safe
`repeat` / `arange` / `zeros` / `full`. Mixing **NumPy + TF** is allowed
(NumPy upcasts onto TF). Mixing **Torch + JAX** (or any two non-NumPy
frameworks) is an error — pick a peer.

Backend objects are **internal**. Public `inf(x)`, `finfo(x)`, `dtype(...)`
are thin functions over `get_backend(x)` so callers never import backend
classes.

---

## Guiding principles (what to expect)

### 1. The input tensor picks the world

Dispatch is **input-adaptive**: the namespace / backend comes from the
arguments you passed, not from a global “set backend” switch.

- Scalars only → NumPy 0-d arrays.
- NumPy only → NumPy.
- Any non-NumPy framework tensor → that framework (NumPy buffers upcast onto
  it).
- Two different non-NumPy frameworks in one call → error.

**Edge implication:** a helper written with AnyTensor does not “return NumPy
because the library imported NumPy.” If the user passed JAX arrays, they get
JAX arrays back.

### 2. NumPy is host data, never a demotion target

When a Torch / JAX / TF tensor meets a NumPy ndarray, we **upcast NumPy onto
the framework** — by **reference** when `asarray(..., copy=False)` works.
We never convert the framework tensor down to NumPy to “make them meet.”

| Situation | Behavior |
|---|---|
| Default | Prefer zero-copy reference |
| Zero-copy impossible | `fallback="copy"` (warn) or `"error"` |
| You will mutate the host buffer | `promote_options(copy=True)` or `@promote(..., copy=True)` |

**Edge implication:** aliasing is intentional. If you mutate the NumPy buffer
after a zero-copy upcast, you may mutate the framework tensor’s storage.

### 3. Reductions return 0-d arrays, not Python scalars

`at.sum(x)` with a full reduce returns a **0-d array** on the same namespace
(`@as_array_result`). That keeps `.shape`, `.dtype`, and further AnyTensor
calls working. Bare `float` / `np.float64` are awkward across backends and
break `type(x) is type(y)` style checks.

**Edge implication:** `float(at.sum(x))` is fine when you truly want a Python
float; don’t assume the raw return is already one.

### 4. Operand kinds separate data, indices, masks, and sizes

`@promote` tags each parameter:

| Kind | Meaning | Edge expectation |
|---|---|---|
| `data` | Numeric payload | Share Array API `result_type` (ints widen beside floats) |
| `index` | Segment ids, `take` indices | **Stay integral** — never widened to float by a float peer |
| `mask` | `where` condition | Cast to bool if needed |
| `shape` | `num_segments`, `total_repeat_length`, … | Python `int` / symbolic / 0-d integral tensor; **not** promoted to a 0-d array |

**Edge implication:** passing float segment ids fails fast (kind `index` or
jaxtyping `Integer[...]` under test-time typecheck). Passing `num_segments` as
a traced JAX scalar without marking it static fails under `jax.jit` — that is
by design (see below).

### 5. Shape-sizes are required and stay static-friendly

`num_segments` is **always required** (JAX convention). We do **not** infer
`max(ids)+1`. The same idea applies to `total_repeat_length` and
`sum_partitions`.

Allowed forms:

- Python `int` (preferred under `jax.jit` / `torch.jit.script` / many TF graphs)
- Framework size symbol / traced constant where the backend accepts it
- 0-d integral tensor scalar

Python ints are **not** wrapped into 0-d tensors by promote, so compilers can
treat them as static.

**Edge implication:** if you omit `num_segments` or pass a traced value into
`jax.jit` without `static_argnames`, you get a concretization error — not a
silent wrong-size output.

### 6. Empty segments use explicit identities

When `num_segments` is larger than the set of ids that appear, empty slots keep
a reduction **identity** (not a random finfo sentinel):

| reduction | floating | integral |
|---|---|---|
| `segment_sum` | `0` | `0` |
| `segment_min` | `+inf` | dtype max |
| `segment_max` | `-inf` | dtype min |

Occupied slots always perform a real reduce (including NaN / ±inf). If empties
should be a finite fill, use `segment_min_or_constant` /
`segment_max_or_constant`.

**Edge implication:** a segment whose only value is `+inf` stays `+inf` under
`segment_min` — we do not collapse it to finito’s max the way stock TF
`unsorted_segment_min` can.

### 7. We standardize TF scatter NaN / empty behavior

Stock TF scatter-min/max ignores NaN updates and uses finfo fills for empties.
AnyTensor’s TF path:

1. Scatter from our empty identities (±inf / iinfo).
2. OR in per-segment NaN so a NaN-only segment becomes NaN.

**Edge implication:** portable code can rely on “NaN in → NaN out” for segment
min/max on TF **eager**. Under **TF XLA** (`jit_compile=True`), NaN vs ±inf can
still diverge from eager — see [Surprising differences](semantics.md); do not
rely on NaN under XLA for portability.

### 8. Compilers see a portable API, with known limits

| Path | Expectation |
|---|---|
| Eager (all backends) | Full public surface |
| `jax.jit` | Mark shape-sizes static; `repeat` / `partition_softmax` may need `total_repeat_length` / `sum_partitions` |
| `tf.function` | Prefer Python ints for sizes; use `at.shape(x)` under polymorphic shapes |
| `torch.compile` | Works for many graphs (fuzzed); Dynamo may graph-break on AAC helpers |
| `torch.jit.script` | Only `segment_sum` / `min` / `max` after `enable_torchscript()` — see below |
| `torch.jit.trace` | Trace tensors only; close over Python ints |

### 9. TorchScript is a divert, not a second public API

`torch.jit.script` cannot follow AAC or Python backend dispatch. Rather than
replace the public API with Torch-only functions when `torch` is imported,
we:

1. Keep eager `at.segment_sum` multi-backend.
2. On `enable_torchscript()`, wrap `segment_sum` / `min` / `max` with
   `if torch.jit.is_scripting(): return torchscript.*`.
3. Bind the original implementations with `torch.jit.ignore` for the eager
   branch.

Libraries call `at.segment_sum` with no Torch knowledge; end users can still
`torch.jit.script` through those call sites. `segment_softmax` and friends are
**not** script-diverted today — use `compile` / `trace`, or script only the
pool.

**Edge implication:** enabling TorchScript does not break NumPy/JAX/TF eager
calls. Under script, `num_segments` must be a Python `int`; ids become int64.

### 10. Typing is for humans; runtime checks are opt-in

Public APIs use [jaxtyping](https://docs.kidger.site/jaxtyping/) shape/dtype
annotations and an `ArrayT` TypeVar — **without** importing Torch/JAX/TF for
typing.

- Runtime checking is **off** by default (no surprise cost in production).
- Opt in with `enable_typecheck()` / `install_import_hook` **before** importing
  modules you want checked.
- Pytest enables the hook for selected submodules (not `torchscript`, so
  scripting still compiles). Symbolic fuzz disables jaxtyping: graph tensors
  with `shape=(None,)` fail strict `SegmentIds` matches even when numerics are
  fine.

**Edge implication:** mismatched segment lengths fail under tests when
typecheck is on; the same call in a normal install relies on backend errors or
silent wrong results depending on the framework — validate ids in your own
code for production.

---

## What we deliberately do **not** unify

These stay **backend-local**. Portable code should not depend on them matching
NumPy:

| Topic | What to expect |
|---|---|
| **Index width** | Torch casts ids to int64 at scatter; JAX (no x64) / TF often int32. Do not assume NumPy int64 ids remain int64. |
| **`sorted=`** | Honored on JAX/TF; **no-op** on NumPy/Torch (always unsorted-safe). |
| **Default float width** | JAX may truncate float64→float32 without `jax_enable_x64`. |
| **`inf *` tiny** | May be `inf` (NumPy, eager TF) or `nan` (JAX, TF XLA) when the tiny underflows to 0. |
| **TF XLA vs eager + NaN** | Min/max-like ops may yield ±inf under XLA where eager yields NaN. |
| **Empty axis `min`/`max`** | Length-0 reductions are framework-defined (often error). Prefer nonempty. |
| **GPU** | No GPU CI. Keep devices matched; float32 compares; tie-breaks under atomics are not portable; empty CUDA / GPU XLA are stricter; MPS ≠ CUDA. |

When in doubt: read [Surprising differences](semantics.md), or check
`anytensor.semantics` and the contract tests.

---

## Testing as design documentation

The suite is layered so “what we promise” stays executable:

| Layer | Role |
|---|---|
| Unit / contract (`pytest -m "not fuzz"`) | Empty identities, promotion, boundaries; coverage gate (omits `backends.py` / `torchscript.py`) |
| Cross-backend fuzz | NumPy reference × random other backend on the `@fuzz_op` registry |
| Symbolic fuzz | Eager vs `jax.jit` / `torch.compile` / `tf.function` (+ XLA); jaxtyping off |
| Docs examples (Sybil) | GAT-style helper + compile/script recipes in [`examples.md`](examples.md) |

If a behavior is not tested at one of these layers, do not assume it is part of
the portability contract.
