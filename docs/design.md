# Design

- **Ordinary ops** → `array_api_compat.array_namespace(x)` (plus
  `@as_array_result` / `@promote` / `@promote_scalars`).
- **Segment ops** → `backends.get_backend(x).segment_reduce(...)`.
- **TensorFlow ordinary ops** → `anytensor.namespace` shim over
  `tf.experimental.numpy`, with graph-safe `repeat` / `arange` / `zeros`.
- Backends are imported lazily; missing optional deps are fine until you pass
  that framework’s tensors. `enable_torchscript()` adds an `is_scripting()`
  divert for libraries that call `at.segment_*` — eager multi-backend dispatch
  is unchanged.
- Backend objects stay **internal**. Public specials (`inf`, `finfo`, …) are
  thin functions over `get_backend(x)`.

## Operand kinds

| Kind | Role |
|---|---|
| `data` | Share `result_type` (ints widen beside floats) |
| `index` | Stay integral (segment ids, `take` indices) |
| `mask` | Boolean (`where` condition) |
| `shape` | Size dim — Python int / symbolic / 0-d integral tensor; **not** promoted to a 0-d array |

## Testing layers

| Layer | What it checks |
|---|---|
| Unit / contract | Semantics, boundaries, backend attrs (`pytest -m "not fuzz"`) |
| Cross-backend fuzz | NumPy × random other backend on `@fuzz_op` registry |
| Symbolic fuzz | Eager vs `jax.jit` / `torch.compile` / `tf.function` (+ XLA) |

Coverage gate omits `backends.py` and excludes fuzz markers.
