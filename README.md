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

### Scalar policy

Full reductions return **0-d arrays** (not bare Python / NumPy scalars), so results always have array methods (`.shape`, `.dtype`, …). Binary ops such as `maximum` / `where` **upcast** Python scalars to 0-d arrays on the other operand’s backend.

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

PRs welcome. Prefer adding portable helpers in `anytensor/core.py` or `segment.py`; only extend `backends.py` when the Array API cannot express the op (e.g. `segment_reduce`). Add coverage in `test/test_ops.py` and backend contracts in `test/test_backend_contracts.py`.

## License

MIT. Backend dispatch patterns adapted from [einops](https://github.com/arogozhnikov/einops); see `NOTICE`.
