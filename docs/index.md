# AnyTensor

Portable tensor ops across **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**,
with a focus on **segment / GNN** primitives.

Ordinary math (`sum`, `exp`, `reshape`, …) uses the
[Python Array API](https://data-apis.org/array-api/latest/) via
[`array-api-compat`](https://github.com/data-apis/array-api-compat). Segment
reductions stay on thin input-adaptive backends, so one function runs on
whatever tensor the caller already has.

**Docs:** [Usage](usage.md) · [Surprising differences](semantics.md) ·
[API reference](api/index.md) · [Design](design.md)

## Install

```bash
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
# or
pip install "anytensor[all]"
```

Requires Python ≥3.10. Backend floors: NumPy ≥1.24, JAX ≥0.4.32, PyTorch ≥2.0,
TensorFlow ≥2.10.

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, num_segments=2)
```

The same call works on JAX / Torch / TF tensors. `num_segments` is required
(JAX convention) — a Python `int`, jit symbolic constant, or 0-d integral
tensor scalar.

See [Usage](usage.md) for promotion rules and [Surprising differences](semantics.md)
for NaN / ±inf / graph gotchas discovered under fuzz.
