# AnyTensor

Portable tensor ops across **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, with a focus on **segment / GNN** primitives.

Ordinary math (`sum`, `exp`, `reshape`, …) uses the [Python Array API](https://data-apis.org/array-api/latest/) via [`array-api-compat`](https://github.com/data-apis/array-api-compat). Segment reductions stay on thin input-adaptive backends, so one function runs on whatever tensor the caller already has.

**Full docs** (usage, surprising differences, API from docstrings): run locally with `uv run --group docs mkdocs serve`, or see [`docs/`](docs/index.md). Planned site: <https://swamidass.github.io/anytensor/>.

## Install

```bash
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
# optional backends (NumPy is a core dependency)
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
```

The same call works on JAX / Torch / TF tensors. `num_segments` is **required** (JAX convention).

Read next:

- [Usage](docs/usage.md) — promotion, segment helpers, TorchScript (`enable_torchscript`), specials
- [Surprising differences](docs/semantics.md) — NaN / ±inf / graph / GPU gotchas from fuzz
- [API reference](docs/api/index.md) — generated from docstrings
- [Contributing](docs/contributing.md) — tests, fuzz, docs build

## Docs / tests (checkout)

```bash
uv sync --extra jax --extra torch --extra tensorflow --group dev --group docs
uv run mkdocs serve
uv run pytest -m "not fuzz" --cov=anytensor --cov-report=term-missing
```

## License

MIT. Backend dispatch patterns adapted from [einops](https://github.com/arogozhnikov/einops); see `NOTICE`.
