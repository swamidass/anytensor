# AnyTensor

Portable tensor ops across **NumPy**, **JAX**, **PyTorch**, and **TensorFlow**, with a focus on **segment / GNN** primitives.

Write a helper once; run it on whatever tensor the caller already has. Ordinary math uses the [Python Array API](https://data-apis.org/array-api/latest/) via [`array-api-compat`](https://github.com/data-apis/array-api-compat). Segment reductions stay on thin input-adaptive backends.

**Docs** (motivation, GAT-style case study, design, API): <https://swamidass.github.io/anytensor/> — or `uv run --group docs mkdocs serve` from a checkout ([`docs/`](docs/index.md)).

We follow [Semantic Versioning](https://semver.org/): breaking changes require a **major** bump. Portability is backed by cross-backend / symbolic fuzz, a coverage gate, and pytest-run docs examples ([Design](docs/design.md#how-we-keep-the-contract-honest)).

## Install

```bash
pip install "anytensor @ git+https://github.com/swamidass/anytensor.git"
# optional backends (NumPy is a core dependency)
pip install "anytensor[jax]" "anytensor[torch]" "anytensor[tensorflow]"
pip install "anytensor[onnx]"   # export helpers (plus tf2onnx for the TF path)
# or
pip install "anytensor[all]"
```

Requires Python ≥3.10. Backend floors: NumPy ≥1.24, JAX ≥0.4.32, PyTorch ≥2.1, TensorFlow ≥2.13.

## Quick start

```python
import anytensor as at
import numpy as np

x = np.arange(12.0).reshape(3, 4)
seg_ids = np.array([0, 0, 1])
y = at.segment_sum(x, seg_ids, num_segments=2)
```

The same call works on JAX / Torch / TF tensors. `num_segments` is **required** (JAX convention).

`anytensor.jraph` is a portable [jraph](https://github.com/google-deepmind/jraph):
`GraphsTuple`, batching/padding, and `GraphNetwork` on any backend. Nested
feature trees use `anytensor.tree` (`jax.tree` API; pure Python, NumPy is the
only binary dep). `anytensor.hetero` adds **heterogeneous graphs** (multiple
node/edge types), DGL-style `multi_update_all` with optional attention, and a
small model zoo (relational GCN, heterogeneous GraphSAGE, HAN, HGT, CompGCN —
see [Hetero](docs/hetero/index.md) for full names and citations). Also see
[Jraph](docs/jraph/index.md) and [Tree](docs/tree/index.md).

Read next:

- [Home / motivation](docs/index.md) — why AnyTensor, GAT neighbor-softmax case study across four backends
- [Jraph](docs/jraph/index.md) — portable GraphsTuple / GraphNetwork
- [Hetero](docs/hetero/index.md) — heterographs, attention, model zoo + citations
- [Tree](docs/tree/index.md) — nest helpers (pure Python + NumPy) for graphs and any structured record
- [Design](docs/design.md) — principles, edge cases, **testing as contract**, SemVer
- [Usage](docs/usage.md) — promotion, segment helpers, `torch.compile`, typing
- [ONNX export](docs/onnx/index.md) — Flax / Lightning / Keras recipes with symbolic lengths
- [Surprising differences](docs/semantics.md) — NaN / ±inf / graph / GPU gotchas from fuzz
- [API reference](docs/api/index.md) — generated from docstrings
- [Contributing](docs/contributing.md) — tests, fuzz, docs build

## Docs / tests (checkout)

```bash
uv sync --extra all --group dev --group docs
uv run mkdocs serve
uv run pytest -m "not fuzz" --cov=anytensor --cov-report=term-missing
```

## License

MIT. Backend dispatch patterns adapted from [einops](https://github.com/arogozhnikov/einops); see `NOTICE`.
