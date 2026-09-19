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
pip install "anytensor[onnx]"   # anytensor.export recipes (plus tf2onnx for the TF path)
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

## Graph libraries on the core

[Jraph](#jraph-anytensorjraph) and [heterogeneous graphs](#heterogeneous-graphs-anytensorhetero)
are example libraries built on AnyTensor’s portable ops (`take`, `segment_*`,
the Array API). They are useful on their own: you do not have to adopt the
rest of the stack to use them, and you keep whatever neural-net library
already owns your weights.

The point of shipping them here is the cross-backend contract. Upstream
[jraph](https://github.com/google-deepmind/jraph) is JAX. DGL and PyG are
excellent heterogeneous-graph stacks **inside one framework**. These modules
are the same *algorithms* when the tensors you already hold are NumPy, JAX,
PyTorch, or TensorFlow — one helper, four callers, no graph-store rewrite.

### Jraph (`anytensor.jraph`)

Homogeneous graphs: one node pool, one edge list. [Jraph](docs/jraph/index.md)
is a portable port of DeepMind’s
[jraph](https://github.com/google-deepmind/jraph) — `GraphsTuple`,
batch/unbatch, pad-with-graphs, and a functional
[`GraphNetwork`](https://arxiv.org/abs/1806.01261) (Battaglia et al.).

Why this is worth a separate library:

- **One sparse layout for a batch.** Several graphs live in one tuple
  (`nodes`, `edges`, `senders`, `receivers`, `n_node`, `n_edge`). Batching
  is concatenation plus sender offsets, not a second graph type.
- **You own the neural net.** Pass any callable (`lambda x: x @ W`, a Flax /
  Haiku / `torch.nn` module). The library does not pick a training framework.
- **Pad to a static shape with graphs**, not a parallel “padded graph” API —
  the usual trick for `jax.jit` / `tf.function` / `torch.compile` without
  changing the message-passing code.
- **Forkable models, not a framework.** `InteractionNetwork`, `GAT`,
  `GraphConvolution`, `DeepSets`, and friends are thin configurations of
  `GraphNetwork`. Copy them when the paper’s variant diverges.

Use upstream jraph when the whole program is JAX. Use `anytensor.jraph` when
the same `GraphsTuple` helper must run on the caller’s backend.

```python
from anytensor.jraph import GraphsTuple, GraphNetwork
```

### Heterogeneous graphs (`anytensor.hetero`)

A **heterograph** has more than one node type and more than one relation:
authors write papers, papers cite papers, authors are affiliated with
institutions. Node ids are local to each type (author `0` is not paper `0`).
[Hetero](docs/hetero/index.md) is that data model plus DGL-style
[`multi_update_all`](https://docs.dgl.ai/generated/dgl.DGLGraph.multi_update_all.html)
on AnyTensor primitives.

Why this is worth a separate library:

- **Typed incidence without a framework graph object.** `HeteroGraphsTuple`
  is maps of arrays keyed by node type and canonical etype
  `(src, relation, dst)`. It sits on tensors you already have — no DGL graph,
  no PyG `HeteroData` conversion just to run one layer.
- **Per-relation message, then an explicit fuse.** Each relation gathers,
  optionally attends (`segment_attention`), and reduces onto destinations.
  Relations that share a destination type are then summed, stacked, or
  otherwise fused. That is the portable core of relational GCN, HAN, and HGT
  without reimplementing scatter per backend.
- **Both directions are first-class etypes.** A reverse such as `written_by`
  is another stored relation, not an undirected flag — so each direction can
  have its own weights.
- **A small zoo as plain functions.** Relational GCN, heterogeneous GraphSAGE,
  HAN, HGT, and CompGCN (citations in the
  [hetero docs](docs/hetero/index.md)). Call sites own `W`; the functions own
  the typed aggregation.

It is not a DGL or PyG replacement: no neighbor sampler, no graph store, no
module zoo that owns parameters. It is the message-passing kernel when those
frameworks’ graphs are the wrong place to live.

```python
from anytensor.hetero import HeteroGraphsTuple, multi_update_all
from anytensor.hetero import relational_graph_convolution
```

## Tree (`anytensor.tree`)

[`anytensor.tree`](docs/tree/index.md) is a **pure Python** nest library. The
only binary dependency is NumPy. No JAX, no C++ pytree extension.

It follows the [`jax.tree`](https://docs.jax.dev/en/latest/pytrees.html) API
(`map`, `flatten`, `unflatten`, and the `tree_*` aliases). `None` is an empty
nest (zero leaves), matching jraph — not a leaf. Arrays and framework tensors
are leaves.

Why it is broadly useful, not only a GNN helper:

- Nested records show up everywhere: a simulation step
  (`{"pos": …, "vel": …}`), a lab row, a checkpoint, a minibatch, and also
  GNN node/edge features. The operations are the same: map a function over
  every array, flatten to leaves, stack records, split them back apart.
- Hand-rolled walks diverge on `None` vs missing keys vs list vs tuple.
  `jax.tree` gets this right but pulls in JAX. `dm-tree` and `optree` are
  fast C++ extensions and treat some of those cases differently. `torch`
  pytrees ship with PyTorch.
- This module is that contract when you do **not** want those dependencies:
  pure Python, NumPy arrays and already-constructed framework tensors as
  leaves, safe to import in a library that must stay backend-agnostic.

Jraph and hetero use it for nested features. You can use it for any structured
record.

```python
import anytensor.tree as tree

record = {"pos": x, "vel": v, "notes": None}
record = tree.map(lambda a: a * 2, record)
```

Read next:

- [Home / motivation](docs/index.md) — why AnyTensor, GAT neighbor-softmax case study across four backends
- [Jraph](docs/jraph/index.md) — portable GraphsTuple / GraphNetwork
- [Hetero](docs/hetero/index.md) — heterographs, attention, model zoo + citations
- [Tree](docs/tree/index.md) — nest helpers (pure Python + NumPy) for graphs and any structured record
- [Design](docs/design.md) — principles, edge cases, **testing as contract**, SemVer
- [Usage](docs/usage.md) — promotion, [caller rules](docs/usage.md#caller-rules), `torch.compile`, typing
- [ONNX](docs/onnx/index.md) — recommended deploy path (ORT); `anytensor.export` recipes (Flax / Lightning / Keras, symbolic lengths)
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
