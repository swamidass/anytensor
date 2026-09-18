# Heterogeneous graphs

`anytensor.hetero` is a small **heterogeneous graph** stack on the caller’s
tensors (NumPy / JAX / PyTorch / TensorFlow). A heterogeneous graph has more
than one **node type** (ntype) and/or **edge type** (etype)—for example
authors, papers, and “writes” / “cites” relations—unlike a
**homogeneous** graph where every node and edge shares one type.

It is **not** part of the jraph-mirroring API — import it explicitly:

```python
from anytensor.hetero import HeteroGraphsTuple, multi_update_all
from anytensor.hetero import relational_graph_convolution
```

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## Why this package

Homogeneous graph neural networks (GNNs) assume one node/edge type. Many
graphs in practice do not: authors write papers, papers cite papers, users
rate movies. [DGL](https://www.dgl.ai/)’s
[`multi_update_all`](https://docs.dgl.ai/generated/dgl.DGLGraph.multi_update_all.html)
pattern — **per-relation message + reduce**, then an explicit
**cross-relation fuse** — is the portable core. This module follows that
shape on AnyTensor primitives (`take`, `segment_*`, optional
`segment_softmax` attention). Destination sizes are shape-sizes
(`at.shape(nodes)[0]`, never `int(shape)` under tracing) — see
[Usage → Caller rules](../usage.md#caller-rules).

Nested features use [`anytensor.tree`](../tree/index.md). Batching is
`tree.batch` / `tree.unbatch` via `HeteroGraphsTuple.__tree_batch__`
(matching ntype/etype keys required; use empty arrays, not `None`, to pad
schemas).

## Data model

| Type | Role |
|---|---|
| `HeteroGraphsTuple` | Nodes / edges keyed by ntype and **canonical etype** `(src_ntype, relation, dst_ntype)` |
| `SendRecvTuple` | One directed send→receive incidence (same ntype = homo view; two ntypes = bipartite) |
| `RelationSpec` | Per-relation message function, reduce name, and optional edge attention |

A **canonical etype** is a triple such as `("author", "writes", "paper")`:
source node type, relation name, destination node type.

## Message passing

`relation_mailbox` runs **one** etype:

1. Gather source features along `senders` (and destination features along
   `receivers` when needed).
2. Optional **attention**: score each edge, normalize with
   `segment_softmax` grouped by destination (`receivers`), weight messages.
3. **Segment reduce** onto destinations (`sum` / `mean` / `max` / `min`).
   Nodes with no incoming edges of that type get `0`.

`multi_update_all` runs many etypes, then fuses mailboxes that share a
destination ntype with a **cross-reducer** (`sum` / `mean` / `max` /
`min` / `stack`). Here a **mailbox** is the per-relation aggregated tensor
at each destination node. `stack` yields shape
`(n_dst, n_relations, …)` in etype-dict insertion order — used when a model
needs to attend **across relations** (see HAN below).

Per-relation attention matches the optional attention path on homo
[`GraphNetwork`](../jraph/index.md) (same `segment_softmax` idea as
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) / GAT). That
is what lets this stack express **Heterogeneous Graph Attention Network**
(HAN) node-level attention and **Heterogeneous Graph Transformer** (HGT)
typed attention.

## Model zoo

Plain functions in `anytensor.hetero.models` (also re-exported from
`anytensor.hetero`). Each takes a graph plus callables for the learnable
pieces — your framework owns the weights (`lambda x: x @ W`, module
`__call__`, etc.).

| Function | Full name / paper | What it does |
|---|---|---|
| `relational_graph_convolution` | **R-GCN** (Relational Graph Convolutional Network) — [Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103) | One linear per relation, mean/sum over neighbors, plus a self term |
| `hetero_sage` | Heterogeneous **GraphSAGE** — [Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.02216) | Mean-aggregate neighbors, concat with self, one combine linear |
| `han` | **HAN** (Heterogeneous Graph Attention Network) — [Wang et al., WWW 2019](https://arxiv.org/abs/1903.07293) | Attention over neighbors on each meta-path/etype, then attention over those path embeddings (`stack`) |
| `hgt` | **HGT** (Heterogeneous Graph Transformer) — [Hu et al., WWW 2020](https://arxiv.org/abs/2003.01332) | Type-aware attention scores and messages, then a target-type output map |
| `comp_gcn` | **CompGCN** (Composition-based Multi-Relational GCN) — [Vashishth et al., ICLR 2020](https://arxiv.org/abs/1911.03082) | Compose source features with edge features (`mult` or `sum`), then relation linear |

A **meta-path** is a typed walk pattern (e.g. author→paper→author). HAN
expects each path you care about to already exist as an etype on the graph
(precompute longer paths offline). For HGT, multi-head query/key/value and
edge-type matrices live inside the callables you pass; `hgt` supplies
neighbor softmax and cross-relation sum.

Helper: `gat_attention_logit` builds a GAT-style edge score
\(\mathrm{LeakyReLU}(a^\top[h_{\mathrm{src}}\|h_{\mathrm{dst}}])\) for use
inside HAN (or any custom logit).

## Relation to DGL

Value parity tests cover batch/unbatch and `multi_update_all` kernels
(including float `u_mul_e` / “source feature times edge weight” cases).
Empty max/min destinations are `0`, matching DGL. Cross-reducers match when
every destination receives every involved etype; with partial coverage,
anytensor always zero-fills then fuses (same as composing DGL per-etype
`update_all` then the same cross reduce).
