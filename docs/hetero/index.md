# Heterogeneous graphs

`anytensor.hetero` is a small heterogeneous-graph stack on the caller’s
tensors (NumPy / JAX / PyTorch / TensorFlow). It is **not** part of the
jraph-mirroring API — import it explicitly:

```python
from anytensor.hetero import HeteroGraphsTuple, multi_update_all
from anytensor.hetero import relational_graph_convolution
```

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## Why this package

Homogeneous GNNs share one node/edge type. Many graphs that show up in
practice do not: authors write papers, papers cite papers, users rate
movies. DGL’s
[`multi_update_all`](https://docs.dgl.ai/generated/dgl.DGLGraph.multi_update_all.html)
pattern — **per-relation message + reduce**, then an explicit
**cross-relation fuse** — is the portable core. This module follows that
shape on AnyTensor primitives (`take`, `segment_*`, optional
`segment_softmax` attention).

Nested features use [`anytensor.tree`](../tree/index.md). Batching is
`tree.batch` / `tree.unbatch` via `HeteroGraphsTuple.__tree_batch__`
(matching ntype/etype keys required; use empty arrays, not `None`, to pad
schemas).

## Data model

| Type | Role |
|---|---|
| `HeteroGraphsTuple` | Nodes / edges keyed by ntype and canonical etype `(src, rel, dst)` |
| `SendRecvTuple` | One directed incidence view (homo or bipartite) |
| `RelationSpec` | Per-relation `message_fn`, `reduce`, optional attention |

Canonical etypes are triples, e.g. `("author", "writes", "paper")`.

## Message passing

`relation_mailbox` runs one etype: gather along `senders`, optional
attention (logit → `segment_softmax` on `receivers` → weight messages),
then segment reduce (`sum` / `mean` / `max` / `min`; empty destinations
`0`).

`multi_update_all` runs many etypes and fuses mailboxes that share a
destination ntype with a **cross-reducer** (`sum` / `mean` / `max` /
`min` / `stack`). `stack` yields `(n_dst, n_relations, …)` in etype-dict
order — the hook for HAN-style semantic attention.

Per-relation attention mirrors homo
[`GraphNetwork`](../jraph/index.md) attention and unlocks full HAN
node-level attention and HGT-style typed attention.

## Model zoo

Plain functions in `anytensor.hetero.models` (re-exported from
`anytensor.hetero`). Each takes a graph plus callables for the learnable
pieces — your framework owns the weights.

| Function | Paper | Idea |
|---|---|---|
| `relational_graph_convolution` | [Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103) | Per-relation linear + mean/sum + self term (R-GCN) |
| `hetero_sage` | [Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.02216) | Mean neighbors, concat self, combine linear |
| `han` | [Wang et al., WWW 2019](https://arxiv.org/abs/1903.07293) | Node-level attn per meta-path etype + semantic attn over `stack` |
| `hgt` | [Hu et al., WWW 2020](https://arxiv.org/abs/2003.01332) | Typed attention logits + messages + target projection |
| `comp_gcn` | [Vashishth et al., ICLR 2020](https://arxiv.org/abs/1911.03082) | Compose `h_src` with edge features (`mult` / `sum`) |

Longer meta-paths for HAN are **precomputed as etypes** on the graph.
HGT’s full multi-head Q/K/V and edge-type matrices fold into the callables
you pass; the function supplies neighbor softmax and cross-sum.

## Relation to DGL

Value parity tests cover batch/unbatch and `multi_update_all` kernels
(including float `u_mul_e` cases). Empty max/min destinations are `0`,
matching DGL. Cross-reducers match when destinations receive every
involved etype; with partial coverage, anytensor always zero-fills then
fuses (equivalent to composing DGL per-etype `update_all` then the same
cross reduce).
