# Heterogeneous graphs

`anytensor.hetero` is a small **heterogeneous graph** stack on the caller’s
tensors (NumPy / JAX / PyTorch / TensorFlow).

It is **not** part of the jraph-mirroring API — import it explicitly:

```python
from anytensor.hetero import HeteroGraphsTuple, multi_update_all
from anytensor.hetero import relational_graph_convolution
```

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## What is a heterogeneous graph?

A **homogeneous** graph has one kind of node and one kind of edge: every
vertex is “a node,” every link is “an edge.” Citation networks treated as
paper→paper, social graphs of users only, and most textbook GNN demos are
homogeneous.

A **heterogeneous graph** (also *heterograph*, *typed graph*, or
*knowledge graph* when relations are labeled facts) has:

- **Node types (ntypes)** — disjoint pools of entities, e.g. `author`,
  `paper`, `venue`.
- **Edge types / relations (etypes)** — typed, directed links between
  those pools, e.g. “an author **writes** a paper,” “a paper **cites** a
  paper,” “a paper is **published_in** a venue.”

Features usually live *per type*: authors might have affiliation vectors,
papers have bag-of-words or embeddings, and edges may carry weights or
relation embeddings. Message passing must respect types — you do not
blindly mix author rows with paper rows in one dense adjacency.

```text
  author ──writes──► paper ──cites──► paper
     ▲                 │
     └──── written_by ─┘   (reverse of writes; often added explicitly)
```

Homogeneous GNNs collapse this to one adjacency. Heterogeneous models keep
the types and run a **separate** transform / aggregate **per relation**,
then combine results at each destination type.

## Direction: messages follow the arrow

In this library (and in DGL / PyG heterographs), each etype is
**directed**. For `("author", "writes", "paper")`:

- Messages flow **author → paper** (along `senders` → `receivers`).
- Papers are updated from their authors.
- **Authors are not updated** by that relation alone.

That matches how the data is stored: one incidence list per etype. There
is no implicit reverse pass.

### Do common architectures propagate both ways?

**Not automatically.** Classic hetero layers — Relational GCN (R-GCN),
heterogeneous GraphSAGE, HAN, HGT, CompGCN — treat each relation as a
one-way channel. If you only register `writes`, information moves into
papers; authors never see paper context through that etype.

### How people get both directions

Almost everyone who needs bidirectional flow **adds reverse relations**
(or builds meta-paths that walk both ways):

| Practice | What you store | Who updates |
|---|---|---|
| Forward only | `("author", "writes", "paper")` | papers ← authors |
| + reverse | also `("paper", "written_by", "author")` | authors ← papers |
| Meta-path (HAN) | e.g. author→paper→author as its own etype or two hops | authors via papers |

Knowledge-graph R-GCNs often invent an inverse predicate for every
relation (`born_in` / `born_in_inv`). Bipartite recommenders and academic
graphs do the same with `writes` / `written_by`. Libraries expose helpers
(DGL `add_reverse_edges`, PyG `ToUndirected` / `to_bidirected`) that
materialize those reverses as **new etypes**, usually with their own
weights — not as a silent undirected multiply.

So: if both ends of a link should update, put **both** directions in the
graph (or a meta-path that uses both). This stack will not invent reverses
for you.

## Why this package

Homogeneous GNNs assume one node/edge type. Many graphs in practice do
not. [DGL](https://www.dgl.ai/)’s
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
| `HeteroGraphsTuple` | Nodes / edges keyed by ntype and **canonical etype** |
| `SendRecvTuple` | One directed send→receive incidence (same ntype = homo view; two ntypes = bipartite relation) |
| `RelationSpec` | Per-relation message function, reduce name, optional edge attention |

A **canonical etype** is a triple `(src_ntype, relation_name, dst_ntype)`,
e.g. `("author", "writes", "paper")`. The same relation name with swapped
endpoints is a *different* etype (`written_by` above). Node ids are
**local to each ntype**: author `0` and paper `0` are unrelated indices.

Fields on `HeteroGraphsTuple`:

| Field | Meaning |
|---|---|
| `nodes[ntype]` | Feature nest for that type, leading axis = number of nodes |
| `edges[etype]` | Optional edge features for that relation (`None` if unused) |
| `senders[etype]` / `receivers[etype]` | Integer endpoints (into the src / dst ntype pools) |
| `n_node[ntype]` / `n_edge[etype]` | Per-graph counts (length = batch size; `1` for a single graph) |

## Message passing

`relation_mailbox` runs **one** etype:

1. Gather source features along `senders` (and destination features along
   `receivers` when needed for attention or edge-wise scores).
2. Optional **attention**: score each edge, normalize with
   `segment_softmax` grouped by destination (`receivers`), weight messages.
3. **Segment reduce** onto destinations (`sum` / `mean` / `max` / `min`).
   Destinations with no incoming edges of that type get `0`.

Only the **destination** ntype of that etype receives a mailbox. Source-only
types are unchanged unless some other etype (often a reverse) targets them.

`multi_update_all` runs many etypes, then fuses mailboxes that share a
destination ntype with a **cross-reducer** (`sum` / `mean` / `max` /
`min` / `stack`). A **mailbox** is the per-relation aggregated tensor at
each destination node. `stack` yields shape `(n_dst, n_relations, …)` in
etype-dict insertion order — used when a model attends **across
relations** (HAN semantic attention).

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
(precompute longer paths offline, often using reverses). For HGT,
multi-head query/key/value and edge-type matrices live inside the callables
you pass; `hgt` supplies neighbor softmax and cross-relation sum.

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
