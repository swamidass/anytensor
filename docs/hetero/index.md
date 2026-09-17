# Heterogeneous graphs

`anytensor.hetero` is a small stack for **heterogeneous graphs** on the
caller’s tensors (NumPy / JAX / PyTorch / TensorFlow).

It is **not** part of the jraph-mirroring API — import it explicitly:

```python
from anytensor.hetero import HeteroGraphsTuple, multi_update_all
from anytensor.hetero import relational_graph_convolution
```

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## What is a heterogeneous graph?

### Homogeneous vs heterogeneous

Most textbook graphs are **homogeneous**: one kind of vertex and one kind of
link. Every node is “a node,” every edge is “an edge.” A citation network
stored only as paper→paper, or a social graph of users only, fits that mold.
A single adjacency matrix (or one `senders` / `receivers` pair) is enough.

A **heterogeneous graph** — also called a *heterograph* or *typed graph* —
has **more than one type of node** and **more than one type of edge**. In
this literature the typed edges are usually called **relations** (or
*predicates*, *link types*). Each relation connects a specific source node
type to a specific destination node type.

| Idea | Homogeneous | Heterogeneous |
|---|---|---|
| Nodes | One pool | Several **node types** (ntypes), e.g. author, paper, institution |
| Links | One edge kind | Several **relations** (etypes), e.g. writes, cites, affiliated_with |
| IDs | Global `0…N-1` | **Local per type** — author `0` ≠ paper `0` |
| Features | One feature table | One table (or nest) **per** node/edge type |
| Message passing | One aggregate | **Per-relation** transform + aggregate, then fuse at the destination type |

When the relations are labeled facts (“born in,” “employs”), people also call
this a **knowledge graph**. Same data model: typed nodes, typed directed
relations.

### Worked example: authors, papers, institutions

Imagine a small academic world:

**Node types**

| ntype | Who | Example features |
|---|---|---|
| `author` | People | embedding, seniority |
| `paper` | Publications | title embedding, year |
| `institution` | Labs / universities | region embedding |

**Relations** (directed; name is the middle string of the canonical triple)

| Canonical etype `(src, relation, dst)` | Meaning |
|---|---|
| `("author", "writes", "paper")` | Author wrote this paper |
| `("paper", "cites", "paper")` | Paper cites another paper |
| `("author", "affiliated_with", "institution")` | Author works at institution |
| `("paper", "published_at", "institution")` | Optional: venue / host org |

Sketch:

```text
 institution
      ▲
      │ affiliated_with
      │
   author ──writes──► paper ──cites──► paper
      ▲                 │
      └──── written_by ─┘     (reverse of writes; see below)
```

Concrete tiny instance:

| Type | Nodes (local ids) |
|---|---|
| author | `0` Ada, `1` Bao, `2` Chen |
| paper | `0` “Graphs 101”, `1` “Hetero GNNs” |
| institution | `0` MIT, `1` ETH |

| Relation | Edges `(sender → receiver)` |
|---|---|
| writes | Ada→Graphs 101, Bao→Graphs 101, Chen→Hetero GNNs |
| cites | Hetero GNNs → Graphs 101 |
| affiliated_with | Ada→MIT, Bao→MIT, Chen→ETH |

In code those are separate arrays keyed by ntype / etype — never one flat
node list. Author id `0` and paper id `0` are different objects.

```python
writes = ("author", "writes", "paper")
cites = ("paper", "cites", "paper")
affil = ("author", "affiliated_with", "institution")
# nodes["author"].shape[0] == 3, nodes["paper"].shape[0] == 2, ...
# senders[writes] == [0, 1, 2], receivers[writes] == [0, 0, 1]
```

A homogeneous GNN would have to fake this (e.g. one big node set with type
flags). A heterogeneous model keeps the types and runs **one update rule per
relation**, then combines contributions that land on the same destination
type (papers get both `writes` and `cites` mail; institutions get
`affiliated_with`; and so on).

## Direction: messages follow the arrow

Each relation is **directed**. For `("author", "writes", "paper")`:

- Messages flow **author → paper**.
- Papers update from their authors.
- **Authors are not updated** by that relation alone.

There is no implicit reverse. That matches DGL / PyG heterographs and this
library: one incidence list per etype.

### Do common architectures propagate both ways?

**Not automatically.** R-GCN, heterogeneous GraphSAGE, HAN, HGT, CompGCN all
treat each relation as a one-way channel. With only `writes`, information
moves into papers; authors never see paper context through that etype.

### How people get both directions

Add an explicit **reverse relation** (or a meta-path that walks both ways):

| Practice | What you store | Who updates |
|---|---|---|
| Forward only | `("author", "writes", "paper")` | papers ← authors |
| + reverse | also `("paper", "written_by", "author")` | authors ← papers |
| Meta-path (HAN) | e.g. author→paper→author | authors via papers |

Same underlying authorship links, stored twice with swapped endpoints — often
with **separate** weights (`W_writes` vs `W_written_by`). Knowledge-graph
R-GCNs invent inverse predicates (`born_in` / `born_in_inv`). DGL
`add_reverse_edges` and PyG `ToUndirected` materialize those reverses as new
etypes; they do not silently make one relation bidirectional.

If both ends should update, put **both** directions in the graph (or a
meta-path). This stack will not invent reverses for you.

In the academic example you might also add
`("institution", "employs", "author")` as the reverse of `affiliated_with`,
and `("paper", "cited_by", "paper")` as the reverse of `cites`.

## Why this package

Homogeneous GNNs assume one node/edge type. Many graphs in practice do not.
[DGL](https://www.dgl.ai/)’s
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
| `SendRecvTuple` | One directed send→receive incidence (same ntype = homo view; two ntypes = one bipartite relation) |
| `RelationSpec` | Per-relation message function, reduce name, optional edge attention |

A **canonical etype** is a triple `(src_ntype, relation_name, dst_ntype)`,
e.g. `("author", "writes", "paper")`. Swapping endpoints yields a different
etype (`written_by`). Node ids are **local to each ntype**.

| Field | Meaning |
|---|---|
| `nodes[ntype]` | Feature nest for that type (leading axis = #nodes of that type) |
| `edges[etype]` | Optional edge features for that relation (`None` if unused) |
| `senders[etype]` / `receivers[etype]` | Integer endpoints into the src / dst ntype pools |
| `n_node[ntype]` / `n_edge[etype]` | Per-graph counts (length = batch size; `[n]` for one graph) |

## Message passing

`relation_mailbox` runs **one** relation:

1. Gather source features along `senders` (and destination features along
   `receivers` when needed for attention).
2. Optional **attention**: score each edge, `segment_softmax` by destination,
   weight messages.
3. **Segment reduce** onto destinations (`sum` / `mean` / `max` / `min`).
   Destinations with no incoming edges of that type get `0`.

Only that relation’s **destination** ntype receives a mailbox.

`multi_update_all` runs many relations, then fuses mailboxes that share a
destination ntype with a **cross-reducer** (`sum` / `mean` / `max` /
`min` / `stack`). A **mailbox** is the per-relation aggregate at each
destination node. `stack` yields `(n_dst, n_relations, …)` in etype-dict
order — used when a model attends **across relations** (HAN).

Per-relation attention matches homo
[`GraphNetwork`](../jraph/index.md) attention (same idea as
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) / GAT), which
is what **Heterogeneous Graph Attention Network** (HAN) node-level attention
and **Heterogeneous Graph Transformer** (HGT) typed attention build on.

## Model zoo

Plain functions in `anytensor.hetero.models` (also re-exported from
`anytensor.hetero`). Each takes a graph plus callables for the learnable
pieces — your framework owns the weights.

| Function | Full name / paper | What it does |
|---|---|---|
| `relational_graph_convolution` | **R-GCN** (Relational Graph Convolutional Network) — [Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103) | One linear per relation, mean/sum over neighbors, plus a self term |
| `hetero_sage` | Heterogeneous **GraphSAGE** — [Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.02216) | Mean-aggregate neighbors, concat with self, one combine linear |
| `han` | **HAN** (Heterogeneous Graph Attention Network) — [Wang et al., WWW 2019](https://arxiv.org/abs/1903.07293) | Attention over neighbors on each meta-path/etype, then attention over those path embeddings (`stack`) |
| `hgt` | **HGT** (Heterogeneous Graph Transformer) — [Hu et al., WWW 2020](https://arxiv.org/abs/2003.01332) | Type-aware attention scores and messages, then a target-type output map |
| `comp_gcn` | **CompGCN** (Composition-based Multi-Relational GCN) — [Vashishth et al., ICLR 2020](https://arxiv.org/abs/1911.03082) | Compose source features with edge features (`mult` or `sum`), then relation linear |

A **meta-path** is a typed walk (e.g. author→paper→author). HAN expects each
path you care about as an etype (often built from forward + reverse hops).
For HGT, typed query/key/value matrices live in the callables you pass.

Helper: `gat_attention_logit` builds a GAT-style edge score for HAN-style
logits.

## Relation to DGL

Value parity tests cover batch/unbatch and `multi_update_all` kernels
(including float `u_mul_e` cases). Empty max/min destinations are `0`,
matching DGL. Cross-reducers match when every destination receives every
involved etype; with partial coverage, anytensor zero-fills then fuses
(same as composing DGL per-etype `update_all` then the same cross reduce).
