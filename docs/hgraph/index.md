# HGraph — heterogeneous graphs

`anytensor.hgraph` is a small **heterogeneous graph** stack on the caller’s
tensors (NumPy / JAX / PyTorch / TensorFlow). A heterogeneous graph has more
than one **node type** (ntype) and/or **edge type** (etype)—for example
authors, papers, and “writes” / “cites” relations—unlike a
**homogeneous** graph where every node and edge shares one type.

## Why this package

Homogeneous GNNs assume one node/edge type. Many real graphs do not: authors
write papers, papers cite papers, users rate movies. Frameworks that own the
full stack ([DGL](https://www.dgl.ai/),
[PyG](https://pytorch-geometric.readthedocs.io/)) already do hetero well —
**inside one backend**.

`anytensor.hgraph` fills a narrower niche:

| | This package | DGL / PyG hetero | `anytensor.jraph` |
|---|---|---|---|
| Backend | Your arrays (NumPy / JAX / Torch / TF) | Framework-native graphs | Portable jraph homo API |
| Core idea | DGL-style `multi_update_all` on `take` / `segment_*` | Full training stack + samplers | GraphsTuple + GraphNetwork |
| Ownership | You own weights / modules | Framework modules | You own weights |
| Scope | Typed incidence + vectorized mailboxes + small zoo | Production hetero tooling | Homogeneous only |

In short: **portable hetero message passing without leaving your tensor
stack** — same kernels on four backends, compile-friendly (no edge Python
loops), and a thin zoo (R-GCN, GraphSAGE, HAN, HGT, CompGCN) as plain
functions. It is **not** a DGL replacement (no neighbor sampling, no graph
store) and **not** part of the jraph-mirroring API — import it explicitly:

```python
from anytensor.hgraph import HeteroGraphsTuple, multi_update_all
from anytensor.hgraph import relational_graph_convolution
```

The portable core mirrors DGL’s
[`multi_update_all`](https://docs.dgl.ai/generated/dgl.DGLGraph.multi_update_all.html):
**per-relation message + reduce**, then an explicit **cross-relation fuse**,
built on AnyTensor primitives (`take`, `segment_*`, `segment_attention`).

Nested features use [`anytensor.tree`](../tree/index.md). Batching is
`tree.batch` / `tree.unbatch` via `HeteroGraphsTuple.__tree_batch__`
(matching ntype/etype keys required; use empty arrays, not `None`, to pad
schemas).

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## Data model

| Type | Role |
|---|---|
| `HeteroGraphsTuple` | Nodes / edges keyed by ntype and **canonical etype** `(src_ntype, relation, dst_ntype)` |
| `SendRecvTuple` | One directed send→receive incidence (same ntype = homo view; two ntypes = bipartite) |
| `RelationSpec` | Per-relation `message_fn`, optional `src_apply` (before gather), reduce, attention |

A **canonical etype** is a triple such as `("author", "writes", "paper")`:
source node type, relation name, destination node type. Edges are
**directed**. Bidirectional traffic is two etypes (see
[Reverse edges](#reverse-edges)), not an undirected flag.

## Reverse edges

Hetero message passing only sees **stored** etype keys in
`senders` / `receivers` / `n_edge`. Two tools:

| API | What it does | Use for |
|---|---|---|
| `relation_view(etype, reverse=True)` | Zero-copy swapped send/recv **aliases** | Inspection, custom loops |
| `add_reverse_edges(graph, …)` | Materializes `(dst, rev_rel, src)` with swapped incidence | R-GCN / HAN / any `etype_dict` that needs both directions |

```python
from anytensor.hgraph import add_reverse_edges, reverse_canonical_etype

writes = ("author", "writes", "paper")
# Default reverse relation name: rev_writes
g2 = add_reverse_edges(g, [writes])
assert ("paper", "rev_writes", "author") in g2.n_edge

# Named reverse (author–paper both ways)
g3 = add_reverse_edges(
    g,
    [writes],
    rev_relation="written_by",  # -> ("paper", "written_by", "author")
)
```

`rev_relation` may also be a callable `etype -> str`. Pass
`skip_existing=True` if some reverses are already present.
`copy_edata=True` (default) reuses the forward edge-feature object on the
reverse key; set `False` to store `None`.

Example: author → paper → institution needs both forward and reverse hops
as stored etypes if you run message passing in both directions or build
HAN meta-paths like author→paper→author offline as separate etypes.

## Message passing

`relation_mailbox` runs **one** etype:

1. Optional **`src_apply`** on `graph.nodes[src]` (size `N_src`).
2. Gather source features along `senders` (and destination features along
   `receivers` when needed).
3. **`message_fn(src, dst, edges) -> messages`** on those **edge-sized**
   tensors (default `copy_u_message` returns `src`).
4. Optional **attention**: score each edge, then
   `segment_attention` (softmax within each destination’s
   neighborhood + weighted `segment_sum`). Edge work is vectorized — there
   is **no Python loop over messages**. Schema-sized loops over etypes only
   (a handful of relations) are fine under `jax.jit` / `tf.function`.
5. **Segment reduce** onto destinations (`sum` / `mean` / `max` / `min`)
   when attention is off. Nodes with no incoming edges of that type get `0`.

### `message_fn` signature

```text
message_fn(src, dst, edges) -> messages
```

Every argument is **edge-aligned** for that etype (leading size `E`):

| Arg | Meaning |
|---|---|
| `src` | Source node features gathered with `senders`. If `src_apply` is set, this is `take(src_apply(nodes[src]), senders)`; otherwise `take(nodes[src], senders)`. |
| `dst` | Destination node features gathered with `receivers` (`take(nodes[dst], receivers)`). |
| `edges` | `graph.edges[etype]`, or `None` if unset. When present, leading size should be `E`. |

Return per-edge `messages` (leading `E`, or a pytree of such tensors). Those
are what get attention-weighted and/or segment-reduced onto destinations.

Built-in: `copy_u_message(src, dst, edges)` → `src` (DGL `fn.copy_u`).

`attention_logit_fn` uses the **same** `(src, dst, edges)` layout, except its
`src` is always gathered from the **raw** source pool (not `src_apply`
output), so scores stay on pre-message features.

`multi_update_all` runs many etypes, then fuses mailboxes that share a
destination ntype with a **cross-reducer** (`sum` / `mean` / `max` /
`min` / `stack`). Here a **mailbox** is the per-relation aggregated tensor
at each destination node. `stack` yields shape
`(n_dst, n_relations, …)` in etype-dict insertion order — used when a model
needs to attend **across relations** (see HAN below).

Per-relation attention matches the optional attention path on homo
[`GraphNetwork`](../jraph/index.md) via the shared
`segment_attention` helper (same idea as
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) / GAT). That
is what lets this stack express **Heterogeneous Graph Attention Network**
(HAN) node-level attention and **Heterogeneous Graph Transformer** (HGT)
typed attention.

## Before gather vs after gather

Source-only maps (typical relation linears) can run in two places:

| Hook | When it runs | Tensor size | Use when |
|---|---|---|---|
| `RelationSpec.src_apply` | **Before** gather | `N_src` nodes | Source-only linear / row map; usually best if `E > N` |
| `RelationSpec.message_fn` | **After** gather | `E` edges | Map needs edge or destination features, or you deliberately want the edge-sized path (e.g. very sparse `E ≪ N`) |

Preferred source-linear pattern:

```python
from anytensor.hgraph import RelationSpec, copy_u_message

RelationSpec(
    message_fn=copy_u_message,  # gather only
    src_apply=lambda h: h @ W,  # matmul on nodes
    reduce="sum",
)
```

Equivalent numerically for a row-wise linear, but costlier when `E > N`:

```python
RelationSpec(
    message_fn=lambda s, d, e: s @ W,  # matmul on edges
    reduce="sum",
)
```

There is no separate “post-gather flag”: **after gather is `message_fn`**.
Zoo models that only transform sources (R-GCN, GraphSAGE, HAN, HGT) use
`src_apply`. CompGCN keeps the linear in `message_fn` because it composes
with edge features first.

Other efficiency rules:

- Neighborhood attention stays **per etype** (`segment_attention` on that
  relation’s ragged edge list). Do not interleave relations into one
  `(n, R, E…)` edge tensor — edge counts differ and would force copies.
- HAN **semantic** attention (after `stack`) is a **dense** softmax on
  schema-sized `(n, R, d)` — not `segment_attention` (that would
  flatten / `repeat` path ids / scatter).

## Model zoo

Plain functions in `anytensor.hgraph.models` (also re-exported from
`anytensor.hgraph`). Each takes a graph plus callables for the learnable
pieces — your framework owns the weights (`lambda x: x @ W`, module
`__call__`, etc.).

| Function | Full name / paper | What it does |
|---|---|---|
| `relational_graph_convolution` | **R-GCN** (Relational Graph Convolutional Network) — [Schlichtkrull et al., ESWC 2018](https://arxiv.org/abs/1703.06103) | One linear per relation, mean/sum over neighbors, plus a self term |
| `hetero_sage` | Heterogeneous **GraphSAGE** — [Hamilton et al., NeurIPS 2017](https://arxiv.org/abs/1706.02216) | Mean-aggregate neighbors, concat with self, one combine linear |
| `han` | **HAN** (Heterogeneous Graph Attention Network) — [Wang et al., WWW 2019](https://arxiv.org/abs/1903.07293) | Per-etype `segment_attention` on neighbors, then dense softmax over stacked path embeddings (`stack`) |
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
