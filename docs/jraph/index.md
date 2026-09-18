# Jraph (portable)

[Jraph](https://github.com/google-deepmind/jraph) (pronounced “giraffe”) is
DeepMind’s lightweight library for graph neural networks in JAX. It gives you
a single sparse data structure, utilities to batch and pad it for `jit`, and
a small zoo of forkable models — without telling you which neural-net library
to use.

`anytensor.jraph` is that same stack on **NumPy / JAX / PyTorch / TensorFlow**:
same `GraphsTuple` layout, same `GraphNetwork` call signature, caller’s
tensors. Nested features use [`anytensor.tree`](../tree/index.md) (`jax.tree`
API; pure Python + NumPy; useful for any nested record, not only graphs).
`None` is an empty pytree, matching jraph.

Runnable recipes: [Examples](examples.md). Generated API: [API](api.md).

## Why jraph

Jraph is small on purpose. The
[`GraphsTuple`](https://github.com/google-deepmind/jraph) is one namedtuple
(`nodes`, `edges`, `senders`, `receivers`, `globals`, `n_node`, `n_edge`) that
holds **one or many** directed graphs. Batching is concatenation with sender
offsets — not a second graph type. Nested dicts of arrays are first-class
because JAX pytrees already know how to walk them.

The core algorithm is a functional
[`GraphNetwork`](https://arxiv.org/abs/1806.01261) (Battaglia et al.): you
pass update functions (typically neural nets, but any JAX — here, any
AnyTensor — callable). Jraph does not prescribe Haiku, Flax, or raw modules.
Thin wrappers (`InteractionNetwork`, `GAT`, `GraphConvolution`, `DeepSets`,
…) are configurations of that one function, meant to be forked.

Padding **with graphs** (a dummy graph plus empty graphs) is the other
design win: static node/edge/graph counts for `jax.jit` without a parallel
“padded graph” API. Masks and `zero_out_padding` keep the dummy off the
loss.

Those choices — sparse tuple, functional message passing, pad-to-static,
pytree features — are why this port follows jraph instead of inventing
another GNN surface. Use upstream jraph when you are JAX-only; use
`anytensor.jraph` when the same helper must run on the caller’s backend.

Docs: [jraph README](https://github.com/google-deepmind/jraph) ·
[jraph.readthedocs.io](https://jraph.readthedocs.io/).

## GraphsTuple

| Field | Meaning |
|---|---|
| `nodes` / `edges` / `globals` | Feature nest, or `None` |
| `senders` / `receivers` | Integer index arrays (absolute in the batched node array) |
| `n_node` / `n_edge` | One integer per graph in the batch |

`senders` / `receivers` may be `None` when there are no edges.

## Batching and padding

`batch` concatenates graphs and **offsets** senders/receivers. `unbatch`
inverts that. Neither is compilable: the output size depends on the list /
batch. These are the same functions as `tree.batch` / `tree.unbatch`;
`GraphsTuple.__tree_batch__` / `__tree_unbatch__` own the graph logic (not
fieldwise array concat). Custom feature objects may define the same methods.
See [Tree](../tree/index.md#batch-and-unbatch).

`pad_with_graphs` appends a dummy graph plus empty graphs so counts hit
static sizes (`n_graph >= 2`). Masks and `zero_out_padding` ignore the dummy.
`dynamically_batch` yields padded batches from an iterator.

## Models

`GraphNetwork` follows Battaglia et al. (sender and receiver aggregations,
optional softmax attention). Thin wrappers match jraph:
`InteractionNetwork`, `GraphMapFeatures`, `RelationNetwork`, `DeepSets`,
`GraphNetGAT`, `GAT`, `GraphConvolution`. Apply is decorated with
`@cache` so `partition_ids` expands the same `n_node` /
`n_edge` vector once (sticky across stacked applies; weakrefs;
`cache["partition"]`; other partition helpers call `partition_ids`).

Segment helpers on this module still require `num_segments` (AnyTensor
contract). `unique_indices` is accepted and ignored.

## Differences from official jraph

| Topic | This port |
|---|---|
| Backends | Caller’s tensors (NumPy / JAX / Torch / TF) |
| `None` features | Empty pytree (jraph / `jax.tree`) |
| Segment ops | `num_segments` required; `partition_softmax` requires `sum_partitions` (3rd positional, official jraph name for core `total_length`) and takes `num_segments` from `shape(partitions)[0]`; `unique_indices` ignored |
| Nest library | [`anytensor.tree`](../tree/index.md) (no JAX runtime dep) |
| Graph concat | Magic methods on `GraphsTuple` (`__tree_batch__` / `__tree_unbatch__`); `jraph.batch` is `tree.batch` |
| Public names | **Every name in official `jraph.__all__`** (unit-tested). Also exports `segment_mean` / `min` / `variance` / `normalize` (on the official module, omitted from its `__all__`) and `sparse_matrix_to_graphs_tuple` (not in upstream jraph). |
| Not in scope | `jraph.experimental` (sharded GraphNet), examples, private `dtype_max_value` / `dtype_min_value` |

Hypothesis parity vs official jraph (when `jraph` + JAX are installed) is
`test/test_jraph_parity_fuzz.py`: batch/unbatch, pad/masks, GraphNetwork,
nested features, segment ops, the model zoo (`GraphMapFeatures`,
`InteractionNetwork`, `RelationNetwork`, `DeepSets`, `GraphNetGAT`, **`GAT`**
with self-edges added rather than skipped, `GraphConvolution` including
`add_self_edges=True`), fully-connected graphs, zero-out padding.
