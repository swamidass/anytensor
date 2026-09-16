# Jraph (portable)

[Jraph](https://github.com/google-deepmind/jraph) (pronounced “giraffe”) is
DeepMind’s lightweight library for graph neural networks in JAX. It gives you
a single sparse data structure, utilities to batch and pad it for `jit`, and
a small zoo of forkable models — without telling you which neural-net library
to use.

`anytensor.jraph` is that same stack on **NumPy / JAX / PyTorch / TensorFlow**:
same `GraphsTuple` layout, same `GraphNetwork` call signature, caller’s
tensors. Nested features use [`anytensor.tree`](../tree/index.md) (`jax.tree`
API). `None` is an empty pytree, matching jraph.

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
batch.

`GraphsTuple.__tree_concat__` / `__tree_split__` implement that graph
batching (not fieldwise array concat). Custom feature objects may define the
same methods so `batch` / `unbatch` (and `tree.concat` / `tree.split`) use
their logic. See [Tree](../tree/index.md#concat-and-split).

`pad_with_graphs` appends a dummy graph plus empty graphs so counts hit
static sizes (`n_graph >= 2`). Masks and `zero_out_padding` ignore the dummy.
`dynamically_batch` yields padded batches from an iterator.

## Models

`GraphNetwork` follows Battaglia et al. (sender and receiver aggregations,
optional softmax attention). Thin wrappers match jraph:
`InteractionNetwork`, `GraphMapFeatures`, `RelationNetwork`, `DeepSets`,
`GraphNetGAT`, `GAT`, `GraphConvolution`.

Segment helpers on this module still require `num_segments` (AnyTensor
contract). `unique_indices` is accepted and ignored.

## Differences from official jraph

| Topic | This port |
|---|---|
| Backends | Caller’s tensors (NumPy / JAX / Torch / TF) |
| `None` features | Empty pytree (jraph / `jax.tree`) |
| Segment ops | `num_segments` required; `unique_indices` ignored |
| Nest library | [`anytensor.tree`](../tree/index.md) (no JAX runtime dep) |
| Graph concat | Magic methods on `GraphsTuple` and feature objects |
