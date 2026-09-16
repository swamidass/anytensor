# Jraph examples

These fenced blocks are executed by pytest (via
[Sybil](https://sybil.readthedocs.io/)). Segment-level GAT recipes stay in
[Worked examples](../examples.md); this page is GraphsTuple / GraphNetwork.

## Batch, unbatch, GraphNetwork

Same sparse graph layout as jraph, on whatever tensor the caller already has.
`batch` offsets senders; `GraphNetwork` with identity updates is a no-op.

```python
import numpy as np
from anytensor import jraph as atj

g1 = atj.GraphsTuple(
    nodes=np.arange(6.0).reshape(3, 2),
    edges=np.arange(10.0).reshape(5, 2),
    senders=np.array([0, 0, 1, 1, 2]),
    receivers=np.array([1, 2, 0, 2, 1]),
    n_node=np.array([3]),
    n_edge=np.array([5]),
    globals=np.array([[1.0, 0.0]]),
)
g2 = atj.GraphsTuple(
    nodes=np.arange(6.0, 10.0).reshape(2, 2),
    edges=np.arange(4.0).reshape(2, 2),
    senders=np.array([0, 1]),
    receivers=np.array([1, 0]),
    n_node=np.array([2]),
    n_edge=np.array([2]),
    globals=np.array([[0.0, 1.0]]),
)
batched = atj.batch([g1, g2])
assert batched.nodes.shape == (5, 2)
assert list(np.asarray(batched.senders[5:])) == [3, 4]
parts = atj.unbatch(batched)
np.testing.assert_allclose(parts[1].nodes, g2.nodes)
np.testing.assert_array_equal(np.asarray(parts[1].senders), g2.senders)

net = atj.GraphNetwork(
    update_edge_fn=lambda e, s, r, g: e,
    update_node_fn=lambda n, s, r, g: n,
    update_global_fn=lambda n, e, g: g,
)
out = net(batched)
np.testing.assert_allclose(out.nodes, batched.nodes)
```

## Padding and masks

`pad_with_graphs` appends a dummy graph plus empty graphs. Masks are `True` for
real nodes / edges / graphs.

```python
import numpy as np
from anytensor import jraph as atj

g = atj.GraphsTuple(
    nodes=np.arange(6.0).reshape(3, 2),
    edges=np.arange(10.0).reshape(5, 2),
    senders=np.array([0, 0, 1, 1, 2]),
    receivers=np.array([1, 2, 0, 2, 1]),
    n_node=np.array([3]),
    n_edge=np.array([5]),
    globals=np.array([[1.0, 0.0]]),
)
padded = atj.pad_with_graphs(g, n_node=6, n_edge=8, n_graph=3)
assert int(np.asarray(padded.n_node).sum()) == 6
assert int(np.asarray(atj.get_node_padding_mask(padded)).sum()) == 3
restored = atj.unpad_with_graphs(padded)
np.testing.assert_allclose(restored.nodes, g.nodes)
```

## `tree.concat` is graph batching

`GraphsTuple.__tree_concat__` calls `batch` (sender offsets), not fieldwise
concat. Feature objects can define `__tree_concat__` / `__tree_split__` the
same way — see [Tree](../tree/index.md#concat-and-split).

```python
import numpy as np
from anytensor import jraph as atj
from anytensor import tree

g1 = atj.GraphsTuple(
    nodes=np.arange(6.0).reshape(3, 2),
    edges=np.arange(4.0).reshape(2, 2),
    senders=np.array([0, 1]),
    receivers=np.array([1, 2]),
    n_node=np.array([3]),
    n_edge=np.array([2]),
    globals=np.array([[1.0, 0.0]]),
)
g2 = atj.GraphsTuple(
    nodes=np.arange(6.0, 10.0).reshape(2, 2),
    edges=np.arange(4.0, 8.0).reshape(2, 2),
    senders=np.array([0, 1]),
    receivers=np.array([1, 0]),
    n_node=np.array([2]),
    n_edge=np.array([2]),
    globals=np.array([[0.0, 1.0]]),
)
batched = tree.concat(g1, g2)
assert list(np.asarray(batched.n_node)) == [3, 2]
assert list(np.asarray(batched.senders[2:])) == [3, 4]
a, b = tree.split(batched, [1, 1])
np.testing.assert_array_equal(np.asarray(b.senders), g2.senders)
```
