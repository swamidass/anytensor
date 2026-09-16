# Tree examples

These fenced blocks are executed by pytest (via
[Sybil](https://sybil.readthedocs.io/)). Graph batching that uses `tree.batch`
on `GraphsTuple` lives in [Jraph examples](../jraph/examples.md). The first
batch example is a nested lab record — the same walk as a GNN feature nest.

## `map`, `flatten`, and `None`

`None` is an empty pytree: `map` does not call the function, and `flatten`
returns no leaves.

```python
import numpy as np
from anytensor import tree

mapped = tree.map(lambda x: x + 1, {"a": np.array([1, 2]), "b": None})
np.testing.assert_array_equal(mapped["a"], [2, 3])
assert mapped["b"] is None
leaves, treedef = tree.flatten({"x": None, "y": np.array([1.0])})
assert len(leaves) == 1
np.testing.assert_array_equal(tree.unflatten(treedef, leaves)["y"], [1.0])
```

## `batch` and `unbatch`

Stack structured records leafwise (experimental runs, time steps, minibatches).
All-`None` stays `None`. `unbatch` yields one record per leading index (same
call as `jraph.unbatch`).

```python
import numpy as np
from anytensor import tree

run_a = {"temp": np.array([20.1]), "ph": np.array([7.1]), "notes": None}
run_b = {"temp": np.array([21.0]), "ph": np.array([6.9]), "notes": None}
joined = tree.batch([run_a, run_b])
np.testing.assert_array_equal(joined["temp"], [20.1, 21.0])
assert joined["notes"] is None
first, second = tree.unbatch(joined)
np.testing.assert_array_equal(first["ph"], [7.1])
np.testing.assert_array_equal(second["temp"], [21.0])
```

## Objects that own batch / unbatch

`__tree_batch__` / `__tree_unbatch__` run before walking children. Use
AnyTensor ops inside them so the same class works on NumPy / JAX / Torch / TF.

```python
import numpy as np
import anytensor as at
from anytensor import tree


class Packed:
    def __init__(self, values):
        self.values = values

    @classmethod
    def __tree_batch__(cls, xs, axis=0):
        return cls(at.concatenate([x.values for x in xs], axis=axis))

    def __tree_unbatch__(self, axis=0):
        n = int(at.shape(self.values)[axis])
        ids = at.arange(n, like=self.values)
        return [
            Packed(at.take(self.values, ids[i : i + 1], axis=axis))
            for i in range(n)
        ]


joined = tree.batch(
    [Packed(np.array([[1.0, 2.0]])), Packed(np.array([[3.0, 4.0]]))]
)
assert isinstance(joined, Packed)
np.testing.assert_array_equal(joined.values, [[1.0, 2.0], [3.0, 4.0]])
a, b = tree.unbatch(joined)
np.testing.assert_array_equal(a.values, [[1.0, 2.0]])
```
