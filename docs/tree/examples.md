# Tree examples

These fenced blocks are executed by pytest (via
[Sybil](https://sybil.readthedocs.io/)). Graph batching that uses `tree.concat`
on `GraphsTuple` lives in [Jraph examples](../jraph/examples.md). The first
concat example is a nested lab record — the same walk as a GNN feature nest.

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

## `concat` and `split`

Stack structured records leafwise (experimental runs, time steps, minibatches).
All-`None` stays `None`.

```python
import numpy as np
from anytensor import tree

run_a = {"temp": np.array([20.1, 20.3]), "ph": np.array([7.1, 7.0]), "notes": None}
run_b = {"temp": np.array([21.0]), "ph": np.array([6.9]), "notes": None}
joined = tree.concat(run_a, run_b)
np.testing.assert_array_equal(joined["temp"], [20.1, 20.3, 21.0])
assert joined["notes"] is None
first, second = tree.split(joined, [2, 1])
np.testing.assert_array_equal(first["ph"], [7.1, 7.0])
np.testing.assert_array_equal(second["temp"], [21.0])
```

## Objects that own concat / split

`__tree_concat__` / `__tree_split__` run before walking children.

```python
import numpy as np
from anytensor import tree


class Packed:
    def __init__(self, values):
        self.values = np.asarray(values)

    @classmethod
    def __tree_concat__(cls, xs, axis=0):
        return cls(np.concatenate([x.values for x in xs], axis=axis))

    def __tree_split__(self, sizes, axis=0):
        start = 0
        out = []
        for n in sizes:
            sl = [slice(None)] * self.values.ndim
            sl[axis] = slice(start, start + n)
            out.append(Packed(self.values[tuple(sl)]))
            start += n
        return out


joined = tree.concat(Packed([[1.0, 2.0]]), Packed([[3.0, 4.0]]))
assert isinstance(joined, Packed)
np.testing.assert_array_equal(joined.values, [[1.0, 2.0], [3.0, 4.0]])
a, b = tree.split(joined, [1, 1])
np.testing.assert_array_equal(a.values, [[1.0, 2.0]])
```
