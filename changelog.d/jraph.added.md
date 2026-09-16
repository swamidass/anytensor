Portable ``anytensor.jraph`` (GraphsTuple, batching, GraphNetwork zoo) and
``anytensor.tree`` (``jax.tree`` API plus ``concat``/``split``). Public names
cover every entry in official ``jraph.__all__``. Hypothesis parity vs
upstream jraph (when JAX is installed) includes batch/pad, GraphNetwork, the
model zoo (GAT with self-edges, not skipped), and segment ops. Custom-type
registration (magic flatten and existing JAX/Torch/optree registries) is beta;
the rest of the tree/jraph API is stable. Tree is pure Python with NumPy as
the only binary dependency, for any nested numeric record, not only graphs.
