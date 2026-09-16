Portable ``anytensor.jraph`` (GraphsTuple, batching, GraphNetwork) and
``anytensor.tree`` (``jax.tree`` API plus ``concat``/``split``). Custom-type
registration (magic flatten and existing JAX/Torch/optree registries) is beta;
the rest of the tree/jraph API is stable. Tree is for any nested numeric
record, not only graphs. Docs: Tree and Jraph sections.
