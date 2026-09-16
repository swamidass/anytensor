# API reference

Generated from package docstrings with
[mkdocstrings](https://mkdocstrings.github.io/).

| Module | Contents |
|---|---|
| [Package root](anytensor.md) | Public re-exports (`anytensor` / `at`) |
| [Core ops](core.md) | Ordinary Array-API ops, promote, specials |
| [Segment ops](segment.md) | Segment reductions and `partition_softmax` |
| [Semantics](semantics_api.md) | `empty_segment_identity` |
| [Namespace](namespace.md) | TF Array-API shim (`array_namespace`) |
| [Optional imports](optional.md) | `module_if_loaded()` — already-imported extras, with callbacks |

Backends (`anytensor.backends`) are internal; use public helpers instead of
`get_backend` unless you are extending the library.
