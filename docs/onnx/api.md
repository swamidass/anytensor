# ONNX helpers

!!! warning "Unstable guide"
    `anytensor.onnx` is **not** a stable library API and **not** an ONNX
    Runtime backend. It is a recipe module for downstream *model* builders.
    Names may change. It is **not** in `anytensor.__all__` — import the
    subpackage explicitly:

    ```python
    from anytensor import onnx
    ```

    Library authors should keep helpers portable (`at.shape`, segment ops)
    and leave serialization to the application.

Narrative: [Overview](index.md). Recipes: [Examples](examples.md).

::: anytensor.onnx
    options:
      members_order: source
      filters:
        - "!^_"
