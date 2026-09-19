# ONNX helpers

!!! warning "Unstable guide"
    `anytensor.export` is **not** a stable library API. It is a recipe module
    for downstream *model* builders. Names may change. It is **not** in
    `anytensor.__all__` — import the subpackage explicitly:

    ```python
    from anytensor import export
    ```

    The recommended target is **ONNX** (ONNX Runtime is well tested for
    deployment). AnyTensor does not run ops on ORT; these helpers serialize a
    Torch or TensorFlow graph. Library authors should keep helpers portable
    (`at.shape`, segment ops) and leave serialization to the application.

Narrative: [Overview](index.md). Recipes: [Examples](examples.md).

::: anytensor.export
    options:
      members_order: source
      filters:
        - "!^_"
