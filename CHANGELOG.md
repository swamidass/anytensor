## 1.0.0 (2026-09-16)

### Added

- Expand design docs into a principles-and-edge-cases explainer. (#design-explainer)
- Add pytest-verified docs examples for GAT helper and jit/compile/script paths. (#docs-examples)
- Expand docs home with motivation and a GAT-style portable neighbor-attention case study. (#docs-intro)
- CI job installs backends at declared minimum versions (`ci/min-versions.txt`). (#min-backends)
- Document SemVer policy and highlight the multi-layer test contract. (#stability)
- Ordinary ops via array-api-compat; segment helpers (mean/count/variance/softmax/partition_softmax); 0-d array scalar policy; Torch/JAX/TF backend patches and version floors. (#1)

### Changed

- Raise Torch (≥2.1), TensorFlow (≥2.13), jaxtyping (≥0.2.34), and array-api-compat (≥1.15) floors for numpy 1.24 / torch.export / TypeVar shape hints / Array API surface. (#backend-floors)
- Prefer `torch.compile` / `torch.export` in docs; treat TorchScript as legacy. (#torch-compile-docs)


# Changelog

All notable changes to this project will be documented in this file.

The format is managed by [towncrier](https://towncrier.readthedocs.io/).
Fragments live in `changelog.d/`.
