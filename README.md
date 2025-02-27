# AnyTensor Library

## Overview
A high-level functional API that works transparently across multiple tensor libraries including NumPy, JAX, Torch, TensorFlow, and Keras. This library follows the architectural approach of einops, allowing you to write tensor operations once and execute them on any supported backend.

## Features
- **Backend agnostic**: Write code once, run on any supported tensor library
- **Simple API**: Clean, functional interface for tensor operations
- **Extensible**: Easily add support for additional tensor libraries
- **Performance**: Leverages native operations of each backend
- **Consistent**: Same behavior across different tensor implementations

## Installation

```bash
pip install anytensor
```

## Usage

```python
import anytensor as at

```