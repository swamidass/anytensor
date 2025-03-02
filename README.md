# AnyTensor Library

## Overview
A high-level functional API that works transparently across multiple tensor libraries including NumPy, JAX, Torch, and TensorFlow. This library follows the architectural approach of einops, allowing you to write tensor operations once and execute them on any supported backend.

## Problem 

There are several tensor libraries currently in use, each with its on quirks, strengths and weaknesses. Because of this, it is difficult to write high-level code that works across all of them. So higher level libraries have to either:

1. Choose a single backend and limit their users to that backend.
2. Implement a backend or plugin structure, which is tricky to build out effectively.

Most libraries take the first approach, and need to be translated into entirely new code to be used with a different backend. Keras 2.0 and Einops are examples of the second approach. Kera 2.0 backend syste is  complex,  difficult to extend, and requires selection of a single backend rather than adapting to the inputs. 

In contrast, the design of Einops is much simpler, more extensible, and transparently adapts to the inputs. The main downside of einops is that it is not complete enough for most use cases. Just as significant, Einops' backend system is not part of its public API, and is not designed to be used by other libraries. So, it is not advisable to use einops backend system directly in other libraries.


## Solution

AnyTensor library makes it possible to write high-level functions that work across multiple tensor libraries, without  creating your own backend or plugin system. Libraries that make use of anytensor should work on all the backends it supports, and adding backends should be fairly striaightforward too. This library follows the architectural approach of einops, with a focus on simplicity and extensibility.


## Installation

For now, install directly from github.

```bash
pip install git+https://github.com/swamidass/anytensor.git
```

In the future, we will publish to PyPI.


## Usage

The primary interface is the functions exported in the main 'anytensor' module. Here is an example of how to use it.

```python
import anytensor as at
import numpy as np

x = np.random.rand(3, 4)

s = at.sum(x, axis=0)
```
This code will work with tensor created by any backend library, not just numpy.

## Segment Functions

One of the primary motivations for creating this library is to provide a unified interface for segment functions. These functions are used to aggregate tensors based on a segment index. For example, given a tensor `x` and a segment index `s`, we can compute the sum of each segment as follows.

```python
import anytensor as at
import numpy as np

x = np.arange(12).reshape(3, 4)
s = np.array([0, 0, 1])

y = at.segment_sum(x, s)
```

Segements are valuable because they enable the construction of models that operate on ragged arrays, where the second dimension of each tensor is not fixed. This is a common pattern in graph neural networks, where each graph has a different number of nodes and edges.


## Einops Compatibility

The library imports all the functions from the einops library, which uses a very similar appraoch to supporting multiple backends.

## Contributing

PRs are welcome.

### Adding a new backend

To add a new backend, you need to implement a class that inherits from the `Backend` class and implements the required methods. 

### Adding a new op or function

To add a new op or function, make sure that it can be implemented using the existing tensor backends. 

If not, you will need to add and test any required functionality to the backends. The operations required by the backend should be minimized, and new higher-level functionality should be pushed to core.py as much as possible. 

New functions with generic utility should  be added to core.py. Key design principles are that the function should begin by infering the backend from the inputs, and then use exclusively backend operations to compute the output. The only exceptions to this are basic element-wise operations that all tensor libraries support natively (e.g. addition, multiplication).

Add testing code for any new ops to ensure that the function works correctly on all backends. The easiest way to do this is by adding to test/test_ops.py. 

### Emulating a new API

We  want to replicate the APIs of important libraries of reference. For example, for graph neural networks, we aim to implement the jraph API as a drop in replacement.

## Road Map

- [x] Implement basic tensor operations
- [x] Implement segment functions
- [ ] implement comprehensive testing
- [ ] Implement jraph API for graph neural networks