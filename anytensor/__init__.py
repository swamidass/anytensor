from . import backends
from .backends import get_backend

# Import einops functions, which work the same way as anytensor.
from einops import einsum, pack, unpack, rearrange, reduce

from .segment import (
    segment_sum,
    segment_max,
    segment_min,
    segment_mean,
    segment_normalize,
)

from .core import (
    repeat,
    take,
    exp,
    log,
    sum,
    min,
    max,
    mean,
    prod,
    shape,
    cumsum,
    reshape,
    transpose,
    concatenate,
    stack,
    maximum,
    minimum,
    sqrt,
    rsqrt,
    where,
    clip,
    astype,
    cast,
    zeros_like,
    ones_like,
    full_like,
    zeros,
    ones,
    full,
    arange,
    matmul,
)

try:
    from ._version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0"

__all__ = [
    "backends",
    "get_backend",
    "einsum",
    "pack",
    "unpack",
    "rearrange",
    "reduce",
    "segment_sum",
    "segment_max",
    "segment_min",
    "segment_mean",
    "segment_normalize",
    "repeat",
    "take",
    "exp",
    "log",
    "sum",
    "min",
    "max",
    "mean",
    "prod",
    "shape",
    "cumsum",
    "reshape",
    "transpose",
    "concatenate",
    "stack",
    "maximum",
    "minimum",
    "sqrt",
    "rsqrt",
    "where",
    "clip",
    "astype",
    "cast",
    "zeros_like",
    "ones_like",
    "full_like",
    "zeros",
    "ones",
    "full",
    "arange",
    "matmul",
    "__version__",
]
