"""Public typing helpers for AnyTensor.

Uses :mod:`jaxtyping` for **dtype + shape** annotations and a :class:`~typing.TypeVar`
(``ArrayT``) so array operands stay on one backend type — without importing
Torch / JAX / TF.

Runtime shape/dtype checking is **off by default**. Annotations are for static
checkers, editors, and docs. To opt in to runtime checks, install ``beartype``
and either::

    from jaxtyping import install_import_hook
    install_import_hook("anytensor", "beartype.beartype")
    import anytensor as at

or call :func:`anytensor.enable_typecheck` **before** importing ``anytensor``
submodules you want checked. You can also set ``JAXTYPING_DISABLE=1`` to force
all jaxtyping runtime checks off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeVar, Union

from jaxtyping import (
    Bool,
    Float,
    Inexact,
    Int,
    Integer,
    Num,
    Real,
    Shaped,
)

# One concrete array type per call (NumPy / JAX / Torch / TF ndarray-like).
# Same TypeVar on multiple parameters ⇒ matching backend array types.
ArrayT = TypeVar("ArrayT")

DtypeLike = Any
# Python int, jit/compile symbolic size, or 0-d integral tensor scalar.
ShapeSize = Any
ShapeLike = Union[int, Sequence[Any], Any]
Axes = Union[int, Sequence[int], None]

# Common AnyTensor layouts (leading space avoids flake8 F821 on bare names).
ShapedArray = Shaped[ArrayT, " ..."]
FloatArray = Float[ArrayT, " ..."]
IntArray = Integer[ArrayT, " ..."]
# Segment reductions: values ``(n, ...)``, ids ``(n,)``, out ``(num_segments, ...)``.
SegmentValues = Shaped[ArrayT, " n ..."]
SegmentIds = Integer[ArrayT, " n"]
SegmentOut = Shaped[ArrayT, " num_segments ..."]

__all__ = [
    "ArrayT",
    "Axes",
    "Bool",
    "DtypeLike",
    "Float",
    "FloatArray",
    "Inexact",
    "Int",
    "IntArray",
    "Integer",
    "Num",
    "Real",
    "SegmentIds",
    "SegmentOut",
    "SegmentValues",
    "ShapeLike",
    "ShapeSize",
    "Shaped",
    "ShapedArray",
    "enable_typecheck",
]


def enable_typecheck(typechecker: str = "beartype.beartype") -> None:
    """Opt-in runtime jaxtyping checks for subsequent ``anytensor`` imports.

    Must be called **before** importing the modules you want checked (or use
    :func:`jaxtyping.install_import_hook` yourself at process start). Default
    AnyTensor imports do **not** install this hook.

    Requires the named typechecker package (default: ``beartype``).
    """
    from jaxtyping import install_import_hook

    install_import_hook("anytensor", typechecker)
