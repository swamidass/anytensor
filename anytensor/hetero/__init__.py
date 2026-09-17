"""Deprecated alias for :mod:`anytensor.hgraph`.

Import from ``anytensor.hgraph`` instead. This module re-exports the same
public API and emits :class:`DeprecationWarning` on import.
"""

from __future__ import annotations

import warnings

warnings.warn(
    "anytensor.hetero is renamed to anytensor.hgraph; "
    "import from anytensor.hgraph instead",
    DeprecationWarning,
    stacklevel=2,
)

from anytensor.hgraph import *  # noqa: F403
from anytensor.hgraph import __all__  # noqa: F401
