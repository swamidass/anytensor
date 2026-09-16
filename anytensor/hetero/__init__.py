"""Heterogeneous graph types and tree batch/unbatch hooks.

Not part of the jraph-mirroring API. Import from here::

    from anytensor.hetero import HeteroGraphsTuple, SendRecvTuple
    import anytensor.tree as tree
    batch = tree.batch([g1, g2])  # -> HeteroBatch
"""

from .graph import (
    ArrayTree,
    CanonicalEtype,
    HeteroBatch,
    HeteroGraphsTuple,
    Ntype,
    SendRecvTuple,
    canonicalize_schema,
    graphs_tuple_as_send_recv,
    schemas_equal,
)

__all__ = [
    "ArrayTree",
    "CanonicalEtype",
    "HeteroBatch",
    "HeteroGraphsTuple",
    "Ntype",
    "SendRecvTuple",
    "canonicalize_schema",
    "graphs_tuple_as_send_recv",
    "schemas_equal",
]
