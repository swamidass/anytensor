"""Heterogeneous graph types and tree batch/unbatch hooks.

Not part of the jraph-mirroring API. Import from here::

    from anytensor.hetero import HeteroGraphsTuple, SendRecvTuple
    import anytensor.tree as tree
    batched = tree.batch([g1, g2])  # same keys required
"""

from .graph import (
    ArrayTree,
    CanonicalEtype,
    HeteroGraphsTuple,
    Ntype,
    SendRecvTuple,
    graphs_tuple_as_send_recv,
    key_schema,
    schemas_equal,
)

__all__ = [
    "ArrayTree",
    "CanonicalEtype",
    "HeteroGraphsTuple",
    "Ntype",
    "SendRecvTuple",
    "graphs_tuple_as_send_recv",
    "key_schema",
    "schemas_equal",
]
