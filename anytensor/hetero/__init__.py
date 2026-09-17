"""Heterogeneous graph types and tree batch/unbatch hooks.

Not part of the jraph-mirroring API. Import from here::

    from anytensor.hetero import HeteroGraphsTuple, multi_update_all
    from anytensor.hetero import relational_graph_convolution
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
from .message import (
    AttentionLogitFn,
    AttentionReduceFn,
    MessageFn,
    RelationSpec,
    SrcApplyFn,
    attention_weight_messages,
    copy_u_message,
    multi_update_all,
    relation_mailbox,
)
from .models import (
    comp_gcn,
    gat_attention_logit,
    han,
    hetero_sage,
    hgt,
    relational_graph_convolution,
)

__all__ = [
    "ArrayTree",
    "AttentionLogitFn",
    "AttentionReduceFn",
    "CanonicalEtype",
    "HeteroGraphsTuple",
    "MessageFn",
    "Ntype",
    "RelationSpec",
    "SendRecvTuple",
    "SrcApplyFn",
    "attention_weight_messages",
    "comp_gcn",
    "copy_u_message",
    "gat_attention_logit",
    "graphs_tuple_as_send_recv",
    "han",
    "hetero_sage",
    "hgt",
    "key_schema",
    "multi_update_all",
    "relation_mailbox",
    "relational_graph_convolution",
    "schemas_equal",
]
