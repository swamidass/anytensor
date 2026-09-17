# Hetero examples

These recipes assume a `HeteroGraphsTuple` graph `g` is already built.
Pass learnable maps as callables (`lambda x: x @ W`, a module `__call__`,
…); the library does not own parameters.

Overview and term definitions: [Heterogeneous graphs](index.md).
API: [Hetero API](api.md).

## Per-relation attention (kernel)

Before the named models, the shared primitive is
`segment_attention`: score each edge, softmax
**within each destination node’s neighborhood**, weight messages, sum.
Same idea as
[Graph Attention Networks](https://arxiv.org/abs/1710.10903) / **GAT**
(Veličković et al., ICLR 2018). Run it **per etype** (relations have
different edge counts — do not interleave into one multi-relation edge
tensor). HAN/HGT reuse this for node-level attention; HAN semantic mixing
over stacked path embeddings stays a dense `(n, R)` softmax.

```python
import numpy as np
from anytensor.hetero import (
    HeteroGraphsTuple,
    RelationSpec,
    gat_attention_logit,
    multi_update_all,
)

writes = ("author", "writes", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 2), dtype=np.float32),
        "paper": np.ones((2, 2), dtype=np.float32),
    },
    edges={writes: None},
    senders={writes: np.array([0, 1, 2])},
    receivers={writes: np.array([0, 0, 1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3])},
)

W = np.eye(2, dtype=np.float32)
a = np.ones((4, 1), dtype=np.float32)
out = multi_update_all(
    g,
    {
        writes: RelationSpec(
            message_fn=lambda s, d, e: s @ W,
            reduce="sum",
            attention_logit_fn=lambda s, d, e: gat_attention_logit(
                s, d, lambda x: x @ a
            ),
        ),
    },
    cross_reducer="sum",
)
assert out.nodes["paper"].shape == (2, 2)
```

## Relational GCN (R-GCN) — Schlichtkrull et al., ESWC 2018

**R-GCN** = Relational Graph Convolutional Network: one weight matrix per
relation, aggregate neighbors (usually mean), add a self/root term.
[arXiv:1703.06103](https://arxiv.org/abs/1703.06103)

```python
import numpy as np
from anytensor.hetero import HeteroGraphsTuple, relational_graph_convolution

writes = ("author", "writes", "paper")
cites = ("paper", "cites", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 4), dtype=np.float32),
        "paper": np.ones((2, 4), dtype=np.float32),
    },
    edges={writes: None, cites: None},
    senders={writes: np.array([0, 1, 2]), cites: np.array([0])},
    receivers={writes: np.array([0, 0, 1]), cites: np.array([1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3]), cites: np.array([1])},
)

d = 4
W_writes = np.eye(d, dtype=np.float32)
W_cites = np.eye(d, dtype=np.float32)
W_author = 0.1 * np.eye(d, dtype=np.float32)
W_paper = 0.1 * np.eye(d, dtype=np.float32)

out = relational_graph_convolution(
    g,
    relation_apply={
        writes: lambda x: x @ W_writes,
        cites: lambda x: x @ W_cites,
    },
    self_apply={
        "author": lambda x: x @ W_author,
        "paper": lambda x: x @ W_paper,
    },
)
assert out.nodes["paper"].shape == (2, 4)
```

## Heterogeneous GraphSAGE — Hamilton et al., NeurIPS 2017

**GraphSAGE** (SAmple and aggreGatE) mean-aggregates neighbor features,
concatenates with the node’s own features, then applies a linear map. The
hetero wrap runs that pattern per relation and sums relation mailboxes.
[arXiv:1706.02216](https://arxiv.org/abs/1706.02216)

```python
import numpy as np
from anytensor.hetero import HeteroGraphsTuple, hetero_sage

writes = ("author", "writes", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 2), dtype=np.float32),
        "paper": np.ones((2, 2), dtype=np.float32),
    },
    edges={writes: None},
    senders={writes: np.array([0, 1, 2])},
    receivers={writes: np.array([0, 0, 1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3])},
)

W_rel = np.eye(2, dtype=np.float32)
W_dst = np.eye(4, 2, dtype=np.float32)
out = hetero_sage(
    g,
    relation_apply={writes: lambda x: x @ W_rel},
    combine_apply={
        "paper": lambda x: x @ W_dst,
        "author": lambda x: x @ W_dst,
    },
)
assert out.nodes["paper"].shape == (2, 2)
```

## Heterogeneous Graph Attention Network (HAN) — Wang et al., WWW 2019

**HAN** = Heterogeneous Graph Attention Network. Two levels:

1. **Node-level attention** — GAT-style weights over neighbors on each
   meta-path (here each canonical etype stands for one path hop).
2. **Semantic attention** — soft weights over those path embeddings
   (`cross_reducer="stack"` then a query vector).

[arXiv:1903.07293](https://arxiv.org/abs/1903.07293)

```python
import numpy as np
from anytensor.hetero import HeteroGraphsTuple, gat_attention_logit, han

writes = ("author", "writes", "paper")
cites = ("paper", "cites", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 2), dtype=np.float32),
        "paper": np.ones((2, 2), dtype=np.float32),
    },
    edges={writes: None, cites: None},
    senders={writes: np.array([0, 1, 2]), cites: np.array([0])},
    receivers={writes: np.array([0, 0, 1]), cites: np.array([1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3]), cites: np.array([1])},
)

W_msg = np.eye(2, dtype=np.float32)
W_attn = np.ones((4, 1), dtype=np.float32)
W_sem = np.eye(2, dtype=np.float32)
q = np.ones((2,), dtype=np.float32)

out = han(
    g,
    meta_path_etypes=[writes, cites],
    node_message={
        writes: lambda x: x @ W_msg,
        cites: lambda x: x @ W_msg,
    },
    node_attention_logit={
        writes: lambda s, d, e: gat_attention_logit(s, d, lambda x: x @ W_attn),
        cites: lambda s, d, e: gat_attention_logit(s, d, lambda x: x @ W_attn),
    },
    semantic_project=lambda x: x @ W_sem,
    semantic_query=q,
)
assert out.nodes["paper"].shape == (2, 2)
```

## Heterogeneous Graph Transformer (HGT) — Hu et al., WWW 2020

**HGT** = Heterogeneous Graph Transformer: attention and messages depend on
source type, edge type, and target type (often multi-head). Pass typed
projections as callables; this function does neighbor softmax, cross-sum
across etypes, then a target-type output map. Fold full query/key/value
and edge-type matrices into those callables as needed.
[arXiv:2003.01332](https://arxiv.org/abs/2003.01332)

```python
import numpy as np
from anytensor.hetero import HeteroGraphsTuple, hgt

writes = ("author", "writes", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 2), dtype=np.float32),
        "paper": np.ones((2, 2), dtype=np.float32),
    },
    edges={writes: None},
    senders={writes: np.array([0, 1, 2])},
    receivers={writes: np.array([0, 0, 1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3])},
)

W_msg = np.eye(2, dtype=np.float32)
W_out = np.eye(2, dtype=np.float32)
out = hgt(
    g,
    message_apply={writes: lambda x: x @ W_msg},
    attention_logit={
        writes: lambda s, d, e: np.sum(s * d, axis=-1, keepdims=True),
    },
    target_apply={"paper": lambda x: x @ W_out},
    scale=2.0,
)
assert out.nodes["paper"].shape == (2, 2)
```

## Composition-based multi-relational GCN (CompGCN) — Vashishth et al., ICLR 2020

**CompGCN** composes each source feature with its edge representation
(multiply or add), applies a relation-specific linear map, aggregates, and
adds a self term. Edge features are required on every used etype.
[arXiv:1911.03082](https://arxiv.org/abs/1911.03082)

```python
import numpy as np
from anytensor.hetero import HeteroGraphsTuple, comp_gcn

writes = ("author", "writes", "paper")
g = HeteroGraphsTuple(
    nodes={
        "author": np.ones((3, 2), dtype=np.float32),
        "paper": np.ones((2, 2), dtype=np.float32),
    },
    edges={writes: np.ones((3, 1), dtype=np.float32)},
    senders={writes: np.array([0, 1, 2])},
    receivers={writes: np.array([0, 0, 1])},
    n_node={"author": np.array([3]), "paper": np.array([2])},
    n_edge={writes: np.array([3])},
)

W_rel = np.eye(2, dtype=np.float32)
Z = np.zeros((2, 2), dtype=np.float32)
out = comp_gcn(
    g,
    relation_apply={writes: lambda x: x @ W_rel},
    self_apply={
        "author": lambda x: x @ Z,
        "paper": lambda x: x @ Z,
    },
    composition="mult",
)
assert out.nodes["paper"].shape == (2, 2)
```
