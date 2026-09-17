### Added

- Hetero per-relation attention on ``relation_mailbox`` /
  ``multi_update_all`` (``RelationSpec``, ``segment_softmax`` on receivers) —
  same pattern as jraph ``GraphNetwork`` attention.
- Hetero model zoo as plain functions: ``relational_graph_convolution``
  (R-GCN), ``hetero_sage``, ``han``, ``hgt``, ``comp_gcn``, with docs and
  citations. Call sites own weights (pass linears / module ``__call__``).
