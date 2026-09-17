"""Tests for hetero attention + model zoo."""

from __future__ import annotations

import numpy as np
import pytest

from anytensor.hetero import (
    HeteroGraphsTuple,
    RelationSpec,
    attention_weight_messages,
    comp_gcn,
    gat_attention_logit,
    han,
    hetero_sage,
    hgt,
    multi_update_all,
    relation_mailbox,
    relational_graph_convolution,
)
from anytensor.segment import segment_attention, segment_softmax, segment_sum


def _author_paper():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    g = HeteroGraphsTuple(
        nodes={
            "author": np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
            ),
            "paper": np.asarray(
                [[0.5, 0.5], [2.0, 0.0]], dtype=np.float32
            ),
        },
        edges={
            writes: np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32),
            cites: np.asarray([[1.0]], dtype=np.float32),
        },
        senders={
            writes: np.asarray([0, 1, 2], dtype=np.int32),
            cites: np.asarray([0], dtype=np.int32),
        },
        receivers={
            writes: np.asarray([0, 0, 1], dtype=np.int32),
            cites: np.asarray([1], dtype=np.int32),
        },
        n_node={
            "author": np.asarray([3], dtype=np.int32),
            "paper": np.asarray([2], dtype=np.int32),
        },
        n_edge={
            writes: np.asarray([3], dtype=np.int32),
            cites: np.asarray([1], dtype=np.int32),
        },
    )
    return g, writes, cites


def _lin(w):
    def apply(x):
        return x @ w

    return apply


def _legacy_segment_attention(messages, logits, segment_ids, num_segments):
    """Pre-refactor hand-roll: segment_softmax → multiply → segment_sum."""
    weights = segment_softmax(logits, segment_ids, num_segments)
    w = weights
    while getattr(w, "ndim", 0) < getattr(messages, "ndim", 0):
        w = np.expand_dims(w, axis=-1)
    return segment_sum(messages * w, segment_ids, num_segments)


def _legacy_dense_softmax_axis1(score):
    """Pre-refactor HAN semantic softmax over dense axis 1."""
    m = np.max(score, axis=1, keepdims=True)
    e = np.exp(score - m)
    return e / np.sum(e, axis=1, keepdims=True)


def test_relation_mailbox_attention_matches_manual_softmax():
    g, writes, _ = _author_paper()

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        # Non-uniform logits so softmax is nontrivial.
        return src[:, :1] * 2.0 - 0.25

    mail = relation_mailbox(
        g,
        writes,
        message_fn=msg,
        reduce="sum",
        attention_logit_fn=logit,
    )
    src = g.nodes["author"][g.senders[writes]]
    logits = src[:, :1] * 2.0 - 0.25
    expected_helper = segment_attention(src, logits, g.receivers[writes], 2)
    expected_legacy = _legacy_segment_attention(src, logits, g.receivers[writes], 2)
    np.testing.assert_allclose(mail, expected_helper, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(mail, expected_legacy, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(expected_helper, expected_legacy, rtol=1e-5, atol=1e-6)


def test_multi_update_all_relation_spec_attention():
    g, writes, cites = _author_paper()

    def msg(src, dst, edges):
        del dst, edges
        return src

    def uniform_logit(src, dst, edges):
        del src, dst, edges
        return np.zeros((3, 1), dtype=np.float32)

    out = multi_update_all(
        g,
        {
            writes: RelationSpec(
                message_fn=msg,
                reduce="sum",
                attention_logit_fn=uniform_logit,
                attention_reduce_fn=attention_weight_messages,
            ),
            cites: (msg, "sum"),
        },
        cross_reducer="sum",
    )
    np.testing.assert_allclose(
        out.nodes["paper"][0],
        0.5 * (g.nodes["author"][0] + g.nodes["author"][1]),
        rtol=1e-5,
    )


def test_relation_mailbox_attention_requires_logit():
    g, writes, _ = _author_paper()
    with pytest.raises(ValueError, match="attention_logit_fn"):
        relation_mailbox(
            g,
            writes,
            attention_reduce_fn=attention_weight_messages,
        )


def test_parse_relation_spec_bad_tuple():
    g, writes, _ = _author_paper()
    with pytest.raises(ValueError, match="etype spec tuple"):
        multi_update_all(g, {writes: (lambda s, d, e: s, "sum", None)})


def test_parse_relation_spec_variants():
    g, writes, cites = _author_paper()

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        return np.zeros((src.shape[0], 1), dtype=np.float32)

    out1 = multi_update_all(g, {writes: (msg,)}, cross_reducer="sum")
    assert out1.nodes["paper"].shape == (2, 2)

    out4 = multi_update_all(
        g,
        {
            writes: (msg, "sum", logit, attention_weight_messages),
            cites: msg,
        },
        cross_reducer="sum",
    )
    assert out4.nodes["paper"].shape == (2, 2)

    with pytest.raises(TypeError, match="etype spec must be"):
        multi_update_all(g, {writes: 123})  # type: ignore[dict-item]


def test_rgcn_shapes_and_values():
    g, writes, cites = _author_paper()
    w_rel = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }
    w_self = {
        "author": _lin(np.zeros((2, 2), dtype=np.float32)),
        "paper": _lin(np.zeros((2, 2), dtype=np.float32)),
    }
    out = relational_graph_convolution(
        g, w_rel, w_self, activation=lambda x: x
    )
    np.testing.assert_allclose(
        out.nodes["paper"][0],
        0.5 * (g.nodes["author"][0] + g.nodes["author"][1]),
    )
    assert out.nodes["paper"].shape == (2, 2)
    assert out.nodes["author"].shape == (3, 2)


def test_heterosage_concat_combine():
    g, writes, cites = _author_paper()
    rel = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }
    w_combine = np.zeros((4, 2), dtype=np.float32)
    w_combine[0, 0] = 1.0
    w_combine[1, 1] = 1.0
    out = hetero_sage(
        g,
        rel,
        {"paper": _lin(w_combine), "author": _lin(w_combine)},
        activation=lambda x: x,
    )
    np.testing.assert_allclose(out.nodes["paper"], g.nodes["paper"])


def test_compgcn_mult_and_sum():
    g, writes, cites = _author_paper()
    rel = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }
    self_z = {
        "author": _lin(np.zeros((2, 2), dtype=np.float32)),
        "paper": _lin(np.zeros((2, 2), dtype=np.float32)),
    }
    out = comp_gcn(g, rel, self_z, composition="mult", activation=lambda x: x)
    assert out.nodes["paper"].shape == (2, 2)
    out_sum = comp_gcn(g, rel, self_z, composition="sum", activation=lambda x: x)
    assert out_sum.nodes["paper"].shape == (2, 2)

    with pytest.raises(ValueError, match="composition"):
        comp_gcn(g, rel, self_z, composition="sub")

    g_no_e = g.update(edges={writes: None, cites: None})
    with pytest.raises(ValueError, match="edge features"):
        comp_gcn(
            g_no_e,
            {writes: rel[writes]},
            {"paper": self_z["paper"]},
            composition="mult",
        )


def test_han_node_and_semantic_attention():
    g, writes, cites = _author_paper()
    node_message = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }

    def zeros_logit(src, dst, edges):
        return np.zeros((src.shape[0], 1), dtype=np.float32)

    out = han(
        g,
        [writes, cites],
        node_message,
        {writes: zeros_logit, cites: zeros_logit},
        _lin(np.eye(2, dtype=np.float32)),
        np.asarray([1.0, 0.0], dtype=np.float32),
        node_activation=lambda x: x,
        semantic_activation=lambda x: x,
    )
    assert out.nodes["paper"].shape == (2, 2)
    assert np.all(np.isfinite(out.nodes["paper"]))


def test_han_matches_legacy_dense_semantic_softmax():
    """HAN semantic path stays dense and matches the frozen axis-1 softmax."""
    g, writes, cites = _author_paper()
    W = np.eye(2, dtype=np.float32)
    q = np.asarray([1.5, -0.5], dtype=np.float32)

    def logit_writes(src, dst, edges):
        del dst, edges
        return src[:, :1] * 2.0 + 0.5

    def logit_cites(src, dst, edges):
        del dst, edges
        return src[:, :1] - 1.0

    got = han(
        g,
        [writes, cites],
        {writes: _lin(W), cites: _lin(W)},
        {writes: logit_writes, cites: logit_cites},
        _lin(W),
        q,
        node_activation=lambda x: x,
        semantic_activation=lambda x: x,
    )

    # Oracle: per-etype segment attention (legacy hand-roll), stack, dense semantic.
    def legacy_mailbox(etype, logit_fn):
        src = g.nodes[etype[0]][g.senders[etype]]
        dst = g.nodes[etype[2]][g.receivers[etype]]
        messages = src @ W
        logits = logit_fn(src, dst, g.edges.get(etype))
        num_dst = g.nodes[etype[2]].shape[0]
        return _legacy_segment_attention(
            messages, logits, g.receivers[etype], num_dst
        )

    h_stack = np.stack(
        [legacy_mailbox(writes, logit_writes), legacy_mailbox(cites, logit_cites)],
        axis=1,
    )
    n, r, d = h_stack.shape
    flat = h_stack.reshape(n * r, d)
    proj = (flat @ W).reshape(n, r, -1)
    score = np.sum(proj * q, axis=-1)
    alpha = _legacy_dense_softmax_axis1(score)
    expected = np.sum(h_stack * alpha[..., None], axis=1)

    np.testing.assert_allclose(got.nodes["paper"], expected, rtol=1e-5, atol=1e-6)

    # Math-only check: dense mix ≡ segment_attention on a flattened view,
    # but the implementation must not take that copy-heavy route.
    path_ids = np.repeat(np.arange(n, dtype=np.int32), r)
    via_segment = segment_attention(
        flat, score.reshape(n * r), path_ids, n
    )
    np.testing.assert_allclose(expected, via_segment, rtol=1e-5, atol=1e-6)


def test_han_defaults_and_empty_reject():
    g, writes, cites = _author_paper()
    node_message = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }
    w_attn = np.ones((4, 1), dtype=np.float32)
    logits = {
        writes: lambda s, d, e: gat_attention_logit(s, d, _lin(w_attn)),
        cites: lambda s, d, e: gat_attention_logit(s, d, _lin(w_attn)),
    }
    out = han(
        g,
        [writes, cites],
        node_message,
        logits,
        _lin(np.eye(2, dtype=np.float32)),
        np.ones((2,), dtype=np.float32),
    )
    assert out.nodes["paper"].shape == (2, 2)

    with pytest.raises(ValueError, match="at least one"):
        han(g, [], {}, {}, _lin(np.eye(2, dtype=np.float32)), np.ones(2))


def test_hgt_typed_attention():
    g, writes, cites = _author_paper()
    msg = {
        writes: _lin(np.eye(2, dtype=np.float32)),
        cites: _lin(np.eye(2, dtype=np.float32)),
    }
    logits = {
        writes: lambda s, d, e: np.ones((s.shape[0], 1), dtype=np.float32),
        cites: lambda s, d, e: np.ones((s.shape[0], 1), dtype=np.float32),
    }
    out = hgt(g, msg, logits, {"paper": _lin(np.eye(2, dtype=np.float32))}, scale=4.0)
    assert out.nodes["paper"].shape == (2, 2)
    np.testing.assert_allclose(out.nodes["author"], g.nodes["author"])

    out2 = hgt(g, {writes: msg[writes]}, {writes: logits[writes]}, {"paper": _lin(np.eye(2, dtype=np.float32))})
    assert out2.nodes["paper"].shape == (2, 2)


def test_rgcn_default_relu():
    g, writes, cites = _author_paper()
    rel = {
        writes: _lin(0.1 * np.eye(2, dtype=np.float32)),
        cites: _lin(0.1 * np.eye(2, dtype=np.float32)),
    }
    self_a = {
        "author": _lin(0.1 * np.eye(2, dtype=np.float32)),
        "paper": _lin(0.1 * np.eye(2, dtype=np.float32)),
    }
    out = relational_graph_convolution(g, rel, self_a)
    assert out.nodes["paper"].shape == (2, 2)


def test_gat_attention_logit_helper():
    src = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    dst = np.asarray([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    w = np.ones((4, 1), dtype=np.float32)
    scores = gat_attention_logit(src, dst, _lin(w))
    assert scores.shape == (2, 1)
    assert float(scores[0, 0]) > 0
