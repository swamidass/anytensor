"""Compile-path checks: hetero message + attention must not unroll edges.

Schema-sized Python loops over etypes are expected (unrolled at compile time).
Edge-level work must stay in ``take`` / ``segment_*`` / ``segment_attention``.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
from helpers import BACKENDS

import anytensor as at
from anytensor.hetero import (
    HeteroGraphsTuple,
    RelationSpec,
    han,
    multi_update_all,
    relation_mailbox,
)
from anytensor.hetero import message as hetero_message
from anytensor.hetero import models as hetero_models
from anytensor.segment import segment_attention


def _author_paper_np():
    writes = ("author", "writes", "paper")
    cites = ("paper", "cites", "paper")
    g = HeteroGraphsTuple(
        nodes={
            "author": np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32
            ),
            "paper": np.asarray([[0.5, 0.5], [2.0, 0.0]], dtype=np.float32),
        },
        edges={
            writes: np.ones((3, 1), dtype=np.float32),
            cites: np.ones((1, 1), dtype=np.float32),
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


def _to_backend(g: HeteroGraphsTuple, xp_asarray):
    return HeteroGraphsTuple(
        nodes={k: xp_asarray(v) for k, v in g.nodes.items()},
        edges={k: xp_asarray(v) for k, v in g.edges.items()},
        senders={k: xp_asarray(v) for k, v in g.senders.items()},
        receivers={k: xp_asarray(v) for k, v in g.receivers.items()},
        n_node={k: xp_asarray(v) for k, v in g.n_node.items()},
        n_edge={k: xp_asarray(v) for k, v in g.n_edge.items()},
    )


def test_default_attention_uses_segment_attention_helper():
    """Neighborhood mailbox uses segment_attention; no edge-index Python loops."""
    mail_src = inspect.getsource(hetero_message.relation_mailbox)
    assert "segment_attention" in mail_src
    assert "for i in range" not in mail_src
    assert "for _ in range" not in mail_src

    # HAN semantic stays dense on (n, R) — do not flatten/repeat into segments.
    han_src = inspect.getsource(hetero_models.han)
    assert "segment_attention(" not in han_src
    assert "_dense_softmax_axis1" in han_src
    assert "repeat(" not in han_src


def _legacy_segment_attention(messages, logits, segment_ids, num_segments):
    """Frozen pre-refactor formula used as the equivalence oracle."""
    weights = at.segment_softmax(logits, segment_ids, num_segments)
    w = np.asarray(weights)
    msgs = np.asarray(messages)
    while w.ndim < msgs.ndim:
        w = np.expand_dims(w, axis=-1)
    return at.segment_sum(msgs * w, segment_ids, num_segments)


def test_segment_attention_matches_manual_softmax_sum():
    messages = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]], dtype=np.float32
    )
    logits = np.asarray([1.0, -0.5, 0.5, 2.0], dtype=np.float32)
    dst = np.asarray([0, 0, 1, 2], dtype=np.int32)
    got = segment_attention(messages, logits, dst, 3)
    expected = _legacy_segment_attention(messages, logits, dst, 3)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    # Also (E, 1) logits must match the (E,) path after broadcast.
    got_col = segment_attention(messages, logits[:, None], dst, 3)
    np.testing.assert_allclose(got_col, expected, rtol=1e-5, atol=1e-6)


def test_relation_mailbox_attention_matches_segment_attention():
    g, writes, _ = _author_paper_np()

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        return src[:, :1]

    mail = relation_mailbox(g, writes, message_fn=msg, attention_logit_fn=logit)
    src = g.nodes["author"][g.senders[writes]]
    expected = segment_attention(src, src[:, :1], g.receivers[writes], 2)
    np.testing.assert_allclose(mail, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_jit_relation_mailbox_attention():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    g_np, writes, _ = _author_paper_np()
    g = _to_backend(g_np, jnp.asarray)

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        return src[:, :1]

    def step(graph):
        return relation_mailbox(
            graph, writes, message_fn=msg, attention_logit_fn=logit
        )

    eager = step(g)
    compiled = jax.jit(step)(g)
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(compiled), rtol=1e-5, atol=1e-6
    )


@pytest.mark.skipif("jax" not in BACKENDS, reason="jax not installed")
def test_jax_jit_multi_update_all_and_han():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    g_np, writes, cites = _author_paper_np()
    g = _to_backend(g_np, jnp.asarray)
    eye = jnp.eye(2, dtype=jnp.float32)

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        return src[:, :1]

    def update(graph):
        return multi_update_all(
            graph,
            {
                writes: RelationSpec(
                    message_fn=msg, reduce="sum", attention_logit_fn=logit
                ),
                cites: msg,
            },
            cross_reducer="sum",
        )

    eager_u = update(g)
    jit_u = jax.jit(update)(g)
    np.testing.assert_allclose(
        np.asarray(eager_u.nodes["paper"]),
        np.asarray(jit_u.nodes["paper"]),
        rtol=1e-5,
        atol=1e-6,
    )

    def han_step(graph):
        return han(
            graph,
            [writes, cites],
            {writes: lambda x: x @ eye, cites: lambda x: x @ eye},
            {
                writes: lambda s, d, e: jnp.zeros((s.shape[0], 1), dtype=s.dtype),
                cites: lambda s, d, e: jnp.zeros((s.shape[0], 1), dtype=s.dtype),
            },
            lambda x: x @ eye,
            jnp.asarray([1.0, 0.0], dtype=jnp.float32),
            node_activation=lambda x: x,
            semantic_activation=lambda x: x,
        )

    eager_h = han_step(g)
    jit_h = jax.jit(han_step)(g)
    np.testing.assert_allclose(
        np.asarray(eager_h.nodes["paper"]),
        np.asarray(jit_h.nodes["paper"]),
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.skipif("tensorflow" not in BACKENDS, reason="tensorflow not installed")
def test_tf_function_relation_mailbox_attention():
    tf = pytest.importorskip("tensorflow")
    g_np, writes, _ = _author_paper_np()
    g = _to_backend(g_np, tf.constant)

    def msg(src, dst, edges):
        del dst, edges
        return src

    def logit(src, dst, edges):
        del dst, edges
        return src[:, :1]

    @tf.function(autograph=False)
    def step(nodes_author, nodes_paper, senders, receivers, edges, n_author, n_paper, n_edge):
        graph = HeteroGraphsTuple(
            nodes={"author": nodes_author, "paper": nodes_paper},
            edges={writes: edges},
            senders={writes: senders},
            receivers={writes: receivers},
            n_node={"author": n_author, "paper": n_paper},
            n_edge={writes: n_edge},
        )
        return relation_mailbox(
            graph, writes, message_fn=msg, attention_logit_fn=logit
        )

    args = (
        g.nodes["author"],
        g.nodes["paper"],
        g.senders[writes],
        g.receivers[writes],
        g.edges[writes],
        g.n_node["author"],
        g.n_node["paper"],
        g.n_edge[writes],
    )
    eager = relation_mailbox(g, writes, message_fn=msg, attention_logit_fn=logit)
    compiled = step(*args)
    np.testing.assert_allclose(
        np.asarray(eager), np.asarray(compiled), rtol=1e-5, atol=1e-6
    )
