"""Verify our adapter integrates with the *real* slime ``RoutingReplay``.

These tests run against the slime package cloned by
``scripts/install_externals.sh slime``. They don't need Megatron, since the
``RoutingReplay`` machinery is pure-python.
"""

from __future__ import annotations

import os
import pytest

slime = pytest.importorskip("slime")
import slime.utils.routing_replay as rr  # type: ignore  # noqa: E402
import torch  # noqa: E402


def test_routing_replay_api_shape():
    """The RoutingReplay API our adapter coded against actually exists."""
    assert hasattr(rr, "RoutingReplay")
    assert hasattr(rr, "set_routing_replay")
    assert hasattr(rr, "register_routing_replay")
    assert hasattr(rr, "get_routing_replay_compute_topk")

    inst = rr.RoutingReplay()
    assert hasattr(inst, "top_indices_list")
    assert hasattr(inst, "forward_index")
    assert hasattr(inst, "record")
    assert hasattr(inst, "pop_forward")


def test_routing_replay_record_then_replay():
    """End-to-end: record, then replay via slime's mechanism (GPU)."""
    if not torch.cuda.is_available():
        pytest.skip("RoutingReplay's pop_forward calls .to(cuda); needs GPU")

    device = torch.device("cuda")
    saved = {k: os.environ.get(k) for k in ("ENABLE_ROUTING_REPLAY", "ROUTING_REPLAY_STAGE")}

    try:
        os.environ["ENABLE_ROUTING_REPLAY"] = "1"

        replay = rr.RoutingReplay()
        rr.set_routing_replay(replay)

        def fake_compute_topk(scores, topk, num_groups=None, group_topk=None):
            top = scores.topk(topk, dim=-1).indices
            return scores.gather(-1, top), top

        wrapped = rr.get_routing_replay_compute_topk(fake_compute_topk)

        # ── record ──
        os.environ["ROUTING_REPLAY_STAGE"] = "record"
        scores1 = torch.randn(8, 16, device=device)
        probs1, idx1 = wrapped(scores1, topk=2)
        assert len(replay.top_indices_list) == 1

        # ── replay_forward ──
        replay.forward_index = 0
        os.environ["ROUTING_REPLAY_STAGE"] = "replay_forward"
        scores2 = torch.randn(8, 16, device=device)  # "router drifted"
        probs2, idx2 = wrapped(scores2, topk=2)

        # The replayed indices must equal the recorded ones, regardless of new scores.
        assert torch.equal(idx2, idx1 := idx1.to(idx2.device))
        # And the returned probs are gather(scores2, idx1)
        assert torch.allclose(probs2, scores2.gather(-1, idx1))
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_qwen3_adapter_imports_with_real_slime():
    from slime_adapter.modeling.qwen3_moe.adapter import Qwen3MoEAdapter
    a = Qwen3MoEAdapter()
    assert a.name == "qwen3_moe"
