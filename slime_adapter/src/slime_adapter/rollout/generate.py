"""Optional override of slime's default generate_rollout.

For v0.1 we use slime's stock ``slime.rollout.sglang_rollout.generate_rollout``
unchanged — all controller-side state is captured by the SGLang patches and
attached to ``Sample.metadata``. Hence this module is just a thin shim that
re-exports slime's default and offers a place to insert pre/post hooks if
needed later (e.g., to enrich Sample with extra controller stats for
debugging).

Wire up via:

    --rollout-function-path slime_adapter.rollout.generate:generate_rollout

(Or the simpler default:
    --rollout-function-path slime.rollout.sglang_rollout.generate_rollout)
"""

from __future__ import annotations


async def generate_rollout(args, rollout_id, data_source, evaluation: bool = False):
    """Pass-through to slime.rollout.sglang_rollout.generate_rollout."""
    from slime.rollout.sglang_rollout import generate_rollout as _slime_default  # type: ignore

    out = await _slime_default(args, rollout_id, data_source, evaluation=evaluation)

    # Hook point: post-process samples to surface controller metadata into top-level fields,
    # if the SGLang patch wrote it under ``Sample.metadata['controller_records']``.
    for sample_group in getattr(out, "samples", []):
        for sample in sample_group if hasattr(sample_group, "__iter__") else [sample_group]:
            md = getattr(sample, "metadata", {}) or {}
            if "controller_records" in md:
                sample.controller_records = md["controller_records"]

    return out
