"""Rollout-side glue.

Wires up the OPD-style KL reward path (``reward_kl.py``) and provides a
no-op generate wrapper (``generate.py``) for users who want to extend
slime's default rollout function with extra metadata extraction.
"""

from .reward_kl import reward_func, post_process_rewards

__all__ = ["reward_func", "post_process_rewards"]
