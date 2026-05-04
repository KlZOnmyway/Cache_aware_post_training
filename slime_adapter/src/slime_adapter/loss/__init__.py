"""Loss-side patches.

Call ``slime_adapter.loss.penalty_loss.apply_patch()`` to monkey-patch
slime's ``policy_loss_function`` and add aux terms on top:

  - ``λ_s · L_switch_pg``   — joint-actor PG for SwitchHead
  - ``λ_h · L_barrier``     — token-level hinge² barrier
  - ``λ_c · L_chunk``       — chunk-wise routing consistency

The patch is NOT auto-applied on import; it requires an explicit
``apply_patch()`` call (done by ``spec.get_spec_with_controller``).
"""

from . import penalty_loss as _penalty_loss  # noqa: F401
from .chunk_consistency import (
    chunk_routing_consistency_loss,
    compute_chunk_consistency,
)

__all__ = ["chunk_routing_consistency_loss", "compute_chunk_consistency"]
