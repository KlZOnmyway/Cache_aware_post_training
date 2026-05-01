"""Loss-side patches.

Importing ``slime_adapter.loss.penalty_loss`` monkey-patches slime's
``policy_loss_function`` to add three new terms on top of the OPD KL:

  - ``λ_b · uniform per-switch cost``
  - ``λ_h · token-level hinge² barrier``
  - ``λ_c · chunk-wise routing consistency``

Side-effecting import: do not import this module unless you want the patch
to apply. Typical usage::

    # at the top of your train.py (after slime is importable)
    import slime_adapter.loss.penalty_loss   # noqa: F401  (applies the patch)
"""

from . import penalty_loss as _penalty_loss  # noqa: F401  (patch on import)
from .chunk_consistency import chunk_routing_consistency_loss

__all__ = ["chunk_routing_consistency_loss"]
