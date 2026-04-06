"""Strategy Card generator — the quant Model Card.

Three tiers:
  Tier 1 (CARD):   Public evaluation — metrics + validation + limitations
  Tier 2 (SPEC):   Private reproduction — code + params + data manifest
  Tier 3 (METHOD): Public methodology — how to discover new strategies

This module generates Tier 1 automatically from validate_strategy() output.
Tier 2 and Tier 3 are documentation conventions (see strategy-card-standard.md).
"""

from causal_edge.card.generate import generate_card, render_card

__all__ = ["generate_card", "render_card"]
