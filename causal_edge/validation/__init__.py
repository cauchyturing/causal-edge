"""Abel Proof validation — metric triangle gate for strategy admission.

Three leverage-invariant, orthogonal dimensions:
  Ratio (Lo-adj Sharpe or Sharpe) — mean/std quality
  Rank  (IC)                      — prediction quality
  Shape (Omega)                   — gain/loss asymmetry

No known transformation improves all three except genuine signal improvement.
"""

from causal_edge.validation.metrics import compute_all_metrics, validate
from causal_edge.validation.gate import validate_strategy

__all__ = ["compute_all_metrics", "validate", "validate_strategy"]
