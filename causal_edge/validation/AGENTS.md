# Validation Subsystem — Abel Proof Gate

Strategies must pass before production admission. Three leverage-invariant dimensions:
- **Ratio** (Lo-adjusted Sharpe) — optimized
- **Rank** (IC) — guardrail, catches concentration
- **Shape** (Omega) — guardrail, catches clipping

## I want to...

### Validate a strategy
    causal-edge validate --strategy <ID> --verbose

### Understand why it failed

| Code | Test | Common cause | Fix |
|------|------|-------------|-----|
| T6 | DSR < 90% | Too many trials (high K) | Reduce parameter search space |
| T7 | PBO > 10% | Overfitting | Simplify model, fewer features |
| T12 | OOS/IS < 0.50 | In-sample overfitting | Shorter train window, regularize |
| T13 | NegRoll > 15% | Regime vulnerability | Add trend filter (SMA) |
| T14 | LossYrs > 2 | Strategy decays | Check if signal source still valid |
| T15-Lo | Lo-adj < 1.0 | Serial correlation | Add persistence penalty |
| T15-Omega | Omega < 1.0 | Return clipping | Use unclipped PnL |
| T15-MaxDD | MaxDD < -20% | Excessive leverage | Reduce position sizing |

### Understand the metric triangle
Read docstring at top of `causal_edge/validation/metrics.py`. No known transformation
improves all three simultaneously except genuine signal improvement.

## Key Files
- `metrics.py` — `compute_all_metrics()`, `validate()`, `decide_keep_discard()`
- `gate.py` — `validate_strategy()` (CSV in -> PASS/FAIL out)
- `profiles/` — YAML threshold configs (crypto_daily, equity_daily, hft)
