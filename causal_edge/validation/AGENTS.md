# Validation — Abel Proof Gate

Three leverage-invariant dimensions: **Lo** (ratio), **IC** (rank), **Omega** (shape).

## I want to...

### Validate a strategy
    causal-edge validate --strategy <ID> --verbose

### Understand why it failed

| Code | Fix |
|------|-----|
| T6 DSR | Fewer param combos (K<50). |
| T7 PBO | Fewer features, shallower trees. |
| T12 OOS/IS | Shorter train window. |
| T13 NegRoll | Add trend filter. |
| T14 LossYrs | Check signal decay. |
| T15-Lo | Persistence penalty. |
| T15-Omega | Use raw returns for PnL. |
| T15-MaxDD | Reduce sizing. |
| R1 pos×return | Fix look-ahead in features. |

### Semantic look-ahead review

When validation shows `semantic_review: required`:
1. Read the strategy source code
2. Apply rules in `look_ahead_rules.md` (7 rules: R-SHIFT through R-EXPANDING)
3. Track data flow through assignments — code regex can't do this, you can
4. Report: `SEMANTIC CLEAN` or `SEMANTIC VIOLATION: R-xxx Lnn — description`

## Key Files
- `gate.py` — `validate_strategy()` (CSV → PASS/FAIL)
- `metrics.py` — `compute_all_metrics()`, `validate()`
- `look_ahead.py` — static T2-T5 + runtime R1/R2
- `look_ahead_rules.md` — semantic rules for agent review
- `profiles/` — threshold configs (crypto_daily, equity_daily, hft)
