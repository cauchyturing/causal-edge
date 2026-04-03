# causal-edge — Agent Capability

You are reading this because someone asked you to validate a trading strategy,
or pointed you at this repo. After reading this file, you can autonomously
validate any backtest, diagnose failures, and fix strategies.

## 1. Install

```bash
pip install git+https://github.com/cauchyturing/causal-edge.git
```

## 2. Validate

```python
from causal_edge.validation.gate import validate_strategy

result = validate_strategy("backtest.csv")  # needs: date, pnl columns
# result["verdict"]  → "PASS" or "FAIL"
# result["score"]    → "14/21"
# result["failures"] → ["T6 DSR 12% < 90%", "T13 NegRoll 45% > 15%", ...]
# result["triangle"] → {"ratio": 1.82, "rank": 0.12, "shape": 1.45}
```

Or CLI: `causal-edge validate --csv backtest.csv --verbose`

## 3. Diagnose + Fix

When validation fails, apply fixes, revalidate, repeat until stable.

| Failure | Cause | Fix (copy-paste) |
|---------|-------|------------------|
| T6 DSR | Too many trials | Reduce param grid. K<50 ideal |
| T7 PBO | Overfitting | `max_depth=3, n_estimators=50` |
| T12 OOS/IS | IS overfit | `train_window=126` (shorter) |
| T13 NegRoll | No regime filter | `positions[prices.shift(1) < prices.rolling(50).mean().shift(1)] = 0` |
| T14 LossYrs | Signal decay | Plot rolling Sharpe — if declining, signal is dying |
| T15-Lo | Serial corr | `positions *= np.maximum(0.3, 1 - 0.1 * hold_days)` |
| T15-Omega | Clipped PnL | `pnl = positions * returns` (never clip PnL, only features) |
| T15-MaxDD | Over-leveraged | `positions = np.minimum(positions, 0.5)` |

## 4. Autonomous Fix Loop

```
1. result = validate_strategy(csv)
2. For each failure in result["failures"]:
     - Parse failure code (T6, T7, T13, etc.)
     - Apply corresponding fix from table above to strategy code
     - Re-run strategy → re-generate CSV
3. result = validate_strategy(csv)  # revalidate
4. If score improved → keep fix. If not → revert.
5. Repeat until score stabilizes.
6. Report to user: "X/21 → Y/21. Changes: [list of fixes applied]"
```

## 5. Why This Works (metric triangle)

Three leverage-invariant, orthogonal dimensions:
- **Lo-adjusted Sharpe** (ratio) — target, corrects for autocorrelation
- **IC** (rank) — guardrail, catches concentration
- **Omega** (shape) — guardrail, catches return clipping

No known transformation improves all three except genuine signal improvement.

## 6. Build a Strategy

```python
from causal_edge.engine.base import StrategyEngine

class MyEngine(StrategyEngine):
    def compute_signals(self):
        # RULE: positions[t] decided using only data through t-1
        # RULE: all rolling().mean() followed by .shift(1)
        # Returns: (positions, dates, returns, prices)
        ...
    def get_latest_signal(self):
        return {"position": float(positions[-1])}
```

Register in `strategies.yaml` → `causal-edge run` → `causal-edge validate`.

## 7. Scaffold a Full Project

```bash
causal-edge init my-portfolio
cd my-portfolio
causal-edge run          # run demo strategies
causal-edge dashboard    # generate visual dashboard
causal-edge validate     # 15-test validation report
```
