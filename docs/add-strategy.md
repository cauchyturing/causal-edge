# Adding a Strategy

## Quick Path

1. Copy the example:

```bash
cp -r examples/sma_crossover/ strategies/my_strategy/
```

2. Edit `strategies/my_strategy/engine.py` — implement your signal logic

3. Add to `strategies.yaml`:

```yaml
strategies:
  - id: my_strategy
    name: "My Strategy"
    asset: ETH
    color: "#FF2D55"
    engine: strategies.my_strategy.engine
    trade_log: "data/trade_log_my_strategy.csv"
```

4. Verify:

```bash
make test                              # structural tests pass
causal-edge run --strategy my_strategy # generates trade log
causal-edge validate --strategy my_strategy  # Abel Proof gate
```

## Engine Interface

Your engine must implement `StrategyEngine` from `causal_edge/engine/base.py`:

```python
class MyEngine(StrategyEngine):
    def compute_signals(self):
        # Returns: (positions, dates, returns, prices)
        # positions: np.ndarray of daily position sizes (0=flat, 1=long)
        # dates: pd.DatetimeIndex
        # returns: np.ndarray of daily asset returns
        # prices: np.ndarray of daily closing prices
        ...

    def get_latest_signal(self):
        # Returns: dict with at least 'position' key
        ...
```

## Rules

- All features must use `shift(1)` — zero look-ahead tolerance
- `rolling().mean()` must be followed by `.shift(1)` before use in decisions
- Clip returns for training features only, use unclipped for PnL
- strategies/ must not import causal_edge/ internals (except base.py)
