# causal-edge

The first agent-native quant framework.

**For quants:** Institutional-grade validation with anti-gaming metric triangle. 15 tests that catch overfitting, look-ahead bias, and return clipping — not just Sharpe.

**For agent developers:** Every subsystem has `AGENTS.md` decision trees. Your AI agent reads them and knows exactly what to do — add strategies, fix validation failures, build dashboards. Structural tests enforce architecture mechanically.

## 5 Minutes to First Dashboard

```bash
pip install git+https://github.com/cauchyturing/causal-edge.git
causal-edge init my-portfolio
cd my-portfolio
causal-edge run
causal-edge dashboard
open dashboard.html
```

Already have a backtest CSV? Skip setup entirely:

```bash
causal-edge validate --csv my_backtest.csv
```

You'll see your first validation report too:

```bash
causal-edge validate
# SMA Crossover: FAIL 11/21 — that's expected.
# Random signals don't pass institutional-grade validation.
```

## Why "Causal"?

Correlation is a property of *data*. Causation is a property of the *data generating process*.

When regimes change (bull→bear, policy shift, crisis):
- Correlations break → correlation-based signals die
- Causal links persist → causal signals survive

This is Pearl's definition: a causal relationship remains invariant under intervention. Regime change *is* intervention on the market. Only causal signals are mathematically guaranteed to survive it.

causal-edge validates strategies against this standard. The optional [Abel plugin](docs/why-causal.md) connects to Abel's causal graph API for automated causal discovery.

## The Metric Triangle

Three leverage-invariant, mathematically orthogonal dimensions:

```
        Lo-adjusted Sharpe (ratio — optimized)
             /           \
       IC (rank —          Omega (shape —
        guardrail)          guardrail)
```

No known transformation improves all three simultaneously except genuine signal improvement:
- **Clipping** inflates Sharpe but tanks Omega (gains clipped harder)
- **Serial correlation** inflates Sharpe but Lo correction catches it
- **Concentration** boosts ratios but IC drops (less position diversity)

Verified across 38 controlled experiments.

## Agent-Native Architecture

```
AGENTS.md (root)           → "I want to add a strategy" → strategies/AGENTS.md
strategies/AGENTS.md       → step-by-step with file paths and make targets
causal_edge/validation/    → failure→fix mapping table
tests/test_structure.py    → 15 tests enforce architecture mechanically
```

An agent that reads `AGENTS.md` can autonomously:
- Add a strategy (copy example → edit → register in YAML → validate)
- Fix a failing validation (read failure code → look up fix → apply)
- Add dashboard components (pure function → register → test verifies)

See [docs/harness-guide.md](docs/harness-guide.md) for the full agent developer guide.

## Demo Strategies

| Strategy | Lines | What it teaches |
|----------|-------|----------------|
| `sma_crossover` | ~40 | StrategyEngine ABC, minimal implementation |
| `momentum_ml` | ~80 | Walk-forward GBDT, look-ahead prevention with shift(1) |

## Commands

```bash
causal-edge init <name>              # scaffold a new project
causal-edge run [--strategy ID]      # run strategies, write trade logs
causal-edge dashboard                # generate dashboard.html
causal-edge validate [--verbose]     # Abel Proof 15-test validation
causal-edge status                   # strategy summary
```

## Project Structure

```
strategies.yaml        → single source of truth (the only file you edit)
causal_edge/engine/    → StrategyEngine ABC + execution
causal_edge/dashboard/ → Jinja2 + Plotly → static HTML
causal_edge/validation/→ Abel Proof metric triangle + 15-test gate
causal_edge/plugins/   → optional (Abel causal discovery)
strategies/            → your strategy engines
examples/              → demo strategies (sma_crossover, momentum_ml)
```

## Documentation

- [Adding a Strategy](docs/add-strategy.md) — step-by-step guide
- [Why Causal?](docs/why-causal.md) — the mathematical argument
- [Agent Developer Guide](docs/harness-guide.md) — how agents operate this framework
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

MIT
