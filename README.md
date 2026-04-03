# causal-edge

**Causation is the only edge that survives.**

Correlations break when regimes change. ML on noise is still noise. Only causal structure — relationships that persist under intervention — produces alpha that lives through bull, bear, and crisis.

This is not theory. Three strategies, same validation framework, same 15 tests:

```
Correlation (SMA)    →  Lo = -0.65   dead on arrival
ML (GBDT)            →  Lo = -0.27   still dead
Causal (Abel graph)  →  Lo = +0.55   alive
```

**causal-edge** is the first agent-native causal quant framework. Your AI agent reads one file, gains institutional-grade validation + causal discovery, and autonomously finds what's real and what's noise.

> **Agents start here → [`CAPABILITY.md`](CAPABILITY.md)**

**For agents:** Read [`CAPABILITY.md`](CAPABILITY.md) — install, validate, diagnose, fix, internalize. One file, full autonomy.

**For quants:** 15-test anti-gaming metric triangle. Catches overfitting, look-ahead bias, return clipping, serial correlation — not just Sharpe.

**For agent developers:** Every subsystem has `AGENTS.md` decision trees. Structural tests enforce architecture mechanically. See [docs/harness-guide.md](docs/harness-guide.md).

## 5 Minutes: Install → Validate → See the Difference

```bash
pip install git+https://github.com/cauchyturing/causal-edge.git
causal-edge init my-portfolio
cd my-portfolio
causal-edge run && causal-edge validate
```

Already have a backtest CSV? One command:

```bash
causal-edge validate --csv my_backtest.csv
```

Three demo strategies show why causal structure matters:

```
SMA Crossover  →  11/21 FAIL  Lo=-0.65  Omega=0.86   (random noise)
Momentum ML    →  11/20 FAIL  Lo=-0.27  Omega=0.93   (ML, still noise)
Causal Voting  →  13/20 FAIL  Lo=+0.55  Omega=1.25   (causal structure = real edge)
```

Same validation framework, same tests. Only the causal strategy produces positive Lo-adjusted Sharpe and Omega > 1. That's the difference between correlation and causation.

## Why Causal?

Correlation is a property of *data*. Causation is a property of the *data generating process*.

When regimes change (bull→bear, policy shift, crisis):
- Correlations break → correlation-based signals die
- Causal links persist → causal signals survive

This is Pearl's definition: a causal relationship remains invariant under intervention. The causal demo uses real causal structure from [Abel's graph API](https://abel.ai) — 5 equity parents and 3 children of TONUSD, each with a validated causal lag. The structure is bundled in `examples/causal_demo/causal_graph.json`. For live causal discovery on any asset: `causal-edge discover <TICKER>` (requires Abel API key).

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

## Demo Strategies

| Strategy | What it is | Score | What it proves |
|----------|-----------|-------|----------------|
| `sma_crossover` | Simple moving average | 11/21 | Random signals fail completely |
| `momentum_ml` | Walk-forward GBDT | 11/20 | ML on noise is still noise |
| `causal_demo` | Abel causal graph voting | 13/20 | Causal structure produces real edge |

Each teaches a different layer: ABC interface → ML + shift(1) → causal discovery + voting.

## Commands

```bash
causal-edge init <name>              # scaffold project with 3 demo strategies
causal-edge run [--strategy ID]      # run strategies, write trade logs
causal-edge dashboard                # generate dark-theme dashboard HTML
causal-edge validate [--verbose]     # Abel Proof 15-test validation
causal-edge validate --csv file.csv  # validate any backtest CSV directly
causal-edge validate --export r.txt  # export report for sharing
causal-edge discover <TICKER>        # find causal parents (Abel API key)
causal-edge status                   # strategy summary
```

## Agent-Native Architecture

```
CAPABILITY.md              → agent reads this, gains full validation capability
AGENTS.md (root)           → "use as tool" or "develop on this repo"
strategies/AGENTS.md       → step-by-step strategy creation
validation/AGENTS.md       → failure→fix mapping with code snippets
tests/test_structure.py    → 15 tests enforce architecture mechanically
```

An agent that reads `CAPABILITY.md` can autonomously:
- Validate any backtest CSV (Python API or CLI)
- Diagnose every failure code with copy-paste fixes
- Run the autonomous fix loop (validate → fix → revalidate → report)
- Internalize the capability permanently (4 levels of persistence)

## Project Structure

```
CAPABILITY.md          → agent capability spec (the front door)
AGENTS.md              → agent routing (use vs develop)
strategies.yaml        → single source of truth for strategies
causal_edge/
  engine/              → StrategyEngine ABC + execution
  dashboard/           → Jinja2 + Plotly → static HTML
  validation/          → Abel Proof metric triangle + 15-test gate
  plugins/             → optional (Abel causal discovery)
examples/
  sma_crossover/       → simple demo (30 lines)
  momentum_ml/         → ML demo (80 lines)
  causal_demo/         → causal demo (100 lines + graph JSON)
```

## Documentation

- [`CAPABILITY.md`](CAPABILITY.md) — agent capability acquisition (start here)
- [Adding a Strategy](docs/add-strategy.md) — three paths: CSV / engine / causal
- [Why Causal?](docs/why-causal.md) — the mathematical argument
- [Agent Developer Guide](docs/harness-guide.md) — how agents operate this framework
- [Contributing](CONTRIBUTING.md) — how to contribute

## License

MIT
