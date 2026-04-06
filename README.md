# causal-edge

**Causation is the only edge that survives. Agents are the only way to find it at scale.**

```
Correlation (SMA)    →  Lo = -0.65   dead on arrival
ML (GBDT)            →  Lo = -0.27   still dead
Causal (Abel graph)  →  Lo = +0.55   alive
```

Same framework. Same 15 tests. Only causal structure produces alpha that lives through regime changes. But discovering causal structure, building strategies from it, validating them, and iterating — that's not a human-speed task.

**Point your agent at this repo.** It reads [`CAPABILITY.md`](CAPABILITY.md), gains institutional-grade causal validation, and autonomously runs the discovery→build→validate→fix loop. No docs to study. No setup to debug. One file, full capability.

That's the edge: **causal signal quality × agent-native capability acquisition.**

## Quick Start

```bash
pip install git+https://github.com/cauchyturing/causal-edge.git
causal-edge init my-portfolio && cd my-portfolio
causal-edge run && causal-edge validate
```

The `run` command uses a generator pipeline — each strategy goes through
a 7-step lifecycle with typed events. Compose it with any consumer:

```python
from causal_edge.harness.pipeline import run_pipeline
for event in run_pipeline(config):
    websocket.send(event)  # real-time dashboard
```

Or validate any backtest instantly: `causal-edge validate --csv my_backtest.csv`

> **Agents → [`CAPABILITY.md`](CAPABILITY.md)** | **Quants → keep reading** | **Agent developers → [harness guide](docs/harness-guide.md)**

## Why Causal?

Correlation is a property of *data*. Causation is a property of the *data generating process*.

Regimes change. Correlations break. Causal links persist. This is Pearl's definition: a causal relationship remains invariant under intervention. Regime change *is* intervention.

The causal demo bundles a real graph from [Abel](https://abel.ai) — 5 equity parents and 3 children of TONUSD with validated causal lags. For live discovery on any asset: `causal-edge discover <TICKER>`.

## The Metric Triangle

Three leverage-invariant, orthogonal dimensions. No known transformation improves all three except genuine signal improvement:

```
        Lo-adjusted Sharpe (ratio — optimized)
             /           \
       IC (rank —          Omega (shape —
        guardrail)          guardrail)
```

Clipping → Omega drops. Serial correlation → Lo catches it. Concentration → IC drops. Verified across 38 controlled experiments.

## Why Agent-Native?

| | Human framework | causal-edge |
|---|---|---|
| Learn | Read 50-page docs | Agent reads 1 file |
| Validate | Configure, run, interpret | `validate_strategy(csv)` → structured result |
| Fix | Google the error | Failure→fix table with code |
| Iterate | Manual, hours | Autonomous loop: validate→fix→revalidate |
| Remember | Bookmark, forget | Self-internalization (4 levels) |

## Demo Strategies

| Strategy | Score | Lo | Omega | What it proves |
|----------|-------|----|-------|----------------|
| `sma_crossover` | 11/21 | -0.65 | 0.86 | Correlation = noise |
| `momentum_ml` | 11/20 | -0.27 | 0.93 | ML on noise = noise |
| `causal_demo` | 13/20 | +0.55 | 1.25 | Causal = real edge |

## Commands

```bash
causal-edge init <name>              # 3 demo strategies (SMA, ML, Causal)
causal-edge run [--strategy ID]      # run strategies, write trade logs
causal-edge dashboard                # dark-theme Plotly dashboard
causal-edge validate [--verbose]     # 15-test validation
causal-edge validate --csv file.csv  # validate any CSV directly
causal-edge discover <TICKER>        # causal parents (Abel API)
```

## Architecture

```
CAPABILITY.md       → agent capability acquisition (the front door)
AGENTS.md           → routing: use as tool / develop on repo
causal_edge/
  validation/       → metric triangle + 15-test gate
  engine/           → StrategyEngine ABC + execution
  dashboard/        → Jinja2 + Plotly → static HTML
  plugins/          → Abel causal discovery (optional)
examples/
  sma_crossover/    → correlation baseline (30 lines)
  momentum_ml/      → ML baseline (80 lines)
  causal_demo/      → causal strategy + graph JSON (100 lines)
tests/              → 15 structural tests enforce architecture
```

## Docs

- [`CAPABILITY.md`](CAPABILITY.md) — agent capability spec
- [Why Causal?](docs/why-causal.md) — Pearl, DGP, intervention invariance
- [Add a Strategy](docs/add-strategy.md) — CSV / engine / causal
- [Harness Guide](docs/harness-guide.md) — agent developer guide
- [Contributing](CONTRIBUTING.md)

## License

MIT
