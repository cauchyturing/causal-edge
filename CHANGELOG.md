# Changelog

## [0.1.0] - 2026-04-02

### Added
- **Framework core**: StrategyEngine ABC, config loader, CLI (init/run/dashboard/validate/discover/status)
- **Abel Proof validation**: 15-test gate with anti-gaming metric triangle (Lo-adjusted Sharpe, IC, Omega)
- **Dashboard**: Dark-theme static HTML with Plotly equity curves and position charts
- **3 demo strategies**: SMA crossover (simple), Momentum ML (walk-forward GBDT), Causal Voting (Abel graph)
- **Causal demo**: Bundled TON causal graph (5 parents + 3 children from Abel), vote² sizing, conviction threshold
- **Agent-native architecture**: CAPABILITY.md for capability acquisition, AGENTS.md decision trees, 15 structural tests
- **Autonomous workflow**: validate → diagnose → fix loop with copy-paste code snippets
- **Self-internalization**: 4 levels (skill, memory, CLAUDE.md, inline knowledge)
- **Project scaffolding**: `causal-edge init` creates harness-contagious project with 3 demos
- **Quick validation**: `causal-edge validate --csv` for instant backtest validation, `--export` for sharing
