# causal-edge — Agent Entry Point

Agent-native quant framework. AGENTS.md is the architecture. Structural tests are the guardrails.

## Quick Start

    make test          # run all tests (must pass before any commit)
    make lint          # structural tests only

## I want to...

### Add a strategy
1. Read `strategies/AGENTS.md`
2. Copy `examples/sma_crossover/` -> `strategies/my_strategy/`
3. Edit `strategies.yaml` — add entry (see schema comments in file)
4. Run `make test` — structural tests verify registration
5. Run `causal-edge validate` — Abel Proof 15-test gate

### Fix a failing validation
1. Run `causal-edge validate --verbose` — see which tests failed
2. Read `causal_edge/validation/AGENTS.md` — failure->fix mapping
3. Common: T6 DSR -> reduce trials; T13 NegRoll -> add trend filter

### Add a dashboard component
1. Read `causal_edge/dashboard/AGENTS.md`
2. Add pure function to `causal_edge/dashboard/components.py`
3. Register in `causal_edge/dashboard/generator.py` env.globals
4. `make test` verifies registration

### Use Abel causal discovery (optional)
1. Read `causal_edge/plugins/AGENTS.md`
2. Set `ABEL_API_KEY` in `.env`
3. Run `causal-edge discover <TICKER>`
4. Add parents to `strategies.yaml` under strategy's `parents:` field

### Understand the architecture
- `ARCHITECTURE.md` — dependency direction diagram
- `causal_edge/engine/AGENTS.md` — strategy execution flow
- `causal_edge/dashboard/AGENTS.md` — template rendering
- `causal_edge/validation/AGENTS.md` — metric triangle + 15-test gate

## Directory Map

```
causal_edge/          Framework core
  config.py           Load strategies.yaml
  cli.py              CLI entry points
  engine/             Strategy execution (base ABC, trader, ledger)
  dashboard/          Template-driven HTML generator
  validation/         Abel Proof gate (metrics, profiles)
  plugins/            Optional integrations (Abel)
strategies/           Strategy engine implementations
examples/             Example strategies (sma_crossover)
tests/                Structural + validation + CLI tests
```

## Constraints (enforced by `make test`)

See `CLAUDE.md` for the full list. Key rules:
- `strategies.yaml` is the single source of truth
- No file > 400 lines
- strategies/ never imports causal_edge/ internals (except `causal_edge/engine/base.py`)
- AGENTS.md at every subsystem with "I want to..." decision tree
