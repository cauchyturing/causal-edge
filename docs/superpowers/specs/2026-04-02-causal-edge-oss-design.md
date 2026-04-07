# causal-edge Open Source Release — Design Spec

**Date**: 2026-04-02
**Status**: Draft
**Author**: bht + Claude

## 1. Project Positioning

> causal-edge — the first agent-native quant framework.
> AGENTS.md is the architecture. Structural tests are the guardrails. Abel Proof is the validation engine.

**Dual audience**:
- **Quant researchers/traders**: Institutional-grade 13-test validation, anti-gaming metric triangle, zero look-ahead tolerance. `causal-edge init → run → validate` in 5 minutes.
- **Agent developers**: Every subsystem has AGENTS.md decision trees. Structural tests mechanically enforce architecture. An agent reads AGENTS.md and autonomously adds strategies, fixes validation failures, builds dashboards.

**Competitive differentiation**: backtrader/zipline/lean have more features. None are agent-native. AGENTS.md as executable architecture spec is the category-defining feature.

## 2. Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Target audience | Quant + Agent developer (dual path) | Unique positioning — no existing OSS does both |
| Abel API | Optional plugin (CaaS) | Framework core has zero Abel dependency; CaaS is the future |
| Repo strategy | New repo, clean history | Zero leak risk; first commit = "initial release" |
| Demo strategies | 3 demos (simple/ML/causal) | Educational gradient without leaking real alpha |
| Approach | Harness-first (Phase 1 = AGENTS.md + tests before code) | Agent-native is the differentiator, not feature count |

## 3. Phase Structure

### Phase 1: Harness Skeleton

**Goal**: Define all interfaces, constraints, and agent routing — zero business logic.

**Deliverables**:
- CLAUDE.md with mechanically enforced constraints
- AGENTS.md decision trees at every subsystem (project root + engine + dashboard + validation + plugins + strategies)
- Structural tests (15 total: 8 existing + 7 new)
- ABC skeletons (all methods raise `NotImplementedError`)
- CLI skeleton (subcommands exist but raise `NotImplementedError`)

**Directory structure**:
```
causal-edge/
├── CLAUDE.md
├── AGENTS.md                    # Root decision tree
├── ARCHITECTURE.md
├── LICENSE                      # MIT
├── Makefile
├── pyproject.toml
├── strategies.yaml              # Empty strategies list, with YAML schema comments showing required fields
│
├── causal_edge/
│   ├── __init__.py
│   ├── cli.py                   # click: init, run, dashboard, validate, status
│   ├── config.py                # Load strategies.yaml + env var expansion
│   ├── engine/
│   │   ├── AGENTS.md
│   │   ├── base.py              # StrategyEngine ABC
│   │   ├── trader.py            # Skeleton
│   │   └── ledger.py            # Skeleton
│   ├── dashboard/
│   │   ├── AGENTS.md
│   │   ├── generator.py         # Skeleton
│   │   ├── components.py        # Skeleton
│   │   ├── _helpers.py
│   │   ├── server.py            # Skeleton
│   │   └── templates/           # Skeleton templates
│   ├── validation/
│   │   ├── AGENTS.md
│   │   ├── metrics.py           # Ported from existing (complete)
│   │   ├── gate.py              # Ported from existing (complete)
│   │   └── profiles/            # Ported from existing (complete)
│   └── plugins/
│       ├── AGENTS.md
│       └── __init__.py          # Plugin discovery + registry
│
├── strategies/
│   └── AGENTS.md
│
├── examples/
│   └── sma_crossover/engine.py  # Existing 30-line example
│
├── tests/
│   ├── test_structure.py        # 13 structural tests
│   ├── test_triangle.py         # Metric triangle invariants (ported)
│   ├── test_validation.py       # Validation gate (ported)
│   └── test_cli.py              # CLI entry point tests
│
└── docs/
    └── add-strategy.md
```

**Structural tests (13)**:

| # | Test | What it enforces |
|---|------|-----------------|
| 1 | TestYamlHasRequiredFields | YAML schema completeness |
| 2 | TestStrategyIdsUnique | No duplicate strategy IDs |
| 3 | TestEngineModuleImportable | Engine paths resolve |
| 4 | TestColorsAreValidHex | Color format |
| 5 | TestFileSizeLimit | No file > 400 lines |
| 6 | TestSubsystemAgentsMd | Every subsystem has AGENTS.md |
| 7 | TestComponentsRegistered | Dashboard components used in generator |
| 8 | TestAgentsMdHasDecisionTree | AGENTS.md contains "I want to..." section |
| 9 | TestPluginsOptional | Removing plugins/ doesn't break tests |
| 10 | TestStrategiesStandalone | strategies/ doesn't import causal_edge/ internals |
| 11 | TestCliEntryPoints | CLI subcommands (init/run/dashboard/validate) callable |
| 12 | TestNoHardcodedPaths | No absolute paths in Python files |
| 13 | TestNoSecrets | No API keys, tokens, or .env values in source |
| 14 | TestAgentsMdReferencesExist | All file paths in AGENTS.md exist |
| 15 | TestAgentsMdSizeBudget | Root ≤80 lines, subsystem ≤60 lines |

**AGENTS.md format — decision tree pattern**:
```markdown
## I want to...

### Add a strategy
1. Read `strategies/AGENTS.md`
2. Copy `examples/sma_crossover/` → `strategies/my_strategy/`
3. Edit `strategies.yaml` — add entry
4. Run `make test` — structural tests verify registration
5. Run `causal-edge validate` — Abel Proof gate

### Fix a failing validation
1. Run `causal-edge validate --verbose`
2. Read `causal_edge/validation/AGENTS.md` — failure → fix mapping
3. Common: T6 DSR low → too many trials; T13 NegRoll → add trend filter

### Add a dashboard component
1. Read `causal_edge/dashboard/AGENTS.md`
2. Add pure function to `components.py`
3. Register in `generator.py` env.globals
4. `make test` — TestComponentsRegistered verifies

### Use Abel causal discovery (optional)
1. Read `causal_edge/plugins/AGENTS.md`
2. Set `ABEL_API_KEY` in `.env`
3. Run `causal-edge discover ETHUSD`
4. Add parents to strategies.yaml
```

**Completion criteria**:
```bash
make test    # All 15 structural tests + validation tests pass
make lint    # Structural tests alone pass
# Agent reads AGENTS.md → can answer "how do I add a strategy?" without reading source
```

### Phase 2: End-to-End MVP

**Goal**: `causal-edge init demo && cd demo && causal-edge run && causal-edge dashboard && causal-edge validate` all work.

**Deliverables**:
- `causal-edge init` scaffolds project with SMA demo + CLAUDE.md + AGENTS.md (harness is contagious)
- `causal-edge run` iterates strategies.yaml, calls compute_signals(), writes trade_log CSV
- `causal-edge dashboard` reads trade_logs → Jinja2 → static dashboard.html
- `causal-edge validate` reads trade_logs → Abel Proof 13-test → PASS/FAIL with exit code

**CLI behavior**:

| Command | Input | Output |
|---------|-------|--------|
| `causal-edge init <name>` | Name | Scaffolded project directory |
| `causal-edge run [--strategy ID]` | strategies.yaml | data/trade_log_*.csv per strategy |
| `causal-edge dashboard` | trade_log CSVs | dashboard.html |
| `causal-edge validate [--strategy ID]` | trade_log CSVs | Report card, exit 0 (PASS) or 1 (FAIL) |
| `causal-edge status` | trade_logs + engines | Per-strategy summary |

**Dashboard MVP** (minimal, not full 126KB version):
- Overview tab: strategy table (name, Sharpe, PnL, MaxDD, badge, latest signal)
- Per-strategy tab: equity curve + position chart (Plotly)
- Dark theme, static HTML, zero websocket
- Templates: `base.html`, `overview.html`, `strategy.html`, `styles.html`

**Init scaffold output**:
```
my-portfolio/
├── strategies.yaml          # Pre-filled with sma_crossover
├── strategies/
│   └── sma_crossover/engine.py
├── .env.example             # ABEL_API_KEY= (commented, optional)
├── CLAUDE.md                # Project-level harness (inherits framework constraints)
└── AGENTS.md                # User project decision tree
```

**Completion criteria**:
```bash
causal-edge init demo && cd demo
causal-edge run                    # Generates data/trade_log_sma_crossover.csv
causal-edge dashboard              # Generates dashboard.html (viewable in browser)
causal-edge validate               # SMA likely FAILs (educational — shows what FAIL looks like)
echo $?                            # 1
make test                          # Still passes
```

### Phase 3: Abel Plugin

**Goal**: Optional Abel CAP API integration. Framework works identically with or without it.

**Plugin architecture**:
```
causal_edge/plugins/
├── AGENTS.md
├── __init__.py              # discover_plugins(), register_plugin(), get_capability()
└── abel/
    ├── AGENTS.md
    ├── __init__.py          # register_plugin("abel", {...})
    ├── client.py            # Abel CAP API client (from cap_probe.py)
    ├── discover.py          # causal-edge discover <TICKER> → parents YAML
    └── overlay.py           # Real-time prediction overlay (optional)
```

**Plugin contract** — direct import with graceful fallback (not registry — YAGNI until second plugin):
```python
# cli.py
try:
    from causal_edge.plugins.abel.discover import discover_parents
    HAS_ABEL = True
except ImportError:
    HAS_ABEL = False
```
See Section 9 for full rationale. Registry pattern deferred until second plugin justifies it.

**strategies.yaml extension** — optional `parents:` field:
```yaml
strategies:
  - id: causal_demo
    parents:                   # Optional, Abel auto-fills or user manually fills
      - ticker: SSTK
        lag: 17
        relationship: parent
```

**Two paths**:
- Has `ABEL_API_KEY` → `causal-edge discover ETHUSD` auto-queries, outputs YAML snippet
- No key → user fills `parents:` manually, or skips (causal features not computed)

**Causal demo strategy** (~60 lines):
- Reads parents from strategies.yaml config
- Dual-lag xcorr at configured lags (public method, generic parameters)
- Binary threshold vs expanding median
- Walk-forward Long/Flat
- Public: methodology. Private: optimal parameter values.

**Completion criteria**:
```bash
# Without key
causal-edge run          # SMA + causal_demo both work
causal-edge validate     # Both get report cards

# With key
export ABEL_API_KEY=xxx
causal-edge discover ETHUSD   # Outputs parents YAML

# Plugin isolation
rm -rf causal_edge/plugins/abel/
make test                      # Still passes (TestPluginsOptional)
```

### Phase 4: ML Demo + Docs

**Goal**: Three demo strategies + documentation for both audiences.

**Momentum ML strategy** (~80 lines):
- Single-asset price data
- Features: lagged returns, rolling vol, rolling momentum, RSI — all shift(1)
- Walk-forward GBDT (sklearn), 126d rolling train, weekly retrain
- Decision threshold from inner validation
- Long/Flat output

**Three-demo educational gradient**:

| Demo | Lines | Teaches |
|------|-------|---------|
| `sma_crossover` | 30 | StrategyEngine ABC minimal implementation |
| `momentum_ml` | 80 | Walk-forward ML + look-ahead prevention |
| `causal_demo` | 60 | Abel plugin + cross-asset xcorr |

**Documentation**:

| File | Audience | Content |
|------|----------|---------|
| `README.md` | Both | "Why causal-edge" + 5-min quickstart + architecture diagram |
| `docs/why-causal.md` | Quant | Pearl/DGP/regime invariance argument (from abel-alpha SKILL.md) |
| `docs/harness-guide.md` | Agent dev | How to read AGENTS.md, how agents operate the framework |
| `docs/add-strategy.md` | Both | Three paths: simple / ML / causal |
| `CONTRIBUTING.md` | Both | Fork → branch → test → PR + harness rules |
| `CHANGELOG.md` | Both | From v0.1.0 |

**README structure** (lead with "why"):
```markdown
# causal-edge

The first agent-native quant framework.

**For quants**: Institutional-grade 13-test validation, anti-gaming metric triangle.
**For agent developers**: AGENTS.md decision trees — your agent reads them, knows what to do.

## 5 Minutes to First Dashboard
  pip install causal-edge && causal-edge init my-portfolio && cd my-portfolio
  causal-edge run && causal-edge dashboard && open dashboard.html

## Why Causal?
  Correlation breaks when regimes change. Causation doesn't. [→ docs/why-causal.md]
```

**Completion criteria**:
```bash
causal-edge init demo && cd demo
ls strategies/          # sma_crossover/ momentum_ml/ causal_demo/
causal-edge run         # All three run
causal-edge validate    # All three get report cards
```

### Phase 5: Open Source Release

**Goal**: `pip install causal-edge` works. CI green. Public repo.

**GitHub Actions**:
```yaml
# ci.yml — on push + PR
- matrix: python 3.11, 3.12
- pip install -e ".[dev]"
- make test

# release.yml — on tag v*
- python -m build
- twine upload dist/*
```

**pyproject.toml**:
```toml
[project]
name = "causal-edge"
version = "0.1.0"
description = "Agent-native quant framework with institutional-grade validation"
license = {text = "MIT"}

[project.optional-dependencies]
abel = ["requests>=2.28"]
dev = ["pytest>=7.0", "pyyaml>=6.0"]
```

**Release checklist**:
- [ ] LICENSE (MIT)
- [ ] .gitignore (Python + .env + data/)
- [ ] GitHub repo topics: quantitative-finance, agent-native, causal-inference, backtesting
- [ ] First tag: v0.1.0
- [ ] PyPI upload
- [ ] README badges: CI status, PyPI version, Python version

**Completion criteria**:
```bash
pip install causal-edge
causal-edge init test && cd test
causal-edge run && causal-edge dashboard && causal-edge validate
# All work from PyPI install, zero clone needed
```

## 4. What Gets Ported vs Written Fresh

| Component | Source | Action |
|-----------|--------|--------|
| `validation/metrics.py` | Existing causal-edge | Port as-is (369 lines, complete) |
| `validation/gate.py` | Existing causal-edge | Port as-is (160 lines, complete) |
| `validation/profiles/` | Existing causal-edge | Port as-is (3 YAML files) |
| `test_triangle.py` | Existing causal-edge | Port as-is (220 lines) |
| `engine/base.py` | Existing causal-edge | Port as-is (49 lines) |
| `examples/sma_crossover/` | Existing causal-edge | Port as-is (42 lines) |
| `config.py` | Existing causal-edge | Port + strip hardcoded paths |
| `CLAUDE.md` | Existing causal-edge | Rewrite (add new structural test refs) |
| All `AGENTS.md` files | Existing (thin) | **Rewrite as decision trees** |
| `cli.py` | Existing (minimal) | **Rewrite (init/run/dashboard/validate)** |
| `dashboard/` | Existing (full) | **Simplify to MVP** (strip to 4 templates) |
| `plugins/abel/` | Existing abel_cap.py + abel_overlay.py | **Refactor into plugin architecture** |
| `strategies/momentum_ml/` | — | **Write fresh** |
| `strategies/causal_demo/` | — | **Write fresh** |
| `test_structure.py` | Existing (7 tests) | **Extend (+7 new tests → 15 total)** |
| `test_cli.py` | — | **Write fresh** |
| All docs/ | — | **Write fresh** |

## 5. Invariants (Must Hold Across All Phases)

1. **`make test` passes at every commit** — no red-green-red cycles across phases
2. **Plugins are optional** — deleting `plugins/` never breaks core
3. **strategies/ never imports causal_edge/ internals** (except base.py ABC)
4. **No file > 400 lines**
5. **No hardcoded paths or secrets in source**
6. **strategies.yaml is the single source of truth** — no strategy-specific code in framework
7. **All PnL uses unclipped returns** — clip features for training only
8. **All features shift(1)** — zero look-ahead tolerance
9. **AGENTS.md has decision tree** — "I want to..." format
10. **Exit codes are meaningful** — validate returns 0 (PASS) or 1 (FAIL) for CI/agent consumption

## 6. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| AGENTS.md quality too low → agents can't operate | Phase 1 completion test: give agent only AGENTS.md, ask "how do I add a strategy?" — must get correct answer |
| Dashboard MVP too minimal → looks unfinished | Dark theme + Plotly charts already look good. Overview table + equity curve is sufficient for v0.1.0 |
| Abel plugin coupling leaks into core | TestPluginsOptional enforces mechanically |
| Demo strategies too trivial → not credible | momentum_ml uses real walk-forward GBDT — same methodology as production, different parameters |
| Look-ahead bugs in demo code | metrics.py already has runtime detection; add static check to CI |
| strategies.yaml empty in Phase 1 → TestYamlHasRequiredFields fails | Phase 1 strategies.yaml has `strategies: []` (empty list). Tests 1-4 skip gracefully when list is empty. Tests validate non-empty entries only. |
| causal_demo needs price data for parents | Demo ships with synthetic parent price CSVs in `examples/data/` (~10KB). No external API needed for demo path. |

## 7. Development Harness (How Agents Build This)

This section defines the harness for *implementing* causal-edge, not the product harness for *using* it.

### 7.1 CLAUDE.md for the New Repo

```markdown
# CLAUDE.md — causal-edge development harness

## Constraints (enforced by tests/test_structure.py)
1. No file > 400 lines. Split at 350.
2. Components are pure functions. Input: arrays/dicts. Output: string.
3. Templates have zero Python logic.
4. strategies/ never imports causal_edge/ internals (except base.py ABC).
5. strategies.yaml is the single source of truth.
6. Every subsystem has AGENTS.md with "I want to..." decision tree.
7. All strategy engines must be importable.
8. No hardcoded absolute paths. No secrets in source.
9. Structural test failure messages include actionable fix instructions.

## Style
- Formatter: ruff format (enforced by `make fmt`)
- Linter: ruff check (enforced by `make check`)
- All public functions have docstrings.
- Chart functions return JSON strings via chart_to_json().

## How to run
make test       # all tests
make lint       # structural tests only
make fmt        # auto-format
make check      # linter
make verify     # phase completion verification (see Makefile)
```

### 7.2 Per-Phase Agent Briefs

Each phase is extracted to a standalone brief that an implementing agent can load without the full spec. Stored at `docs/phases/phase-N.md`. Each brief contains:
- Goal (1 sentence)
- Files to create/modify (exhaustive list)
- Completion criteria (bash commands)
- What NOT to touch (scope fence)

This keeps agent context under 200 lines per phase.

### 7.3 Worktree + Parallel Agent Strategy

**Within Phase 2** (the largest), these are independent and can be dispatched to parallel agents in separate worktrees:

| Agent | Scope | Touches | Does NOT touch |
|-------|-------|---------|----------------|
| CLI agent | `cli.py` init/run/dashboard/validate subcommands | cli.py | engine/, dashboard/, validation/ |
| Engine agent | `engine/trader.py`, `engine/ledger.py` | engine/ | cli.py, dashboard/ |
| Dashboard agent | `generator.py`, `components.py`, templates/ | dashboard/ | cli.py, engine/ |

**Merge order**: Engine → Dashboard → CLI (CLI depends on engine + dashboard being functional).

Phase 1, 3, 4, 5 are small enough for single-agent sequential execution.

### 7.4 Makefile Subsystem Targets

```makefile
test:          pytest tests/ -v
test-struct:   pytest tests/test_structure.py -v
test-valid:    pytest tests/test_triangle.py tests/test_validation.py -v
test-cli:      pytest tests/test_cli.py -v
fmt:           ruff format .
check:         ruff check .
lint:          make test-struct

# Phase verification
verify-1:      make test
verify-2:      make test && causal-edge init /tmp/_ce_verify && cd /tmp/_ce_verify && causal-edge run && causal-edge dashboard && causal-edge validate; rm -rf /tmp/_ce_verify
verify-3:      make verify-2 && python -c "from causal_edge.plugins.abel import discover" 2>/dev/null || echo "Abel plugin not installed (OK)"
```

### 7.5 Structural Test Error Messages

All structural tests must include actionable fix instructions in their assertion message. Pattern:

```python
assert not violations, (
    f"Problem description:\n"
    + "\n".join(f"  {v}" for v in violations)
    + f"\nFix: [specific action to resolve]."
    + f"\nSee: [reference file or doc]."
)
```

## 8. Subsystem AGENTS.md Specifications

Each AGENTS.md must be ≤ 80 lines (root) or ≤ 60 lines (subsystem). Every referenced file path must exist (enforced by TestAgentsMdReferencesExist).

### 8.1 validation/AGENTS.md — Failure→Fix Mapping

```markdown
## I want to...

### Validate a strategy
  causal-edge validate --strategy <ID> --verbose

### Understand why it failed
Failure codes and fixes:

| Code | Test | Common cause | Fix |
|------|------|-------------|-----|
| T6  | DSR < 90% | Too many trials (high K) | Reduce parameter search space |
| T7  | PBO > 10% | Overfitting | Simplify model, fewer features |
| T12 | OOS/IS < 0.50 | In-sample overfitting | Shorter train window, regularize |
| T13 | NegRoll > 15% | Regime vulnerability | Add trend filter (SMA) |
| T14 | LossYrs > 2 | Strategy decays | Check if signal source still valid |
| T15-Lo | Lo-adj < 1.0 | Serial correlation inflating Sharpe | Add persistence penalty |
| T15-Omega | Omega < 1.0 | Return clipping or asymmetric losses | Use unclipped PnL |
| T15-MaxDD | MaxDD < -20% | Excessive leverage | Reduce position sizing |

### Understand the metric triangle
Read the docstring at top of metrics.py. Three orthogonal, leverage-invariant dimensions:
- Ratio (Lo-adjusted Sharpe) — optimized
- Rank (IC) — guardrail
- Shape (Omega) — guardrail, catches clipping
```

### 8.2 engine/AGENTS.md — Core Content

```markdown
## I want to...

### Create a new engine
1. Copy `examples/sma_crossover/engine.py`
2. Implement `compute_signals()` → (positions, dates, returns, prices)
3. Implement `get_latest_signal()` → dict with 'position' key
4. All features must use shift(1) — zero look-ahead tolerance

### Understand the execution flow
  strategies.yaml → config.py → trader.py (iterates, calls engines) → ledger.py (writes CSV)

### Debug "engine not importable"
1. Check strategies.yaml engine path matches module path
2. Check __init__.py exists in strategy directory
3. Run: python -c "import strategies.my_strategy.engine"
```

### 8.3 dashboard/AGENTS.md — Core Content

```markdown
## I want to...

### Add a chart component
1. Add pure function to components.py: data in → JSON string out (via chart_to_json)
2. Register in generator.py env.globals
3. make test — TestComponentsRegistered verifies

### Modify a template
Templates are in templates/. Rules:
- Zero Python logic — only Jinja2 loops/conditionals + component calls
- One template renders N strategies — never copy-paste per strategy
- Strategies come from strategies.yaml, not hardcoded

### Debug "component not registered"
1. make test — TestComponentsRegistered shows which function is unused
2. Add it to env.globals dict in generator.py
3. Or prefix with _ to mark private
```

### 8.4 plugins/AGENTS.md — Core Content

```markdown
## I want to...

### Use Abel causal discovery
1. Set ABEL_API_KEY in .env
2. Run: causal-edge discover <TICKER>
3. Copy output parents YAML into strategies.yaml
4. No key? Fill parents: manually — framework works identically

### Understand plugin isolation
- Plugins are optional. Deleting plugins/abel/ must not break anything.
- Framework uses try/except import, not registry scan.
- TestPluginsOptional enforces this mechanically.

### Build a new plugin (future)
- Create causal_edge/plugins/<name>/
- Expose capabilities via top-level functions
- Framework discovers via try/except import in cli.py
- No registry until second plugin exists (YAGNI)
```

## 9. Plugin Architecture Simplification

Phase 3 plugin system uses **direct import with graceful fallback**, not a registration pattern:

```python
# cli.py
try:
    from causal_edge.plugins.abel.discover import discover_parents
    HAS_ABEL = True
except ImportError:
    HAS_ABEL = False

@cli.command()
def discover(ticker):
    if not HAS_ABEL:
        click.echo("Abel plugin not installed. pip install causal-edge[abel]")
        sys.exit(1)
    parents = discover_parents(ticker)
    click.echo(yaml.dump(parents))
```

Registry pattern deferred until a second plugin justifies it. This satisfies TestPluginsOptional without premature abstraction.

## 10. Automated Agent-Readability Test

```python
# test_structure.py
class TestAgentsMdReferencesExist:
    """Every file path referenced in AGENTS.md must exist."""
    def test_all_referenced_files_exist(self):
        import re
        agents_files = list(ROOT.rglob("AGENTS.md"))
        missing = []
        for af in agents_files:
            content = af.read_text()
            # Match backtick-quoted paths like `strategies/AGENTS.md`
            paths = re.findall(r'`([a-zA-Z_./]+\.[a-z]+)`', content)
            for p in paths:
                resolved = (af.parent / p) if not p.startswith("causal_edge") else (ROOT / p)
                if not resolved.exists() and not (ROOT / p).exists():
                    missing.append(f"{af.relative_to(ROOT)}: references {p}")
        assert not missing, (
            f"AGENTS.md files reference non-existent paths:\n"
            + "\n".join(f"  {m}" for m in missing)
            + "\nFix: Update the path or create the missing file."
        )

class TestAgentsMdSizeBudget:
    """AGENTS.md files must stay concise for agent context efficiency."""
    def test_root_agents_md_under_80_lines(self):
        lines = (ROOT / "AGENTS.md").read_text().splitlines()
        assert len(lines) <= 80, (
            f"Root AGENTS.md is {len(lines)} lines (max 80). "
            f"Fix: Move details to subsystem AGENTS.md files."
        )
    
    def test_subsystem_agents_md_under_60_lines(self):
        for af in ROOT.rglob("AGENTS.md"):
            if af == ROOT / "AGENTS.md":
                continue
            lines = af.read_text().splitlines()
            assert len(lines) <= 60, (
                f"{af.relative_to(ROOT)} is {len(lines)} lines (max 60). "
                f"Fix: Split into sub-sections or link to docs/."
            )
```

This brings the structural test count to **15** (13 original + TestAgentsMdReferencesExist + TestAgentsMdSizeBudget).
