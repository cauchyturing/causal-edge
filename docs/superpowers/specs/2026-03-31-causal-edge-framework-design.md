# causal-edge Framework Design Spec

**Date:** 2026-03-31
**Status:** Draft
**Goal:** Refactor monolithic paper trading dashboard into a reusable, pip-installable framework where adding a strategy = YAML config + engine.py, zero dashboard code.

---

## 1. Problem

The current `dashboard.py` is 2849 lines of interleaved Python + HTML. Adding AAPL required modifying 19 locations. Each new strategy costs ~200 lines of copy-paste across tab buttons, overview table, backtest tab, live tab, chart data, JS tab switching, real-time config, and consensus. This does not scale.

## 2. Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Migration approach | Hybrid (C) | New framework, reuse existing engines unchanged |
| Target audience | Both internal + open-source (C) | Engines in strategies/, framework has no dependency on them |
| Template engine | Jinja2 (A) | `{% for s in strategies %}` is the core value; industry standard |
| Real-time prices | Server-side proxy + JS polling (A) | Simple, proven, sufficient for 5-10 strategies at 30s intervals |
| Abel-proof integration | Badge + gate on add (A+C) | Lightweight display, enforced quality at entry |

## 3. Architecture

```
causal-edge/
├── pyproject.toml
├── strategies.yaml               # Strategy registry + global settings
├── causal_edge/
│   ├── __init__.py
│   ├── cli.py                    # CLI entry point
│   ├── config.py                 # Load + validate strategies.yaml
│   │
│   ├── engine/
│   │   ├── base.py               # StrategyEngine ABC
│   │   ├── trader.py             # Iterate strategies, call engines, write trade logs
│   │   ├── ledger.py             # Trade-level lifecycle tracking
│   │   └── scheduler.py          # Cron setup helper
│   │
│   ├── dashboard/
│   │   ├── generator.py          # Load data + render templates → dashboard.html
│   │   ├── server.py             # ThreadingHTTPServer + /api/prices
│   │   ├── components.py         # Reusable Plotly chart builders
│   │   └── templates/
│   │       ├── base.html         # Head, CSS, tab bar, JS framework
│   │       ├── overview.html     # Summary table, consensus, real-time bar
│   │       ├── backtest.html     # BT tab (one template, rendered per strategy)
│   │       └── live.html         # Live tab (one template, rendered per strategy)
│   │
│   ├── validation/
│   │   └── gate.py               # Abel-proof gate on add, badge generation
│   │
│   └── plugins/                  # Optional integrations (not a plugin system — direct imports)
│       └── abel_overlay.py       # Abel CAP position adjustment (ported from paper_trading/)
│
├── strategies/                   # Production engines (not a framework dependency)
│   ├── alpha_max/engine.py
│   ├── seven_comp/engine.py
│   ├── ton/engine.py
│   ├── meta/engine.py
│   └── aapl/engine.py
│
├── examples/
│   └── sma_crossover/
│       ├── engine.py             # 30-line example (also CI integration test)
│       └── README.md
│
├── data/                         # Trade logs, price caches
│   └── *.csv
│
└── tests/
    ├── test_config.py
    ├── test_components.py
    ├── test_generator.py
    └── test_sma_example.py       # End-to-end with sma_crossover
```

## 4. Strategy Registry (strategies.yaml)

```yaml
settings:
  capital: 100000
  port: 8088
  refresh_seconds: 300
  fmp_api_key: ${FMP_API_KEY}
  theme: dark

strategies:
  - id: alpha_max
    name: "Alpha Max"
    asset: ETH
    color: "#0A84FF"
    engine: strategies.alpha_max.engine
    trade_log: data/trade_log_alpha_max.csv
    realtime:
      ticker: ETHUSD
      source: coingecko
    badge: "15/15"

  - id: seven_comp
    name: "7-Component"
    asset: ETH
    color: "#30D158"
    engine: strategies.seven_comp.engine
    trade_log: data/trade_log_7comp.csv
    realtime:
      ticker: ETHUSD
      source: coingecko
    badge: "15/15"

  - id: ton
    name: "TON Multi"
    asset: TON
    color: "#FF6B35"
    engine: strategies.ton.engine
    trade_log: data/trade_log_ton.csv
    realtime:
      ticker: TONUSD
      source: coingecko
    badge: "15/15"

  - id: meta
    name: "META Multi"
    asset: META
    color: "#FF9500"
    engine: strategies.meta.engine
    trade_log: data/trade_log_meta.csv
    realtime:
      ticker: META
      source: fmp
    badge: "15/15"

  - id: aapl
    name: "AAPL Multi"
    asset: AAPL
    color: "#FF2D55"
    engine: strategies.aapl.engine
    trade_log: data/trade_log_aapl.csv
    realtime:
      ticker: AAPL
      source: fmp
    badge: "15/15"

plugins:
  - abel_overlay    # Optional Abel CAP position adjustment (hardcoded call, not plugin system)
```

**Environment variables:** `${VAR}` syntax expanded at load time.
- **Required:** `fmp_api_key` (data fetching fails without it — hard error)
- **Optional:** plugin-specific keys like `ABEL_API_KEY` (warning, plugin skipped)

**Naming convention:** Strategy IDs use snake_case. Directory names match IDs exactly.
The current `sevencomp_engine.py` wraps `signal_engine.py` (8 causal pairs) + depends on
`ml_engine.py`. In the new structure, `strategies/eight_comp/engine.py` internalizes all
three. Name changed from "7-Component" to "8-Component" to match CLAUDE.md.

## 5. Engine Interface

```python
# causal_edge/engine/base.py

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Any

class StrategyEngine(ABC):
    """Base class for all trading strategy engines.

    Engines are self-contained: they manage their own data loading,
    model training, and prediction. The framework provides a context
    dict at init time with shared resources (price cache paths, API keys).
    """

    def __init__(self, context: dict | None = None):
        """Initialize with optional context.

        Args:
            context: dict with keys like 'prices_csv', 'fmp_cache_dir',
                     'fmp_api_key', 'data_dir'. Engines that need shared
                     data read from here. Self-contained engines ignore it.
        """
        self.context = context or {}

    @abstractmethod
    def compute_signals(self) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Full backtest computation. Engine loads its own data.

        Returns:
            pnl: Daily PnL array (unclipped returns * positions)
            dates: DatetimeIndex aligned with pnl
            positions: Daily position array
            asset_prices: Asset close prices aligned with dates
        """

    @abstractmethod
    def get_latest_signal(self) -> dict:
        """Get today's trading signal.

        Returns dict with CANONICAL keys:
            date: Latest date (pd.Timestamp or None)
            position: Position size (float, 0 = flat)
            pnl_today: Today's PnL (float, scalar, NOT array)
            sharpe: Running Sharpe ratio (float)
            asset_close: Latest asset price (float)
            details: dict with strategy-specific info
        """

    def on_retrain(self) -> None:
        """Called by trader on retrain schedule (e.g., weekly Monday).

        Override if the engine has ML models that need periodic retraining.
        Default: no-op. Engines that don't retrain ignore this.
        """
        pass
```

### Engine context and dependencies

Engines that need shared state (e.g., 7-component needs ML predictions from ml_engine)
handle this internally. The `context` dict provides paths and API keys but NOT
pre-computed data. This keeps engines self-contained and testable in isolation.

For the 7-component engine specifically, the wrapper internalizes the ML dependency:

```python
# strategies/seven_comp/engine.py

class SevenCompEngine(StrategyEngine):
    def compute_signals(self):
        # Loads prices, runs ML internally, computes all 8 components
        from paper_trading.ml_engine import load_predictions
        from paper_trading.signal_engine import compute_full_signals
        prices = pd.read_csv(self.context.get("prices_csv", "data/prices.csv"), ...)
        pred_mainwf = load_predictions("mainwf")
        pred_lag3 = load_predictions("lag3")
        pnl, dates, positions, px = compute_full_signals(prices, pred_mainwf, pred_lag3)
        return pnl, dates, positions, px

    def on_retrain(self):
        """Weekly ML retrain (called by trader on Monday)."""
        from paper_trading.ml_engine import retrain_models
        retrain_models()
```

### ML retraining lifecycle

The trader calls `engine.on_retrain()` on the configured schedule:

```yaml
strategies:
  - id: seven_comp
    retrain: monday     # trader calls on_retrain() on Mondays
    # ...
  - id: aapl
    retrain: null        # no retraining needed
```

New strategies implement the ABC directly. The `sma_crossover` example shows the minimal implementation (no `on_retrain`, no context needed).

## 6. Dashboard Templates

### 6.1 base.html (~120 lines)

Contains: HTML head, CSS variables, Plotly script tag, header bar, tab buttons, tab content containers, JS tab switching + real-time polling.

Key loop:
```html
<div class="tabs">
  <button class="tab active" onclick="switchTab('overview')">Overview</button>
  {% for s in strategies %}
  <button class="tab" onclick="switchTab('{{s.id}}live')" style="border-bottom-color:{{s.color}}">
    {{s.name}} Live
  </button>
  <button class="tab" onclick="switchTab('{{s.id}}bt')">{{s.name}} BT</button>
  {% endfor %}
</div>

{% include 'overview.html' %}
{% for s in strategies %}
  {% include 'backtest.html' %}
  {% include 'live.html' %}
{% endfor %}
```

### 6.2 overview.html (~80 lines)

Summary table iterating strategies. Real-time price bar. Consensus display. Each strategy row is generated from the same template block.

### 6.3 backtest.html (~100 lines)

Single template, rendered per strategy. Uses component functions:

```html
<div id="tab-{{s.id}}bt" class="tab-content">
  <div class="section-label">{{s.name}} — Backtest</div>
  {{ metrics_cards(s.bt) }}
  <div class="chart-card"><div id="c-{{s.id}}bt-equity"></div></div>
  <div class="row-2">
    <div class="chart-card"><div id="c-{{s.id}}bt-pos"></div></div>
    <div class="chart-card"><div id="c-{{s.id}}bt-dd"></div></div>
  </div>
  <!-- ... rolling sharpe, distribution, heatmap, year-by-year, blotter -->
  {{ yearly_table(s.bt.yearly) }}
  {{ trade_blotter(s.id, s.bt.trades) }}
</div>

<script type="application/json" id="d-{{s.id}}bt-equity">{{ s.bt.charts.equity }}</script>
<!-- ... other chart data -->
```

### 6.4 live.html (~120 lines)

Same structure as backtest but for live/paper trading data. Hero section with current position, live metrics, trade table, analysis charts.

## 7. Components (components.py)

Port of existing `_make_*` functions, each returning Plotly JSON:

| Function | Current source | Input | Output |
|----------|---------------|-------|--------|
| `equity_chart(dates, cum_pnl, color, title)` | `_make_equity_chart` | Arrays | Plotly JSON string |
| `position_chart(dates, positions, color)` | `_make_position_chart` | Arrays | Plotly JSON string |
| `drawdown_chart(dates, cum_pnl)` | `_make_drawdown` | Arrays | Plotly JSON string |
| `rolling_sharpe(dates, pnl, window)` | `_make_rolling_sharpe` | Arrays | Plotly JSON string |
| `monthly_heatmap(dates, pnl)` | `_make_monthly_heatmap` | Arrays | Plotly JSON string |
| `pnl_distribution(pnl)` | `_make_pnl_dist` | Array | Plotly JSON string |
| `metrics_cards(metrics_dict)` | inline HTML | Dict | HTML string |
| `yearly_table(yearly_data)` | `_yearly_html` | List | HTML string |
| `trade_blotter(strategy_id, trades_df, ledger_df)` | `_trade_summary_html` | DataFrames | HTML string |
| `live_analysis(trade_log, ledger, bt_sharpe)` | `_live_analysis_block` | DataFrames | HTML string |

Each function is stateless, takes data in, returns string out. No strategy-specific logic. Registered as Jinja2 globals so templates can call them directly.

## 8. Generator (generator.py)

```python
def generate(config_path="strategies.yaml", output="dashboard.html"):
    """Main entry point. Reads config, loads data, renders templates."""

    cfg = load_config(config_path)
    strategies = []

    for s_cfg in cfg["strategies"]:
        trade_log = pd.read_csv(s_cfg["trade_log"], parse_dates=["date"])
        bt_data = prepare_backtest(trade_log, s_cfg)
        live_data = prepare_live(trade_log, s_cfg)
        strategies.append({
            "id": s_cfg["id"],
            "name": s_cfg["name"],
            "color": s_cfg["color"],
            "asset": s_cfg["asset"],
            "badge": s_cfg.get("badge", ""),
            "realtime": s_cfg.get("realtime", {}),
            "bt": bt_data,
            "live": live_data,
        })

    env = jinja2.Environment(loader=PackageLoader("causal_edge", "dashboard/templates"))
    env.globals.update({
        "equity_chart": components.equity_chart,
        "drawdown_chart": components.drawdown_chart,
        "position_chart": components.position_chart,
        "rolling_sharpe": components.rolling_sharpe,
        "monthly_heatmap": components.monthly_heatmap,
        "pnl_distribution": components.pnl_distribution,
        "metrics_cards": components.metrics_cards,
        "yearly_table": components.yearly_table,
        "trade_blotter": components.trade_blotter,
        "live_analysis": components.live_analysis,
    })

    html = env.get_template("base.html").render(
        strategies=strategies,
        settings=cfg["settings"],
    )
    Path(output).write_text(html)
```

### prepare_backtest(trade_log, s_cfg) → dict

Computes everything the backtest tab needs from the trade log:
- `metrics`: dict with sharpe, cum_pnl, max_dd, win_rate, n_trades, calmar
- `charts`: dict with Plotly JSON for each chart (equity, position, drawdown, rolling_sharpe, distribution, monthly)
- `yearly`: list of per-year dicts (year, sharpe, return, maxdd, win_rate, n_trades)
- `trades`: DataFrame for trade blotter (from ledger)
- `dates`, `pnl`, `positions`, `cum_pnl`: raw arrays for custom rendering

This is a port of the inline computation in current dashboard.py (~200 lines), made generic.

### prepare_live(trade_log, s_cfg) → dict

Same structure but filtered to `source="live"` rows only. Adds:
- `hero`: dict with current position, direction, today's PnL, live days count
- `bt_sharpe`: backtest Sharpe for tracking comparison
- `extras`: live analysis metrics (drawdown from live peak, rolling Sharpe, etc.)

Port of `_compute_live_extras` + `_live_analysis_block` (~200 lines), made generic.

### Realistic line count

| Component | Estimated lines | Current equivalent |
|-----------|----------------|-------------------|
| Templates (4 files) | ~400 | ~1800 (HTML in dashboard.py) |
| components.py | ~300 | ~400 (_make_* functions) |
| generator.py | ~200 | ~600 (data prep + orchestration) |
| JS (in base.html) | ~200 | ~600 (tab switching, real-time) |
| **Total** | **~1100** | **~2849** |

The JS is embedded in base.html template, not in Python. Target is ~1100 lines total (62% reduction), not 700 as originally claimed. The reduction comes from eliminating per-strategy duplication, not from removing functionality.

## 9. Trader (trader.py)

Replaces `paper_trader.py`'s hardcoded TLOG dict. Iterates strategies.yaml with proper lifecycle.

### Daily update pipeline

```python
def daily_update(config_path="strategies.yaml"):
    cfg = load_config(config_path)
    context = build_context(cfg)  # prices_csv, fmp_cache_dir, api keys

    # Step 1: Data refresh (fetch latest prices, update caches)
    refresh_price_data(cfg)

    # Step 2: ML retraining (Monday only, for strategies that need it)
    if is_retrain_day():
        for s_cfg in cfg["strategies"]:
            if s_cfg.get("retrain") and should_retrain(s_cfg["retrain"]):
                engine = import_engine(s_cfg["engine"], context)
                engine.on_retrain()
                print(f"  Retrained: {s_cfg['name']}")

    # Step 3: Compute signals (ordered — each engine is independent)
    for s_cfg in cfg["strategies"]:
        try:
            engine = import_engine(s_cfg["engine"], context)
            sig = engine.get_latest_signal()
            if sig["date"] is not None:
                record_signal(s_cfg["trade_log"], sig)
        except Exception as e:
            print(f"  {s_cfg['name']} error: {e}")
            # Continue to next strategy — failure isolation

    # Step 4: Optional plugins (e.g., Abel overlay)
    if "abel_overlay" in cfg.get("plugins", []):
        from causal_edge.plugins.abel_overlay import apply_overlays
        apply_overlays(cfg)

    # Step 5: Regenerate dashboard
    from causal_edge.dashboard.generator import generate
    generate(config_path)
```

### Trade log management (record_signal)

Port of `paper_trader._record_signal` (~70 lines). Handles three cases:
1. **Already live today** — skip (idempotent)
2. **Promoting backfill** — replace last backfill row with live, recompute cum_pnl
3. **New append** — append with source="live", compute cum_pnl from previous

Also integrates with `ledger.py` for trade-level lifecycle tracking (open/close/hold).

### Data fetching

`refresh_price_data(cfg)` ports the data refresh logic from `data_fetcher.py`:
- Update FMP price cache for equity tickers
- Update crypto prices (CoinGecko/Binance)
- Merge into prices.csv

This runs once per daily update, before any engine. Engines read from cached files,
not live APIs (except during their own signal computation if needed).

### Error handling and logging

- Each strategy is wrapped in try/except — one failure doesn't stop others
- Logging to stdout (captured by cron wrapper) + optional file log
- The cron wrapper (`causal-edge cron`) uses flock for locking and handles log rotation

## 10. CLI (cli.py)

```
causal-edge init <project-name>    Create project scaffold with strategies.yaml + sma_crossover example
causal-edge add <engine-path>      Add strategy to strategies.yaml, run abel-proof gate
causal-edge backfill [--strategy]  Run compute_signals() for all/one strategy, write trade logs
causal-edge update                 Daily signal update (cron target)
causal-edge dashboard [--port]     Generate dashboard.html + start server
causal-edge validate <strategy>    Run abel-proof audit, display report card
causal-edge cron --daily <time>    Install/update system cron job
causal-edge status                 Show all strategies with latest position/PnL
```

## 11. Validation Gate (gate.py)

When `causal-edge add` is run:
1. Import the engine, call `compute_signals()`
2. Run abel-proof audit on the resulting PnL
3. If 15/15 PASS: add to strategies.yaml with badge
4. If FAIL: print report card, ask user to confirm (can override with `--force`)

Badge is stored in strategies.yaml and displayed on dashboard strategy cards.

## 12. Migration Path

| Step | Action | Risk |
|------|--------|------|
| 1 | Create `causal-edge/` project structure, `pyproject.toml`, empty modules | None |
| 2 | Implement `config.py` (load strategies.yaml with env var expansion) | None |
| 3 | Implement `engine/base.py` (ABC) | None |
| 4 | Implement `components.py` (port all `_make_*` functions from dashboard.py) | Low — pure functions, easy to test |
| 5 | Create Jinja2 templates (base, overview, backtest, live) | Medium — must match current visual design |
| 6 | Implement `generator.py` (wire config + components + templates) | Medium — core integration point |
| 7 | Implement `server.py` (port existing, add ThreadingMixIn) | Low — already clean |
| 8 | Create strategy wrappers in `strategies/` (thin ABC adapters) | Low — delegate to existing engines |
| 9 | Implement `trader.py` (port paper_trader.py to iterate YAML) | Medium — orchestration logic |
| 10 | Implement `cli.py` (click-based CLI) | Low |
| 11 | Implement `gate.py` (abel-proof integration) | Low |
| 12 | Create `sma_crossover` example + tests | Low |
| 13 | Verify: generate dashboard, compare visually to current | High — must match exactly |
| 14 | Cutover: point cron to new system, archive old `paper_trading/` | Medium |

## 13. Dependencies

```toml
[project]
dependencies = [
    "pandas>=2.0",
    "numpy>=1.24",
    "plotly>=5.0",
    "jinja2>=3.1",
    "pyyaml>=6.0",
    "scikit-learn>=1.3",
    "scipy>=1.10",
    "requests>=2.28",
    "click>=8.0",
]
```

## 14. What Stays the Same

- All engine internals (alpha_max, signal, ton, meta, apple) — unchanged
- Trade log CSV format (date, position, pnl, cum_pnl, asset_close, source) — unchanged
- Dashboard visual design (dark theme, Apple-style, Plotly charts) — unchanged
- Real-time architecture (server proxy + 30s JS polling) — unchanged
- Data directory structure — unchanged

## 15. Success Criteria

1. `causal-edge dashboard` produces visually identical output to current dashboard (verified by Playwright screenshot diff)
2. Adding a 6th strategy requires only strategies.yaml edit + engine.py
3. `sma_crossover` example works end-to-end: init → add → backfill → dashboard
4. All 5 existing strategies pass abel-proof after migration
5. Dashboard code: ~1100 lines (templates + components + generator + JS) vs current 2849
6. `pip install causal-edge` works from clean environment
7. All structural tests pass: YAML schema, engine-YAML alignment, component registration

## 16. Mechanical Enforcement

### Structural tests (run in CI and before every dashboard generation)

```python
# tests/test_structure.py

def test_yaml_schema():
    """Every strategy must have required fields: id, name, asset, color, engine, trade_log."""

def test_engine_exists():
    """Every strategy in YAML must have a matching importable engine module."""

def test_engine_implements_abc():
    """Every engine module must export a class that subclasses StrategyEngine."""

def test_components_registered():
    """Every component function in components.py must be registered in generator.py globals."""

def test_trade_log_schema():
    """Every trade log CSV must have columns: date, position, pnl, cum_pnl, asset_close, source."""

def test_template_ids_unique():
    """No two strategies can have the same id (would create duplicate HTML element IDs)."""
```

### Lint rules (custom, with fix instructions)

```
# Strategy YAML lint
ERROR: Strategy 'foo' has no 'engine' field.
  Fix: Add 'engine: strategies.foo.engine' to the strategy block in strategies.yaml.
  See: docs/add-strategy.md

# Engine interface lint
ERROR: strategies/foo/engine.py does not export a StrategyEngine subclass.
  Fix: Create a class that inherits from causal_edge.engine.base.StrategyEngine
  and implements compute_signals() and get_latest_signal().
  See: examples/sma_crossover/engine.py for minimal example.

# Component registration lint
ERROR: components.py defines 'new_chart()' but it is not registered in generator.py.
  Fix: Add 'new_chart': components.new_chart to env.globals.update() in generator.py.
```

## 17. Agent Navigation (AGENTS.md hierarchy)

```
causal-edge/
├── AGENTS.md                          # Root entry (~60 lines)
│   Points to: ARCHITECTURE.md, strategies.yaml, docs/add-strategy.md
│
├── ARCHITECTURE.md                    # Domain layers + dependency direction
│   config.py → engine/ → dashboard/
│   strategies/ has NO dependency on causal_edge/ (importable but not required)
│
├── causal_edge/
│   ├── engine/AGENTS.md               # Engine subsystem conventions
│   │   "All engines implement StrategyEngine ABC. See base.py for interface."
│   │
│   └── dashboard/AGENTS.md            # Dashboard subsystem conventions
│       "Templates use Jinja2. Components are stateless. See components.py for chart API."
│
├── strategies/AGENTS.md               # How to add a strategy
│   "Each strategy is a directory with engine.py. See examples/sma_crossover/."
│
└── docs/
    └── add-strategy.md                # Step-by-step guide for new strategies
```

## 18. Monolith Porting Map

Concrete function-by-function decomposition of dashboard.py (2849 lines):

### → components.py

| Current function | Lines | New function | Notes |
|-----------------|-------|-------------|-------|
| `_metrics(pnl)` | 17-37 | `compute_metrics(pnl)` | Returns dict |
| `_chart_json(fig)` | 38-40 | `chart_to_json(fig)` | Plotly → JSON string |
| `_chart_layout(...)` | 42-60 | `chart_layout(...)` | Shared Plotly layout |
| `_make_equity_chart(...)` | 61-71 | `equity_chart(...)` | |
| `_make_daily_pnl_bars(...)` | 77-84 | `daily_pnl_bars(...)` | |
| `_make_drawdown(...)` | 86-100 | `drawdown_chart(...)` | |
| `_make_rolling_sharpe(...)` | 101-115 | `rolling_sharpe(...)` | |
| `_make_components(...)` | 116-130 | `component_chart(...)` | 7comp-specific, keep generic |
| `_make_position_chart(...)` | 131-148 | `position_chart(...)` | |
| `_make_monthly_heatmap(...)` | 149-170 | `monthly_heatmap(...)` | |
| `_make_pnl_dist(...)` | 171-181 | `pnl_distribution(...)` | |
| `_yearly_metrics(...)` | 192-202 | `yearly_metrics(...)` | |
| `_yearly_html(...)` | 203-218 | `yearly_table(...)` | Returns HTML |
| `_trade_summary_html(...)` | 219-438 | `trade_blotter(...)` | Largest function (~220 lines) |

### → generator.py

| Current code region | Lines | New function | Notes |
|-------------------|-------|-------------|-------|
| Per-strategy data processing | 784-1050 (×5) | `prepare_backtest()` | Generic, called per strategy |
| Live analysis computation | `_compute_live_extras` 439-510 | `prepare_live()` | Generic |
| Live analysis HTML | `_live_analysis_block` 511-634 | Moved to live.html template | |
| Overview data | `_overview_strategy_data` 635-713 | `prepare_overview()` | Per-strategy |
| Overview HTML | `_overview_html` 714-783 | Moved to overview.html template | |
| Realtime config | `_build_realtime_config` (in paper_trader) | `build_realtime_config()` | |

### → templates/

| Current code region | Approx lines | Template | Notes |
|-------------------|-------------|----------|-------|
| HTML head + CSS + header | ~300 | base.html (top) | One copy, not ×5 |
| Tab buttons | ~15 ×5 = 75 | base.html ({% for %}) | |
| Overview section | ~200 | overview.html | |
| BT tab per strategy | ~200 ×5 = 1000 | backtest.html (×1) | Biggest savings |
| Live tab per strategy | ~250 ×5 = 1250 | live.html (×1) | Biggest savings |
| JS (tab switch, realtime, charts) | ~600 | base.html (bottom) | Must be generic |

### Parallelizable implementation

| Agent A | Agent B | Agent C |
|---------|---------|---------|
| components.py (port _make_* functions) | templates/ (extract HTML patterns) | config.py + base.py (framework core) |
| test_components.py | — | test_config.py, test_structure.py |

After A+B+C merge → Agent D implements generator.py (wires everything together).

## 19. Visual Regression Testing

```python
# tests/test_visual.py

def test_dashboard_visual_match():
    """Compare new dashboard output against reference screenshot."""
    # 1. Generate dashboard with current system → reference.png
    # 2. Generate dashboard with causal-edge → candidate.png
    # 3. Playwright screenshot both at same viewport
    # 4. Pixel diff with tolerance (< 5% difference = PASS)

    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800})

        page.goto("file:///path/to/reference/dashboard.html")
        page.screenshot(path="reference.png", full_page=True)

        page.goto("file:///path/to/candidate/dashboard.html")
        page.screenshot(path="candidate.png", full_page=True)

        # Compare with perceptual diff
        # PASS if structural match (tabs, charts, tables present)
```
