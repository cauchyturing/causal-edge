# causal-edge Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the 2849-line monolithic dashboard.py into a template-driven framework where adding a strategy = YAML config + engine.py.

**Architecture:** `config.py` → `engine/` → `dashboard/`. Strategies are data (YAML), not code. Jinja2 templates loop over strategies. Components are stateless pure functions. See `ARCHITECTURE.md`.

**Tech Stack:** Python 3.12, Jinja2, Plotly, Click, PyYAML, pandas, numpy

**Source to port from:** `~/Claude/New project/paper_trading/dashboard.py` (2849 lines). Porting map in spec Section 18.

---

## Chunk 1: Foundation (config + engine ABC + project scaffold)

### Task 1: Project scaffold + pyproject.toml

**Files:**
- Create: `causal_edge/__init__.py`
- Create: `causal_edge/engine/__init__.py`
- Create: `causal_edge/dashboard/__init__.py`
- Create: `causal_edge/validation/__init__.py`
- Create: `causal_edge/plugins/__init__.py`
- Create: `strategies/__init__.py`
- Create: `examples/__init__.py`
- Create: `tests/__init__.py`
- Create: `pyproject.toml`
- Create: `data/.gitkeep`

- [ ] **Step 1: Create directory structure**

```bash
cd ~/Claude/causal-edge
mkdir -p causal_edge/{engine,dashboard/templates,validation,plugins}
mkdir -p strategies examples/sma_crossover tests data
touch causal_edge/__init__.py causal_edge/engine/__init__.py
touch causal_edge/dashboard/__init__.py causal_edge/validation/__init__.py
touch causal_edge/plugins/__init__.py
touch strategies/__init__.py examples/__init__.py tests/__init__.py
touch data/.gitkeep
```

- [ ] **Step 2: Write pyproject.toml**

```toml
[project]
name = "causal-edge"
version = "0.1.0"
description = "Paper trading dashboard framework — add a strategy with YAML + engine.py"
requires-python = ">=3.11"
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

[project.scripts]
causal-edge = "causal_edge.cli:main"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends._legacy:_Backend"

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 3: Commit scaffold**

```bash
git init
git add -A
git commit -m "scaffold: project structure + pyproject.toml"
```

---

### Task 2: config.py — Load + validate strategies.yaml

**Files:**
- Create: `causal_edge/config.py`
- Create: `tests/test_config.py`
- Create: `strategies.yaml` (with SMA crossover example only for now)

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import pytest
from pathlib import Path

def test_load_config_returns_settings_and_strategies(tmp_path):
    yaml_content = """
settings:
  capital: 100000
  port: 8088

strategies:
  - id: test_strat
    name: "Test"
    asset: BTC
    color: "#FF0000"
    engine: examples.sma_crossover.engine
    trade_log: data/trade_log_test.csv
"""
    cfg_file = tmp_path / "strategies.yaml"
    cfg_file.write_text(yaml_content)

    from causal_edge.config import load_config
    cfg = load_config(str(cfg_file))

    assert cfg["settings"]["capital"] == 100000
    assert len(cfg["strategies"]) == 1
    assert cfg["strategies"][0]["id"] == "test_strat"
    assert cfg["strategies"][0]["color"] == "#FF0000"


def test_env_var_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    yaml_content = """
settings:
  api_key: ${MY_KEY}
strategies: []
"""
    cfg_file = tmp_path / "strategies.yaml"
    cfg_file.write_text(yaml_content)

    from causal_edge.config import load_config
    cfg = load_config(str(cfg_file))
    assert cfg["settings"]["api_key"] == "secret123"


def test_missing_required_field(tmp_path):
    yaml_content = """
settings: {}
strategies:
  - id: bad
    name: "Bad"
"""
    cfg_file = tmp_path / "strategies.yaml"
    cfg_file.write_text(yaml_content)

    from causal_edge.config import load_config
    with pytest.raises(ValueError, match="asset"):
        load_config(str(cfg_file))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ~/Claude/causal-edge && python -m pytest tests/test_config.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement config.py**

```python
# causal_edge/config.py
"""Load and validate strategies.yaml with env var expansion."""

import os
import re
import yaml
from pathlib import Path

REQUIRED_STRATEGY_FIELDS = ["id", "name", "asset", "color", "engine", "trade_log"]

DEFAULT_SETTINGS = {
    "capital": 100000,
    "port": 8088,
    "refresh_seconds": 300,
    "theme": "dark",
}


def _expand_env_vars(value):
    """Expand ${VAR} patterns in string values."""
    if not isinstance(value, str):
        return value
    def replacer(match):
        var_name = match.group(1)
        val = os.environ.get(var_name)
        if val is None:
            return match.group(0)  # Leave unexpanded
        return val
    return re.sub(r"\$\{(\w+)\}", replacer, value)


def _expand_recursive(obj):
    """Recursively expand env vars in all string values."""
    if isinstance(obj, dict):
        return {k: _expand_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_recursive(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    return obj


def load_config(path: str = "strategies.yaml") -> dict:
    """Load and validate strategies.yaml.

    Returns dict with 'settings', 'strategies', 'plugins' keys.
    Raises ValueError if required fields are missing.
    """
    raw = yaml.safe_load(Path(path).read_text())
    cfg = _expand_recursive(raw)

    # Merge defaults into settings
    settings = {**DEFAULT_SETTINGS, **(cfg.get("settings") or {})}
    cfg["settings"] = settings

    # Validate strategies
    strategies = cfg.get("strategies") or []
    for i, s in enumerate(strategies):
        for field in REQUIRED_STRATEGY_FIELDS:
            if field not in s:
                raise ValueError(
                    f"Strategy #{i} ('{s.get('id', '?')}') missing required field: '{field}'. "
                    f"Required fields: {REQUIRED_STRATEGY_FIELDS}. "
                    f"See docs/add-strategy.md for example."
                )
    cfg["strategies"] = strategies
    cfg["plugins"] = cfg.get("plugins") or []

    return cfg
```

- [ ] **Step 4: Run tests**

Run: `cd ~/Claude/causal-edge && python -m pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add causal_edge/config.py tests/test_config.py
git commit -m "feat: config.py — load + validate strategies.yaml with env var expansion"
```

---

### Task 3: engine/base.py — StrategyEngine ABC

**Files:**
- Create: `causal_edge/engine/base.py`
- Create: `tests/test_engine_base.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_engine_base.py
import pytest
import numpy as np
import pandas as pd
from causal_edge.engine.base import StrategyEngine


def test_cannot_instantiate_abc():
    with pytest.raises(TypeError):
        StrategyEngine()


def test_concrete_engine_works():
    class DummyEngine(StrategyEngine):
        def compute_signals(self):
            dates = pd.date_range("2024-01-01", periods=10)
            pnl = np.random.randn(10) * 0.01
            positions = np.ones(10) * 0.5
            prices = np.linspace(100, 110, 10)
            return pnl, dates, positions, prices

        def get_latest_signal(self):
            return {"date": pd.Timestamp("2024-01-10"), "position": 0.5,
                    "pnl_today": 0.01, "sharpe": 2.0, "asset_close": 110.0,
                    "details": {}}

    engine = DummyEngine()
    pnl, dates, pos, px = engine.compute_signals()
    assert len(pnl) == 10
    assert isinstance(dates, pd.DatetimeIndex)

    sig = engine.get_latest_signal()
    assert sig["position"] == 0.5
    assert "pnl_today" in sig


def test_on_retrain_default_noop():
    class MinimalEngine(StrategyEngine):
        def compute_signals(self):
            return np.array([]), pd.DatetimeIndex([]), np.array([]), np.array([])
        def get_latest_signal(self):
            return {"date": None}

    engine = MinimalEngine()
    engine.on_retrain()  # Should not raise


def test_context_passed_to_engine():
    class ContextEngine(StrategyEngine):
        def compute_signals(self):
            return np.array([]), pd.DatetimeIndex([]), np.array([]), np.array([])
        def get_latest_signal(self):
            return {"date": None, "path": self.context.get("prices_csv")}

    engine = ContextEngine(context={"prices_csv": "/data/prices.csv"})
    sig = engine.get_latest_signal()
    assert sig["path"] == "/data/prices.csv"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_engine_base.py -v`
Expected: FAIL

- [ ] **Step 3: Implement base.py**

```python
# causal_edge/engine/base.py
"""Strategy engine abstract base class."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class StrategyEngine(ABC):
    """Base class for all trading strategy engines.

    Engines are self-contained: they manage their own data loading,
    model training, and prediction. The framework provides a context
    dict at init time with shared resources (price cache paths, API keys).
    """

    def __init__(self, context: dict | None = None):
        self.context = context or {}

    @abstractmethod
    def compute_signals(self) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Full backtest computation.

        Returns:
            pnl: Daily PnL array
            dates: DatetimeIndex aligned with pnl
            positions: Daily position array
            asset_prices: Asset close prices aligned with dates
        """

    @abstractmethod
    def get_latest_signal(self) -> dict:
        """Get today's trading signal.

        Returns dict with CANONICAL keys:
            date, position, pnl_today (scalar), sharpe, asset_close, details
        """

    def on_retrain(self) -> None:
        """Called by trader on retrain schedule. Override if needed."""
        pass
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_engine_base.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add causal_edge/engine/base.py tests/test_engine_base.py
git commit -m "feat: StrategyEngine ABC with context + on_retrain hook"
```

---

### Task 4: Structural tests

**Files:**
- Create: `tests/test_structure.py`
- Create: `strategies.yaml` (minimal, with sma_crossover placeholder)

- [ ] **Step 1: Write structural tests**

```python
# tests/test_structure.py
"""Structural tests — mechanical enforcement of invariants."""
import pytest
import importlib
from pathlib import Path
from causal_edge.config import load_config

CONFIG_PATH = Path(__file__).parent.parent / "strategies.yaml"


@pytest.fixture
def cfg():
    if not CONFIG_PATH.exists():
        pytest.skip("strategies.yaml not found")
    return load_config(str(CONFIG_PATH))


def test_yaml_has_required_fields(cfg):
    """Every strategy must have all required fields."""
    for s in cfg["strategies"]:
        for field in ["id", "name", "asset", "color", "engine", "trade_log"]:
            assert field in s, (
                f"Strategy '{s.get('id', '?')}' missing '{field}'. "
                f"Fix: Add '{field}' to the strategy block in strategies.yaml."
            )


def test_strategy_ids_unique(cfg):
    """No duplicate strategy IDs (would create duplicate HTML element IDs)."""
    ids = [s["id"] for s in cfg["strategies"]]
    dupes = [x for x in ids if ids.count(x) > 1]
    assert not dupes, (
        f"Duplicate strategy IDs: {set(dupes)}. "
        f"Fix: Each strategy must have a unique 'id' in strategies.yaml."
    )


def test_engine_module_importable(cfg):
    """Every engine in YAML must be importable."""
    for s in cfg["strategies"]:
        engine_path = s["engine"]
        try:
            importlib.import_module(engine_path)
        except ImportError:
            pytest.fail(
                f"Cannot import engine '{engine_path}' for strategy '{s['id']}'. "
                f"Fix: Create {engine_path.replace('.', '/')}.py with a StrategyEngine subclass. "
                f"See examples/sma_crossover/engine.py for minimal example."
            )


def test_colors_are_valid_hex(cfg):
    """Strategy colors must be valid hex codes."""
    import re
    for s in cfg["strategies"]:
        assert re.match(r"^#[0-9A-Fa-f]{6}$", s["color"]), (
            f"Strategy '{s['id']}' has invalid color '{s['color']}'. "
            f"Fix: Use a 6-digit hex color like '#FF2D55'."
        )
```

- [ ] **Step 2: Create minimal strategies.yaml**

```yaml
# strategies.yaml — Strategy registry
settings:
  capital: 100000
  port: 8088
  refresh_seconds: 300
  theme: dark

strategies:
  - id: sma_crossover
    name: "SMA Crossover"
    asset: DEMO
    color: "#0A84FF"
    engine: examples.sma_crossover.engine
    trade_log: data/trade_log_sma.csv
    realtime:
      ticker: DEMO
      source: none
    badge: ""
```

- [ ] **Step 3: Create placeholder SMA engine** (so import test passes)

```python
# examples/sma_crossover/__init__.py
# (empty)
```

```python
# examples/sma_crossover/engine.py
"""Minimal SMA crossover strategy — example + integration test."""

import numpy as np
import pandas as pd
from causal_edge.engine.base import StrategyEngine


class SMAEngine(StrategyEngine):
    def compute_signals(self):
        # Generate synthetic price data for demo
        np.random.seed(42)
        dates = pd.bdate_range("2023-01-01", periods=504)
        returns = np.random.randn(504) * 0.01 + 0.0003
        prices = 100 * np.exp(np.cumsum(returns))

        # SMA crossover: long when price > SMA(50)
        sma50 = pd.Series(prices).rolling(50).mean().values
        positions = np.where(
            (prices > sma50) & ~np.isnan(sma50), 1.0, 0.0
        )
        # Shift to avoid look-ahead
        positions = np.roll(positions, 1)
        positions[0] = 0

        pnl = positions * returns
        return pnl, dates, positions, prices

    def get_latest_signal(self):
        pnl, dates, positions, prices = self.compute_signals()
        return {
            "date": dates[-1],
            "position": float(positions[-1]),
            "pnl_today": float(pnl[-1]),
            "sharpe": float(np.mean(pnl) / np.std(pnl, ddof=1) * np.sqrt(252)) if np.std(pnl, ddof=1) > 0 else 0,
            "asset_close": float(prices[-1]),
            "details": {"direction_label": "LONG" if positions[-1] > 0 else "FLAT"},
        }
```

- [ ] **Step 4: Run structural tests**

Run: `python -m pytest tests/test_structure.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add tests/test_structure.py strategies.yaml examples/
git commit -m "feat: structural tests + SMA crossover example engine"
```

---

## Chunk 2: Dashboard Components (port pure functions)

### Task 5: components.py — Port all chart builders

**Files:**
- Create: `causal_edge/dashboard/components.py`
- Create: `tests/test_components.py`

Port from: `~/Claude/New project/paper_trading/dashboard.py` lines 17-218 (see spec Section 18).

- [ ] **Step 1: Write failing tests for core components**

```python
# tests/test_components.py
import numpy as np
import pandas as pd
import json
import pytest

from causal_edge.dashboard.components import (
    compute_metrics, chart_to_json, equity_chart,
    drawdown_chart, rolling_sharpe, monthly_heatmap,
    pnl_distribution, position_chart, daily_pnl_bars,
    yearly_metrics, yearly_table,
)


@pytest.fixture
def sample_data():
    dates = pd.bdate_range("2023-01-01", periods=252)
    pnl = np.random.RandomState(42).randn(252) * 0.01 + 0.001
    positions = np.where(pnl > 0, 0.5, 0.0)
    cum_pnl = np.cumsum(pnl)
    return dates, pnl, positions, cum_pnl


def test_compute_metrics_basic(sample_data):
    _, pnl, _, _ = sample_data
    m = compute_metrics(pnl)
    assert "sharpe" in m
    assert "cum_pnl" in m
    assert "max_dd" in m
    assert "win_rate" in m
    assert m["n_days"] == 252
    assert m["sharpe"] != 0


def test_compute_metrics_empty():
    m = compute_metrics(np.array([]))
    assert m["sharpe"] == 0
    assert m["n_days"] == 0


def test_equity_chart_returns_valid_json(sample_data):
    dates, _, _, cum_pnl = sample_data
    result = equity_chart(dates, cum_pnl, color="#0A84FF")
    parsed = json.loads(result)
    assert "data" in parsed
    assert "layout" in parsed


def test_drawdown_chart_returns_valid_json(sample_data):
    dates, _, _, cum_pnl = sample_data
    result = drawdown_chart(dates, cum_pnl)
    parsed = json.loads(result)
    assert "data" in parsed


def test_rolling_sharpe_returns_valid_json(sample_data):
    dates, pnl, _, _ = sample_data
    result = rolling_sharpe(dates, pnl, window=60)
    parsed = json.loads(result)
    assert "data" in parsed


def test_monthly_heatmap_returns_valid_json(sample_data):
    dates, pnl, _, _ = sample_data
    result = monthly_heatmap(dates, pnl)
    parsed = json.loads(result)
    assert "data" in parsed


def test_position_chart_returns_valid_json(sample_data):
    dates, _, positions, _ = sample_data
    result = position_chart(dates, positions, color="#64D2FF")
    parsed = json.loads(result)
    assert "data" in parsed


def test_pnl_distribution_returns_valid_json(sample_data):
    _, pnl, _, _ = sample_data
    result = pnl_distribution(pnl)
    parsed = json.loads(result)
    assert "data" in parsed


def test_yearly_metrics(sample_data):
    dates, pnl, positions, _ = sample_data
    df = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions})
    yearly = yearly_metrics(df)
    assert len(yearly) >= 1
    assert "year" in yearly[0]
    assert "sharpe" in yearly[0]


def test_yearly_table_returns_html(sample_data):
    dates, pnl, positions, _ = sample_data
    df = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions})
    yearly = yearly_metrics(df)
    html = yearly_table(yearly)
    assert "<table" in html
    assert "2023" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_components.py -v`
Expected: FAIL (import error)

- [ ] **Step 3: Implement components.py**

Port all `_make_*` and `_metrics` functions from `~/Claude/New project/paper_trading/dashboard.py` lines 17-218. Each function becomes a public function that returns a string (JSON or HTML). Read the source file line by line during implementation.

Key mappings:
- `_metrics()` → `compute_metrics()`
- `_chart_json()` → `chart_to_json()`
- `_chart_layout()` → `chart_layout()` (internal helper)
- `_hex_to_rgb()` → `_hex_to_rgb()` (internal helper)
- `_make_equity_chart()` → `equity_chart()` — returns JSON string via `chart_to_json()`
- `_make_daily_pnl_bars()` → `daily_pnl_bars()`
- `_make_drawdown()` → `drawdown_chart()`
- `_make_rolling_sharpe()` → `rolling_sharpe()`
- `_make_position_chart()` → `position_chart()`
- `_make_monthly_heatmap()` → `monthly_heatmap()`
- `_make_pnl_dist()` → `pnl_distribution()`
- `_yearly_metrics()` → `yearly_metrics()`
- `_yearly_html()` → `yearly_table()`

All functions are stateless: data in → string out. No strategy-specific logic.

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_components.py -v`
Expected: All passed

- [ ] **Step 5: Commit**

```bash
git add causal_edge/dashboard/components.py tests/test_components.py
git commit -m "feat: dashboard components — port all chart builders from dashboard.py"
```

---

### Task 6: trade_blotter component (largest function, separate task)

**Files:**
- Modify: `causal_edge/dashboard/components.py` (add `trade_blotter()`)
- Create: `tests/test_trade_blotter.py`

Port from: `~/Claude/New project/paper_trading/dashboard.py` lines 219-438 (`_trade_summary_html`).

- [ ] **Step 1: Write failing test**

```python
# tests/test_trade_blotter.py
import pandas as pd
import numpy as np
from causal_edge.dashboard.components import trade_blotter


def test_trade_blotter_returns_html():
    # Minimal trade ledger
    ledger = pd.DataFrame({
        "strategy": ["test"] * 4,
        "entry_date": pd.to_datetime(["2024-01-01", "2024-01-05", "2024-01-10", "2024-01-15"]),
        "exit_date": pd.to_datetime(["2024-01-03", "2024-01-08", "2024-01-12", "2024-01-18"]),
        "direction": ["Long"] * 4,
        "entry_price": [100, 105, 110, 108],
        "exit_price": [103, 102, 115, 112],
        "gross_pnl": [0.03, -0.03, 0.05, 0.04],
        "net_pnl": [0.029, -0.031, 0.049, 0.039],
        "hold_days": [2, 3, 2, 3],
    })
    html = trade_blotter("test", ledger)
    assert "<table" in html or "<div" in html
    assert "Win Rate" in html


def test_trade_blotter_empty():
    ledger = pd.DataFrame()
    html = trade_blotter("test", ledger)
    assert "No trades" in html or html == ""
```

- [ ] **Step 2-4: Implement + test + commit**

Port `_trade_summary_html` (~220 lines). This is the largest single function.
Keep the HTML structure identical to current output.

```bash
git commit -m "feat: trade_blotter component — port from _trade_summary_html"
```

---

## Chunk 3: Jinja2 Templates + Generator

### Task 7: base.html template

**Files:**
- Create: `causal_edge/dashboard/templates/base.html`

- [ ] **Step 1: Extract CSS from dashboard.py**

Read `~/Claude/New project/paper_trading/dashboard.py` lines ~1250-1600 (the CSS block in the f-string). Port to base.html `<style>` section.

- [ ] **Step 2: Create base.html with Jinja2 loops**

Key structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Causal Edge — Paper Trading</title>
  <script src="https://cdn.plot.ly/plotly-3.4.0.min.js"></script>
  <style>/* ... ported CSS ... */</style>
</head>
<body>
  <div class="header">
    <div class="header-left">
      <div class="logo"><div class="logo-dot"></div> Causal Edge</div>
      <div class="tabs">
        <button class="tab active" onclick="switchTab('overview')">Overview</button>
        {% for s in strategies %}
        <button class="tab" onclick="switchTab('{{s.id}}live')">{{s.name}} Live</button>
        <button class="tab" onclick="switchTab('{{s.id}}bt')">{{s.name}} BT</button>
        {% endfor %}
      </div>
    </div>
  </div>

  {% include 'overview.html' %}
  {% for s in strategies %}
    {% include 'backtest.html' %}
    {% include 'live.html' %}
  {% endfor %}

  <!-- JS: tab switching, real-time, chart rendering -->
  <script>/* ... ported JS ... */</script>
</body>
</html>
```

- [ ] **Step 3: Port JS** from dashboard.py lines ~4176-4587 (switchTab, countdown, renderChart, staticCharts, initLiveTab, fetchPrices). Make all strategy references use template variables.

- [ ] **Step 4: Commit**

```bash
git add causal_edge/dashboard/templates/base.html
git commit -m "feat: base.html template with CSS + JS + Jinja2 strategy loops"
```

---

### Task 8: overview.html template

**Files:**
- Create: `causal_edge/dashboard/templates/overview.html`

- [ ] **Step 1: Create overview template**

Port from dashboard.py `_overview_html` (lines 714-783) + overview section of `generate_dashboard`. Make strategy rows iterate via `{% for s in strategies %}`.

```html
<div id="tab-overview" class="tab-content active">
  <!-- Market status banner -->
  <div class="panel" style="...">
    <span class="status-dot"></span> Market Active
    <span>Data through {{latest_date}}</span>
  </div>

  <!-- Real-time price bar -->
  <div class="panel hero">
    <div class="hero-title">ETH $<span id="live-eth-price">--</span></div>
    <div>
      {% for s in strategies %}
      <span style="color:{{s.color}}">{{s.name}}: <span id="rt-overview-{{s.id}}">--</span></span>
      {% endfor %}
    </div>
  </div>

  <!-- Strategy table -->
  <table class="overview-table">
    <thead>
      <tr><th>Strategy</th><th>Position</th><th>Action</th><th>Today</th>
          <th>Cumulative</th><th>Sharpe</th><th>MaxDD</th><th>Last 7d</th></tr>
    </thead>
    <tbody>
    {% for s in strategies %}
      <tr>
        <td style="color:{{s.color}};font-weight:600">{{s.name}}</td>
        <td>{{s.overview.position_label}}</td>
        <td>{{s.overview.action_label}}</td>
        <td>{{s.overview.today_pnl}}</td>
        <td>{{s.overview.cum_pnl}}</td>
        <td>{{s.overview.sharpe}}</td>
        <td>{{s.overview.maxdd}}</td>
        <td>{{s.overview.last_7d_html}}</td>
      </tr>
    {% endfor %}
    </tbody>
  </table>
</div>
```

- [ ] **Step 2: Commit**

```bash
git add causal_edge/dashboard/templates/overview.html
git commit -m "feat: overview.html template with strategy loop"
```

---

### Task 9: backtest.html + live.html templates

**Files:**
- Create: `causal_edge/dashboard/templates/backtest.html`
- Create: `causal_edge/dashboard/templates/live.html`

- [ ] **Step 1: Create backtest.html** (port from BT tab HTML in dashboard.py)

One template, rendered per strategy inside `{% for s in strategies %}` loop in base.html:

```html
<div id="tab-{{s.id}}bt" class="tab-content">
  <div class="section-label">{{s.name}} — Backtest</div>

  <div class="metrics">
    {{ metrics_cards(s.bt.metrics, s.color) }}
  </div>

  <div class="chart-card"><div id="c-{{s.id}}bt-equity"></div></div>
  <div class="row-2">
    <div class="chart-card"><div id="c-{{s.id}}bt-pos"></div></div>
    <div class="chart-card"><div id="c-{{s.id}}bt-dd"></div></div>
  </div>
  <div class="row-2">
    <div class="chart-card"><div id="c-{{s.id}}bt-sharpe"></div></div>
    <div class="chart-card"><div id="c-{{s.id}}bt-dist"></div></div>
  </div>
  <div class="chart-card"><div id="c-{{s.id}}bt-monthly-wrap"></div></div>

  {{ yearly_table(s.bt.yearly) }}
  {{ trade_blotter(s.id, s.bt.trades) }}

  <!-- Chart data (embedded JSON) -->
  <script type="application/json" id="d-{{s.id}}bt-equity">{{s.bt.charts.equity}}</script>
  <script type="application/json" id="d-{{s.id}}bt-pos">{{s.bt.charts.position}}</script>
  <script type="application/json" id="d-{{s.id}}bt-dd">{{s.bt.charts.drawdown}}</script>
  <script type="application/json" id="d-{{s.id}}bt-sharpe">{{s.bt.charts.rolling_sharpe}}</script>
  <script type="application/json" id="d-{{s.id}}bt-dist">{{s.bt.charts.distribution}}</script>
  <script type="application/json" id="d-{{s.id}}bt-monthly">{{s.bt.charts.monthly}}</script>
</div>
```

- [ ] **Step 2: Create live.html** (same pattern, with hero section + live trades table)

- [ ] **Step 3: Commit**

```bash
git add causal_edge/dashboard/templates/backtest.html causal_edge/dashboard/templates/live.html
git commit -m "feat: backtest.html + live.html templates"
```

---

### Task 10: generator.py — Wire everything together

**Files:**
- Create: `causal_edge/dashboard/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_generator.py
from pathlib import Path
import pandas as pd
import numpy as np

def test_generate_produces_html(tmp_path):
    # Create minimal trade log
    dates = pd.bdate_range("2024-01-01", periods=252)
    df = pd.DataFrame({
        "date": dates, "position": np.random.choice([0, 0.5], 252),
        "pnl": np.random.randn(252) * 0.01,
        "cum_pnl": np.cumsum(np.random.randn(252) * 0.01),
        "asset_close": np.linspace(100, 120, 252),
        "source": "backfill",
    })
    log_path = tmp_path / "trade_log.csv"
    df.to_csv(log_path, index=False)

    # Create strategies.yaml
    yaml_content = f"""
settings:
  capital: 100000
strategies:
  - id: test
    name: "Test Strategy"
    asset: DEMO
    color: "#0A84FF"
    engine: examples.sma_crossover.engine
    trade_log: {log_path}
"""
    cfg_path = tmp_path / "strategies.yaml"
    cfg_path.write_text(yaml_content)

    from causal_edge.dashboard.generator import generate
    out_path = tmp_path / "dashboard.html"
    generate(str(cfg_path), str(out_path))

    html = out_path.read_text()
    assert "Test Strategy" in html
    assert "tab-testbt" in html
    assert "tab-testlive" in html
    assert "Plotly" in html or "plotly" in html
```

- [ ] **Step 2: Implement generator.py**

Port data preparation logic from dashboard.py `generate_dashboard()` (lines 784-2835).
Key functions: `prepare_backtest()`, `prepare_live()`, `prepare_overview()`, `generate()`.

Each `prepare_*` function computes metrics + chart JSON using components.py, returns a dict consumed by templates.

- [ ] **Step 3: Run tests**

Run: `python -m pytest tests/test_generator.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add causal_edge/dashboard/generator.py tests/test_generator.py
git commit -m "feat: generator.py — wire config + components + templates → dashboard.html"
```

---

## Chunk 4: Server + CLI + Strategy Wrappers + Integration

### Task 11: server.py

**Files:**
- Create: `causal_edge/dashboard/server.py`

- [ ] **Step 1: Port from existing** `~/Claude/New project/paper_trading/server.py` (255 lines). Already clean — just copy and adjust imports.

- [ ] **Step 2: Commit**

```bash
git add causal_edge/dashboard/server.py
git commit -m "feat: dashboard server with ThreadingMixIn + /api/prices"
```

---

### Task 12: cli.py

**Files:**
- Create: `causal_edge/cli.py`

- [ ] **Step 1: Implement CLI with Click**

```python
# causal_edge/cli.py
import click
from pathlib import Path

@click.group()
def main():
    """causal-edge: Paper trading dashboard framework."""
    pass

@main.command()
@click.option("--port", default=8088)
@click.option("--config", default="strategies.yaml")
def dashboard(port, config):
    """Generate dashboard and start server."""
    from causal_edge.dashboard.generator import generate
    from causal_edge.dashboard.server import main as serve
    generate(config)
    serve(port)

@main.command()
@click.option("--config", default="strategies.yaml")
@click.option("--strategy", default=None)
def backfill(config, strategy):
    """Run backfill for all or one strategy."""
    from causal_edge.engine.trader import run_backfill
    run_backfill(config, strategy_filter=strategy)

@main.command()
@click.option("--config", default="strategies.yaml")
def update(config):
    """Daily signal update (cron target)."""
    from causal_edge.engine.trader import daily_update
    daily_update(config)

@main.command()
@click.option("--config", default="strategies.yaml")
def status(config):
    """Show strategy status."""
    from causal_edge.engine.trader import show_status
    show_status(config)

@main.command()
@click.argument("engine_path")
@click.option("--config", default="strategies.yaml")
@click.option("--force", is_flag=True)
def add(engine_path, config, force):
    """Add a strategy (runs abel-proof gate)."""
    from causal_edge.validation.gate import add_strategy
    add_strategy(engine_path, config, force=force)
```

- [ ] **Step 2: Commit**

```bash
git add causal_edge/cli.py
git commit -m "feat: CLI with dashboard/backfill/update/status/add commands"
```

---

### Task 13: Strategy wrappers (production engines)

**Files:**
- Create: `strategies/alpha_max/engine.py`
- Create: `strategies/eight_comp/engine.py`
- Create: `strategies/ton/engine.py`
- Create: `strategies/meta/engine.py`
- Create: `strategies/aapl/engine.py`
- Create: `strategies/*/\__init__.py` for each

- [ ] **Step 1: Create thin ABC wrappers for each engine**

Each wrapper imports from `~/Claude/New project/paper_trading/*_engine.py` and delegates.
Example pattern (Alpha Max):

```python
# strategies/alpha_max/engine.py
import sys, os
sys.path.insert(0, os.path.expanduser("~/Claude/New project"))

from causal_edge.engine.base import StrategyEngine

class AlphaMaxEngine(StrategyEngine):
    def compute_signals(self):
        from paper_trading.alpha_max_engine import compute_alpha_max_signals
        pnl, dates, positions, px, _ = compute_alpha_max_signals()
        return pnl, dates, positions, px

    def get_latest_signal(self):
        pnl, dates, positions, px = self.compute_signals()
        return {
            "date": dates[-1], "position": float(positions[-1]),
            "pnl_today": float(pnl[-1]),
            "sharpe": float(__import__('numpy').mean(pnl) / __import__('numpy').std(pnl, ddof=1) * (252**0.5)),
            "asset_close": float(px[-1]),
            "details": {},
        }
```

Repeat for each of the 5 strategies, adapting imports.

- [ ] **Step 2: Update strategies.yaml with all 5 production strategies**

- [ ] **Step 3: Run structural tests**

Run: `python -m pytest tests/test_structure.py -v`
Expected: All pass (all engines importable, all fields present)

- [ ] **Step 4: Commit**

```bash
git add strategies/
git commit -m "feat: production strategy wrappers (alpha_max, 8comp, ton, meta, aapl)"
```

---

### Task 14: End-to-end integration test

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write E2E test**

```python
# tests/test_e2e.py
"""End-to-end: SMA crossover → backfill → generate dashboard → verify HTML."""
from pathlib import Path
import subprocess

def test_sma_crossover_e2e(tmp_path):
    # 1. Backfill SMA strategy
    from examples.sma_crossover.engine import SMAEngine
    engine = SMAEngine()
    pnl, dates, positions, prices = engine.compute_signals()

    # 2. Write trade log
    import pandas as pd, numpy as np
    df = pd.DataFrame({
        "date": dates, "position": positions, "pnl": pnl,
        "cum_pnl": np.cumsum(pnl), "asset_close": prices, "source": "backfill",
    })
    log_path = tmp_path / "trade_log_sma.csv"
    df.to_csv(log_path, index=False)

    # 3. Generate dashboard
    cfg = f"""
settings:
  capital: 100000
strategies:
  - id: sma_crossover
    name: "SMA Crossover"
    asset: DEMO
    color: "#0A84FF"
    engine: examples.sma_crossover.engine
    trade_log: {log_path}
"""
    cfg_path = tmp_path / "strategies.yaml"
    cfg_path.write_text(cfg)

    from causal_edge.dashboard.generator import generate
    out = tmp_path / "dashboard.html"
    generate(str(cfg_path), str(out))

    # 4. Verify HTML
    html = out.read_text()
    assert "SMA Crossover" in html
    assert "tab-sma_crossoverbt" in html
    assert len(html) > 10000  # Not trivially small
```

- [ ] **Step 2: Run E2E**

Run: `python -m pytest tests/test_e2e.py -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end integration test with SMA crossover"
```

---

### Task 15: AGENTS.md hierarchy + docs

**Files:**
- Create: `causal_edge/engine/AGENTS.md`
- Create: `causal_edge/dashboard/AGENTS.md`
- Create: `strategies/AGENTS.md`
- Create: `docs/add-strategy.md`

- [ ] **Step 1: Create subsystem AGENTS.md files** (~30 lines each)

- [ ] **Step 2: Create add-strategy.md guide**

```markdown
# How to Add a Strategy

1. Create `strategies/my_strategy/engine.py` implementing `StrategyEngine`
2. Run `causal-edge add strategies.my_strategy.engine`
3. Abel-proof validates → badge added to strategies.yaml
4. Run `causal-edge backfill --strategy my_strategy`
5. Run `causal-edge dashboard` → your strategy appears automatically
```

- [ ] **Step 3: Commit**

```bash
git add causal_edge/engine/AGENTS.md causal_edge/dashboard/AGENTS.md
git add strategies/AGENTS.md docs/add-strategy.md
git commit -m "docs: AGENTS.md hierarchy + add-strategy guide"
```

---

### Task 16: Visual regression capture + final verification

**Files:**
- Create: `tests/reference_dashboard.png` (captured from current system)

- [ ] **Step 1: Capture reference screenshot**

Use Playwright to screenshot current dashboard at `http://192.168.0.7:8088/dashboard.html` as reference.

- [ ] **Step 2: Generate new dashboard with all 5 strategies**

Run: `python -m causal_edge.cli dashboard --config strategies.yaml`

- [ ] **Step 3: Visual comparison**

Compare side by side. Tabs, charts, metrics, overview table should match structurally.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "v0.1.0: causal-edge framework — complete migration from monolith"
```
