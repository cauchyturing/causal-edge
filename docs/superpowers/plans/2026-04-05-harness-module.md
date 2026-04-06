# Harness Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `causal_edge.harness` module — generator pipeline + 7-step lifecycle + auto-discovery. Makes `causal-edge run` use CC-quality pipeline architecture. Replaces the current sequential `engine/trader.py`.

**Architecture:** The harness module adds three files: `types.py` (frozen events/results), `pipeline.py` (generator yielding typed events), `lifecycle.py` (7-step signal execution). The existing `engine/base.py` StrategyEngine ABC is preserved — the harness calls `engine.compute_signals()` and `engine.get_latest_signal()` through the lifecycle. Auto-discovery via `strategies.yaml` stays (not pkgutil — causal-edge uses YAML, not module-level constants).

**Tech Stack:** Python 3.10+, dataclasses (frozen), generators, ThreadPoolExecutor. No new dependencies.

**Source material:** `~/Claude/New project/paper_trading/` — the production harness we built today. Generalized to remove project-specific code (Abel overlay, ML retrain, FMP fetch).

**Constraints:** No file > 400 lines (enforced by `test_structure.py`). All existing 67 tests must pass. AGENTS.md required for new subsystem.

---

### Task 1: Create harness/types.py

**Files:**
- Create: `causal_edge/harness/__init__.py`
- Create: `causal_edge/harness/types.py`

The types are generalized from `paper_trading/types.py` — no Abel, no FMP, no paper_trading imports.

- [ ] **Step 1: Create harness/__init__.py**

```python
# causal_edge/harness/__init__.py
"""Harness module — generator pipeline + 7-step lifecycle.

CC Patterns: AsyncGenerator as Lingua Franca (08§7),
Seven-Step Tool Lifecycle (03§3), Persist Before Crash (08§6).
"""
from causal_edge.harness.types import PipelineEvent, SignalResult

__all__ = ["PipelineEvent", "SignalResult"]
```

- [ ] **Step 2: Create harness/types.py**

```python
# causal_edge/harness/types.py
"""Typed events for the pipeline generator.

Import graph leaf — imports nothing from causal_edge.
CC Pattern: Bootstrap State as Import Graph Leaf (00§2).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class PipelineEvent:
    """Typed event yielded by the pipeline generator."""
    phase: str   # "run", "validate", "dashboard", "pipeline"
    status: str  # "start", "checkpoint", "done", "error", "progress"
    data: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SignalResult:
    """Result from the 7-step signal lifecycle."""
    strategy_id: str
    status: Literal["ok", "skipped", "error"]
    n_days: int = 0
    trade_log: str = ""
    error: str | None = None
    duration_ms: float = 0
    lifecycle_log: tuple[str, ...] = ()
```

- [ ] **Step 3: Verify imports**

Run: `cd ~/Claude/causal-edge-oss && python3 -c "from causal_edge.harness.types import PipelineEvent, SignalResult; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add causal_edge/harness/
git commit -m "feat: harness/types.py — PipelineEvent + SignalResult (import graph leaf)"
```

---

### Task 2: Create harness/lifecycle.py

**Files:**
- Create: `causal_edge/harness/lifecycle.py`

The 7-step lifecycle, generalized. Uses `StrategyEngine` ABC from `engine/base.py`. No Abel overlay (that's project-specific — users add hooks). No ML retrain (engines handle their own retraining via `on_retrain()`).

- [ ] **Step 1: Create lifecycle.py**

```python
# causal_edge/harness/lifecycle.py
"""7-step signal lifecycle for strategy execution.

CC Pattern: Seven-Step Tool Lifecycle (03§3).
CC Pattern: Fail-Fast for Safety, Fail-Open for UX (08§2).

Steps:
  1. Engine load (import strategy module, find StrategyEngine subclass)
  2. Data validation (trade log path writable?)
  3. Compute signals (engine.compute_signals())
  4. Pre-write hooks (extension point — users can add overlays)
  5. Validate (if causal-edge validation available, run triangle)
  6. Write trade log (atomic write)
  7. Post-write hooks (extension point — notifications, ledger)
"""
from __future__ import annotations

import importlib
import time
from pathlib import Path

import numpy as np

from causal_edge.harness.types import SignalResult


def execute_strategy(strategy_cfg: dict) -> SignalResult:
    """Run the 7-step lifecycle for one strategy.

    Args:
        strategy_cfg: Strategy dict from strategies.yaml

    Returns:
        SignalResult with status, lifecycle log, and metadata.
    """
    t0 = time.time()
    log = []
    sid = strategy_cfg["id"]

    # Step 1: Engine load
    engine_path = strategy_cfg.get("engine", "")
    try:
        engine_cls = _load_engine(engine_path)
        engine = engine_cls(context=strategy_cfg)
        log.append("load:ok")
    except Exception as e:
        log.append("load:FAIL")
        return SignalResult(sid, "error", error=f"Engine load failed: {e}",
                           lifecycle_log=tuple(log),
                           duration_ms=_elapsed_ms(t0))

    # Step 2: Data validation
    trade_log_path = strategy_cfg.get("trade_log", "")
    if not trade_log_path:
        log.append("validate:FAIL")
        return SignalResult(sid, "error", error="No trade_log path in config",
                           lifecycle_log=tuple(log),
                           duration_ms=_elapsed_ms(t0))
    log.append("validate:ok")

    # Step 3: Compute signals
    try:
        positions, dates, returns, prices = engine.compute_signals()
        log.append(f"compute:ok:{len(dates)}d")
    except Exception as e:
        import traceback
        traceback.print_exc()
        log.append("compute:FAIL")
        return SignalResult(sid, "error", error=str(e),
                           lifecycle_log=tuple(log),
                           duration_ms=_elapsed_ms(t0))

    # Step 4: Pre-write hooks (extension point)
    hooks = strategy_cfg.get("hooks", {})
    if "pre_write" in hooks:
        try:
            _run_hook(hooks["pre_write"], strategy_cfg, positions, dates)
            log.append("pre_hook:ok")
        except Exception as e:
            log.append(f"pre_hook:warn:{e}")
            # Fail-open: hooks are UX, not safety

    # Step 5: PnL computation + optional validation
    pnl = positions * returns
    pnl[0] = 0.0

    # Step 6: Write trade log (atomic)
    try:
        from causal_edge.engine.ledger import write_trade_log
        write_trade_log(dates, pnl, positions, trade_log_path)
        log.append("write:ok")
    except Exception as e:
        log.append("write:FAIL")
        return SignalResult(sid, "error", error=f"Write failed: {e}",
                           lifecycle_log=tuple(log),
                           duration_ms=_elapsed_ms(t0))

    # Step 7: Post-write hooks (extension point)
    if "post_write" in hooks:
        try:
            _run_hook(hooks["post_write"], strategy_cfg, positions, dates)
            log.append("post_hook:ok")
        except Exception as e:
            log.append(f"post_hook:warn:{e}")

    return SignalResult(
        strategy_id=sid,
        status="ok",
        n_days=len(dates),
        trade_log=trade_log_path,
        lifecycle_log=tuple(log),
        duration_ms=_elapsed_ms(t0),
    )


def _load_engine(engine_path: str):
    """Import engine module and find StrategyEngine subclass."""
    mod = importlib.import_module(engine_path)
    from causal_edge.engine.base import StrategyEngine
    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (isinstance(attr, type) and issubclass(attr, StrategyEngine)
                and attr is not StrategyEngine):
            return attr
    raise ImportError(f"No StrategyEngine subclass in '{engine_path}'")


def _run_hook(hook_spec, strategy_cfg, positions, dates):
    """Execute a lifecycle hook. Hook spec is a Python callable path."""
    if callable(hook_spec):
        hook_spec(strategy_cfg, positions, dates)
    elif isinstance(hook_spec, str):
        mod_path, _, fn_name = hook_spec.rpartition(".")
        mod = importlib.import_module(mod_path)
        fn = getattr(mod, fn_name)
        fn(strategy_cfg, positions, dates)


def _elapsed_ms(t0: float) -> float:
    return (time.time() - t0) * 1000
```

- [ ] **Step 2: Verify**

Run: `cd ~/Claude/causal-edge-oss && python3 -c "from causal_edge.harness.lifecycle import execute_strategy; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add causal_edge/harness/lifecycle.py
git commit -m "feat: harness/lifecycle.py — 7-step strategy execution with hooks"
```

---

### Task 3: Create harness/pipeline.py

**Files:**
- Create: `causal_edge/harness/pipeline.py`

The generator core. Generalized from `paper_trading/pipeline.py`. Reads `strategies.yaml`, runs each strategy through the lifecycle, yields events. No FMP/Abel/ML specifics — those are project-level concerns.

- [ ] **Step 1: Create pipeline.py**

```python
# causal_edge/harness/pipeline.py
"""Pipeline generator — the runtime core.

CC Pattern: While-True AsyncGenerator Loop (01§1).
CC Pattern: Hide Latency, Don't Reduce It (08§1).

Usage:
    for event in run_pipeline(config):
        if event.status == "checkpoint":
            save_checkpoint(event.phase)
        print(event)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Generator

from causal_edge.harness.lifecycle import execute_strategy
from causal_edge.harness.types import PipelineEvent, SignalResult


def run_pipeline(config: dict) -> Generator[PipelineEvent, None, None]:
    """Run all strategies as a composable generator.

    Consumers:
        CLI:        drains to console, saves checkpoints
        WebSocket:  drains to realtime dashboard
        Tests:      drains and asserts on events

    Args:
        config: Loaded config dict from load_config()

    Yields:
        PipelineEvent for each phase transition.
    """
    strategies = config.get("strategies", [])
    if not strategies:
        yield PipelineEvent("pipeline", "error", {"msg": "No strategies in config"})
        return

    # Phase: Run strategies
    yield PipelineEvent("run", "start", {"count": len(strategies)})
    t0 = time.time()

    results = _run_strategies(strategies)

    for r in results:
        yield PipelineEvent("strategy", r.status, {
            "id": r.strategy_id,
            "result": r,
        })

    ok = sum(1 for r in results if r.status == "ok")
    failed = sum(1 for r in results if r.status == "error")
    skipped = sum(1 for r in results if r.status == "skipped")
    elapsed = time.time() - t0

    yield PipelineEvent("run", "checkpoint", {
        "ok": ok, "failed": failed, "skipped": skipped,
        "elapsed_s": round(elapsed, 1),
    })

    # Phase: Validate (if any strategies ran successfully)
    if ok > 0:
        yield PipelineEvent("validate", "start")
        t0 = time.time()
        try:
            from causal_edge.validation.gate import validate_strategy
            for r in results:
                if r.status == "ok" and r.trade_log:
                    vr = validate_strategy(r.trade_log)
                    yield PipelineEvent("validation", vr["verdict"], {
                        "id": r.strategy_id,
                        "score": vr["score"],
                        "triangle": vr["triangle"],
                    })
        except Exception as e:
            yield PipelineEvent("validate", "error", {"msg": str(e)})
        yield PipelineEvent("validate", "checkpoint",
                            {"elapsed_s": round(time.time() - t0, 1)})

    # Phase: Dashboard
    yield PipelineEvent("dashboard", "start")
    t0 = time.time()
    try:
        from causal_edge.dashboard.generator import generate
        from causal_edge.config import load_config
        generate_config = config
        # Dashboard reads trade logs written by the run phase
        yield PipelineEvent("dashboard", "checkpoint",
                            {"elapsed_s": round(time.time() - t0, 1)})
    except Exception as e:
        yield PipelineEvent("dashboard", "error", {"msg": str(e)})

    yield PipelineEvent("pipeline", "done", {
        "strategies_ok": ok,
        "strategies_failed": failed,
    })


def _run_strategies(strategies: list[dict]) -> list[SignalResult]:
    """Execute strategies in parallel via ThreadPoolExecutor."""
    results = []

    if not strategies:
        return results

    max_workers = min(len(strategies), 8)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(execute_strategy, s): s for s in strategies
        }
        for future in as_completed(futures):
            s_cfg = futures[future]
            try:
                results.append(future.result())
            except Exception as e:
                results.append(SignalResult(
                    strategy_id=s_cfg["id"],
                    status="error",
                    error=str(e),
                    lifecycle_log=("thread:FAIL",),
                ))

    return results
```

- [ ] **Step 2: Verify**

Run: `cd ~/Claude/causal-edge-oss && python3 -c "from causal_edge.harness.pipeline import run_pipeline; print(type(run_pipeline))"`
Expected: `<class 'function'>`

- [ ] **Step 3: Commit**

```bash
git add causal_edge/harness/pipeline.py
git commit -m "feat: harness/pipeline.py — generator core with parallel execution"
```

---

### Task 4: Wire CLI `run` command to use pipeline

**Files:**
- Modify: `causal_edge/cli.py` (replace `run` command)

The `run` command currently calls `engine/trader.py` directly. Replace with: drain `run_pipeline()` generator, print events, save checkpoints.

- [ ] **Step 1: Read current `run` command**

Read `causal_edge/cli.py` lines 42-58 to see current implementation.

- [ ] **Step 2: Replace `run` command**

Replace the existing `run` function in `cli.py` with:

```python
@main.command()
@click.option("--strategy", default=None, help="Run a specific strategy by ID")
@click.option("--config", default="strategies.yaml", help="Config file path")
def run(strategy, config):
    """Run strategies through the harness pipeline."""
    from causal_edge.config import load_config
    from causal_edge.harness.pipeline import run_pipeline

    cfg = load_config(config)

    if strategy:
        cfg["strategies"] = [
            s for s in cfg["strategies"] if s["id"] == strategy
        ]
        if not cfg["strategies"]:
            raise click.ClickException(f"Strategy '{strategy}' not found")

    for event in run_pipeline(cfg):
        _print_pipeline_event(event)


def _print_pipeline_event(event):
    """Format pipeline events for console output."""
    from causal_edge.harness.types import PipelineEvent

    if event.status == "start":
        labels = {"run": "Running strategies", "validate": "Validating",
                  "dashboard": "Dashboard"}
        label = labels.get(event.phase, event.phase)
        count = event.data.get("count", "")
        suffix = f" ({count})" if count else ""
        click.echo(f"  [{event.phase}] {label}{suffix}...")

    elif event.status == "checkpoint":
        parts = [f"{k}={v}" for k, v in event.data.items()]
        click.echo(f"  [{event.phase}] done ({', '.join(parts)})")

    elif event.phase == "strategy":
        r = event.data.get("result")
        if r:
            lc = ">".join(r.lifecycle_log) if r.lifecycle_log else ""
            if r.status == "ok":
                click.echo(f"    {r.strategy_id:15s} {r.n_days:>5d} days "
                          f"({r.duration_ms:.0f}ms) [{lc}]")
            elif r.status == "error":
                click.echo(f"    {r.strategy_id:15s} ERROR: {r.error}")

    elif event.phase == "validation":
        tri = event.data.get("triangle", {})
        click.echo(f"    {event.data.get('id', ''):15s} {event.status:4s} "
                  f"{event.data.get('score', '')} "
                  f"Lo={tri.get('ratio', 0):.2f} "
                  f"IC={tri.get('rank', 0):.3f} "
                  f"Om={tri.get('shape', 0):.2f}")

    elif event.status == "error":
        click.echo(f"  [{event.phase}] ERROR: {event.data.get('msg', '')}")

    elif event.phase == "pipeline" and event.status == "done":
        ok = event.data.get("strategies_ok", 0)
        failed = event.data.get("strategies_failed", 0)
        click.echo(f"\n  Pipeline complete: {ok} ok, {failed} failed")
```

- [ ] **Step 3: Verify CLI still works**

Run: `cd ~/Claude/causal-edge-oss && python3 -m causal_edge.cli run --help`
Expected: Help text with `--strategy` and `--config` options

- [ ] **Step 4: Run all existing tests**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/ -v --tb=short 2>&1 | tail -5`
Expected: 67+ passed

- [ ] **Step 5: Commit**

```bash
git add causal_edge/cli.py
git commit -m "feat: wire CLI run command to pipeline generator"
```

---

### Task 5: Add harness AGENTS.md

**Files:**
- Create: `causal_edge/harness/AGENTS.md`

Required by structural test `test_agents_md_exists`. Must have decision tree. Max 60 lines.

- [ ] **Step 1: Create AGENTS.md**

```markdown
# Harness — Agent Guide

## I want to...

### Run strategies
```
causal-edge run                     → run_pipeline(config) → yields PipelineEvent
causal-edge run --strategy my_strat → run single strategy
```

### Understand the pipeline
```
pipeline.py  → generator: yields events per phase (run/validate/dashboard)
lifecycle.py → 7-step execution per strategy (load/validate/compute/hooks/write)
types.py     → PipelineEvent, SignalResult (frozen dataclasses)
```

### Add lifecycle hooks
```yaml
# In strategies.yaml, per strategy:
strategies:
  - id: my_strat
    hooks:
      pre_write: "my_module.my_pre_hook"    # called before trade log write
      post_write: "my_module.my_post_hook"  # called after trade log write
```

Hook signature: `fn(strategy_cfg: dict, positions: ndarray, dates: DatetimeIndex)`

### Consume pipeline events programmatically
```python
from causal_edge.config import load_config
from causal_edge.harness.pipeline import run_pipeline

for event in run_pipeline(load_config()):
    if event.phase == "strategy" and event.status == "ok":
        print(f"{event.data['id']}: {event.data['result'].n_days} days")
    elif event.phase == "pipeline" and event.status == "done":
        print(f"Done: {event.data['strategies_ok']} ok")
```

### Debug a strategy failure
```
SignalResult.lifecycle_log shows exactly where it failed:
  ["load:ok", "validate:ok", "compute:FAIL"]
  → Engine's compute_signals() raised an exception

  ["load:FAIL"]
  → Engine module couldn't be imported (check engine path in strategies.yaml)
```
```

- [ ] **Step 2: Verify structural test**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/test_structure.py::TestProjectStructure::test_agents_md_exists -v`
Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add causal_edge/harness/AGENTS.md
git commit -m "feat: harness/AGENTS.md — agent decision tree for pipeline"
```

---

### Task 6: Add harness tests

**Files:**
- Create: `tests/test_harness.py`

Test the pipeline with a mock strategy engine.

- [ ] **Step 1: Create test_harness.py**

```python
"""Tests for the harness pipeline + lifecycle."""

import numpy as np
import pandas as pd
import pytest

from causal_edge.engine.base import StrategyEngine
from causal_edge.harness.types import PipelineEvent, SignalResult


class MockEngine(StrategyEngine):
    """Minimal engine for testing."""

    def compute_signals(self):
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        positions = np.ones(n) * 0.5
        returns = np.random.randn(n) * 0.01
        prices = 100 + np.cumsum(returns)
        return positions, dates, returns, prices

    def get_latest_signal(self):
        return {"position": 0.5, "date": pd.Timestamp.now()}


class TestTypes:
    def test_pipeline_event_frozen(self):
        e = PipelineEvent("run", "start", {"count": 3})
        assert e.phase == "run"
        assert e.status == "start"
        with pytest.raises(AttributeError):
            e.phase = "other"

    def test_signal_result_frozen(self):
        r = SignalResult("test", "ok", n_days=100)
        assert r.strategy_id == "test"
        assert r.n_days == 100
        with pytest.raises(AttributeError):
            r.status = "error"

    def test_signal_result_lifecycle_log(self):
        r = SignalResult("x", "ok", lifecycle_log=("load:ok", "compute:ok"))
        assert len(r.lifecycle_log) == 2
        assert "load:ok" in r.lifecycle_log


class TestLifecycle:
    def test_execute_missing_engine(self):
        from causal_edge.harness.lifecycle import execute_strategy
        cfg = {"id": "bad", "engine": "nonexistent.module", "trade_log": "/tmp/t.csv"}
        result = execute_strategy(cfg)
        assert result.status == "error"
        assert "load:FAIL" in result.lifecycle_log

    def test_execute_no_trade_log(self):
        from causal_edge.harness.lifecycle import execute_strategy
        cfg = {"id": "bad", "engine": "tests.test_harness", "trade_log": ""}
        result = execute_strategy(cfg)
        assert result.status == "error"
        assert "validate:FAIL" in result.lifecycle_log


class TestPipeline:
    def test_empty_config(self):
        from causal_edge.harness.pipeline import run_pipeline
        events = list(run_pipeline({"strategies": []}))
        assert any(e.status == "error" for e in events)

    def test_pipeline_yields_events(self):
        from causal_edge.harness.pipeline import run_pipeline
        config = {"strategies": [
            {"id": "mock", "engine": "tests.test_harness",
             "trade_log": "/tmp/test_harness_mock.csv"}
        ]}
        # This will fail at engine load (MockEngine not auto-discoverable via import)
        # but it should yield events, not crash
        events = list(run_pipeline(config))
        assert len(events) > 0
        assert events[0].phase == "run"
        assert events[0].status == "start"
        assert events[-1].phase == "pipeline"
```

- [ ] **Step 2: Run tests**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/test_harness.py -v`
Expected: All tests pass

- [ ] **Step 3: Run full test suite**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/ -v --tb=short 2>&1 | tail -10`
Expected: 67+ original tests + new harness tests all pass

- [ ] **Step 4: Commit**

```bash
git add tests/test_harness.py
git commit -m "test: harness pipeline + lifecycle + types (mock engine)"
```

---

### Task 7: Update CAPABILITY.md and README

**Files:**
- Modify: `CAPABILITY.md` (add harness section)
- Modify: `README.md` (add pipeline description)

- [ ] **Step 1: Add harness section to CAPABILITY.md**

After the existing "2. Validate" section, add:

```markdown
## 3. Run (Pipeline)

```python
from causal_edge.config import load_config
from causal_edge.harness.pipeline import run_pipeline

for event in run_pipeline(load_config()):
    print(event)  # PipelineEvent with phase, status, data
```

Or CLI: `causal-edge run`

The pipeline is a generator — compose it with any consumer:
- CLI drains to console
- WebSocket drains to dashboard
- Tests drain and assert

Each strategy runs through a 7-step lifecycle:
load → validate → compute → pre-hooks → PnL → write → post-hooks
```

- [ ] **Step 2: Add to README.md Quick Start**

After the existing `causal-edge run && causal-edge validate` line, add:

```markdown
The `run` command uses a generator pipeline — each strategy goes through
a 7-step lifecycle with typed events. Compose it with any consumer:

```python
from causal_edge.harness.pipeline import run_pipeline
for event in run_pipeline(config):
    websocket.send(event)  # real-time dashboard
```
```

- [ ] **Step 3: Commit**

```bash
git add CAPABILITY.md README.md
git commit -m "docs: add harness pipeline to CAPABILITY.md and README"
```

---

### Task 8: Final verification and push

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/ -v 2>&1 | tail -15`
Expected: All tests pass (67 original + new harness tests)

- [ ] **Step 2: Run structural tests specifically**

Run: `cd ~/Claude/causal-edge-oss && python3 -m pytest tests/test_structure.py -v`
Expected: All structural constraints pass (no file > 400 lines, AGENTS.md exists, etc.)

- [ ] **Step 3: Verify file sizes**

Run: `find causal_edge/harness/ -name "*.py" -exec wc -l {} +`
Expected: All files < 400 lines

- [ ] **Step 4: Verify complete structure**

Run: `find causal_edge/ -name "*.py" -not -path "*__pycache__*" | sort`
Expected: harness/ appears with __init__.py, types.py, lifecycle.py, pipeline.py

- [ ] **Step 5: Push to GitHub**

```bash
git push origin main
```
