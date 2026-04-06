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

    # Step 5: PnL computation
    try:
        pnl = positions * returns
        pnl[0] = 0.0
        log.append("pnl:ok")
    except Exception as e:
        log.append("pnl:FAIL")
        return SignalResult(sid, "error", error=f"PnL computation failed: {e}",
                           lifecycle_log=tuple(log),
                           duration_ms=_elapsed_ms(t0))

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
