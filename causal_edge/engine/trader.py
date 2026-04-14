"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations

import importlib

import click
import numpy as np

from causal_edge.engine.ledger import write_trade_log, append_live_row


def _load_engine(engine_path: str):
    """Import engine module and find the StrategyEngine subclass."""
    mod = importlib.import_module(engine_path)
    from causal_edge.engine.base import StrategyEngine

    for attr_name in dir(mod):
        attr = getattr(mod, attr_name)
        if (
            isinstance(attr, type)
            and issubclass(attr, StrategyEngine)
            and attr is not StrategyEngine
        ):
            return attr
    raise ImportError(
        f"No StrategyEngine subclass found in '{engine_path}'. "
        f"Fix: Ensure your engine.py defines a class inheriting StrategyEngine."
    )


def run_one(strategy_cfg: dict) -> dict:
    """Run a single strategy and write its trade log.

    Two modes:
    1. Backfill: rewrite entire trade log (for initial setup)
    2. Live append: compute signals, append ONLY the latest day with timestamp

    In production (cron), this always appends the latest day as a live record.
    Historical rows are preserved — live rows with timestamps are immutable.
    """
    sid = strategy_cfg["id"]
    engine_path = strategy_cfg["engine"]
    trade_log_path = strategy_cfg["trade_log"]

    click.echo(f"  Running {sid}...")
    engine_cls = _load_engine(engine_path)
    engine = engine_cls(context=strategy_cfg)

    positions, dates, returns, prices = engine.compute_signals()

    # PnL: positions[t] * returns[t] is correct because the engine contract
    # requires positions[t] to be decided using data through t-1 only.
    pnl = positions * returns
    pnl[0] = 0.0

    # Write full backfill (preserves existing timestamped live rows)
    write_trade_log(dates, pnl, positions, trade_log_path)

    # Append latest day as timestamped live record (immutable)
    if len(dates) > 0:
        append_live_row(
            date=dates[-1],
            position=float(positions[-1]),
            pnl=float(pnl[-1]),
            path=trade_log_path,
        )

    return {"id": sid, "n_days": len(dates), "trade_log": trade_log_path}


def run_all(config: dict, strategy_id: str | None = None) -> list[dict]:
    """Run all strategies (or one specific strategy) from config."""
    strategies = config["strategies"]
    if strategy_id:
        strategies = [s for s in strategies if s["id"] == strategy_id]
        if not strategies:
            raise ValueError(
                f"Strategy '{strategy_id}' not found in strategies.yaml. "
                f"Available: {[s['id'] for s in config['strategies']]}"
            )

    results = []
    for s_cfg in strategies:
        result = run_one(s_cfg)
        results.append(result)
        click.echo(f"    → {result['n_days']} days written to {result['trade_log']}")

    return results
