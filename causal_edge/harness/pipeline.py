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

    # Dashboard: not auto-run in pipeline. Use `causal-edge dashboard` separately.
    # The pipeline is for compute + validate. Dashboard is a presentation concern.

    yield PipelineEvent("pipeline", "done", {
        "strategies_ok": ok,
        "strategies_failed": failed,
    })


def _run_strategies(strategies: list[dict]) -> list[SignalResult]:
    """Execute strategies sequentially.

    Serial by design: engines use ProcessPoolExecutor + joblib internally, and
    fork() from a multi-threaded parent deadlocks on CPython. Each strategy
    runs in turn with full access to all CPU cores.
    """
    results = []
    for s_cfg in strategies:
        try:
            results.append(execute_strategy(s_cfg))
        except Exception as e:
            results.append(SignalResult(
                strategy_id=s_cfg["id"],
                status="error",
                error=str(e),
                lifecycle_log=("serial:FAIL",),
            ))
    return results
