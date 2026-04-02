"""Strategy execution orchestrator. Iterates strategies.yaml, calls engines."""

from __future__ import annotations


def run_all(config: dict) -> None:
    """Run all strategies from config and write trade logs."""
    raise NotImplementedError("trader.run_all coming in Phase 2.")


def run_one(config: dict, strategy_id: str) -> None:
    """Run a single strategy by ID."""
    raise NotImplementedError("trader.run_one coming in Phase 2.")
