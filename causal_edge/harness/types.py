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
