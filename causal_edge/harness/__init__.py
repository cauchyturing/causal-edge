# causal_edge/harness/__init__.py
"""Harness module — generator pipeline + 7-step lifecycle.

CC Patterns: AsyncGenerator as Lingua Franca (08§7),
Seven-Step Tool Lifecycle (03§3), Persist Before Crash (08§6).
"""
from causal_edge.harness.types import PipelineEvent, SignalResult

__all__ = ["PipelineEvent", "SignalResult"]
