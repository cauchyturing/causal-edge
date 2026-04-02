"""Stateless chart-builder functions for the dashboard.

Every public function: data in (arrays/dicts) → string out (JSON or HTML).
No side effects. No strategy-specific logic.
"""

from __future__ import annotations

import json

import numpy as np
import plotly.graph_objects as go


def _chart_to_json(fig: go.Figure) -> str:
    """Convert Plotly figure to JSON string."""
    return json.dumps(fig.to_dict(), default=str)


def equity_chart(dates, cum_pnl, name: str, color: str) -> str:
    """Equity curve chart. Returns JSON string."""
    raise NotImplementedError("equity_chart coming in Phase 2.")


def position_chart(dates, positions, name: str, color: str) -> str:
    """Position history chart. Returns JSON string."""
    raise NotImplementedError("position_chart coming in Phase 2.")
