"""Stateless chart-builder functions for the dashboard.

Every public function: data in (arrays/dicts) → string out (JSON or HTML).
No side effects. No strategy-specific logic.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def _chart_to_json(fig: go.Figure) -> str:
    """Convert Plotly figure to JSON string."""
    return json.dumps(fig.to_dict(), default=str)


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert a 6-digit hex color to an rgba() string with the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def compute_metrics(pnl: np.ndarray) -> dict:
    """Compute standard metrics from a PnL array of daily log returns.

    Returns dict with: sharpe, cum_pnl, max_dd, win_rate, n_trades, n_days.
    """
    if len(pnl) == 0:
        return dict(sharpe=0, cum_pnl=0, max_dd=0, win_rate=0, n_trades=0, n_days=0)

    cum = np.cumsum(pnl)
    std = np.std(pnl, ddof=1) if len(pnl) > 1 else 0.0
    sharpe = float(np.mean(pnl) / std * np.sqrt(252)) if std > 0 else 0.0

    # MaxDD on equity curve
    equity = np.exp(cum)
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / peak
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0

    active = pnl[np.abs(pnl) > 1e-10]
    win_rate = float(np.mean(active > 0)) if len(active) > 0 else 0.0
    n_trades = int(np.sum(np.abs(pnl) > 1e-10))

    return dict(
        sharpe=round(sharpe, 2),
        cum_pnl=round(float(cum[-1]), 4),
        max_dd=round(max_dd, 4),
        win_rate=round(win_rate, 3),
        n_trades=n_trades,
        n_days=len(pnl),
    )


def equity_chart(dates, cum_pnl, name: str, color: str) -> str:
    """Equity curve chart. Returns JSON string.

    Args:
        dates: array-like of dates
        cum_pnl: array-like of cumulative PnL values
        name: strategy name for legend
        color: hex color string
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates),
        y=list(cum_pnl),
        mode="lines",
        name=name,
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.12),
    ))
    fig.update_layout(
        title=f"{name} — Equity Curve",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL",
        yaxis_tickformat=".1%",
        template="plotly_dark",
        height=400,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _chart_to_json(fig)


def position_chart(dates, positions, name: str, color: str) -> str:
    """Position history chart. Returns JSON string.

    Args:
        dates: array-like of dates
        positions: array-like of position sizes (0=flat, 1=long)
        name: strategy name for legend
        color: hex color string
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(dates),
        y=list(positions),
        mode="lines",
        name="Position",
        line=dict(color=color, width=1),
        fill="tozeroy",
        fillcolor=_hex_to_rgba(color, 0.19),
    ))
    fig.update_layout(
        title=f"{name} — Position",
        xaxis_title="Date",
        yaxis_title="Position Size",
        template="plotly_dark",
        height=300,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return _chart_to_json(fig)
