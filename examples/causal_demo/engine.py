"""Causal voting strategy — demo using Abel's causal graph for TONUSD.

Educational demo showing:
- Causal parent/child structure from Abel graph (bundled in causal_graph.json)
- Directional signal at each parent's causal lag (shift by tau)
- Vote² sizing: position = vote_fraction² (amplifies high conviction)
- Conviction threshold: only trade when ≥75% of components agree
- Parent signals: sign(parent_return.rolling(window).shift(tau))
- Child signals: sign(target_return.rolling(window).shift(tau))

The causal structure is REAL (from Abel's graph API). The price data is
SYNTHETIC (demo does not require market data API). To use real data,
replace _generate_synthetic_prices() with actual price fetching.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine

GRAPH_PATH = Path(__file__).parent / "causal_graph.json"
CONVICTION_MIN = 0.75  # ≥75% of votes must agree to trade


class CausalDemoEngine(StrategyEngine):
    """Vote-based strategy using Abel causal graph structure."""

    def __init__(self, context: dict | None = None, n_days: int = 600) -> None:
        super().__init__(context=context)
        self.n_days = n_days
        with open(GRAPH_PATH) as f:
            self.graph = json.load(f)

    def compute_signals(self):
        """Generate signals from causal graph components."""
        components = self.graph["parents"] + self.graph["children"]
        rng = np.random.default_rng(seed=55)

        # Synthetic target (TON-like crypto volatility)
        target_ret = rng.normal(0.0004, 0.025, self.n_days)
        target_prices = 5.0 * np.exp(np.cumsum(target_ret))
        dates = pd.bdate_range(end="2026-01-01", periods=self.n_days)

        # Generate correlated parent/child prices with realistic causal lags
        component_returns = {}
        for comp in components:
            ticker = comp["ticker"]
            tau = comp["tau"]
            # Parent leads target by tau days — inject lagged correlation
            noise = rng.normal(0, 0.02, self.n_days)
            if comp["type"] == "parent":
                # Parent return at t correlates with target return at t+tau
                signal = np.zeros(self.n_days)
                signal[: self.n_days - tau] = target_ret[tau:] * 0.15
                component_returns[ticker] = signal + noise
            else:
                # Child return at t+tau correlates with target return at t
                signal = np.zeros(self.n_days)
                signal[tau:] = target_ret[: self.n_days - tau] * 0.15
                component_returns[ticker] = signal + noise

        # Compute per-component directional signals
        sig_matrix = []
        for comp in components:
            ticker = comp["ticker"]
            tau = comp["tau"]
            win = comp["window"]
            ret = pd.Series(component_returns[ticker])

            if comp["type"] == "parent":
                # Parent signal: direction of parent's recent returns, shifted by tau
                if win > 1:
                    sig = np.sign(ret.rolling(win).sum().shift(tau)).values
                else:
                    sig = np.sign(ret.shift(tau)).values
            else:
                # Child signal: direction of target's recent returns, shifted by tau
                tr = pd.Series(target_ret)
                if win > 1:
                    sig = np.sign(tr.rolling(win).sum().shift(tau)).values
                else:
                    sig = np.sign(tr.shift(tau)).values

            sig_matrix.append(np.nan_to_num(sig, nan=0.0))

        sig_matrix = np.array(sig_matrix)  # (n_components, n_days)

        # Vote² sizing with conviction threshold
        n_up = (sig_matrix > 0).sum(axis=0)
        n_down = (sig_matrix < 0).sum(axis=0)
        n_active = (sig_matrix != 0).sum(axis=0)
        vote_frac = np.where(n_active > 0, n_up / n_active, 0.5)

        positions = np.zeros(self.n_days)
        bull = n_up > n_down
        positions[bull] = vote_frac[bull] ** 2

        # Conviction filter: go flat if vote not strong enough
        weak = bull & (vote_frac < CONVICTION_MIN)
        positions[weak] = 0.0

        # Long-only (no short in demo)
        positions = np.maximum(positions, 0.0)

        return positions, dates, target_ret, target_prices

    def get_latest_signal(self):
        """Return latest causal voting signal."""
        positions, dates, _, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }
