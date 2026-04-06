"""Causal voting strategy — demo using Abel's causal graph for TONUSD.

Shows the full production architecture validated by causal-edge (13/13 PASS):
- 23 Abel parents (10 direct + 10 Markov blanket + 3 crypto peers)
- Vote² sizing with 60% conviction filter
- WTM dual-lag xcorr (21+42) overlay: binary 1.50/0.50 vs expanding median
- RSI(20) contrarian: overbought(>70) 0.60x, oversold(<30) 1.40x
- Position persistence penalty: day2 start, -0.10/day, floor 0.30

The causal structure is REAL (from Abel CAP, queried 2026-04-05).
Price data is SYNTHETIC (demo requires no API key). To use real data,
replace _generate_synthetic_prices() with yfinance or FMP fetching.

Production metrics (on real FMP data, 2021-09 to 2026-04):
  Sharpe=2.10, Lo=1.65, IC=0.17, Omega=2.70, MaxDD=-10.2%
  DSR=98.5%, PBO=3.3%, 0 loss years, 0% negative rolling windows
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.engine.base import StrategyEngine

GRAPH_PATH = Path(__file__).parent / "causal_graph.json"

# Architecture params (matching production)
CONVICTION_MIN = 0.60
XCORR_UP = 1.50
XCORR_DOWN = 0.50
CLIP = 0.05
LAGS = [3, 5, 7, 10, 14, 21, 30, 42, 50]
WINDOWS = [1, 3, 5]


class CausalDemoEngine(StrategyEngine):
    """Vote² strategy with conviction filter and overlays from Abel graph."""

    def __init__(self, context: dict | None = None, n_days: int = 1200) -> None:
        super().__init__(context=context)
        self.n_days = n_days
        with open(GRAPH_PATH) as f:
            self.graph = json.load(f)

    def _generate_synthetic_prices(self, rng):
        """Generate synthetic prices for all tickers in the causal graph.

        Returns dict of ticker -> pd.Series of daily returns.
        """
        target_ret = rng.normal(0.0005, 0.035, self.n_days)
        all_tickers = []
        for group in ["parents", "blanket", "crypto_peers"]:
            for comp in self.graph.get(group, []):
                all_tickers.append(comp["ticker"])

        returns = {"target": target_ret}
        for ticker in all_tickers:
            # Inject weak lagged correlation with target (realistic)
            tau = rng.integers(3, 50)
            noise = rng.normal(0, 0.02, self.n_days)
            signal = np.zeros(self.n_days)
            signal[: self.n_days - tau] = target_ret[tau:] * 0.12
            returns[ticker] = signal + noise

        return returns, target_ret

    def compute_signals(self):
        """Generate signals from causal graph with full overlay stack."""
        rng = np.random.default_rng(seed=42)
        returns, target_ret = self._generate_synthetic_prices(rng)
        dates = pd.bdate_range(end="2026-04-01", periods=self.n_days)
        target_prices = 3.0 * np.exp(np.cumsum(target_ret))
        target_ret_clip = np.clip(target_ret, -CLIP, CLIP)

        # Collect all parent tickers
        tickers = []
        for group in ["parents", "blanket", "crypto_peers"]:
            for comp in self.graph.get(group, []):
                tickers.append(comp["ticker"])

        # Scan for robust components (3/3 time-split)
        t1 = self.n_days // 3
        t2 = 2 * self.n_days // 3
        selected = []

        for ticker in tickers:
            if ticker not in returns:
                continue
            p_ret = returns[ticker]
            best_sh, best_lag, best_win = -1, 0, 0

            for lag in LAGS:
                for win in WINDOWS:
                    sig = np.sign(
                        pd.Series(p_ret).rolling(win).sum().shift(lag)
                        .fillna(0).values
                    )
                    pnl = sig * target_ret_clip
                    start = lag + win + 5
                    if self.n_days - start < 300:
                        continue
                    std = np.std(pnl[start:], ddof=1)
                    if std <= 0:
                        continue
                    sh = float(np.mean(pnl[start:]) / std * np.sqrt(252))
                    if sh <= 0.2:
                        continue

                    # 3/3 time-split
                    splits = [pnl[start:t1], pnl[t1:t2], pnl[t2:]]
                    n_pos = sum(
                        1 for s in splits
                        if len(s) > 30 and np.std(s, ddof=1) > 0
                        and np.mean(s) / np.std(s, ddof=1) * np.sqrt(252) > 0
                    )
                    if n_pos >= 2 and sh > best_sh:
                        best_sh, best_lag, best_win = sh, lag, win

            if best_lag > 0:
                selected.append((ticker, best_lag, best_win, best_sh))

        # Take top 15 by Sharpe
        selected.sort(key=lambda x: x[3], reverse=True)
        selected = selected[:15]

        if not selected:
            positions = np.zeros(self.n_days)
            return positions, dates, target_ret, target_prices

        # Build component signals
        comp_sigs = []
        for ticker, lag, win, _ in selected:
            p_ret = returns[ticker]
            sig = np.sign(
                pd.Series(p_ret).rolling(win).sum().shift(lag)
                .fillna(0).values
            ) if win > 1 else np.sign(
                pd.Series(p_ret).shift(lag).fillna(0).values
            )
            comp_sigs.append(sig)

        sig_matrix = np.array(comp_sigs)
        n_up = (sig_matrix > 0).sum(axis=0)
        n_down = (sig_matrix < 0).sum(axis=0)
        n_active = np.maximum((sig_matrix != 0).sum(axis=0), 1)
        vote_frac = n_up / n_active

        # Vote² + conviction filter
        positions = np.zeros(self.n_days)
        bull = n_up > n_down
        positions[bull] = vote_frac[bull] ** 2
        positions[bull & (vote_frac < CONVICTION_MIN)] = 0.0
        positions = np.clip(positions, 0, 1.0)

        # WTM xcorr overlay (dual-lag 21+42)
        xcorr_ticker = selected[0][0] if selected else None
        if xcorr_ticker and xcorr_ticker in returns:
            xr = pd.Series(returns[xcorr_ticker])
            tr = pd.Series(target_ret_clip)
            xc_a = xr.shift(21).rolling(60).corr(tr.shift(1)).shift(1)
            xc_b = xr.shift(42).rolling(60).corr(tr.shift(1)).shift(1)
            xc_avg = ((xc_a + xc_b) / 2).fillna(0).values
            xc_med = (
                pd.Series(xc_avg).expanding().median().shift(1).fillna(0).values
            )
            for i in range(self.n_days):
                if np.isnan(xc_med[i]) or xc_avg[i] > xc_med[i]:
                    positions[i] *= XCORR_UP
                else:
                    positions[i] *= XCORR_DOWN

        # RSI(20) contrarian
        tr_s = pd.Series(target_ret_clip)
        gains = tr_s.clip(lower=0).rolling(20).mean().shift(1)
        losses = (-tr_s.clip(upper=0)).rolling(20).mean().shift(1)
        rs = gains / (losses + 1e-10)
        rsi = (100 - 100 / (1 + rs)).fillna(50).values
        for i in range(self.n_days):
            if rsi[i] > 70:
                positions[i] *= 0.60
            elif rsi[i] < 30:
                positions[i] *= 1.40

        # Persistence penalty (day2, -0.10/d, min 0.30)
        prev, days = 0.0, 0
        for i in range(self.n_days):
            if positions[i] > 0.01:
                if prev > 0.01:
                    days += 1
                    if days >= 2:
                        positions[i] *= max(0.30, 1.0 - 0.10 * (days - 1))
                else:
                    days = 1
            else:
                days = 0
            prev = positions[i]

        positions = np.clip(positions, 0, 1.0)
        return positions, dates, target_ret, target_prices

    def get_latest_signal(self):
        """Return latest causal voting signal."""
        positions, dates, _, prices = self.compute_signals()
        return {
            "position": float(positions[-1]),
            "date": str(dates[-1].date()),
            "price": float(prices[-1]),
        }
