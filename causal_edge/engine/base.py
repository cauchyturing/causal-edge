"""Abstract base class for strategy engines.

Every strategy engine must implement compute_signals() and get_latest_signal().
Engines are standalone: strategies/ never imports causal_edge/ internals.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class StrategyEngine(ABC):
    """Base class for all strategy engines.

    Subclasses must implement:
        compute_signals() -> tuple of (positions, dates, returns, prices)
        get_latest_signal() -> dict with at least 'position' key
    """

    def __init__(self, context: dict | None = None) -> None:
        self.context = context

    @abstractmethod
    def compute_signals(
        self,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, np.ndarray, np.ndarray]:
        """Compute full signal history.

        Returns:
            Tuple of (positions, dates, returns, prices) where:
                positions: np.ndarray of daily position sizes (0=flat, 1=long).
                    IMPORTANT: positions[t] must be decided using only data through
                    day t-1. Apply .shift(1) to any indicators before using them
                    to determine positions. This prevents look-ahead bias.
                dates: pd.DatetimeIndex of trading dates
                returns: np.ndarray of daily asset returns
                prices: np.ndarray of daily closing prices
        """

    @abstractmethod
    def get_latest_signal(self) -> dict:
        """Return the most recent signal as a dict.

        Must include at least a 'position' key.
        """

    def on_retrain(self) -> None:
        """Hook for periodic model retraining.  Default is no-op."""
