"""Trade log read/write. Single source of truth for trade log CSV format."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = ("date", "pnl", "position", "cum_pnl", "source")


def read_trade_log(path: str | Path) -> pd.DataFrame:
    """Read a trade log CSV. Returns DataFrame with standard columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def write_trade_log(df: pd.DataFrame, path: str | Path) -> None:
    """Write a trade log CSV. Ensures required columns exist."""
    raise NotImplementedError("ledger.write_trade_log coming in Phase 2.")
