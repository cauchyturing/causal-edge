"""Trade log read/write. Single source of truth for trade log CSV format."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = ("date", "pnl", "position", "cum_pnl", "source")


def read_trade_log(path: str | Path) -> pd.DataFrame:
    """Read a trade log CSV. Returns DataFrame with standard columns."""
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def write_trade_log(
    dates: pd.DatetimeIndex,
    pnl: np.ndarray,
    positions: np.ndarray,
    path: str | Path,
    source: str = "backfill",
) -> None:
    """Write trade log: append new live rows, preserve existing live rows.

    Backfill rows (source != 'live') are always recomputed.
    Live rows with a timestamp are NEVER overwritten — they are real-time records.

    Args:
        dates: Trading dates from compute_signals()
        pnl: Daily PnL (position * returns)
        positions: Daily position sizes
        path: Output CSV path
        source: "backfill" or "live"
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Build new full-history DataFrame
    new_df = pd.DataFrame({
        "date": dates,
        "pnl": pnl,
        "position": positions,
        "cum_pnl": np.cumsum(pnl),
        "source": source,
    })

    if not path.exists():
        new_df.to_csv(path, index=False)
        return

    # Read existing trade log
    existing = pd.read_csv(path, parse_dates=["date"])

    # Preserve live rows that have timestamps (real-time recorded)
    if "timestamp" in existing.columns:
        live_locked = existing[existing["timestamp"].notna()].copy()
    else:
        live_locked = pd.DataFrame()

    if len(live_locked) == 0:
        # No locked live rows — overwrite entirely (backward compatible)
        new_df.to_csv(path, index=False)
        return

    # Merge: use locked live rows for their dates, new_df for everything else
    locked_dates = set(live_locked["date"].dt.date)
    new_non_locked = new_df[
        ~new_df["date"].apply(lambda d: d.date() in locked_dates)
    ]

    merged = pd.concat([new_non_locked, live_locked], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["cum_pnl"] = merged["pnl"].cumsum()
    merged.to_csv(path, index=False)


def append_live_row(
    date: pd.Timestamp,
    position: float,
    pnl: float,
    path: str | Path,
) -> None:
    """Write the timestamped live record for a given date.

    Invariant: after this call, the trade log has EXACTLY ONE row for `date`.
    If a backfill row exists for the same date, it is replaced by this live row.
    If an existing timestamped live row exists, it is preserved (immutable).

    This guarantees downstream readers never see duplicate (date) rows, so
    they don't need to dedup — any `df["pnl"].sum()` is correct.
    """
    path = Path(path)
    row = pd.DataFrame([{
        "date": date,
        "pnl": pnl,
        "position": position,
        "cum_pnl": 0.0,  # recomputed below
        "source": "live",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }])

    if not path.exists():
        row.to_csv(path, index=False)
        return

    existing = pd.read_csv(path, parse_dates=["date"])
    target_date = pd.Timestamp(date).date()

    # Preserve existing timestamped live row (immutable audit record)
    if "timestamp" in existing.columns:
        same_day_live = existing[
            (existing["date"].dt.date == target_date)
            & (existing["timestamp"].notna())
        ]
        if len(same_day_live) > 0:
            return

    # Remove any same-date row (backfill) before appending live.
    # Enforces uniqueness-on-date invariant at the write layer.
    existing = existing[existing["date"].dt.date != target_date]

    merged = pd.concat([existing, row], ignore_index=True)
    merged = merged.sort_values("date").reset_index(drop=True)
    merged["cum_pnl"] = merged["pnl"].cumsum()
    merged.to_csv(path, index=False)
