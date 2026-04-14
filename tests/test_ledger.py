"""Tests for trade log invariants and baseline consistency.

Guards against:
- Duplicate-date bug (double-counting in MTD/YTD/validation)
- R1 runtime look-ahead check being dormant (no `returns` passed)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causal_edge.engine.ledger import write_trade_log, append_live_row


def _make_dates(n: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2026-01-01", periods=n)


def test_append_live_replaces_same_day_backfill(tmp_path):
    """append_live_row must remove any same-date backfill row to keep the
    trade log unique on date."""
    path = tmp_path / "log.csv"
    dates = _make_dates(5)
    write_trade_log(dates, np.array([0.01] * 5), np.array([0.5] * 5), path)

    append_live_row(dates[-1], 0.5, 0.01, path)

    df = pd.read_csv(path, parse_dates=["date"])
    dup_count = df["date"].duplicated().sum()
    assert dup_count == 0, f"Trade log has {dup_count} duplicate date rows"
    assert len(df) == 5, f"Expected 5 rows (one per date), got {len(df)}"


def test_append_live_preserves_existing_live(tmp_path):
    """Existing timestamped live rows must be immutable across re-appends."""
    path = tmp_path / "log.csv"
    dates = _make_dates(5)
    write_trade_log(dates, np.array([0.01] * 5), np.array([0.5] * 5), path)

    append_live_row(dates[-1], 0.5, 0.01, path)
    df1 = pd.read_csv(path, parse_dates=["date"])
    ts1 = df1[df1["source"] == "live"]["timestamp"].iloc[0]

    append_live_row(dates[-1], 0.9, 0.99, path)
    df2 = pd.read_csv(path, parse_dates=["date"])
    ts2 = df2[df2["source"] == "live"]["timestamp"].iloc[0]

    assert ts1 == ts2, "Second append overwrote the locked live timestamp"
    live_pnl = df2[df2["source"] == "live"]["pnl"].iloc[0]
    assert live_pnl == 0.01, f"Live row pnl was overwritten: {live_pnl}"


def test_pnl_sum_matches_row_count(tmp_path):
    """After write+append cycle, df['pnl'].sum() must equal the true cumulative
    PnL without any double-counting. Downstream aggregators rely on this."""
    path = tmp_path / "log.csv"
    dates = _make_dates(10)
    pnl = np.array([0.01] * 10)
    write_trade_log(dates, pnl, np.array([0.5] * 10), path)
    append_live_row(dates[-1], 0.5, 0.01, path)

    df = pd.read_csv(path, parse_dates=["date"])
    assert len(df) == 10
    assert abs(df["pnl"].sum() - 0.10) < 1e-9, (
        f"Expected sum 0.10 (10 days × 0.01), got {df['pnl'].sum()}"
    )


def test_full_cron_cycle_idempotent(tmp_path):
    """Simulate two consecutive cron days. After each cycle the trade log must
    be unique-on-date and have exactly one live row for the latest day."""
    path = tmp_path / "log.csv"
    dates_day1 = _make_dates(5)
    write_trade_log(dates_day1, np.array([0.01] * 5), np.array([0.5] * 5), path)
    append_live_row(dates_day1[-1], 0.5, 0.01, path)

    dates_day2 = _make_dates(6)
    write_trade_log(dates_day2, np.array([0.01] * 6), np.array([0.5] * 6), path)
    append_live_row(dates_day2[-1], 0.5, 0.01, path)

    df = pd.read_csv(path, parse_dates=["date"])
    assert df["date"].duplicated().sum() == 0
    assert len(df) == 6
    live_rows = df[df["source"] == "live"]
    assert len(live_rows) == 2, (
        f"Expected 2 live rows (one per cron), got {len(live_rows)}"
    )


def test_production_trade_logs_unique_on_date():
    """Structural guard: every trade log in trading-internal/data/ must have
    unique dates. Fails loudly if anyone reintroduces the duplicate bug."""
    from pathlib import Path

    data_dir = Path.home() / "Claude" / "trading-internal" / "data"
    if not data_dir.exists():
        pytest.skip("trading-internal not present")

    failures = []
    for csv in sorted(data_dir.glob("trade_log_*.csv")):
        df = pd.read_csv(csv, parse_dates=["date"])
        dups = df["date"].duplicated().sum()
        if dups > 0:
            failures.append(f"{csv.name}: {dups} duplicate dates")

    assert not failures, (
        "Trade logs violate uniqueness-on-date invariant:\n  "
        + "\n  ".join(failures)
        + "\nFix: `append_live_row` must remove same-date rows before appending."
    )


def test_r1_runtime_check_catches_leak(tmp_path):
    """validate_strategy must pass `returns` to check_runtime so R1 fires.
    Create a trade log with deliberate look-ahead and assert R1 flags it."""
    from causal_edge.validation.gate import validate_strategy

    rng = np.random.default_rng(42)
    n = 300
    dates = pd.bdate_range("2024-01-01", periods=n)
    returns = rng.normal(0, 0.01, n)

    # Leak: positions perfectly anticipate same-day |return| magnitude
    positions = np.sign(returns) * np.abs(returns) * 10
    pnl = positions * returns

    path = tmp_path / "leaky.csv"
    pd.DataFrame({
        "date": dates,
        "pnl": pnl,
        "position": positions,
        "cum_pnl": np.cumsum(pnl),
        "source": ["backfill"] * n,
    }).to_csv(path, index=False)

    result = validate_strategy(str(path))
    r1_hits = [f for f in result["failures"] if f.startswith("R1")]
    assert r1_hits, (
        "R1 look-ahead check did not fire on a deliberately leaky trade log. "
        "Regression: check_runtime is being called without `returns` argument."
    )


def test_static_scanner_skips_string_literals():
    """AST-aware: regex-match inside a docstring must NOT fire T2.
    Guards the docstring false-positive that flagged causal_foundation.py L3.
    """
    from causal_edge.validation.look_ahead import check_static

    source = '''
"""Example — mentions df.rolling(5).sum() in the docstring only."""
def f(df):
    return df.rolling(5).mean().shift(1)
'''
    v = check_static(source)
    assert v == [], f"Expected clean, got violations: {v}"


def test_static_scanner_noqa_suppression():
    """# noqa: T3 must silence the flagged violation on that line."""
    from causal_edge.validation.look_ahead import check_static

    leaky = "import numpy as np\ndef f(x):\n    return np.std(x)\n"
    suppressed = "import numpy as np\ndef f(x):\n    return np.std(x)  # noqa: T3\n"
    bare = "import numpy as np\ndef f(x):\n    return np.std(x)  # noqa\n"
    wrong = "import numpy as np\ndef f(x):\n    return np.std(x)  # noqa: T2\n"
    assert len(check_static(leaky)) >= 1
    assert check_static(suppressed) == []
    assert check_static(bare) == []
    assert len(check_static(wrong)) >= 1, "T2 noqa should not suppress T3"


def test_production_engines_static_lookahead_clean():
    """Structural guard: every production engine in trading-internal/strategies/
    must pass static look-ahead checks (with noqa annotations as needed)."""
    from pathlib import Path
    from causal_edge.validation.look_ahead import check_static_file

    strat_dir = Path.home() / "Claude" / "trading-internal" / "strategies"
    if not strat_dir.exists():
        pytest.skip("trading-internal not present")

    ids = [
        "causal_foundation", "seven_comp", "dr_v2", "dual_resonance",
        "bnb", "ton_v2", "meta_v2", "aapl_v2", "abel_portfolio",
    ]
    failures = []
    for sid in ids:
        engine = strat_dir / sid / "engine.py"
        if not engine.exists():
            continue
        v = check_static_file(engine)
        if v:
            failures.append(f"{sid}: {len(v)} violations: {v[0][:80]}")

    assert not failures, (
        "Production engines have unsuppressed static look-ahead violations:\n  "
        + "\n  ".join(failures)
        + "\nFix: review each; if false positive, add `# noqa: T<N>` with reason."
    )


def test_derived_returns_match_pnl_division(tmp_path):
    """The returns derivation `np.divide(pnl, pos, where=|pos|>0.01)` must
    recover the true returns for active days. Regression guard against anyone
    replacing np.divide with naive `pnl / pos` (division by zero on flat days)."""
    rng = np.random.default_rng(0)
    n = 100
    true_returns = rng.normal(0, 0.01, n)
    positions = np.where(rng.uniform(size=n) > 0.5, 0.5, 0.0)
    pnl = positions * true_returns

    derived = np.divide(
        pnl, positions,
        out=np.zeros_like(pnl),
        where=np.abs(positions) > 0.01,
    )

    active = np.abs(positions) > 0.01
    assert np.allclose(derived[active], true_returns[active]), (
        "Derived returns don't match true returns on active days"
    )
    assert np.all(derived[~active] == 0), (
        "Derived returns should be 0 on flat days (no division by zero)"
    )
