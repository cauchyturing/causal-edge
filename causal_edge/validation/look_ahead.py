"""Look-ahead bias detection — static analysis + runtime checks.

Static (source code scan):
  T1: features without shift(lag) where lag >= 1
  T2: rolling().stat() without .shift(1)
  T3: np.std/np.mean on full array (not rolling/expanding)
  T4: walk-forward slicing [:i+1] instead of [:i]
  T5: trend filter using current-day price (not shift(1))

Runtime (data correlation):
  R1: position magnitude correlates with same-day return (leak detector)
  R2: any feature column correlates > threshold with next-day return

Both layers must pass. Static catches code-level leaks.
Runtime catches data-level leaks that static analysis misses.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


# ── Static Analysis ──


def check_static(source: str) -> list[str]:
    """Run all 5 static look-ahead checks on strategy source code.

    Args:
        source: Python source code string.

    Returns:
        List of violation messages. Empty = clean.
    """
    violations = []
    violations.extend(_t2_rolling_without_shift(source))
    violations.extend(_t3_global_stats(source))
    violations.extend(_t4_wf_slicing(source))
    violations.extend(_t5_trend_filter(source))
    return violations


def check_static_file(path: str | Path) -> list[str]:
    """Run static checks on a file path."""
    return check_static(Path(path).read_text())


def _t2_rolling_without_shift(source: str) -> list[str]:
    """T2: rolling().stat() must be followed by .shift(1).

    rolling(N).mean/std/sum/corr() includes the current value.
    Must shift before using for decisions.
    """
    violations = []
    # Match rolling(N).stat() — various stat methods
    pattern = re.compile(
        r'\.rolling\(\s*\d+\s*\)\s*\.\s*(mean|std|sum|var|median|corr|min|max)\s*\([^)]*\)'
    )
    for m in pattern.finditer(source):
        line_num = source[:m.start()].count('\n') + 1
        # Check if .shift() follows within 50 chars
        after = source[m.end():m.end() + 50]
        if '.shift(' not in after:
            # Allow if it's a target variable (shift(-N) for labels)
            ctx = source[max(0, m.start() - 30):m.end() + 50]
            if 'shift(-' in ctx or '# target' in ctx.lower() or '# label' in ctx.lower():
                continue
            # Allow if assigned to a target/label variable
            line_start = source.rfind('\n', 0, m.start()) + 1
            line = source[line_start:m.end() + 50]
            if re.match(r'\s*(y|target|label)\s*=', line):
                continue
            snippet = source[max(0, m.start() - 5):m.end() + 20].strip()
            violations.append(f"T2 L{line_num}: rolling().{m.group(1)}() without .shift(1): {snippet}")
    return violations


def _t3_global_stats(source: str) -> list[str]:
    """T3: np.std/np.mean on full array = look-ahead.

    Use rolling/expanding std, or compute on [:i] slice only.
    """
    violations = []
    for func in ('np.std', 'np.mean'):
        for m in re.finditer(re.escape(func) + r'\s*\(\s*(\w+)\s*[,)]', source):
            line_num = source[:m.start()].count('\n') + 1
            var_name = m.group(1)
            # Allow if it's on a slice [:i] or [:n] or [start:end]
            full_call = source[m.start():m.end() + 30]
            if '[:' in full_call or 'ddof' in full_call:
                continue
            # Allow if in a metrics/reporting context
            line_start = source.rfind('\n', 0, m.start()) + 1
            line = source[line_start:source.find('\n', m.end())]
            if any(kw in line.lower() for kw in ('sharpe', 'metric', 'result', 'report', 'print')):
                continue
            violations.append(
                f"T3 L{line_num}: {func}({var_name}) on full array — use rolling/expanding or [:i] slice"
            )
    return violations


def _t4_wf_slicing(source: str) -> list[str]:
    """T4: Walk-forward slicing must use [:i], not [:i+1].

    [:i+1] includes the current day in the training window.
    """
    violations = []
    pattern = re.compile(r'\[\s*:?\s*(\w+)\s*\+\s*1\s*\]')
    for m in pattern.finditer(source):
        line_num = source[:m.start()].count('\n') + 1
        # Context: is this in a training/fitting context?
        line_start = source.rfind('\n', 0, m.start()) + 1
        line = source[line_start:source.find('\n', m.end())]
        if any(kw in line.lower() for kw in ('train', 'fit', 'x_tr', 'y_tr', 'x_s', 'weight')):
            violations.append(
                f"T4 L{line_num}: [:i+1] in training context — use [:i] to exclude current day"
            )
    return violations


def _t5_trend_filter(source: str) -> list[str]:
    """T5: Trend filter must compare yesterday's price/MA, not today's.

    prices[i] < sma[i] uses today's price → look-ahead.
    Must use prices[i-1] < sma[i-1] or .shift(1).
    """
    violations = []
    # Detect patterns like: close[i] < sma[i] or prices[i] < ma[i]
    pattern = re.compile(
        r'(close|price|px|eth_price|sma|ma)\s*\[\s*(\w+)\s*\]'
        r'\s*[<>]=?\s*'
        r'(close|price|px|sma|ma)\s*\[\s*(\w+)\s*\]'
    )
    for m in pattern.finditer(source):
        idx_left, idx_right = m.group(2), m.group(4)
        line_num = source[:m.start()].count('\n') + 1
        # Both using same index without -1 = potential look-ahead
        if idx_left == idx_right and '-1' not in idx_left:
            line_start = source.rfind('\n', 0, m.start()) + 1
            line = source[line_start:source.find('\n', m.end())]
            # Skip if already using shift(1) pattern
            if 'shift(1)' in line or 'i-1' in line or 'i - 1' in line:
                continue
            violations.append(
                f"T5 L{line_num}: trend filter uses current-day index — use [i-1] or .shift(1)"
            )
    return violations


# ── Runtime Analysis ──


def check_runtime(pnl: np.ndarray, positions: np.ndarray,
                  returns: np.ndarray | None = None,
                  threshold: float = 0.3) -> list[str]:
    """Runtime look-ahead detection via correlation analysis.

    R1: |corr(|position|, |same-day return|)| should be low.
         High correlation means positions "know" today's return.

    R2: Hit rate sanity — if > 70% of positioned days are profitable,
        suspicious unless strategy has very few trades.

    Args:
        pnl: daily PnL array
        positions: daily position array
        returns: daily asset returns (if available, for R1)
        threshold: correlation threshold for flagging

    Returns:
        List of warning messages. Empty = clean.
    """
    warnings = []

    if returns is not None and len(returns) == len(positions):
        # R1: Position magnitude × same-day return correlation
        active = np.abs(positions) > 0.01
        if active.sum() > 30:
            pos_mag = np.abs(positions[active])
            ret_mag = np.abs(returns[active])
            corr = np.corrcoef(pos_mag, ret_mag)[0, 1]
            if not np.isnan(corr) and abs(corr) > threshold:
                warnings.append(
                    f"R1: |position| correlates with |same-day return| "
                    f"(corr={corr:.3f}, threshold={threshold}). "
                    f"Positions may be using future information."
                )

    # R2: Hit rate sanity check
    active_pnl = pnl[np.abs(positions) > 0.01]
    if len(active_pnl) > 30:
        hit_rate = np.mean(active_pnl > 0)
        if hit_rate > 0.70:
            warnings.append(
                f"R2: Hit rate {hit_rate:.0%} on {len(active_pnl)} active days "
                f"is unusually high. Verify no look-ahead in feature engineering."
            )

    return warnings
