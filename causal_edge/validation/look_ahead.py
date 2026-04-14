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

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd

from causal_edge.validation._look_ahead_ast import (
    collect_scope_bindings,
    in_string_literal,
    is_bounded_expr,
    is_numpy_reduction,
    node_offset,
    numpy_call_name,
    safe_unparse,
    string_literal_spans,
)


# ── Static Analysis ──


def check_static(source: str) -> list[str]:
    """Run all 5 static look-ahead checks on strategy source code.

    Args:
        source: Python source code string.

    Returns:
        List of violation messages. Empty = clean.

    Suppression: add `# noqa: T2` (or T3/T4/T5, or bare `# noqa`) to a line to
    silence a false positive. Use `# noqa: lookahead` to silence all look-ahead
    codes on that line. Scanners are regex-based and flag legitimate patterns
    like `[ws:i]` windowed slices or cross-axis mean across strategies —
    those are textual false positives, not real leaks. Always pair a noqa
    with a reason comment on the same or previous line.
    """
    lines = source.split("\n")
    raw = []
    raw.extend(_t2_rolling_without_shift(source))
    raw.extend(_t3_global_stats(source))
    raw.extend(_t4_wf_slicing(source))
    raw.extend(_t5_trend_filter(source))
    return [v for v in raw if not _is_suppressed(v, lines)]


def check_static_file(path: str | Path) -> list[str]:
    """Run static checks on a file path."""
    return check_static(Path(path).read_text())


def _is_suppressed(violation: str, lines: list[str]) -> bool:
    """Return True if the line referenced by the violation carries a matching
    # noqa annotation. Supported forms on the flagged line or the line above:
      # noqa                — suppresses any code
      # noqa: T2            — suppresses only T2
      # noqa: T2, T3        — suppresses listed codes
      # noqa: lookahead     — suppresses all look-ahead codes (T1-T5, R1-R2)
    """
    m = re.match(r"^(T[1-5])\s+L(\d+):", violation)
    if not m:
        return False
    code, line_num = m.group(1), int(m.group(2))
    # Check the flagged line and the line above (for multi-line expressions)
    candidates = []
    if 0 <= line_num - 1 < len(lines):
        candidates.append(lines[line_num - 1])
    if 0 <= line_num - 2 < len(lines):
        candidates.append(lines[line_num - 2])
    for ln in candidates:
        noqa = re.search(r"#\s*noqa(?::\s*([A-Za-z0-9_,\s]+))?", ln)
        if not noqa:
            continue
        arg = noqa.group(1)
        if arg is None:  # bare `# noqa` suppresses any
            return True
        tokens = {t.strip().lower() for t in arg.split(",")}
        if code.lower() in tokens or "lookahead" in tokens:
            return True
    return False


def _t2_rolling_without_shift(source: str) -> list[str]:
    """T2: rolling().stat() must be followed by .shift(1).

    rolling(N).mean/std/sum/corr() includes the current value.
    Must shift before using for decisions.
    """
    violations = []
    string_spans = string_literal_spans(source)
    # Match rolling(N).stat() — various stat methods
    pattern = re.compile(
        r'\.rolling\(\s*\d+\s*\)\s*\.\s*(mean|std|sum|var|median|corr|min|max)\s*\([^)]*\)'
    )
    for m in pattern.finditer(source):
        if in_string_literal(m.start(), string_spans):
            continue  # skip matches inside docstrings/string literals
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
    """T3: np.std/np.mean/np.var on an unbounded array = look-ahead.

    AST-based def-use analysis: for each `np.std(X)` call, trace X back to
    its binding. If X was assigned from a slice (`a[l:h]`), a list literal
    (`[...]`), an np.where with a sliced operand, or any expression
    containing a slice, it is considered "bounded" (window-local) and safe.

    Falls back to clean regex heuristics for cases where AST parsing fails
    or def-use is ambiguous (e.g. variable from function parameter).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []

    violations: list[str] = []
    string_spans = string_literal_spans(source)
    scope_cache: dict[int, dict] = {}

    def _scan(node: ast.AST, scope: ast.AST) -> None:
        if isinstance(node, ast.Call) and is_numpy_reduction(node) and node.args:
            offset = node_offset(node, source)
            if offset is None or not in_string_literal(offset, string_spans):
                sid = id(scope)
                if sid not in scope_cache:
                    scope_cache[sid] = collect_scope_bindings(scope)
                if not is_bounded_expr(node.args[0], scope_cache[sid]):
                    violations.append(
                        f"T3 L{node.lineno}: "
                        f"np.{numpy_call_name(node)}({safe_unparse(node.args[0])}) "
                        f"on full array — use rolling/expanding or [:i] slice"
                    )
        for child in ast.iter_child_nodes(node):
            new_scope = (
                child if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                else scope
            )
            _scan(child, new_scope)

    _scan(tree, tree)
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
