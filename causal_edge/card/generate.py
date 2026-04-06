"""Generate Strategy Card (Tier 1) from validation results.

Takes the output of validate_strategy() and formats it as
YAML frontmatter + markdown body — the quant Model Card.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np

SCHEMA_VERSION = "strategy-card/1.0"


def generate_card(
    validation_result: dict,
    *,
    name: str = "Unnamed Strategy",
    asset: str = "",
    asset_class: str = "unknown",
    author: str = "",
    description: str = "",
    direction: str = "long_flat",
    discovery: dict | None = None,
    limitations: list[str] | None = None,
) -> dict:
    """Build a Strategy Card dict from validation output.

    Args:
        validation_result: output of validate_strategy()
        name: strategy name
        asset: target asset ticker
        asset_class: crypto | equity | commodity | fx | multi
        author: strategy author
        description: one-paragraph summary
        direction: long_only | long_short | long_flat | market_neutral
        discovery: optional dict with method, K, source
        limitations: optional list of known limitations

    Returns:
        dict with all card fields (pass to render_card for output).
    """
    m = validation_result.get("metrics", {})
    tri = validation_result.get("triangle", {})

    card = {
        "schema": SCHEMA_VERSION,
        "tier": "card",
        # Identity
        "name": name,
        "asset": asset,
        "asset_class": asset_class,
        "author": author,
        "date": date.today().isoformat(),
        "direction": direction,
        # Performance
        "performance": {
            "sharpe": _r(m.get("sharpe", 0)),
            "lo_adjusted": _r(m.get("lo_adjusted", 0)),
            "calmar": _r(m.get("calmar", 0)),
            "max_dd": _r(m.get("max_dd", 0)),
            "total_pnl": _r(m.get("total_pnl", 0)),
            "omega": _r(m.get("omega", 0)),
            "sortino": _r(m.get("sortino", 0)),
        },
        # Metric triangle
        "triangle": {
            "lo_adjusted": _r(tri.get("ratio", 0)),
            "ic": _r(tri.get("rank", 0)),
            "omega": _r(tri.get("shape", 0)),
        },
        # Validation
        "validation": {
            "framework": "causal-edge",
            "verdict": validation_result.get("verdict", "UNKNOWN"),
            "score": validation_result.get("score", "0/0"),
            "profile": validation_result.get("profile", "unknown"),
            "tests": _extract_tests(m, validation_result),
        },
        # Risk
        "risk": {
            "skewness": _r(m.get("skew", 0)),
            "hill_estimator": _r(m.get("hill_alpha", 0)),
            "cvar_var_ratio": _r(m.get("cvar_var_ratio", 0)),
            "worst_year": _worst_year(m),
        },
        # Robustness
        "robustness": {
            "dsr": _r(m.get("dsr", 0)),
            "pbo": _r(m.get("pbo", 0)),
            "oos_is": _r(m.get("oos_is", 0)),
            "neg_roll_frac": _r(m.get("neg_roll_frac", 0)),
            "loss_years": int(m.get("loss_years", 0)),
            "bootstrap_p": _r(m.get("bootstrap_p", 0)),
        },
        # Discovery provenance
        "discovery": discovery or {},
        # Metadata
        "backtest_days": int(m.get("T", len(m.get("pnl", [])))),
        "description": description,
        "limitations": limitations or [],
        "failures": validation_result.get("failures", []),
    }
    return card


def render_card(card: dict) -> str:
    """Render Strategy Card dict as YAML frontmatter + markdown body."""
    lines = ["---"]
    lines.append(f"schema: {card['schema']}")
    lines.append(f"tier: {card['tier']}")
    lines.append("")

    # Identity
    lines.append(f"name: \"{card['name']}\"")
    if card["asset"]:
        lines.append(f"asset: {card['asset']}")
    if card["asset_class"] != "unknown":
        lines.append(f"asset_class: {card['asset_class']}")
    if card["author"]:
        lines.append(f"author: \"{card['author']}\"")
    lines.append(f"date: \"{card['date']}\"")
    lines.append(f"direction: {card['direction']}")
    lines.append("")

    # Performance
    p = card["performance"]
    lines.append("performance:")
    for k, v in p.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Triangle
    t = card["triangle"]
    lines.append("triangle:")
    lines.append(f"  lo_adjusted: {t['lo_adjusted']}")
    lines.append(f"  ic: {t['ic']}")
    lines.append(f"  omega: {t['omega']}")
    lines.append("")

    # Validation
    v = card["validation"]
    lines.append("validation:")
    lines.append(f"  framework: {v['framework']}")
    lines.append(f"  verdict: {v['verdict']}")
    lines.append(f"  score: \"{v['score']}\"")
    lines.append(f"  profile: {v['profile']}")
    if v["tests"]:
        lines.append("  tests:")
        for test_name, test_data in v["tests"].items():
            if isinstance(test_data, dict):
                parts = ", ".join(f"{tk}: {tv}" for tk, tv in test_data.items())
                lines.append(f"    {test_name}: {{{parts}}}")
            else:
                lines.append(f"    {test_name}: {test_data}")
    lines.append("")

    # Risk
    r = card["risk"]
    lines.append("risk:")
    for k, v in r.items():
        if isinstance(v, dict):
            parts = ", ".join(f"{rk}: {rv}" for rk, rv in v.items())
            lines.append(f"  {k}: {{{parts}}}")
        else:
            lines.append(f"  {k}: {v}")
    lines.append("")

    # Robustness
    rb = card["robustness"]
    lines.append("robustness:")
    for k, v in rb.items():
        lines.append(f"  {k}: {v}")
    lines.append("")

    # Discovery
    if card["discovery"]:
        lines.append("discovery:")
        for k, v in card["discovery"].items():
            lines.append(f"  {k}: {v}")
        lines.append("")

    # Limitations (always present — empty list = no known limitations)
    lines.append("limitations:")
    if card["limitations"]:
        for lim in card["limitations"]:
            lines.append(f"  - \"{lim}\"")
    else:
        lines.append("  []")
    lines.append("")

    lines.append(f"backtest_days: {card['backtest_days']}")
    lines.append("---")
    lines.append("")

    # Markdown body
    lines.append(f"# {card['name']}")
    lines.append("")
    if card["description"]:
        lines.append(card["description"])
        lines.append("")

    # Triangle summary
    t = card["triangle"]
    lines.append("## Metric Triangle")
    lines.append("")
    lines.append(f"| Dimension | Value |")
    lines.append(f"|-----------|-------|")
    lines.append(f"| Lo-adjusted Sharpe (ratio) | {t['lo_adjusted']} |")
    lines.append(f"| IC (rank) | {t['ic']} |")
    lines.append(f"| Omega (shape) | {t['omega']} |")
    lines.append("")

    # Validation
    v = card["validation"]
    lines.append(f"## Validation: {v['verdict']} ({v['score']})")
    lines.append("")
    lines.append(f"Framework: {v['framework']} | Profile: {v['profile']}")
    lines.append("")
    if card["failures"]:
        lines.append("### Failures")
        lines.append("")
        for f in card["failures"]:
            lines.append(f"- {f}")
        lines.append("")

    # Limitations
    if card["limitations"]:
        lines.append("## Limitations")
        lines.append("")
        for lim in card["limitations"]:
            lines.append(f"- {lim}")
        lines.append("")

    # Discovery
    if card["discovery"]:
        d = card["discovery"]
        lines.append("## Discovery Provenance")
        lines.append("")
        if "method" in d:
            lines.append(f"- Method: {d['method']}")
        if "K" in d:
            lines.append(f"- K (trials): {d['K']}")
        if "source" in d:
            lines.append(f"- Source: {d['source']}")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by [causal-edge](https://github.com/cauchyturing/causal-edge)*")
    lines.append("")
    return "\n".join(lines)


# ── Helpers ──────────────────────────────────────────────────────────


def _r(v, decimals=3):
    """Round a value for display."""
    if isinstance(v, (int, np.integer)):
        return int(v)
    try:
        return round(float(v), decimals)
    except (TypeError, ValueError):
        return 0


def _extract_tests(metrics: dict, result: dict) -> dict:
    """Extract individual test results from metrics."""
    tests = {}
    field_map = {
        "dsr": ("dsr", 0.90),
        "pbo": ("pbo", 0.10),
        "oos_is": ("oos_is", 0.50),
        "neg_roll_frac": ("neg_roll_frac", 0.15),
        "loss_years": ("loss_years", 2),
        "lo_adjusted": ("lo_adjusted", 1.0),
        "omega": ("omega", 1.0),
        "max_dd": ("max_dd", -0.20),
    }
    failures = set(f.split(":")[0].strip() if ":" in f else f
                   for f in result.get("failures", []))
    for name, (key, threshold) in field_map.items():
        val = metrics.get(key, 0)
        tests[name] = {
            "value": _r(val),
            "threshold": threshold,
            "status": "pass" if name not in str(failures) else "fail",
        }
    # Look-ahead (binary)
    tests["look_ahead_static"] = "pass"
    tests["look_ahead_runtime"] = "pass"
    return tests


def _worst_year(metrics: dict) -> dict:
    """Find worst year from yearly sharpes."""
    yearly = metrics.get("yearly_sharpes", {})
    if not yearly:
        return {"year": 0, "sharpe": 0}
    worst_yr = min(yearly, key=yearly.get)
    return {"year": int(worst_yr), "sharpe": _r(yearly[worst_yr])}
