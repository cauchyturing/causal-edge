"""Dashboard HTML generator — reads config + trade logs, renders Jinja2 templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader

from causal_edge.config import load_config
from causal_edge.dashboard._helpers import fmt_pnl_pct, fmt_dollar
from causal_edge.dashboard.components import (
    compute_metrics,
    yearly_metrics,
    live_metrics,
    equity_chart,
    drawdown_chart,
    rolling_sharpe_chart,
    daily_pnl_chart,
    monthly_heatmap,
    pnl_distribution,
    position_chart,
)

TEMPLATES_DIR = Path(__file__).parent / "templates"


def _load_trade_log(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path, parse_dates=["date"])
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None


def _prepare_strategy(s_cfg: dict) -> dict:
    df = _load_trade_log(s_cfg["trade_log"])
    empty = {
        "id": s_cfg["id"], "name": s_cfg["name"],
        "color": s_cfg["color"], "asset": s_cfg["asset"],
        "has_data": False, "metrics": {},
    }
    if df is None or len(df) == 0:
        return empty

    pnl = df["pnl"].values.astype(float)
    positions = (df["position"].values.astype(float)
                 if "position" in df.columns else np.zeros(len(pnl)))
    source = (df["source"].values if "source" in df.columns
              else np.array(["backfill"] * len(pnl)))
    asset_close = (df["asset_close"].values.astype(float)
                   if "asset_close" in df.columns else None)
    dates = pd.DatetimeIndex(df["date"])
    cum_pnl = np.cumsum(pnl)
    name = s_cfg["name"]
    color = s_cfg["color"]

    # Run causal-edge validation
    validation = None
    try:
        from causal_edge.validation.gate import validate_strategy
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            positions_arr = positions
            vdf = pd.DataFrame({"date": dates, "pnl": pnl, "position": positions_arr})
            vdf.to_csv(f.name, index=False)
            vr = validate_strategy(f.name, positions_col="position")
            Path(f.name).unlink(missing_ok=True)
            validation = {
                "verdict": vr["verdict"],
                "score": vr["score"],
                "failures": vr.get("failures", []),
                "triangle": vr.get("triangle", {}),
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in vr.get("metrics", {}).items()
                            if k in ("sharpe", "lo_adjusted", "ic", "omega",
                                     "max_dd", "dsr", "pbo", "oos_is",
                                     "neg_roll_frac", "loss_years", "total_pnl",
                                     "calmar", "bootstrap_p", "ic_stability",
                                     "sharpe_lo_ratio", "sortino",
                                     "ic_hit_rate", "ic_monthly_mean")},
            }
    except Exception:
        pass

    return {
        "id": s_cfg["id"], "name": name,
        "color": color, "asset": s_cfg["asset"],
        "badge": s_cfg.get("badge", ""),
        "has_data": True,
        "metrics": compute_metrics(pnl),
        "live": live_metrics(dates, pnl, positions, source, asset_close),
        "validation": validation,
        "yearly": yearly_metrics(dates, pnl),
        # Charts (all JSON strings)
        "equity_json": equity_chart(dates, cum_pnl, name, color),
        "drawdown_json": drawdown_chart(dates, cum_pnl, name),
        "rolling_sharpe_json": rolling_sharpe_chart(dates, pnl, name),
        "daily_pnl_json": daily_pnl_chart(dates, pnl, name),
        "monthly_json": monthly_heatmap(dates, pnl, name),
        "dist_json": pnl_distribution(pnl, name),
        "position_json": position_chart(dates, positions, name, color),
        "latest_date": str(dates[-1].date()),
        "latest_position": float(positions[-1]),
    }


def generate(config_path: str, output_path: str) -> None:
    cfg = load_config(config_path)
    strat_list = cfg["strategies"]
    if len(strat_list) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(len(strat_list), 8)) as pool:
            strategies = list(pool.map(_prepare_strategy, strat_list))
    else:
        strategies = [_prepare_strategy(s) for s in strat_list]

    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=False)
    env.globals["fmt_pnl_pct"] = fmt_pnl_pct
    env.globals["fmt_dollar"] = fmt_dollar

    template = env.get_template("base.html")
    html = template.render(
        strategies=strategies,
        settings=cfg["settings"],
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )
    Path(output_path).write_text(html)
