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
    dates = pd.DatetimeIndex(df["date"])
    cum_pnl = np.cumsum(pnl)
    name = s_cfg["name"]
    color = s_cfg["color"]

    return {
        "id": s_cfg["id"], "name": name,
        "color": color, "asset": s_cfg["asset"],
        "badge": s_cfg.get("badge", ""),
        "has_data": True,
        "metrics": compute_metrics(pnl),
        "live": live_metrics(dates, pnl, source),
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
    strategies = [_prepare_strategy(s) for s in cfg["strategies"]]

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
