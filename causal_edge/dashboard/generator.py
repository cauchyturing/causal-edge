"""Dashboard HTML generator — reads config + trade logs, renders Jinja2 templates."""

from __future__ import annotations

from causal_edge.dashboard.components import equity_chart, position_chart


def generate(config_path: str, output_path: str) -> None:
    """Generate dashboard.html from config and trade logs."""
    raise NotImplementedError("generator.generate coming in Phase 2.")
