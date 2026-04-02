"""Command-line interface for causal-edge."""

from __future__ import annotations

import click


@click.group()
@click.version_option()
def main():
    """causal-edge: Agent-native quant framework."""


@main.command()
@click.argument("name")
def init(name):
    """Scaffold a new causal-edge project."""
    raise NotImplementedError(
        "causal-edge init is not yet implemented. Coming in Phase 2."
    )


@main.command()
@click.option("--strategy", default=None, help="Run a specific strategy by ID")
@click.option("--config", default="strategies.yaml", help="Config file path")
def run(strategy, config):
    """Run strategies and write trade logs."""
    raise NotImplementedError(
        "causal-edge run is not yet implemented. Coming in Phase 2."
    )


@main.command()
@click.option("--config", default="strategies.yaml", help="Config file path")
@click.option("--output", default="dashboard.html", help="Output HTML path")
def dashboard(config, output):
    """Generate dashboard HTML."""
    raise NotImplementedError(
        "causal-edge dashboard is not yet implemented. Coming in Phase 2."
    )


@main.command()
@click.option("--strategy", default=None, help="Validate a specific strategy by ID")
@click.option("--verbose", is_flag=True, help="Show detailed failure info")
@click.option("--config", default="strategies.yaml", help="Config file path")
def validate(strategy, verbose, config):
    """Run Abel Proof 15-test validation on strategies."""
    raise NotImplementedError(
        "causal-edge validate is not yet implemented. Coming in Phase 2."
    )


@main.command()
@click.option("--config", default="strategies.yaml", help="Config file path")
def status(config):
    """Show strategy status summary."""
    from causal_edge.config import load_config

    cfg = load_config(config)
    click.echo(f"Strategies: {len(cfg['strategies'])}")
    for s in cfg["strategies"]:
        click.echo(f"  {s['name']:20s}  {s['asset']:6s}  {s.get('badge', '?')}")


if __name__ == "__main__":
    main()
