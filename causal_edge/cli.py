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
    from causal_edge.scaffold import scaffold_project

    try:
        root = scaffold_project(name)
    except FileExistsError as e:
        raise click.ClickException(str(e))

    click.echo(f"Created {root}/")
    click.echo(f"  strategies.yaml")
    click.echo(f"  strategies/sma_crossover/engine.py")
    click.echo(f"  CLAUDE.md")
    click.echo(f"  AGENTS.md")
    click.echo(f"  .env.example")
    click.echo(f"")
    click.echo(f"Next:")
    click.echo(f"  cd {name}")
    click.echo(f"  causal-edge run")
    click.echo(f"  causal-edge dashboard")
    click.echo(f"  causal-edge validate")


@main.command()
@click.option("--strategy", default=None, help="Run a specific strategy by ID")
@click.option("--config", default="strategies.yaml", help="Config file path")
def run(strategy, config):
    """Run strategies and write trade logs."""
    from causal_edge.config import load_config
    from causal_edge.engine.trader import run_all

    cfg = load_config(config)
    if not cfg["strategies"]:
        click.echo("No strategies configured. Add strategies to strategies.yaml.")
        return

    click.echo(f"Running {len(cfg['strategies'])} strategies...")
    results = run_all(cfg, strategy_id=strategy)
    click.echo(f"Done. {len(results)} strategies executed.")


@main.command()
@click.option("--config", default="strategies.yaml", help="Config file path")
@click.option("--output", default="dashboard.html", help="Output HTML path")
def dashboard(config, output):
    """Generate dashboard HTML."""
    from causal_edge.dashboard.generator import generate

    generate(config, output)
    click.echo(f"Dashboard generated: {output}")


@main.command()
@click.option("--strategy", default=None, help="Validate a specific strategy by ID")
@click.option("--verbose", is_flag=True, help="Show detailed failure info")
@click.option("--config", default="strategies.yaml", help="Config file path")
def validate(strategy, verbose, config):
    """Run Abel Proof 15-test validation on strategies."""
    import sys

    from causal_edge.config import load_config
    from causal_edge.validation.gate import validate_strategy, print_validation_report

    cfg = load_config(config)
    strategies = cfg["strategies"]
    if strategy:
        strategies = [s for s in strategies if s["id"] == strategy]

    if not strategies:
        click.echo("No strategies to validate.")
        return

    results = {}
    for s_cfg in strategies:
        sid = s_cfg["id"]
        log_path = s_cfg.get("trade_log", "")
        result = validate_strategy(log_path)
        results[sid] = result

    print_validation_report(results)

    all_pass = all(r["verdict"] == "PASS" for r in results.values())
    sys.exit(0 if all_pass else 1)


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
