"""Command-line interface for causal-edge."""

from __future__ import annotations

from pathlib import Path

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
    click.echo(f"  strategies.yaml          (3 demo strategies)")
    click.echo(f"  strategies/sma_crossover (simple SMA)")
    click.echo(f"  strategies/momentum_ml   (walk-forward GBDT)")
    click.echo(f"  strategies/causal_demo   (Abel causal graph voting)")
    click.echo(f"  CLAUDE.md + AGENTS.md    (agent harness)")
    click.echo(f"  .env.example             (Abel API key, optional)")
    click.echo(f"")
    click.echo(f"Next:")
    click.echo(f"  cd {name}")
    click.echo(f"  causal-edge run")
    click.echo(f"  causal-edge dashboard")
    click.echo(f"  causal-edge validate")


@main.command()
@click.option("--strategy", default=None, help="Run a specific strategy by ID")
@click.option("--config", default=None, help="Config file path (default: strategies.local.yaml > strategies.yaml)")
def run(strategy, config):
    """Run strategies through the harness pipeline."""
    from causal_edge.config import load_config
    from causal_edge.harness.pipeline import run_pipeline

    cfg = load_config(config)

    if strategy:
        cfg["strategies"] = [
            s for s in cfg["strategies"] if s["id"] == strategy
        ]
        if not cfg["strategies"]:
            raise click.ClickException(f"Strategy '{strategy}' not found")

    for event in run_pipeline(cfg):
        _print_pipeline_event(event)


def _print_pipeline_event(event):
    """Format pipeline events for console output."""
    if event.status == "start":
        labels = {"run": "Running strategies", "validate": "Validating"}
        label = labels.get(event.phase, event.phase)
        count = event.data.get("count", "")
        suffix = f" ({count})" if count else ""
        click.echo(f"  [{event.phase}] {label}{suffix}...")

    elif event.status == "checkpoint":
        parts = [f"{k}={v}" for k, v in event.data.items()]
        click.echo(f"  [{event.phase}] done ({', '.join(parts)})")

    elif event.phase == "strategy":
        r = event.data.get("result")
        if r:
            lc = ">".join(r.lifecycle_log) if r.lifecycle_log else ""
            if r.status == "ok":
                click.echo(f"    {r.strategy_id:15s} {r.n_days:>5d} days "
                          f"({r.duration_ms:.0f}ms) [{lc}]")
            elif r.status == "error":
                click.echo(f"    {r.strategy_id:15s} ERROR: {r.error}")

    elif event.phase == "validation":
        tri = event.data.get("triangle", {})
        click.echo(f"    {event.data.get('id', ''):15s} {event.status:4s} "
                  f"{event.data.get('score', '')} "
                  f"Lo={tri.get('ratio', 0):.2f} "
                  f"IC={tri.get('rank', 0):.3f} "
                  f"Om={tri.get('shape', 0):.2f}")

    elif event.status == "error":
        click.echo(f"  [{event.phase}] ERROR: {event.data.get('msg', '')}")

    elif event.phase == "pipeline" and event.status == "done":
        ok = event.data.get("strategies_ok", 0)
        failed = event.data.get("strategies_failed", 0)
        click.echo(f"\n  Pipeline complete: {ok} ok, {failed} failed")


@main.command()
@click.option("--config", default=None, help="Config file path (default: strategies.local.yaml > strategies.yaml)")
@click.option("--output", default="dashboard.html", help="Output HTML path")
def dashboard(config, output):
    """Generate dashboard HTML."""
    from causal_edge.dashboard.generator import generate

    generate(config, output)
    click.echo(f"Dashboard generated: {output}")


@main.command()
@click.option("--strategy", default=None, help="Validate a specific strategy by ID")
@click.option("--verbose", is_flag=True, help="Show detailed failure info")
@click.option("--csv", "csv_path", default=None, help="Validate a standalone CSV (date,pnl columns)")
@click.option("--export", "export_path", default=None, help="Export report to file")
@click.option("--card", "card_path", default=None, help="Generate Strategy Card (YAML+markdown)")
@click.option("--config", default=None, help="Config file path (default: strategies.local.yaml > strategies.yaml)")
def validate(strategy, verbose, csv_path, export_path, card_path, config):
    """Run Abel Proof 13-test validation on strategies."""
    import io
    import sys

    from causal_edge.validation.gate import validate_strategy, print_validation_report

    results = {}

    if csv_path:
        # Quick path: validate a standalone CSV without strategies.yaml
        if not Path(csv_path).exists():
            raise click.ClickException(f"CSV not found: {csv_path}")
        result = validate_strategy(csv_path)
        results[Path(csv_path).stem] = result
    else:
        from causal_edge.config import load_config

        cfg = load_config(config)
        strategies_list = cfg["strategies"]
        if strategy:
            strategies_list = [s for s in strategies_list if s["id"] == strategy]

        if not strategies_list:
            click.echo("No strategies to validate.")
            return

        for s_cfg in strategies_list:
            sid = s_cfg["id"]
            log_path = s_cfg.get("trade_log", "")
            if not Path(log_path).exists():
                results[sid] = {
                    "verdict": "SKIP",
                    "score": "0/0",
                    "failures": [f"Trade log not found: {log_path}. Run 'causal-edge run' first."],
                    "metrics": {},
                    "triangle": {"ratio": 0, "rank": 0, "shape": 0},
                    "profile": "unknown",
                }
                continue
            results[sid] = validate_strategy(log_path)

    # Capture output for --export
    if export_path:
        old_stdout = sys.stdout
        sys.stdout = capture = io.StringIO()

    print_validation_report(results)

    if verbose:
        print()
        for sid, r in results.items():
            if r.get("metrics"):
                print(f"  {sid} metrics:")
                m = r["metrics"]
                for key in ("sharpe", "lo_adjusted", "sortino", "total_pnl",
                            "max_dd", "calmar", "dsr", "pbo", "oos_is",
                            "omega", "ic", "ic_hit_rate"):
                    if key in m:
                        print(f"    {key:20s} {m[key]:.4f}")
                if "yearly_sharpes" in m:
                    print(f"    yearly_sharpes:")
                    for yr, sh in sorted(m["yearly_sharpes"].items()):
                        print(f"      {yr}: {sh:.2f}")

    if export_path:
        sys.stdout = old_stdout
        report_text = capture.getvalue()
        click.echo(report_text, nl=False)  # also print to terminal
        Path(export_path).write_text(report_text)
        click.echo(f"\n  Report exported to {export_path}")

    # Strategy Card generation
    if card_path:
        from causal_edge.card import generate_card, render_card

        for sid, r in results.items():
            if r["verdict"] == "SKIP":
                continue
            card = generate_card(r, name=sid)
            card_text = render_card(card)
            out = Path(card_path)
            if len(results) > 1:
                out = out.parent / f"{out.stem}_{sid}{out.suffix}"
            out.write_text(card_text)
            click.echo(f"  Strategy Card → {out}")

    all_pass = all(r["verdict"] in ("PASS", "SKIP") for r in results.values())
    sys.exit(0 if all_pass else 1)


@main.command()
@click.argument("ticker")
def discover(ticker):
    """Discover causal parents for an asset via Abel API (requires ABEL_API_KEY)."""
    try:
        from causal_edge.plugins.abel.discover import discover_parents
    except ImportError:
        raise click.ClickException(
            "Abel plugin not installed. See: causal_edge/plugins/AGENTS.md"
        )
    parents = discover_parents(ticker)
    click.echo(parents)


@main.command()
@click.option("--config", default=None, help="Config file path (default: strategies.local.yaml > strategies.yaml)")
def status(config):
    """Show strategy status summary."""
    from causal_edge.config import load_config

    cfg = load_config(config)
    click.echo(f"Strategies: {len(cfg['strategies'])}")
    for s in cfg["strategies"]:
        click.echo(f"  {s['name']:20s}  {s['asset']:6s}  {s.get('badge', '?')}")


@main.group()
def research():
    """Autonomous research loop with L1 enforcement."""


@research.command("init")
@click.argument("ticker")
@click.option("--workdir", default=None, help="Workspace directory")
def research_init(ticker, workdir):
    """Initialize research workspace for a ticker."""
    from causal_edge.research.workspace import init_workspace

    ws = init_workspace(ticker, workdir)
    click.echo(f"Research workspace: {ws}/")
    click.echo(f"  strategy.py  — fill in run_strategy()")
    click.echo(f"  results.tsv  — experiment log (append-only)")
    click.echo(f"  memory.md    — agent memory")
    click.echo(f"  discovery.json — Abel parents")
    click.echo()
    click.echo("Next: edit strategy.py, then:")
    click.echo(f"  causal-edge research run --workdir {ws}")


@research.command("run")
@click.option("--workdir", default=".", help="Research workspace dir")
@click.option("--mode", default="exploit", type=click.Choice(["exploit", "explore"]))
@click.option("--description", "-d", default="", help="Experiment description")
def research_run(workdir, mode, description):
    """Run strategy.py through the immutable evaluation harness."""
    from pathlib import Path
    from causal_edge.research.evaluate import run_evaluation, append_results_tsv

    result = run_evaluation(workdir)

    # Print result
    import json
    verdict = result.get("verdict", "ERROR")
    score = result.get("score", "?/?")
    K = result.get("K", "?")
    tri = result.get("triangle", {})

    click.echo(f"\n  Verdict: {verdict}")
    click.echo(f"  Score:   {score}")
    click.echo(f"  K:       {K} (auto-computed)")
    click.echo(f"  Triangle: Lo={tri.get('ratio', 0):.2f}  "
               f"IC={tri.get('rank', 0):.3f}  "
               f"Om={tri.get('shape', 0):.2f}")

    m = result.get("metrics", {})
    if m:
        click.echo(f"  Sharpe={m.get('sharpe', 0):.2f}  "
                   f"MaxDD={m.get('max_dd', 0)*100:.1f}%  "
                   f"PnL={m.get('total_pnl', 0)*100:.1f}%")

    fails = result.get("failures", [])
    if fails:
        click.echo(f"\n  Failures:")
        for f in fails:
            click.echo(f"    - {f}")

    # Determine status
    if verdict == "PASS":
        status = "keep"
        click.echo(f"\n  PASS — recording as KEEP")
    else:
        status = "discard"
        click.echo(f"\n  {verdict} — recording as DISCARD")

    if not description:
        description = f"{mode}: {verdict} {score}"

    append_results_tsv(Path(workdir), result, status, mode, description)
    click.echo(f"  Appended to results.tsv")


@research.command("status")
@click.option("--workdir", default=".", help="Research workspace dir")
def research_status(workdir):
    """Show research progress summary."""
    from pathlib import Path
    import csv

    ws = Path(workdir)
    tsv = ws / "results.tsv"
    if not tsv.exists():
        click.echo("No results.tsv found. Run 'causal-edge research init' first.")
        return

    rows = list(csv.DictReader(open(tsv), delimiter="\t"))
    n_total = len(rows)
    n_keep = sum(1 for r in rows if r.get("status") == "keep")
    n_discard = n_total - n_keep

    click.echo(f"  Experiments: {n_total} ({n_keep} keep, {n_discard} discard)")

    if rows:
        latest = rows[-1]
        click.echo(f"  Latest: {latest.get('description', '?')} "
                   f"[{latest.get('score', '?')}] {latest.get('status', '?')}")

    keeps = [r for r in rows if r.get("status") == "keep"]
    if keeps:
        best = keeps[-1]
        click.echo(f"  Baseline: Sharpe={best.get('sharpe', '?')} "
                   f"Lo={best.get('lo_adj', '?')} "
                   f"IC={best.get('ic', '?')} "
                   f"K={best.get('K', '?')} "
                   f"[{best.get('score', '?')}]")


if __name__ == "__main__":
    main()
