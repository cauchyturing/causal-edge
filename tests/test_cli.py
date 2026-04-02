"""CLI entry point tests."""

from click.testing import CliRunner

from causal_edge.cli import main


def test_help():
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "causal-edge" in result.output


def test_version():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output


def test_status_empty():
    """Status with empty strategies.yaml should show 0 strategies."""
    result = CliRunner().invoke(main, ["status"])
    assert result.exit_code == 0
    assert "Strategies: 0" in result.output


def test_init_not_implemented():
    result = CliRunner().invoke(main, ["init", "test_project"])
    assert result.exit_code != 0


def test_run_not_implemented():
    result = CliRunner().invoke(main, ["run"])
    assert result.exit_code != 0


def test_dashboard_not_implemented():
    result = CliRunner().invoke(main, ["dashboard"])
    assert result.exit_code != 0


def test_validate_not_implemented():
    result = CliRunner().invoke(main, ["validate"])
    assert result.exit_code != 0
