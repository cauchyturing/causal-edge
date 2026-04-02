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


def test_init_creates_project(tmp_path):
    """init should create a project directory with expected files."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code == 0, result.output
        from pathlib import Path
        root = Path("myproject")
        assert root.is_dir()
        assert (root / "strategies.yaml").exists()
        assert (root / "strategies" / "sma_crossover" / "engine.py").exists()
        assert (root / "strategies" / "sma_crossover" / "__init__.py").exists()
        assert (root / "data").is_dir()
        assert (root / ".env.example").exists()
        assert (root / "CLAUDE.md").exists()
        assert (root / "AGENTS.md").exists()


def test_init_fails_if_dir_exists(tmp_path):
    """init should fail with a clear error if the directory already exists."""
    runner = CliRunner()
    with runner.isolated_filesystem(temp_dir=tmp_path):
        runner.invoke(main, ["init", "myproject"])
        result = runner.invoke(main, ["init", "myproject"])
        assert result.exit_code != 0
        assert "already exists" in result.output


def test_run_empty():
    """Run with no strategies should print message, not crash."""
    result = CliRunner().invoke(main, ["run"])
    assert result.exit_code == 0
    assert "No strategies" in result.output


def test_dashboard_empty():
    """Dashboard with no strategies should generate HTML without error."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        # Create minimal strategies.yaml
        from pathlib import Path
        Path("strategies.yaml").write_text("settings: {}\nstrategies: []\n")
        result = runner.invoke(main, ["dashboard"])
        assert result.exit_code == 0
        assert Path("dashboard.html").exists()


def test_validate_empty():
    """Validate with no strategies should print message, not crash."""
    result = CliRunner().invoke(main, ["validate"])
    assert result.exit_code == 0
    assert "No strategies" in result.output
