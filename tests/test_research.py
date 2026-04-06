"""Tests for the research module (L1 enforcement)."""

import json
from pathlib import Path

import numpy as np
import pytest

from causal_edge.research.workspace import init_workspace
from causal_edge.research.evaluate import compute_k, check_look_ahead, run_evaluation, append_results_tsv


class TestInitWorkspace:
    def test_creates_files(self, tmp_path):
        ws = init_workspace("SOLUSD", tmp_path / "sol")
        assert (ws / "strategy.py").exists()
        assert (ws / "results.tsv").exists()
        assert (ws / "memory.md").exists()
        assert (ws / "discovery.json").exists()

    def test_strategy_has_run_strategy(self, tmp_path):
        ws = init_workspace("TSLA", tmp_path / "tsla")
        src = (ws / "strategy.py").read_text()
        assert "def run_strategy()" in src
        assert "TSLA" in src

    def test_results_has_header(self, tmp_path):
        ws = init_workspace("ETH", tmp_path / "eth")
        header = (ws / "results.tsv").read_text().strip()
        assert "commit" in header
        assert "score" in header
        assert "K" in header

    def test_idempotent(self, tmp_path):
        ws = init_workspace("SOL", tmp_path / "sol")
        (ws / "strategy.py").write_text("# modified")
        init_workspace("SOL", tmp_path / "sol")
        assert (ws / "strategy.py").read_text() == "# modified"


class TestComputeK:
    def test_counts_tickers_and_lags(self, tmp_path):
        code = '''
PARENTS = [("BTCUSD", 3, 5), ("ETHUSD", 21, 1), ("AVAXUSD", 7, 5)]
def run_strategy():
    p_al.shift(3)
    p_al.shift(21)
    p_al.shift(7)
'''
        p = tmp_path / "strategy.py"
        p.write_text(code)
        K, tickers, lags = compute_k(p)
        assert "BTCUSD" in tickers
        assert "ETHUSD" in tickers
        assert 3 in lags
        assert 21 in lags
        assert K > 0

    def test_filters_market_factors(self, tmp_path):
        code = '''
tickers = ["BTCUSD", "SPY", "QQQ"]
def run_strategy():
    p.shift(5)
'''
        p = tmp_path / "strategy.py"
        p.write_text(code)
        K, tickers, lags = compute_k(p)
        assert "SPY" not in tickers
        assert "BTCUSD" in tickers

    def test_empty_strategy(self, tmp_path):
        p = tmp_path / "strategy.py"
        p.write_text("def run_strategy(): pass")
        K, _, _ = compute_k(p)
        assert K >= 1  # minimum K


class TestCheckLookAhead:
    def test_catches_rolling_without_shift(self, tmp_path):
        code = "x = ret.rolling(20).mean()\npositions = x > 0"
        p = tmp_path / "strategy.py"
        p.write_text(code)
        violations = check_look_ahead(p)
        assert len(violations) > 0

    def test_allows_rolling_with_shift(self, tmp_path):
        code = "x = ret.rolling(20).mean().shift(1)\npositions = x > 0"
        p = tmp_path / "strategy.py"
        p.write_text(code)
        violations = check_look_ahead(p)
        assert len(violations) == 0


class TestAppendResults:
    def test_keep_requires_pass(self, tmp_path):
        ws = init_workspace("TEST", tmp_path / "t")
        result = {"verdict": "FAIL", "score": "10/13", "metrics": {}}
        with pytest.raises(ValueError, match="Cannot KEEP"):
            append_results_tsv(ws, result, "keep", "exploit", "test")

    def test_discard_always_works(self, tmp_path):
        ws = init_workspace("TEST", tmp_path / "t")
        result = {"verdict": "FAIL", "score": "10/13", "metrics": {}, "K": 50}
        append_results_tsv(ws, result, "discard", "exploit", "test")
        content = (ws / "results.tsv").read_text()
        assert "discard" in content

    def test_schema_has_k_column(self, tmp_path):
        ws = init_workspace("TEST", tmp_path / "t")
        result = {"verdict": "PASS", "score": "13/13",
                  "metrics": {"lo_adjusted": 1.5, "ic": 0.1, "omega": 2.0,
                              "sharpe": 2.0, "total_pnl": 1.0},
                  "K": 25}
        append_results_tsv(ws, result, "keep", "exploit", "baseline")
        content = (ws / "results.tsv").read_text()
        assert "25" in content


class TestRunEvaluation:
    def test_missing_strategy(self, tmp_path):
        result = run_evaluation(tmp_path)
        assert result["verdict"] == "ERROR"

    def test_unimplemented_strategy(self, tmp_path):
        ws = init_workspace("TEST", tmp_path / "t")
        result = run_evaluation(ws)
        assert result["verdict"] == "ERROR"
        assert any("failed" in f.lower() for f in result["failures"])
