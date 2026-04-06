"""Tests for the harness pipeline + lifecycle."""

import numpy as np
import pandas as pd
import pytest

from causal_edge.engine.base import StrategyEngine
from causal_edge.harness.types import PipelineEvent, SignalResult


class MockEngine(StrategyEngine):
    """Minimal engine for testing."""

    def compute_signals(self):
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        positions = np.ones(n) * 0.5
        returns = np.random.randn(n) * 0.01
        prices = 100 + np.cumsum(returns)
        return positions, dates, returns, prices

    def get_latest_signal(self):
        return {"position": 0.5, "date": pd.Timestamp.now()}


class TestTypes:
    def test_pipeline_event_frozen(self):
        e = PipelineEvent("run", "start", {"count": 3})
        assert e.phase == "run"
        assert e.status == "start"
        with pytest.raises(AttributeError):
            e.phase = "other"

    def test_signal_result_frozen(self):
        r = SignalResult("test", "ok", n_days=100)
        assert r.strategy_id == "test"
        assert r.n_days == 100
        with pytest.raises(AttributeError):
            r.status = "error"

    def test_signal_result_lifecycle_log(self):
        r = SignalResult("x", "ok", lifecycle_log=("load:ok", "compute:ok"))
        assert len(r.lifecycle_log) == 2
        assert "load:ok" in r.lifecycle_log


class TestLifecycle:
    def test_execute_missing_engine(self):
        from causal_edge.harness.lifecycle import execute_strategy
        cfg = {"id": "bad", "engine": "nonexistent.module", "trade_log": "/tmp/t.csv"}
        result = execute_strategy(cfg)
        assert result.status == "error"
        assert "load:FAIL" in result.lifecycle_log

    def test_execute_no_trade_log(self):
        from causal_edge.harness.lifecycle import execute_strategy
        cfg = {"id": "bad", "engine": "tests.test_harness", "trade_log": ""}
        result = execute_strategy(cfg)
        assert result.status == "error"
        assert "validate:FAIL" in result.lifecycle_log


class TestPipeline:
    def test_empty_config(self):
        from causal_edge.harness.pipeline import run_pipeline
        events = list(run_pipeline({"strategies": []}))
        assert any(e.status == "error" for e in events)

    def test_pipeline_yields_events(self):
        from causal_edge.harness.pipeline import run_pipeline
        config = {"strategies": [
            {"id": "mock", "engine": "tests.test_harness",
             "trade_log": "/tmp/test_harness_mock.csv"}
        ]}
        # This will fail at engine load (MockEngine not auto-discoverable via import)
        # but it should yield events, not crash
        events = list(run_pipeline(config))
        assert len(events) > 0
        assert events[0].phase == "run"
        assert events[0].status == "start"
        assert events[-1].phase == "pipeline"
