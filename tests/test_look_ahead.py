"""Tests for look-ahead bias detection."""

import numpy as np
import pytest

from causal_edge.validation.look_ahead import check_static, check_runtime


class TestT2RollingWithoutShift:
    def test_catches_rolling_mean(self):
        code = "x = ret.rolling(20).mean()\npositions = x > 0"
        assert any("T2" in v for v in check_static(code))

    def test_allows_rolling_with_shift(self):
        code = "x = ret.rolling(20).mean().shift(1)\npositions = x > 0"
        assert not any("T2" in v for v in check_static(code))

    def test_catches_rolling_std(self):
        code = "vol = ret.rolling(20).std()\npositions = vol < 0.02"
        assert any("T2" in v for v in check_static(code))

    def test_allows_target_variable(self):
        code = "y = ret.rolling(5).sum()  # target\nmodel.fit(X, y)"
        assert not any("T2" in v for v in check_static(code))

    def test_catches_rolling_corr(self):
        code = "xc = a.rolling(60).corr(b)\npositions = xc > 0"
        assert any("T2" in v for v in check_static(code))


class TestT3GlobalStats:
    def test_catches_np_std_full(self):
        code = "s = np.std(pnl_array)\nsharpe = mean / s"
        v = check_static(code)
        assert any("T3" in x for x in v)

    def test_allows_np_std_on_slice_with_ddof(self):
        code = "s = np.std(pnl[:i], ddof=1)"
        assert not any("T3" in v for v in check_static(code))

    def test_allows_np_std_on_slice(self):
        code = "s = np.std(pnl[:i])"
        assert not any("T3" in v for v in check_static(code))

    def test_allows_def_use_slice_var(self):
        code = "def f(pnl, i):\n    w = pnl[:i]\n    s = np.std(w)"
        assert not any("T3" in v for v in check_static(code))

    def test_allows_list_comprehension(self):
        code = "m = np.mean([x for x in items if x > 0])"
        assert not any("T3" in v for v in check_static(code))

    def test_allows_ml_validation_params(self):
        code = (
            "def score(x_fit, y_fit, x_val, y_val):\n"
            "    prob = model.predict(x_val)\n"
            "    return np.mean((prob - y_val) ** 2)"
        )
        assert not any("T3" in v for v in check_static(code))

    def test_flags_unbounded_in_metrics_context(self):
        code = "def f(pnl):\n    return np.mean(pnl) / np.std(pnl)"
        assert any("T3" in v for v in check_static(code))


class TestT4WFSlicing:
    def test_catches_i_plus_1_in_train(self):
        code = "X_train = X[:i+1]\nmodel.fit(X_train, y_train)"
        v = check_static(code)
        assert any("T4" in x for x in v)

    def test_allows_normal_slicing(self):
        code = "X_train = X[:i]\nmodel.fit(X_train, y)"
        assert not any("T4" in v for v in check_static(code))


class TestT5TrendFilter:
    def test_catches_current_day_comparison(self):
        code = "if close[i] < sma[i]: positions[i] = 0"
        v = check_static(code)
        assert any("T5" in x for x in v)

    def test_allows_shifted_comparison(self):
        code = "if close[i-1] < sma[i-1]: positions[i] = 0"
        assert not any("T5" in v for v in check_static(code))


class TestRuntimeR1:
    def test_clean_strategy(self):
        rng = np.random.default_rng(42)
        positions = rng.choice([0, 0.5, 1.0], size=200)
        returns = rng.normal(0, 0.02, 200)
        pnl = positions * returns
        warnings = check_runtime(pnl, positions, returns)
        assert not any("R1" in w for w in warnings)

    def test_catches_leaking_positions(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.02, 200)
        # Positions perfectly sized to match return magnitude = leak
        positions = np.abs(returns) * 50
        pnl = positions * returns
        warnings = check_runtime(pnl, positions, returns)
        assert any("R1" in w for w in warnings)


class TestRuntimeR2:
    def test_normal_hit_rate(self):
        rng = np.random.default_rng(42)
        pnl = rng.normal(0.001, 0.02, 200)
        positions = np.ones(200) * 0.5
        warnings = check_runtime(pnl, positions)
        assert not any("R2" in w for w in warnings)

    def test_suspicious_hit_rate(self):
        pnl = np.abs(np.random.default_rng(42).normal(0, 0.02, 200))  # all positive
        positions = np.ones(200) * 0.5
        warnings = check_runtime(pnl, positions)
        assert any("R2" in w for w in warnings)
