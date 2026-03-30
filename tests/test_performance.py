"""
Test 8: compute_metrics with known inputs.
"""

import numpy as np
import pandas as pd
import pytest

from src.analysis.performance import compute_metrics, compute_metrics_with_positions

TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pnl(values):
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Constant positive PnL
# ---------------------------------------------------------------------------

class TestConstantPositivePnL:
    def setup_method(self):
        self.pnl = _pnl([1.0] * TRADING_DAYS)
        self.m = compute_metrics(self.pnl)

    def test_sharpe_is_nan_or_very_large(self):
        # std = 0 → Sharpe is undefined; implementation returns NaN
        assert np.isnan(self.m["sharpe"])

    def test_max_drawdown_is_zero(self):
        assert self.m["max_drawdown"] == 0.0

    def test_annual_return_positive(self):
        assert self.m["annual_return"] == pytest.approx(TRADING_DAYS * 1.0)

    def test_win_rate_is_one(self):
        assert self.m["win_rate"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Alternating +1 / -1 PnL
# ---------------------------------------------------------------------------

class TestAlternatingPnL:
    def setup_method(self):
        values = [1.0 if i % 2 == 0 else -1.0 for i in range(TRADING_DAYS)]
        self.pnl = _pnl(values)
        self.m = compute_metrics(self.pnl)

    def test_sharpe_near_zero(self):
        assert abs(self.m["sharpe"]) < 0.1

    def test_win_rate_near_half(self):
        assert self.m["win_rate"] == pytest.approx(0.5)

    def test_max_drawdown_negative(self):
        assert self.m["max_drawdown"] < 0

    def test_annual_return_near_zero(self):
        assert abs(self.m["annual_return"]) < 1.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_series_returns_all_nan():
    m = compute_metrics(_pnl([]))
    for key in ("sharpe", "max_drawdown", "annual_return", "win_rate", "turnover"):
        assert np.isnan(m[key]), f"{key} should be NaN for empty series"


def test_all_nan_series_returns_all_nan():
    m = compute_metrics(_pnl([np.nan, np.nan, np.nan]))
    for key in ("sharpe", "max_drawdown", "annual_return", "win_rate"):
        assert np.isnan(m[key])


def test_all_zeros_win_rate_is_nan():
    # No active (non-zero) days → win rate undefined
    m = compute_metrics(_pnl([0.0] * 20))
    assert np.isnan(m["win_rate"])


def test_sharpe_sign_matches_return():
    pos_pnl = _pnl([0.1] * 50 + [-0.01] * 50)
    neg_pnl = _pnl([-0.1] * 50 + [0.01] * 50)
    assert compute_metrics(pos_pnl)["sharpe"] > 0
    assert compute_metrics(neg_pnl)["sharpe"] < 0


def test_max_drawdown_known_sequence():
    # Equity: 0 → 3 → 2 → 5 → 1  →  max DD = 1 - 5 = -4
    pnl = _pnl([3.0, -1.0, 3.0, -4.0])
    m = compute_metrics(pnl)
    assert m["max_drawdown"] == pytest.approx(-4.0)


# ---------------------------------------------------------------------------
# Turnover via compute_metrics_with_positions
# ---------------------------------------------------------------------------

def test_turnover_full_reversal_every_day():
    # Position alternates +1 / -1 → daily change = 2 → turnover = 2
    pos = _pnl([1.0, -1.0, 1.0, -1.0, 1.0])
    pnl = _pnl([0.1] * 5)
    m = compute_metrics_with_positions(pnl, pos)
    assert m["turnover"] == pytest.approx(2.0)


def test_turnover_no_trades():
    pos = _pnl([1.0, 1.0, 1.0, 1.0])
    pnl = _pnl([0.1] * 4)
    m = compute_metrics_with_positions(pnl, pos)
    assert m["turnover"] == pytest.approx(0.0)
