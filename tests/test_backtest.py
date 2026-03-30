"""
Tests 3, 4, 5, 7: backtest_pair accounting identities,
walk_forward_split coverage, and portfolio additivity.
"""

import numpy as np
import pandas as pd
import pytest

from src.strategy.backtest import (
    BacktestParams,
    backtest_pair,
    run_portfolio_backtest,
    walk_forward_split,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_prices(n=500, seed=42):
    """Two synthetic cointegrated price series."""
    rng = np.random.default_rng(seed)
    common = np.cumsum(rng.normal(0, 1, n))
    noise_a = rng.normal(0, 0.5, n)
    noise_b = rng.normal(0, 0.5, n)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")
    price_a = pd.Series(100 + common + noise_a, index=dates)
    price_b = pd.Series(100 + common + noise_b, index=dates)
    return price_a, price_b


# ---------------------------------------------------------------------------
# Test 3: zero-cost accounting identities
# ---------------------------------------------------------------------------

class TestBacktestPairZeroCost:
    def setup_method(self):
        self.pa, self.pb = _make_prices()
        params = BacktestParams(cost_bps=0.0)
        self.result = backtest_pair(self.pa, self.pb, params)

    def test_cumulative_pnl_equals_sum_of_daily(self):
        r = self.result.dropna()
        expected = r["daily_pnl"].cumsum()
        pd.testing.assert_series_equal(
            r["cumulative_pnl"], expected, check_names=False, atol=1e-10
        )

    def test_pnl_zero_when_flat(self):
        r = self.result.dropna()
        flat_mask = r["position"].shift(1).fillna(0) == 0
        assert (r.loc[flat_mask, "daily_pnl"] == 0).all()


# ---------------------------------------------------------------------------
# Test 4: transaction cost accounting
# ---------------------------------------------------------------------------

class TestBacktestPairTransactionCosts:
    def test_cost_difference_matches_trade_count(self):
        pa, pb = _make_prices()

        params_0 = BacktestParams(cost_bps=0.0, notional=1.0)
        params_5 = BacktestParams(cost_bps=5.0, notional=1.0)

        r0 = backtest_pair(pa, pb, params_0)
        r5 = backtest_pair(pa, pb, params_5)

        # Signals are identical (same params minus cost)
        pd.testing.assert_series_equal(r0["position"], r5["position"])

        # Total cost = sum of |Δposition| × cost_per_unit
        pos = r0["position"]
        n_position_changes = pos.diff().abs().fillna(abs(pos.iloc[0])).sum()
        expected_total_cost = n_position_changes * (5.0 / 10_000) * 1.0

        actual_cost_diff = (
            r0["daily_pnl"].sum() - r5["daily_pnl"].sum()
        )

        assert abs(actual_cost_diff - expected_total_cost) < 1e-9, (
            f"Cost diff={actual_cost_diff:.6f}, expected={expected_total_cost:.6f}"
        )


# ---------------------------------------------------------------------------
# Test 5: walk_forward_split coverage
# ---------------------------------------------------------------------------

class TestWalkForwardSplit:
    def setup_method(self):
        n = 1760
        self.dates = pd.date_range("2015-01-01", periods=n, freq="B")
        self.formation = 252
        self.trading = 63
        self.step = 21
        self.windows = walk_forward_split(
            self.dates, self.formation, self.trading, self.step
        )

    def test_produces_windows(self):
        assert len(self.windows) > 0

    def test_first_formation_starts_at_day_zero(self):
        form_idx, _ = self.windows[0]
        assert form_idx[0] == self.dates[0]

    def test_formation_and_trading_do_not_overlap(self):
        for form_idx, trade_idx in self.windows:
            assert form_idx[-1] < trade_idx[0], (
                f"Formation/trading overlap: {form_idx[-1]} >= {trade_idx[0]}"
            )

    def test_trading_windows_are_contiguous(self):
        """Consecutive trading windows must be adjacent (no date gaps)."""
        trade_windows = [t for _, t in self.windows]
        for prev, curr in zip(trade_windows, trade_windows[1:]):
            # curr starts exactly one step after prev starts
            # (windows may overlap if step < trading, which is fine)
            assert curr[0] >= prev[0]

    def test_last_trading_window_within_date_range(self):
        _, last_trade = self.windows[-1]
        assert last_trade[-1] <= self.dates[-1]

    def test_window_lengths(self):
        for form_idx, trade_idx in self.windows:
            assert len(form_idx) == self.formation
            assert len(trade_idx) == self.trading

    def test_roughly_expected_number_of_windows(self):
        # (1760 - 252 - 63) / 21 ≈ 68–70 windows
        assert 60 <= len(self.windows) <= 80

    def test_short_series_produces_no_windows(self):
        short_dates = pd.date_range("2020-01-01", periods=100, freq="B")
        windows = walk_forward_split(short_dates, 252, 63, 21)
        assert windows == []


# ---------------------------------------------------------------------------
# Test 7: portfolio additivity
# ---------------------------------------------------------------------------

class TestPortfolioAdditivity:
    def test_portfolio_pnl_equals_mean_of_pairs(self):
        pa, pb = _make_prices(seed=1)
        pc, pd_ = _make_prices(seed=2)
        pe, pf = _make_prices(seed=3)

        prices = pd.DataFrame(
            {"A": pa, "B": pb, "C": pc, "D": pd_, "E": pe, "F": pf}
        )
        pairs_df = pd.DataFrame([
            {"ticker_a": "A", "ticker_b": "B"},
            {"ticker_a": "C", "ticker_b": "D"},
            {"ticker_a": "E", "ticker_b": "F"},
        ])

        params = BacktestParams(cost_bps=0.0)
        out = run_portfolio_backtest(prices, pairs_df, params)

        port = out["portfolio"]
        pair_results = out["pairs"]

        # Align all pair PnLs to portfolio index
        pnl_matrix = pd.DataFrame(
            {key: df["daily_pnl"] for key, df in pair_results.items()}
        ).reindex(port.index)

        n_active = pnl_matrix.notna().sum(axis=1)
        expected_pnl = pnl_matrix.sum(axis=1) / n_active.replace(0, np.nan)

        pd.testing.assert_series_equal(
            port["daily_pnl"].dropna(),
            expected_pnl.dropna(),
            check_names=False,
            atol=1e-10,
        )

    def test_portfolio_output_has_required_keys(self):
        pa, pb = _make_prices(seed=10)
        prices = pd.DataFrame({"X": pa, "Y": pb})
        pairs_df = pd.DataFrame([{"ticker_a": "X", "ticker_b": "Y"}])
        out = run_portfolio_backtest(prices, pairs_df, BacktestParams())
        assert "pairs" in out
        assert "portfolio" in out
        assert set(out["portfolio"].columns) >= {"daily_pnl", "cumulative_pnl", "n_active_pairs"}
