"""
Backtesting engine for statistical arbitrage pairs.

Three levels of backtesting:
    backtest_pair          -- single pair, full-sample (no walk-forward)
    walk_forward_pair      -- single pair, rolling formation/trading windows
    run_portfolio_backtest -- multi-pair portfolio orchestrator
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.pairs.spread import build_spread, compute_pair_spread, kalman_hedge_ratio, zscore
from src.models.ou_process import calibrate_ou
from src.strategy.signals import generate_signals

logger = logging.getLogger(__name__)


@dataclass
class BacktestParams:
    # Signal thresholds (z-score units)
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float = 3.5

    # Spread construction
    zscore_window: int = 252        # rolling window for z-score normalisation
    trans_cov: float = 1e-4         # Kalman process noise Q

    # Transaction costs
    cost_bps: float = 5.0           # one-way cost in basis points
    notional: float = 1.0           # notional per unit of spread position

    # Walk-forward windows (in trading days)
    formation: int = 252
    trading: int = 63
    step: int = 21


# ---------------------------------------------------------------------------
# Function 1: single pair, full-sample backtest
# ---------------------------------------------------------------------------

def backtest_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    params: BacktestParams,
) -> pd.DataFrame:
    """
    Full-sample, single-pair backtest.

    Steps
    -----
    1. Kalman hedge ratio -> spread -> z-score via compute_pair_spread
    2. generate_signals on z-score
    3. daily spread returns:     DeltaS_t = S_t - S_{t-1}
    4. daily PnL:                position_{t-1} x DeltaS_t
    5. transaction cost:         |Deltaposition| x cost_bps/10000 x notional
    6. net PnL and cumulative PnL

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Aligned price series for legs A and B.
    params : BacktestParams

    Returns
    -------
    pd.DataFrame
        Columns: spread, zscore, position, daily_pnl, cumulative_pnl
    """
    spread_df = compute_pair_spread(
        prices_a, prices_b,
        zscore_window=params.zscore_window,
        trans_cov=params.trans_cov,
    )

    spread = spread_df["spread"]
    z = spread_df["zscore"]

    # Drop NaN rows (first zscore_window-1 rows) before generating signals
    valid = z.notna()
    position = pd.Series(0, index=z.index, dtype=int)
    position[valid] = generate_signals(
        z[valid], params.entry_z, params.exit_z, params.stop_z
    )

    # Spread returns (NaN on first row)
    spread_ret = spread.diff()

    # PnL: yesterday's position x today's spread move
    daily_pnl = position.shift(1) * spread_ret

    # Transaction costs on position changes
    cost_per_unit = params.cost_bps / 10_000 * params.notional
    trade_cost = position.diff().abs() * cost_per_unit
    trade_cost.iloc[0] = abs(position.iloc[0]) * cost_per_unit  # first bar

    daily_pnl = daily_pnl.fillna(0.0) - trade_cost.fillna(0.0)

    cumulative_pnl = daily_pnl.cumsum()

    return pd.DataFrame(
        {
            "spread": spread,
            "zscore": z,
            "position": position,
            "daily_pnl": daily_pnl,
            "cumulative_pnl": cumulative_pnl,
        }
    )


# ---------------------------------------------------------------------------
# Function 2: walk-forward index splitter
# ---------------------------------------------------------------------------

def walk_forward_split(
    dates: pd.DatetimeIndex,
    formation: int,
    trading: int,
    step: int,
) -> list[tuple[pd.DatetimeIndex, pd.DatetimeIndex]]:
    """
    Generate rolling (formation, trading) window index pairs.

    The formation window ends where the trading window begins.
    Windows advance by `step` days each iteration.

    Parameters
    ----------
    dates : pd.DatetimeIndex
        Full date range of the price data.
    formation : int
        Length of the formation (training) window in days.
    trading : int
        Length of the trading (test) window in days.
    step : int
        Number of days to advance each iteration.

    Returns
    -------
    list of (formation_dates, trading_dates)
        Each element is a tuple of DatetimeIndex slices.
    """
    n = len(dates)
    windows = []
    start = 0

    while start + formation + trading <= n:
        form_idx = dates[start : start + formation]
        trade_idx = dates[start + formation : start + formation + trading]
        windows.append((form_idx, trade_idx))
        start += step

    logger.debug(
        f"walk_forward_split: {len(windows)} windows "
        f"(formation={formation}, trading={trading}, step={step}, n={n})"
    )
    return windows


# ---------------------------------------------------------------------------
# Function 3: single pair, walk-forward backtest
# ---------------------------------------------------------------------------

def walk_forward_pair(
    prices_a: pd.Series,
    prices_b: pd.Series,
    params: BacktestParams,
) -> pd.DataFrame:
    """
    Walk-forward single-pair backtest.

    For each window:
      - Formation window: run Kalman filter on prices_a/prices_b,
        calibrate z-score normalisation (rolling mean/std from spread).
      - Trading window: generate signals using formation-window z-score
        stats, compute PnL.

    The Kalman filter state carries forward across windows (no reset).
    Z-score normalisation (mean and std) is re-estimated each formation window.

    Parameters
    ----------
    prices_a, prices_b : pd.Series
        Full-history aligned price series.
    params : BacktestParams

    Returns
    -------
    pd.DataFrame
        Same columns as backtest_pair: spread, zscore, position,
        daily_pnl, cumulative_pnl -- covering only trading-window dates.
    """
    windows = walk_forward_split(
        prices_a.index, params.formation, params.trading, params.step
    )
    if not windows:
        raise ValueError(
            f"No windows generated: need at least {params.formation + params.trading} "
            f"observations, got {len(prices_a)}."
        )

    # Run Kalman filter over the full history once so state carries forward.
    # We extract beta_t for arbitrary sub-windows from this single pass.
    full_beta = kalman_hedge_ratio(prices_a, prices_b, trans_cov=params.trans_cov)
    full_spread = build_spread(prices_a, prices_b, full_beta)

    pieces: list[pd.DataFrame] = []

    for form_idx, trade_idx in windows:
        # --- Formation: estimate z-score normalisation params ---
        form_spread = full_spread.loc[form_idx]
        form_mu = form_spread.mean()
        form_std = form_spread.std(ddof=1)

        if form_std == 0 or np.isnan(form_std):
            logger.warning(
                f"Zero/NaN std in formation window ending {form_idx[-1].date()}; skipping."
            )
            continue

        # --- Trading: apply formation-window normalisation ---
        trade_spread = full_spread.loc[trade_idx]
        trade_z = (trade_spread - form_mu) / form_std

        trade_pos = generate_signals(
            trade_z, params.entry_z, params.exit_z, params.stop_z
        )

        # Spread returns within the trading window
        # Include the last formation-window spread value to get a return on day 1
        last_form_spread = form_spread.iloc[-1]
        spread_with_prev = pd.concat(
            [pd.Series([last_form_spread], index=[form_idx[-1]]), trade_spread]
        )
        spread_ret = spread_with_prev.diff().iloc[1:]  # drop the prepended row

        daily_pnl = trade_pos.shift(1).fillna(0) * spread_ret

        cost_per_unit = params.cost_bps / 10_000 * params.notional
        trade_cost = trade_pos.diff().abs() * cost_per_unit
        trade_cost.iloc[0] = abs(trade_pos.iloc[0]) * cost_per_unit

        daily_pnl = daily_pnl.fillna(0.0) - trade_cost.fillna(0.0)

        pieces.append(
            pd.DataFrame(
                {
                    "spread": trade_spread,
                    "zscore": trade_z,
                    "position": trade_pos,
                    "daily_pnl": daily_pnl,
                }
            )
        )

    if not pieces:
        raise ValueError("All walk-forward windows were skipped.")

    result = pd.concat(pieces)

    # De-duplicate overlapping trading windows (keep last assignment)
    result = result[~result.index.duplicated(keep="last")]
    result = result.sort_index()
    result["cumulative_pnl"] = result["daily_pnl"].cumsum()

    return result


# ---------------------------------------------------------------------------
# Function 4: portfolio orchestrator
# ---------------------------------------------------------------------------

def run_portfolio_backtest(
    prices: pd.DataFrame,
    pairs_df: pd.DataFrame,
    params: BacktestParams,
) -> dict[str, Any]:
    """
    Walk-forward backtest across all pairs; aggregate into an equal-weight portfolio.

    Parameters
    ----------
    prices : pd.DataFrame
        Full price matrix, columns are tickers.
    pairs_df : pd.DataFrame
        Output of test_all_pairs -- must have columns ticker_a and ticker_b.
    params : BacktestParams

    Returns
    -------
    dict with keys:
        "pairs"     -- dict mapping "{ticker_a}/{ticker_b}" -> per-pair DataFrame
        "portfolio" -- pd.DataFrame with columns: daily_pnl, cumulative_pnl,
                      n_active_pairs (number of pairs with non-NaN PnL on each date)
    """
    pair_results: dict[str, pd.DataFrame] = {}

    for _, row in pairs_df.iterrows():
        a, b = row["ticker_a"], row["ticker_b"]
        pair_key = f"{a}/{b}"

        if a not in prices.columns or b not in prices.columns:
            logger.warning(f"Skipping {pair_key}: tickers not in price matrix.")
            continue

        logger.info(f"Backtesting {pair_key} ...")
        try:
            result = walk_forward_pair(prices[a], prices[b], params)
            pair_results[pair_key] = result
        except Exception as exc:
            logger.warning(f"Skipping {pair_key}: {exc}")
            continue

    if not pair_results:
        raise ValueError("No pairs backtested successfully.")

    # --- Aggregate into equal-weight portfolio ---
    pnl_matrix = pd.DataFrame(
        {key: df["daily_pnl"] for key, df in pair_results.items()}
    )

    n_active = pnl_matrix.notna().sum(axis=1)
    portfolio_pnl = pnl_matrix.sum(axis=1) / n_active.replace(0, np.nan)

    portfolio = pd.DataFrame(
        {
            "daily_pnl": portfolio_pnl,
            "cumulative_pnl": portfolio_pnl.cumsum(),
            "n_active_pairs": n_active,
        }
    )

    logger.info(
        f"Portfolio backtest complete: {len(pair_results)} pairs, "
        f"{len(portfolio)} trading days."
    )

    return {"pairs": pair_results, "portfolio": portfolio}
