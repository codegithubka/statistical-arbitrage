"""
Post-backtest performance metrics.
"""

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def compute_metrics(pnl_series: pd.Series) -> dict:
    """
    Compute performance metrics from a daily PnL series.

    Parameters
    ----------
    pnl_series : pd.Series
        Daily net PnL (not cumulative). NaN values are dropped.

    Returns
    -------
    dict with keys:
        sharpe          -- annualised Sharpe ratio (assumes risk-free rate = 0)
        max_drawdown    -- peak-to-trough drawdown (in same units as PnL)
        annual_return   -- annualised mean daily PnL
        win_rate        -- fraction of non-zero trading days with positive PnL
        turnover        -- mean absolute daily change in position (requires
                          position_series kwarg; NaN if not supplied)
    """
    pnl = pnl_series.dropna()

    if len(pnl) == 0:
        return {
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "annual_return": np.nan,
            "win_rate": np.nan,
            "turnover": np.nan,
        }

    # Annualised Sharpe (risk-free = 0)
    mean_daily = pnl.mean()
    std_daily = pnl.std(ddof=1)
    sharpe = (mean_daily / std_daily * np.sqrt(TRADING_DAYS_PER_YEAR)
              if std_daily > 0 else np.nan)

    # Maximum drawdown (peak-to-trough on cumulative PnL)
    cum = pnl.cumsum()
    running_peak = cum.cummax()
    drawdown = cum - running_peak          # always <= 0
    max_drawdown = drawdown.min()          # most negative value

    # Annualised return
    annual_return = mean_daily * TRADING_DAYS_PER_YEAR

    # Win rate: fraction of active (non-zero) days with positive PnL
    active = pnl[pnl != 0]
    win_rate = (active > 0).mean() if len(active) > 0 else np.nan

    # Turnover: not computable from PnL alone -- caller must pass position_series
    turnover = np.nan

    return {
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "annual_return": annual_return,
        "win_rate": win_rate,
        "turnover": turnover,
    }


def compute_metrics_with_positions(
    pnl_series: pd.Series,
    position_series: pd.Series,
) -> dict:
    """
    compute_metrics plus turnover, computed from the position series.

    Turnover is the mean absolute daily position change -- a value of 1.0
    means the full position is turned over every day on average.

    Parameters
    ----------
    pnl_series : pd.Series
        Daily net PnL.
    position_series : pd.Series
        Position labels (+1, 0, -1), aligned to pnl_series.
    """
    metrics = compute_metrics(pnl_series)
    pos = position_series.reindex(pnl_series.dropna().index).fillna(0)
    metrics["turnover"] = pos.diff().abs().mean()
    return metrics
