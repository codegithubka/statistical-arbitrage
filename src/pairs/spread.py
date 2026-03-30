"""
Spread construction using a Kalman filter hedge ratio.

State-space model (random-walk hedge ratio):
    beta_t  = beta_{t-1} + w_t        w_t ~ N(0, Q)   [state equation]
    P_A(t) = beta_t * P_B(t) + v_t  v_t ~ N(0, R)   [observation equation]

The Kalman filter estimates beta_t at each step.  Spread and z-score follow.
"""

import logging
import numpy as np
import pandas as pd
from pykalman import KalmanFilter

logger = logging.getLogger(__name__)


def kalman_hedge_ratio(
    price_a: pd.Series,
    price_b: pd.Series,
    trans_cov: float = 1e-4,
    obs_cov: float | None = None,
) -> pd.Series:
    """
    Estimate a time-varying hedge ratio beta_t via Kalman filter.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Aligned price series for the two legs of the pair.
    trans_cov : float
        Process (transition) noise Q -- controls how fast beta_t can drift.
        Larger -> faster adaptation; smaller -> smoother beta.
    obs_cov : float, optional
        Observation noise R.  Defaults to the variance of OLS residuals.

    Returns
    -------
    pd.Series
        Time-varying hedge ratio beta_t, same index as inputs.
    """
    a = price_a.values.astype(float)
    b = price_b.values.astype(float)
    n = len(a)

    # Bootstrap initial state from OLS
    beta_ols = np.polyfit(b, a, 1)[0]

    if obs_cov is None:
        residuals = a - beta_ols * b
        obs_cov = float(np.var(residuals))

    # Observation matrix is time-varying: H_t = [[P_B(t)]]
    obs_mat = b.reshape(n, 1, 1)

    kf = KalmanFilter(
        transition_matrices=np.array([[1.0]]),
        observation_matrices=obs_mat,
        transition_covariance=np.array([[trans_cov]]),
        observation_covariance=np.array([[obs_cov]]),
        initial_state_mean=np.array([beta_ols]),
        initial_state_covariance=np.array([[1.0]]),
    )

    state_means, _ = kf.filter(a)
    beta_t = state_means[:, 0]

    logger.debug(
        f"Kalman hedge ratio: OLS beta0={beta_ols:.4f}, "
        f"final beta_T={beta_t[-1]:.4f}, range=[{beta_t.min():.4f}, {beta_t.max():.4f}]"
    )

    return pd.Series(beta_t, index=price_a.index, name="hedge_ratio")


def build_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    beta_t: pd.Series,
) -> pd.Series:
    """
    Construct the spread: S_t = P_A(t) - beta_t * P_B(t).

    Parameters
    ----------
    price_a, price_b : pd.Series
        Aligned price series.
    beta_t : pd.Series
        Time-varying hedge ratio (same index).

    Returns
    -------
    pd.Series
        Spread series S_t.
    """
    spread = price_a - beta_t * price_b
    spread.name = "spread"
    return spread


def zscore(spread: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling z-score of the spread.

    Z_t = (S_t - mu_t) / sigma_t
    where mu_t and sigma_t are the rolling mean and std over `window` days.

    Parameters
    ----------
    spread : pd.Series
        Raw spread series.
    window : int
        Lookback window in trading days (default 252 = 1 year).

    Returns
    -------
    pd.Series
        Z-scored spread, NaN for the first `window - 1` observations.
    """
    mu = spread.rolling(window).mean()
    sigma = spread.rolling(window).std(ddof=1)
    z = (spread - mu) / sigma
    z.name = "zscore"
    return z


def compute_pair_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    zscore_window: int = 252,
    trans_cov: float = 1e-4,
    obs_cov: float | None = None,
) -> pd.DataFrame:
    """
    Full pipeline for a single pair: Kalman beta_t -> spread -> z-score.

    Parameters
    ----------
    price_a, price_b : pd.Series
        Aligned price series for tickers A and B.
    zscore_window : int
        Rolling window for z-score normalisation.
    trans_cov : float
        Kalman process noise Q.
    obs_cov : float, optional
        Kalman observation noise R (defaults to OLS residual variance).

    Returns
    -------
    pd.DataFrame
        Columns: hedge_ratio, spread, zscore.
    """
    beta_t = kalman_hedge_ratio(price_a, price_b, trans_cov=trans_cov, obs_cov=obs_cov)
    spread = build_spread(price_a, price_b, beta_t)
    z = zscore(spread, window=zscore_window)

    return pd.DataFrame({"hedge_ratio": beta_t, "spread": spread, "zscore": z})
