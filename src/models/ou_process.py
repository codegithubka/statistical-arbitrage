"""
Ornstein-Uhlenbeck process calibration for spread modelling.

Reference: Ornstein & Uhlenbeck (1930); Avellaneda & Lee (2010).
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OUParams:
    mu: float      # long-run mean
    theta: float   # mean-reversion speed (per day)
    sigma: float   # diffusion volatility (annualised)
    half_life: float  # ln(2) / theta, in trading days


def calibrate_ou(spread: pd.Series | np.ndarray) -> OUParams:
    """
    Calibrate OU parameters from discrete spread observations via MLE.

    Parameters
    ----------
    spread : array-like
        Time series of spread values (e.g. from build_spread).

    Returns
    -------
    OUParams

    Raises
    ------
    ValueError
        If b >= 1 (non-stationary) or theta <= 0 (no mean reversion detected).
    """
    s = np.asarray(spread, dtype=float)
    s = s[~np.isnan(s)]

    if len(s) < 30:
        raise ValueError(f"Spread has only {len(s)} observations; need >= 30 for reliable calibration.")

    # AR(1) OLS: regress S_{t+1} on [1, S_t]
    y = s[1:]
    x = s[:-1]
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
    a, b = coeffs

    if b >= 1.0:
        raise ValueError(
            f"AR(1) coefficient b={b:.4f} >= 1: spread is non-stationary, OU model not applicable."
        )

    sigma_eps = np.std(y - X @ coeffs, ddof=2)

    # Recover OU parameters (dt = 1 trading day)
    theta = -np.log(b)
    mu = a / (1.0 - b)
    # Annualise sigma: sigma_OU = sigma_eps * sqrt(2theta / (1 - e^{-2theta})) ~ sigma_eps * sqrt(2theta / (1 - b^2))
    sigma = sigma_eps * np.sqrt(2.0 * theta / (1.0 - b**2))

    hl = half_life(theta)

    logger.debug(
        f"OU calibration: mu={mu:.4f}, theta={theta:.6f}/day, "
        f"sigma={sigma:.6f}, half-life={hl:.1f} days"
    )

    return OUParams(mu=mu, theta=theta, sigma=sigma, half_life=hl)


def half_life(theta: float) -> float:
    """
    Mean-reversion half-life in trading days.

    Half-life = ln(2) / theta

    A half-life of 5-30 days is generally considered tradeable.
    Too short (< 3 days) -> transaction costs dominate.
    Too long (> 60 days) -> capital locked up, low edge.

    Parameters
    ----------
    theta : float
        Mean-reversion speed from calibrate_ou (per day).

    Returns
    -------
    float
        Half-life in trading days.
    """
    if theta <= 0:
        return np.inf
    return np.log(2.0) / theta


def ou_optimal_threshold(
    theta: float,
    sigma: float,
    cost: float,
) -> float:
    """
    Analytically optimal entry threshold for an OU spread, net of costs.

    Derived from Avellaneda & Lee (2010), eq. for the optimal entry level c*
    that maximises expected P&L per unit time net of transaction costs:

        c* = sigma / sqrt(2theta) * 1 / sqrt(1 - 2theta*cost^2 / sigma^2)

    This is the level (in spread units) at which to enter a position.
    The corresponding z-score entry is c* / (sigma / sqrt(2theta)).

    Parameters
    ----------
    theta : float
        Mean-reversion speed (per day).
    sigma : float
        OU diffusion parameter (per day, same units as spread).
    cost : float
        One-way transaction cost in spread units
        (e.g. bid-ask + market impact, expressed as a spread-level move).

    Returns
    -------
    float
        Optimal entry threshold c* in spread units.
        Returns np.nan if cost is too large relative to the OU dynamics
        (i.e. trading is not profitable at any threshold).
    """
    if theta <= 0 or sigma <= 0:
        return np.nan

    # Equilibrium spread width: sigma_eq = sigma / sqrt(2theta)
    sigma_eq = sigma / np.sqrt(2.0 * theta)

    discriminant = 1.0 - 2.0 * theta * cost**2 / sigma**2
    if discriminant <= 0:
        logger.warning(
            f"Transaction cost ({cost:.6f}) exceeds profitable threshold for "
            f"theta={theta:.6f}, sigma={sigma:.6f}. No profitable entry exists."
        )
        return np.nan

    c_star = sigma_eq / np.sqrt(discriminant)

    logger.debug(
        f"OU optimal threshold: c*={c_star:.4f} spread units "
        f"(sigma_eq={sigma_eq:.4f}, cost={cost:.6f})"
    )

    return c_star
