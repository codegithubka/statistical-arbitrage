"""
Pair selection via Engle-Granger cointegration testing (Gatev et al. 2006).
"""

import logging
from itertools import combinations
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

logger = logging.getLogger(__name__)


def test_all_pairs(
    prices: pd.DataFrame,
    pvalue_threshold: float = 0.05,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Run Engle-Granger cointegration test on all unique pairs.

    Parameters
    ----------
    prices : pd.DataFrame
        Cleaned price matrix (DatetimeIndex × tickers).
    pvalue_threshold : float
        Maximum p-value to keep a pair.
    top_n : int
        Return only the top N pairs ranked by p-value.

    Returns
    -------
    pd.DataFrame
        Columns: ticker_a, ticker_b, coint_pvalue, hedge_ratio_ols.
        Sorted by coint_pvalue ascending.
    """
    tickers = prices.columns.tolist()
    n_pairs = len(tickers) * (len(tickers) - 1) // 2
    logger.info(f"Testing {n_pairs} pairs for cointegration...")

    results = []
    for i, (a, b) in enumerate(combinations(tickers, 2)):
        if (i + 1) % 500 == 0:
            logger.info(f"  ... {i + 1}/{n_pairs} pairs tested")

        score, pvalue, _ = coint(prices[a].values, prices[b].values)

        if pvalue > pvalue_threshold:
            continue

        beta = np.polyfit(prices[b].values, prices[a].values, 1)[0]

        results.append({
            "ticker_a": a,
            "ticker_b": b,
            "coint_pvalue": round(pvalue, 6),
            "hedge_ratio_ols": round(beta, 6),
        })

    df = (
        pd.DataFrame(results)
        .sort_values("coint_pvalue")
        .head(top_n)
        .reset_index(drop=True)
    )
    logger.info(f"Found {len(df)} cointegrated pairs (p < {pvalue_threshold}), returning top {top_n}")
    return df