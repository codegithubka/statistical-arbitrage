"""
Download and cache daily price data for a US equity universe.
"""

import logging
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

logger = logging.getLogger(__name__)

# S&P 100 tickers (OEX constituents as of late 2024)
SP100_TICKERS = [
    "AAPL", "ABBV", "ABT", "ACN", "ADBE", "AIG", "AMD", "AMGN", "AMT", "AMZN",
    "AVGO", "AXP", "BA", "BAC", "BK", "BKNG", "BLK", "BMY", "BRK-B", "C",
    "CAT", "CHTR", "CL", "CMCSA", "COF", "COP", "COST", "CRM", "CSCO", "CVS",
    "CVX", "DE", "DHR", "DIS", "DOW", "DUK", "EMR", "EXC", "F", "FDX",
    "GD", "GE", "GILD", "GM", "GOOG", "GS", "HD", "HON", "IBM", "INTC",
    "JNJ", "JPM", "KHC", "KO", "LIN", "LLY", "LMT", "LOW", "MA", "MCD",
    "MDLZ", "MDT", "MET", "META", "MMM", "MO", "MRK", "MS", "MSFT", "NEE",
    "NFLX", "NKE", "NVDA", "ORCL", "PEP", "PFE", "PG", "PM", "PYPL", "QCOM",
    "RTX", "SBUX", "SCHW", "SO", "SPG", "T", "TGT", "TMO", "TMUS", "TXN",
    "UNH", "UNP", "UPS", "USB", "V", "VZ", "WBA", "WFC", "WMT", "XOM",
]


def load_config(path: str = "config/params.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def get_universe(tag: str) -> list[str]:
    """Return ticker list for a given universe tag."""
    if tag == "sp100":
        return SP100_TICKERS
    raise ValueError(f"Unknown universe: {tag}")


def download_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: str = "data",
) -> pd.DataFrame:
    """
    Download adjusted close prices from Yahoo Finance.
    Caches result to CSV to avoid repeated API calls.

    Returns
    -------
    pd.DataFrame
        DatetimeIndex, columns = tickers, values = adjusted close.
    """
    cache_path = Path(cache_dir) / f"prices_{start}_{end}.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info(f"Loading cached prices from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    logger.info(f"Downloading {len(tickers)} tickers from Yahoo Finance...")
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, threads=True)

    # yf.download returns MultiIndex columns: (field, ticker)
    prices = raw["Close"].copy()
    prices.to_csv(cache_path)
    logger.info(f"Cached prices to {cache_path}")

    return prices


def clean_prices(
    prices: pd.DataFrame,
    min_history_pct: float = 0.95,
) -> pd.DataFrame:
    """
    Clean price data:
    1. Drop tickers with too many missing observations.
    2. Forward-fill remaining gaps (weekends/holidays already excluded by yfinance).
    3. Drop any residual NaN rows.
    """
    n_days = len(prices)
    threshold = int(n_days * min_history_pct)

    valid = prices.columns[prices.notna().sum() >= threshold]
    dropped = set(prices.columns) - set(valid)
    if dropped:
        logger.warning(f"Dropped {len(dropped)} tickers (insufficient history): {dropped}")

    out = prices[valid].ffill().dropna()
    logger.info(f"Clean price matrix: {out.shape[0]} days × {out.shape[1]} tickers")
    return out


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log returns from price levels."""
    import numpy as np
    return np.log(prices / prices.shift(1)).dropna()


def load_data(config_path: str = "config/params.yaml") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline: load config → download → clean → compute returns.

    Returns
    -------
    prices : pd.DataFrame
    log_returns : pd.DataFrame
    """
    cfg = load_config(config_path)["data"]

    tickers = get_universe(cfg["universe"])
    prices = download_prices(tickers, cfg["start_date"], cfg["end_date"], cfg["cache_dir"])
    prices = clean_prices(prices, cfg["min_history_pct"])
    log_returns = compute_log_returns(prices)

    return prices, log_returns