"""
data_ingestion.py
------------------
Fetches OHLCV price data for one or more tickers, cleans it, and normalizes
timestamps. Uses yfinance when available and network-reachable; otherwise
falls back to a reproducible synthetic Geometric-Brownian-Motion price
generator so the rest of the pipeline can always be exercised offline.

Design notes
------------
- All output DataFrames share the same schema: a DatetimeIndex (tz-naive,
  daily frequency) and columns [Open, High, Low, Close, Volume].
- Missing data points are forward-filled (never back-filled, to avoid
  lookahead bias) and any leading NaNs are dropped.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataConfig:
    tickers: list[str]
    start: str = "2019-01-01"
    end: str | None = None
    interval: str = "1d"          # yfinance-style interval string
    seed: int = 42                # only used for synthetic fallback


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize timestamps and handle missing data without lookahead bias."""
    df = df.copy()
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    # Forward-fill only: using a future price to patch a past gap is lookahead.
    df[REQUIRED_COLUMNS] = df[REQUIRED_COLUMNS].ffill()
    df = df.dropna(subset=REQUIRED_COLUMNS)
    return df


def _fetch_yfinance(ticker: str, cfg: DataConfig) -> pd.DataFrame | None:
    try:
        import yfinance as yf
    except ImportError:
        logger.info("yfinance not installed; using synthetic data for %s", ticker)
        return None

    try:
        raw = yf.download(
            ticker,
            start=cfg.start,
            end=cfg.end,
            interval=cfg.interval,
            progress=False,
            auto_adjust=True,
        )
        if raw is None or raw.empty:
            logger.warning("yfinance returned no data for %s; using synthetic data", ticker)
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        return _clean(raw[REQUIRED_COLUMNS])
    except Exception as exc:  # network errors, rate limits, etc.
        logger.warning("yfinance fetch failed for %s (%s); using synthetic data", ticker, exc)
        return None


def _synthetic_ohlcv(ticker: str, cfg: DataConfig) -> pd.DataFrame:
    """
    Generate a reproducible synthetic price series using Geometric Brownian
    Motion, then derive plausible Open/High/Low/Volume around the daily
    Close. Seeded per-ticker so different tickers don't move identically.
    """
    dates = pd.bdate_range(start=cfg.start, end=cfg.end or pd.Timestamp.today())
    n = len(dates)
    rng = np.random.default_rng(abs(hash(ticker)) % (2**32) ^ cfg.seed)

    mu, sigma = 0.0004, 0.018          # daily drift / vol, roughly equity-like
    s0 = 100 * (1 + rng.uniform(-0.5, 2.0))
    shocks = rng.normal(mu - 0.5 * sigma**2, sigma, size=n)
    close = s0 * np.exp(np.cumsum(shocks))

    intraday_range = np.abs(rng.normal(0, sigma, size=n)) * close
    open_ = close * (1 + rng.normal(0, sigma / 2, size=n))
    high = np.maximum(open_, close) + intraday_range * rng.uniform(0.1, 0.6, size=n)
    low = np.minimum(open_, close) - intraday_range * rng.uniform(0.1, 0.6, size=n)
    volume = rng.integers(1_000_000, 8_000_000, size=n)

    df = pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )
    return _clean(df)


def load_prices(cfg: DataConfig) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for every ticker in cfg.tickers."""
    out: dict[str, pd.DataFrame] = {}
    for ticker in cfg.tickers:
        df = _fetch_yfinance(ticker, cfg)
        if df is None:
            df = _synthetic_ohlcv(ticker, cfg)
        out[ticker] = df
    return out


def align(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Restrict every ticker's frame to the shared trading-day index."""
    common_index = None
    for df in frames.values():
        common_index = df.index if common_index is None else common_index.intersection(df.index)
    return {t: df.loc[common_index] for t, df in frames.items()}
