"""
signals.py
----------
Vectorized signal-generation engine. Every function takes a price
DataFrame (must contain 'Close') and returns an integer pd.Series aligned
to the same index, valued in {-1, 0, 1} for Sell / Hold / Buy.

All computations use pandas/numpy vectorized operations only -- no
row-by-row Python loops -- so a strategy can be evaluated across years of
data and hundreds of tickers quickly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def moving_average_crossover(prices: pd.DataFrame, fast: int = 20, slow: int = 50) -> pd.Series:
    """Classic dual moving-average crossover.

    Buy when the fast MA crosses above the slow MA, sell on the reverse
    cross. Signal is held constant between crossovers (position-style
    signal, not an event pulse).
    """
    close = prices["Close"]
    fast_ma = close.rolling(fast, min_periods=fast).mean()
    slow_ma = close.rolling(slow, min_periods=slow).mean()
    signal = np.where(fast_ma > slow_ma, 1, -1)
    signal = pd.Series(signal, index=close.index)
    signal[fast_ma.isna() | slow_ma.isna()] = 0
    return signal.astype(int)


def rsi(prices: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder's Relative Strength Index (0-100), vectorized via EWMA."""
    close = prices["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_val = 100 - (100 / (1 + rs))
    return rsi_val.fillna(50)


def rsi_signal(prices: pd.DataFrame, period: int = 14, low: int = 30, high: int = 70) -> pd.Series:
    """Mean-reversion signal: buy when oversold, sell when overbought."""
    r = rsi(prices, period)
    signal = pd.Series(0, index=prices.index)
    signal[r < low] = 1
    signal[r > high] = -1
    return signal.ffill().fillna(0).astype(int)


def mean_reversion_zscore(prices: pd.DataFrame, window: int = 20, entry_z: float = 1.0) -> pd.Series:
    """
    Z-score mean reversion: go long when price is entry_z std-devs below its
    rolling mean (expecting reversion up), short when it's entry_z above.
    """
    close = prices["Close"]
    mean = close.rolling(window, min_periods=window).mean()
    std = close.rolling(window, min_periods=window).std()
    z = (close - mean) / std.replace(0, np.nan)

    signal = pd.Series(0, index=close.index)
    signal[z < -entry_z] = 1
    signal[z > entry_z] = -1
    signal[z.abs() <= entry_z * 0.2] = 0  # exit near the mean
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)


def pairs_zscore_signal(price_a: pd.Series, price_b: pd.Series, window: int = 30, entry_z: float = 2.0) -> pd.Series:
    """
    Cointegration-style pairs signal on the price ratio (a simple, fast
    proxy for full Engle-Granger cointegration testing). +1 means long the
    spread (long A / short B), -1 means the opposite, 0 is flat.
    """
    spread = np.log(price_a) - np.log(price_b)
    mean = spread.rolling(window, min_periods=window).mean()
    std = spread.rolling(window, min_periods=window).std()
    z = (spread - mean) / std.replace(0, np.nan)

    signal = pd.Series(0, index=spread.index)
    signal[z < -entry_z] = 1
    signal[z > entry_z] = -1
    signal[z.abs() <= 0.3] = 0
    return signal.replace(0, np.nan).ffill().fillna(0).astype(int)
