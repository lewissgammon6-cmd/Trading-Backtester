"""
execution.py
------------
Turns a signal series into a simulated equity curve, applying realistic
frictions: slippage, transaction fees, and a position-sizing rule. Care is
taken to avoid lookahead bias -- a signal computed using data through day
t is executed at day t+1's open-to-close return, never at day t's own
return.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ExecutionConfig:
    fee_bps: float = 5.0          # transaction fee in basis points per trade (one-way)
    slippage_bps: float = 2.0     # extra cost in basis points when a position changes
    position_sizing: str = "fixed"  # "fixed" or "vol_target"
    fixed_fraction: float = 1.0     # fraction of capital deployed when "fixed"
    vol_target_annual: float = 0.15  # target annualized vol when "vol_target"
    vol_lookback: int = 20
    initial_capital: float = 100_000.0


def _position_size(returns: pd.Series, cfg: ExecutionConfig) -> pd.Series:
    if cfg.position_sizing == "fixed":
        return pd.Series(cfg.fixed_fraction, index=returns.index)
    if cfg.position_sizing == "vol_target":
        realized_vol = returns.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std() * np.sqrt(252)
        realized_vol = realized_vol.replace(0, np.nan)
        size = (cfg.vol_target_annual / realized_vol).clip(upper=2.0).fillna(0.0)
        return size
    raise ValueError(f"Unknown position_sizing: {cfg.position_sizing}")


def simulate(prices: pd.DataFrame, signal: pd.Series, cfg: ExecutionConfig) -> pd.DataFrame:
    """
    Simulate trade execution given a daily target position signal
    (-1 / 0 / 1). Returns a DataFrame with per-day strategy returns,
    costs, position size, and the resulting equity curve.
    """
    close = prices["Close"]
    daily_return = close.pct_change().fillna(0.0)

    # The signal known at close of day t can only be acted on starting the
    # next bar -- shift by 1 to prevent lookahead bias.
    executable_signal = signal.shift(1).fillna(0)

    size = _position_size(daily_return, cfg)
    target_position = executable_signal * size

    # Trading cost is charged whenever the position changes, proportional
    # to the size of the change (turnover), combining fee + slippage.
    turnover = target_position.diff().abs().fillna(target_position.abs())
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    costs = turnover * cost_rate

    strategy_return = target_position.shift(0) * daily_return - costs
    # position is applied to *today's* return since target_position already
    # reflects information available at the start of today (see shift above)
    equity = cfg.initial_capital * (1 + strategy_return).cumprod()

    return pd.DataFrame(
        {
            "close": close,
            "daily_return": daily_return,
            "position": target_position,
            "turnover": turnover,
            "cost": costs,
            "strategy_return": strategy_return,
            "equity": equity,
        }
    )
