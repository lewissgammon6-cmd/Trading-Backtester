import numpy as np
import pandas as pd
import pytest

from src import data_ingestion as di
from src import execution as ex
from src import performance as perf
from src import signals as sig


@pytest.fixture
def prices():
    cfg = di.DataConfig(tickers=["TEST"], start="2020-01-01", end="2020-12-31")
    return di.load_prices(cfg)["TEST"]


def test_synthetic_data_schema(prices):
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        assert col in prices.columns
    assert prices.index.is_monotonic_increasing
    assert not prices["Close"].isna().any()


def test_ma_crossover_signal_values(prices):
    signal = sig.moving_average_crossover(prices, fast=5, slow=10)
    assert set(signal.unique()).issubset({-1, 0, 1})
    assert len(signal) == len(prices)


def test_rsi_bounds(prices):
    r = sig.rsi(prices, period=14)
    assert (r >= 0).all() and (r <= 100).all()


def test_no_lookahead_bias_in_execution(prices):
    """The signal at time t must not affect the return realized at time t
    before the position had a chance to be entered (shift-by-one check)."""
    signal = pd.Series(0, index=prices.index)
    signal.iloc[50] = 1  # a single, isolated buy signal

    cfg = ex.ExecutionConfig(fee_bps=0, slippage_bps=0)
    sim = ex.simulate(prices, signal, cfg)

    # The position resulting from the signal at index 50 must first appear
    # no earlier than index 51 (i.e. shifted forward by at least one bar).
    assert sim["position"].iloc[50] == 0
    assert sim["position"].iloc[51] != 0


def test_execution_costs_reduce_return(prices):
    signal = sig.moving_average_crossover(prices, fast=5, slow=10)
    cfg_no_cost = ex.ExecutionConfig(fee_bps=0, slippage_bps=0)
    cfg_cost = ex.ExecutionConfig(fee_bps=50, slippage_bps=50)

    sim_no_cost = ex.simulate(prices, signal, cfg_no_cost)
    sim_cost = ex.simulate(prices, signal, cfg_cost)

    assert sim_cost["equity"].iloc[-1] <= sim_no_cost["equity"].iloc[-1]


def test_max_drawdown_is_non_positive():
    equity = pd.Series([100, 110, 90, 95, 120], index=pd.date_range("2020-01-01", periods=5))
    mdd, duration = perf.max_drawdown(equity)
    assert mdd <= 0
    assert duration >= 0


def test_sharpe_zero_when_no_variance():
    returns = pd.Series(np.zeros(30))
    assert perf.sharpe_ratio(returns) == 0.0


def test_full_backtest_report_fields(prices):
    signal = sig.moving_average_crossover(prices, fast=5, slow=10)
    sim = ex.simulate(prices, signal, ex.ExecutionConfig())
    report = perf.build_report(sim)
    assert isinstance(report.sharpe, float)
    assert isinstance(report.max_drawdown, float)
    assert report.max_drawdown <= 0
