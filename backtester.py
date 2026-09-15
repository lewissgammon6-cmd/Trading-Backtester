"""
backtester.py
-------------
Orchestrates the full pipeline: data ingestion -> signal generation ->
execution simulation -> performance analytics, for a single ticker
strategy benchmarked against SPY (or another chosen ticker).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from . import data_ingestion as di
from . import execution as ex
from . import performance as perf
from . import signals as sig

STRATEGY_REGISTRY = {
    "ma_crossover": lambda prices, **kw: sig.moving_average_crossover(prices, **kw),
    "rsi": lambda prices, **kw: sig.rsi_signal(prices, **kw),
    "mean_reversion": lambda prices, **kw: sig.mean_reversion_zscore(prices, **kw),
}


@dataclass
class BacktestResult:
    sim: pd.DataFrame
    report: "perf.PerformanceReport"
    benchmark_equity: pd.Series
    elapsed_seconds: float


def run_backtest(
    ticker: str,
    strategy: str,
    strategy_kwargs: dict | None = None,
    benchmark: str = "SPY",
    start: str = "2019-01-01",
    end: str | None = None,
    exec_cfg: ex.ExecutionConfig | None = None,
) -> BacktestResult:
    strategy_kwargs = strategy_kwargs or {}
    exec_cfg = exec_cfg or ex.ExecutionConfig()

    t0 = time.perf_counter()

    cfg = di.DataConfig(tickers=[ticker, benchmark], start=start, end=end)
    frames = di.align(di.load_prices(cfg))
    prices, bench_prices = frames[ticker], frames[benchmark]

    signal_fn = STRATEGY_REGISTRY[strategy]
    signal = signal_fn(prices, **strategy_kwargs)

    sim = ex.simulate(prices, signal, exec_cfg)

    benchmark_equity = exec_cfg.initial_capital * (1 + bench_prices["Close"].pct_change().fillna(0)).cumprod()
    report = perf.build_report(sim, benchmark_equity)

    elapsed = time.perf_counter() - t0
    return BacktestResult(sim=sim, report=report, benchmark_equity=benchmark_equity, elapsed_seconds=elapsed)


def run_universe_benchmark(tickers: list[str], start: str = "2019-01-01", end: str | None = None) -> float:
    """
    Measures raw vectorized-indicator throughput across a universe of
    tickers (used for the README's performance claim), independent of any
    single strategy's execution simulation.
    """
    t0 = time.perf_counter()
    cfg = di.DataConfig(tickers=tickers, start=start, end=end)
    frames = di.load_prices(cfg)
    for prices in frames.values():
        sig.moving_average_crossover(prices)
        sig.rsi(prices)
        sig.mean_reversion_zscore(prices)
    return time.perf_counter() - t0
