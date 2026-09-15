#!/usr/bin/env python3
"""
main.py
-------
Command-line entry point.

Examples
--------
    python main.py --ticker AAPL --strategy ma_crossover --start 2019-01-01
    python main.py --ticker MSFT --strategy rsi --fast 10 --slow 30
    python main.py --benchmark-only --tickers AAPL MSFT GOOG AMZN META
"""

from __future__ import annotations

import argparse
import sys

from src.backtester import run_backtest, run_universe_benchmark
from src.execution import ExecutionConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Algorithmic Trading Strategy Backtester")
    p.add_argument("--ticker", default="AAPL", help="Ticker to trade")
    p.add_argument("--benchmark", default="SPY", help="Benchmark ticker")
    p.add_argument("--strategy", default="ma_crossover", choices=["ma_crossover", "rsi", "mean_reversion"])
    p.add_argument("--start", default="2019-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--fast", type=int, default=20, help="Fast MA window (ma_crossover)")
    p.add_argument("--slow", type=int, default=50, help="Slow MA window (ma_crossover)")
    p.add_argument("--rsi-period", type=int, default=14)
    p.add_argument("--zscore-window", type=int, default=20)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=2.0)
    p.add_argument("--position-sizing", default="fixed", choices=["fixed", "vol_target"])
    p.add_argument("--initial-capital", type=float, default=100_000.0)
    p.add_argument("--plot-out", default="equity_curve.png")
    p.add_argument("--benchmark-only", action="store_true", help="Run the throughput benchmark and exit")
    p.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "META"])
    return p.parse_args()


def strategy_kwargs_from_args(args: argparse.Namespace) -> dict:
    if args.strategy == "ma_crossover":
        return {"fast": args.fast, "slow": args.slow}
    if args.strategy == "rsi":
        return {"period": args.rsi_period}
    if args.strategy == "mean_reversion":
        return {"window": args.zscore_window}
    return {}


def main() -> int:
    args = parse_args()

    if args.benchmark_only:
        elapsed = run_universe_benchmark(args.tickers, start=args.start, end=args.end)
        print(f"Computed indicators for {len(args.tickers)} tickers in {elapsed:.3f}s")
        return 0

    exec_cfg = ExecutionConfig(
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        position_sizing=args.position_sizing,
        initial_capital=args.initial_capital,
    )

    result = run_backtest(
        ticker=args.ticker,
        strategy=args.strategy,
        strategy_kwargs=strategy_kwargs_from_args(args),
        benchmark=args.benchmark,
        start=args.start,
        end=args.end,
        exec_cfg=exec_cfg,
    )

    print(f"\n=== {args.ticker} | {args.strategy} vs {args.benchmark} ===")
    print(result.report)
    print(f"\nBacktest computed in {result.elapsed_seconds:.3f}s")

    from src.performance import plot_equity_curve

    plot_equity_curve(
        result.sim,
        result.benchmark_equity,
        args.plot_out,
        title=f"{args.ticker} ({args.strategy}) vs {args.benchmark}",
    )
    print(f"Saved equity curve plot to {args.plot_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
