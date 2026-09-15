"""
performance.py
---------------
Quantitative performance analytics on a strategy's daily-return series:
Sharpe, Sortino, max drawdown (+ duration), cumulative return, and a
benchmark-relative equity plot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceReport:
    total_return: float
    annualized_return: float
    annualized_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    max_drawdown_duration_days: int
    benchmark_total_return: float | None = None

    def to_dict(self) -> dict:
        return self.__dict__

    def __str__(self) -> str:
        lines = [
            f"Total Return:            {self.total_return:8.2%}",
            f"Annualized Return:       {self.annualized_return:8.2%}",
            f"Annualized Volatility:   {self.annualized_vol:8.2%}",
            f"Sharpe Ratio:            {self.sharpe:8.2f}",
            f"Sortino Ratio:           {self.sortino:8.2f}",
            f"Max Drawdown:            {self.max_drawdown:8.2%}",
            f"Max Drawdown Duration:   {self.max_drawdown_duration_days:8d} days",
        ]
        if self.benchmark_total_return is not None:
            lines.append(f"Benchmark Total Return:  {self.benchmark_total_return:8.2%}")
        return "\n".join(lines)


def sharpe_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = returns - risk_free_annual / TRADING_DAYS
    if excess.std() == 0 or excess.dropna().empty:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / excess.std())


def sortino_ratio(returns: pd.Series, risk_free_annual: float = 0.0) -> float:
    excess = returns - risk_free_annual / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std()
    if not downside_std or np.isnan(downside_std) or downside_std == 0:
        return 0.0
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / downside_std)


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """Returns (max_drawdown_fraction, duration_in_days_of_the_worst_drawdown)."""
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    mdd = float(drawdown.min())

    # duration of the drawdown that produced the trough
    trough_idx = drawdown.idxmin()
    peak_idx = equity.loc[:trough_idx].idxmax()
    recovery = equity.loc[trough_idx:][equity.loc[trough_idx:] >= running_max.loc[peak_idx]]
    end_idx = recovery.index[0] if len(recovery) else equity.index[-1]
    duration = (end_idx - peak_idx).days
    return mdd, int(duration)


def build_report(sim: pd.DataFrame, benchmark_equity: pd.Series | None = None) -> PerformanceReport:
    returns = sim["strategy_return"]
    equity = sim["equity"]

    n_days = len(returns)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = max(n_days / TRADING_DAYS, 1e-9)
    annualized_return = float((1 + total_return) ** (1 / years) - 1)
    annualized_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    mdd, duration = max_drawdown(equity)

    bench_total = None
    if benchmark_equity is not None and len(benchmark_equity):
        bench_total = float(benchmark_equity.iloc[-1] / benchmark_equity.iloc[0] - 1.0)

    return PerformanceReport(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_vol=annualized_vol,
        sharpe=sharpe_ratio(returns),
        sortino=sortino_ratio(returns),
        max_drawdown=mdd,
        max_drawdown_duration_days=duration,
        benchmark_total_return=bench_total,
    )


def plot_equity_curve(sim: pd.DataFrame, benchmark_equity: pd.Series | None, out_path: str, title: str = "Strategy vs Benchmark") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    strat_norm = sim["equity"] / sim["equity"].iloc[0]
    ax.plot(strat_norm.index, strat_norm.values, label="Strategy", linewidth=1.8)

    if benchmark_equity is not None and len(benchmark_equity):
        bench_norm = benchmark_equity / benchmark_equity.iloc[0]
        ax.plot(bench_norm.index, bench_norm.values, label="Benchmark (SPY)", linewidth=1.4, linestyle="--")

    ax.set_title(title)
    ax.set_ylabel("Growth of $1")
    ax.set_xlabel("Date")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
