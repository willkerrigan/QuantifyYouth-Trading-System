"""Vectorized backtesting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from math import isnan

import pandas as pd


TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a simple close-to-close strategy backtest."""

    initial_cash: float = 10_000.0
    transaction_cost: float = 0.001
    annualization_factor: int = TRADING_DAYS_PER_YEAR


@dataclass(frozen=True)
class BacktestResult:
    """Backtest equity curve and summary metrics."""

    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def run_backtest(
    data: pd.DataFrame,
    signal: pd.Series,
    config: BacktestConfig | None = None,
) -> BacktestResult:
    """Run a close-to-close vectorized backtest.

    The strategy position is shifted by one bar so today's signal is executed
    on the next day's return. This avoids look-ahead bias for end-of-day data.
    """

    config = config or BacktestConfig()
    validate_inputs(data, signal, config)

    close = data["Close"].astype(float)
    aligned_signal = signal.reindex(close.index).ffill().fillna(0).clip(-1, 1)
    position = aligned_signal.shift(1).fillna(0)

    market_return = close.pct_change(fill_method=None).fillna(0)
    trades = position.diff().abs().fillna(position.abs())
    cost_drag = trades * config.transaction_cost
    strategy_return = (position * market_return) - cost_drag

    equity = pd.DataFrame(index=close.index)
    equity["close"] = close
    equity["signal"] = aligned_signal
    equity["position"] = position
    equity["market_return"] = market_return
    equity["strategy_return"] = strategy_return
    equity["equity"] = config.initial_cash * (1 + strategy_return).cumprod()
    equity["buy_hold_equity"] = config.initial_cash * (1 + market_return).cumprod()
    equity["drawdown"] = equity["equity"] / equity["equity"].cummax() - 1

    return BacktestResult(equity_curve=equity, metrics=calculate_metrics(equity, config))


def calculate_metrics(equity: pd.DataFrame, config: BacktestConfig) -> dict[str, float]:
    """Calculate common strategy performance statistics."""

    returns = equity["strategy_return"]
    market_returns = equity["market_return"]
    periods = max(len(equity), 1)
    years = periods / config.annualization_factor

    total_return = equity["equity"].iloc[-1] / config.initial_cash - 1
    buy_hold_return = equity["buy_hold_equity"].iloc[-1] / config.initial_cash - 1
    annual_return = annualize_return(total_return, years)
    annual_volatility = returns.std() * (config.annualization_factor**0.5)
    sharpe = annual_return / annual_volatility if annual_volatility else 0.0
    max_drawdown = equity["drawdown"].min()

    active_returns = returns[equity["position"] != 0]
    win_rate = (active_returns > 0).mean() if len(active_returns) else 0.0
    exposure = (equity["position"] != 0).mean()
    number_of_trades = (equity["position"].diff().abs().fillna(0) > 0).sum()

    market_correlation = returns.corr(market_returns)
    if pd.isna(market_correlation) or isnan(float(market_correlation)):
        market_correlation = 0.0

    return {
        "total_return": float(total_return),
        "buy_hold_return": float(buy_hold_return),
        "annual_return": float(annual_return),
        "annual_volatility": float(annual_volatility),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "exposure": float(exposure),
        "number_of_trades": float(number_of_trades),
        "final_equity": float(equity["equity"].iloc[-1]),
        "final_buy_hold_equity": float(equity["buy_hold_equity"].iloc[-1]),
        "market_correlation": float(market_correlation),
    }


def annualize_return(total_return: float, years: float) -> float:
    if years <= 0:
        return 0.0
    ending_value = 1 + total_return
    if ending_value <= 0:
        return -1.0
    return ending_value ** (1 / years) - 1


def validate_inputs(data: pd.DataFrame, signal: pd.Series, config: BacktestConfig) -> None:
    if "Close" not in data.columns:
        raise ValueError("data must contain a Close column")
    if data.empty:
        raise ValueError("data must not be empty")
    if signal.empty:
        raise ValueError("signal must not be empty")
    if config.initial_cash <= 0:
        raise ValueError("initial_cash must be positive")
    if config.transaction_cost < 0:
        raise ValueError("transaction_cost must be non-negative")
