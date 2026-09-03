"""Signal-generation strategies for the trading system."""

from __future__ import annotations

import pandas as pd

from indicators import sma


def moving_average_crossover(
    data: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    long_only: bool = True,
) -> pd.Series:
    """Create trading signals from a fast/slow simple moving-average crossover.

    Returns a signal series where ``1`` means long, ``0`` means flat, and ``-1``
    means short when ``long_only`` is false.
    """

    if "Close" not in data.columns:
        raise ValueError("data must contain a Close column")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    fast = sma(data["Close"], fast_window)
    slow = sma(data["Close"], slow_window)
    signal = pd.Series(0, index=data.index, name="signal", dtype="float64")
    signal.loc[fast > slow] = 1
    if not long_only:
        signal.loc[fast < slow] = -1
    return signal.fillna(0)


def buy_and_hold(data: pd.DataFrame) -> pd.Series:
    """Always-long benchmark signal."""

    if "Close" not in data.columns:
        raise ValueError("data must contain a Close column")
    return pd.Series(1, index=data.index, name="signal", dtype="float64")
