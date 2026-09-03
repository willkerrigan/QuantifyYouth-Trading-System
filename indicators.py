"""Technical indicators used by the example strategies."""

from __future__ import annotations

import pandas as pd


def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""

    validate_window(window)
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average."""

    validate_window(span)
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """Relative strength index using Wilder-style exponential smoothing."""

    validate_window(window)
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = losses.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    return 100 - (100 / (1 + relative_strength))


def add_technical_indicators(
    data: pd.DataFrame,
    *,
    fast_window: int = 20,
    slow_window: int = 50,
    rsi_window: int = 14,
) -> pd.DataFrame:
    """Return a copy of price data with common indicators attached."""

    if "Close" not in data.columns:
        raise ValueError("data must contain a Close column")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    enriched = data.copy()
    close = enriched["Close"]
    enriched[f"SMA_{fast_window}"] = sma(close, fast_window)
    enriched[f"SMA_{slow_window}"] = sma(close, slow_window)
    enriched[f"EMA_{fast_window}"] = ema(close, fast_window)
    enriched[f"RSI_{rsi_window}"] = rsi(close, rsi_window)
    return enriched


def validate_window(window: int) -> None:
    if window < 1:
        raise ValueError("window must be at least 1")
