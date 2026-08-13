from typing import Dict
import pandas as pd

from .indicators import compute_rsi

RSI2_PERIOD = 2


def prepare_rsi2_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds the 'rsi' column engine.run needs before the day-by-day loop starts."""
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"], period=RSI2_PERIOD)
    return df


def rsi2_strategy(daily_data: Dict, open_positions: Dict, params: Dict) -> Dict[str, str]:
    """Connors-style RSI(2) mean reversion: buy oversold, sell overbought.

    params:
      rsi_buy_threshold  - enter when RSI(2) drops below this (default 5)
      rsi_sell_threshold - exit when RSI(2) rises above this (default 70)
    """
    buy_threshold = params.get("rsi_buy_threshold", 5)
    sell_threshold = params.get("rsi_sell_threshold", 70)

    signals = {}
    for symbol, row in daily_data.items():
        rsi = row.get("rsi")
        if rsi is None or pd.isna(rsi):
            signals[symbol] = "HOLD"
        elif symbol not in open_positions and rsi < buy_threshold:
            signals[symbol] = "BUY"
        elif symbol in open_positions and rsi > sell_threshold:
            signals[symbol] = "SELL"
        else:
            signals[symbol] = "HOLD"
    return signals
