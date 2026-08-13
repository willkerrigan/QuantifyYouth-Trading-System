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


def prepare_orb_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'opening_range_high' and 'is_session_close' columns for orb_strategy.
    Assumes each row is one intraday bar and the bar interval equals the opening
    range length (e.g. 15m bars for a 15-minute opening range), so the first bar
    of each session already covers the whole opening range."""
    df = df.copy()
    session = df.index.normalize()
    df["opening_range_high"] = df.groupby(session)["high"].transform("first")
    next_session = pd.Series(session, index=df.index).shift(-1)
    df["is_session_close"] = (session != next_session) | next_session.isna()
    return df


def orb_strategy(daily_data: Dict, open_positions: Dict, params: Dict) -> Dict[str, str]:
    """Opening range breakout: buy when price breaks above the opening range high,
    flat by the end of the session (no overnight holds).

    Note: close <= high always holds for a single bar, so this naturally can't fire
    on the opening bar itself (its own high defines opening_range_high).
    """
    signals = {}
    for symbol, row in daily_data.items():
        if symbol in open_positions:
            signals[symbol] = "SELL" if row.get("is_session_close", False) else "HOLD"
            continue
        orb_high = row.get("opening_range_high")
        if orb_high is not None and not pd.isna(orb_high) and row["close"] > orb_high:
            signals[symbol] = "BUY"
        else:
            signals[symbol] = "HOLD"
    return signals


def prepare_vwap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Adds 'vwap', 'vwap_deviation' and 'is_session_close' columns for vwap_strategy.

    VWAP is session-anchored: the cumulative sums reset at the start of each trading
    session, so an early bar of one session never inherits the previous session's
    volume profile. Typical price is (high + low + close) / 3.

    Bars whose cumulative session volume is zero would divide by zero, so their VWAP
    is left as NaN and vwap_strategy treats them as 'no opinion' (HOLD).
    """
    df = df.copy()
    session = df.index.normalize()

    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cum_pv = (typical_price * df["volume"]).groupby(session).cumsum()
    cum_volume = df["volume"].groupby(session).cumsum()

    # A zero (or NaN) cumulative volume yields NaN rather than inf/ZeroDivisionError.
    safe_volume = cum_volume.where(cum_volume > 0)
    df["vwap"] = cum_pv / safe_volume
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].where(df["vwap"] != 0)

    next_session = pd.Series(session, index=df.index).shift(-1)
    df["is_session_close"] = (session != next_session) | next_session.isna()
    return df


def vwap_strategy(daily_data: Dict, open_positions: Dict, params: Dict) -> Dict[str, str]:
    """Intraday VWAP mean reversion: buy when price is stretched below the
    session VWAP, exit when it reverts back to VWAP, flat by the end of the
    session (no overnight holds).

    params:
      vwap_entry_deviation - enter when close sits this far below VWAP as a
                             fraction, e.g. 0.003 = 0.3% below (default 0.003)
    """
    entry_deviation = params.get("vwap_entry_deviation", 0.003)

    signals = {}
    for symbol, row in daily_data.items():
        deviation = row.get("vwap_deviation")
        has_deviation = deviation is not None and not pd.isna(deviation)

        if symbol in open_positions:
            if row.get("is_session_close", False):
                signals[symbol] = "SELL"
            elif has_deviation and deviation >= 0:
                signals[symbol] = "SELL"
            else:
                signals[symbol] = "HOLD"
            continue

        if has_deviation and deviation < -entry_deviation:
            signals[symbol] = "BUY"
        else:
            signals[symbol] = "HOLD"
    return signals


STRATEGIES = {
    "rsi2": (rsi2_strategy, prepare_rsi2_data),
    "orb": (orb_strategy, prepare_orb_data),
    "vwap": (vwap_strategy, prepare_vwap_data),
}


def get_strategy(name: str):
    try:
        return STRATEGIES[name]
    except KeyError:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGIES)}")
