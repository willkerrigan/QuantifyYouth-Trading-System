import pandas as pd
from backtester.indicators import compute_rsi
from backtester.strategies import rsi2_strategy, prepare_rsi2_data


def test_compute_rsi_all_gains_hits_100():
    close = pd.Series([100, 101, 102, 103, 104, 105])
    rsi = compute_rsi(close, period=2)
    assert rsi.iloc[-1] == 100


def test_compute_rsi_all_losses_hits_0():
    close = pd.Series([105, 104, 103, 102, 101, 100])
    rsi = compute_rsi(close, period=2)
    assert rsi.iloc[-1] == 0


def test_prepare_rsi2_data_adds_column():
    df = pd.DataFrame({"close": [100, 99, 98, 97, 96]})
    prepared = prepare_rsi2_data(df)
    assert "rsi" in prepared.columns
    assert "rsi" not in df.columns  # original frame untouched


def test_rsi2_strategy_buys_when_oversold():
    daily_data = {"SPY": pd.Series({"close": 100, "rsi": 3})}
    signals = rsi2_strategy(daily_data, open_positions={}, params={})
    assert signals["SPY"] == "BUY"


def test_rsi2_strategy_sells_when_overbought_and_holding():
    daily_data = {"SPY": pd.Series({"close": 100, "rsi": 80})}
    signals = rsi2_strategy(daily_data, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "SELL"


def test_rsi2_strategy_holds_when_no_signal():
    daily_data = {"SPY": pd.Series({"close": 100, "rsi": 50})}
    signals = rsi2_strategy(daily_data, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_rsi2_strategy_does_not_buy_if_already_open():
    daily_data = {"SPY": pd.Series({"close": 100, "rsi": 3})}
    signals = rsi2_strategy(daily_data, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "HOLD"


def test_rsi2_strategy_respects_custom_thresholds():
    daily_data = {"SPY": pd.Series({"close": 100, "rsi": 10})}
    signals = rsi2_strategy(daily_data, open_positions={}, params={"rsi_buy_threshold": 15})
    assert signals["SPY"] == "BUY"
