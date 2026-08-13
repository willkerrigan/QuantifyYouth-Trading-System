import pandas as pd
from backtester.indicators import compute_rsi
from backtester.strategies import rsi2_strategy, prepare_rsi2_data, orb_strategy, prepare_orb_data, get_strategy


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


def _two_session_intraday_df():
    # Session 1: opening bar high=101, then a breakout bar, then the session-close bar.
    # Session 2: fresh opening range, independent of session 1.
    day1 = pd.date_range("2024-01-02 09:30", periods=3, freq="15min", tz="UTC")
    day2 = pd.date_range("2024-01-03 09:30", periods=2, freq="15min", tz="UTC")
    index = day1.append(day2)
    return pd.DataFrame({
        "open":  [100, 101, 102, 100, 100],
        "high":  [101, 102, 103, 100, 100],
        "low":   [99,  100, 101, 99,  99],
        "close": [100, 102, 103, 100, 100],
    }, index=index)


def test_prepare_orb_data_opening_range_resets_per_session():
    prepared = prepare_orb_data(_two_session_intraday_df())
    assert list(prepared["opening_range_high"]) == [101, 101, 101, 100, 100]


def test_prepare_orb_data_marks_last_bar_of_each_session():
    prepared = prepare_orb_data(_two_session_intraday_df())
    assert list(prepared["is_session_close"]) == [False, False, True, False, True]


def test_orb_strategy_buys_on_breakout_above_opening_range():
    row = pd.Series({"close": 102, "opening_range_high": 101, "is_session_close": False})
    signals = orb_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "BUY"


def test_orb_strategy_holds_below_opening_range():
    row = pd.Series({"close": 100.5, "opening_range_high": 101, "is_session_close": False})
    signals = orb_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_orb_strategy_exits_flat_at_session_close():
    row = pd.Series({"close": 103, "opening_range_high": 101, "is_session_close": True})
    signals = orb_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "SELL"


def test_orb_strategy_holds_open_position_mid_session():
    row = pd.Series({"close": 103, "opening_range_high": 101, "is_session_close": False})
    signals = orb_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "HOLD"


def test_get_strategy_returns_registered_pair():
    strategy_func, prepare_func = get_strategy("orb")
    assert strategy_func is orb_strategy
    assert prepare_func is prepare_orb_data


def test_get_strategy_raises_on_unknown_name():
    try:
        get_strategy("does_not_exist")
        assert False, "expected ValueError"
    except ValueError:
        pass
