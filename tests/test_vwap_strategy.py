import pandas as pd
from backtester.strategies import vwap_strategy, prepare_vwap_data, get_strategy


def _two_session_intraday_df():
    """Hand-checkable intraday bars. Typical price = (high + low + close) / 3.

    Session 1 (2024-01-02):
      bar1 tp=(102+98+100)/3=100,  vol=100 -> cum_pv=10000, cum_vol=100 -> vwap=100.00
      bar2 tp=(106+100+103)/3=103, vol=100 -> cum_pv=20300, cum_vol=200 -> vwap=101.50
      bar3 tp=(110+104+104)/3=106, vol=200 -> cum_pv=41500, cum_vol=400 -> vwap=103.75
    Session 2 (2024-01-03) restarts the cumulative sums from scratch:
      bar4 tp=(92+88+90)/3=90,     vol=100 -> cum_pv=9000,  cum_vol=100 -> vwap=90.00
      bar5 tp=(94+88+91)/3=91,     vol=100 -> cum_pv=18100, cum_vol=200 -> vwap=90.50
    """
    day1 = pd.date_range("2024-01-02 09:30", periods=3, freq="15min", tz="UTC")
    day2 = pd.date_range("2024-01-03 09:30", periods=2, freq="15min", tz="UTC")
    index = day1.append(day2)
    return pd.DataFrame({
        "open":   [100, 100, 104, 90, 90],
        "high":   [102, 106, 110, 92, 94],
        "low":    [98,  100, 104, 88, 88],
        "close":  [100, 103, 104, 90, 91],
        "volume": [100, 100, 200, 100, 100],
    }, index=index)


def test_prepare_vwap_data_math_matches_hand_computed_values():
    prepared = prepare_vwap_data(_two_session_intraday_df())
    assert list(prepared["vwap"]) == [100.0, 101.5, 103.75, 90.0, 90.5]


def test_prepare_vwap_data_resets_each_session():
    prepared = prepare_vwap_data(_two_session_intraday_df())
    # First bar of session 2 must equal its own typical price, not a blend with
    # session 1's much higher prices.
    assert prepared["vwap"].iloc[3] == 90.0


def test_prepare_vwap_data_deviation_is_fraction_of_vwap():
    prepared = prepare_vwap_data(_two_session_intraday_df())
    # bar2: close=103, vwap=101.5 -> (103 - 101.5) / 101.5
    assert prepared["vwap_deviation"].iloc[1] == (103 - 101.5) / 101.5
    # bar1 closes exactly on VWAP.
    assert prepared["vwap_deviation"].iloc[0] == 0.0


def test_prepare_vwap_data_marks_last_bar_of_each_session():
    prepared = prepare_vwap_data(_two_session_intraday_df())
    assert list(prepared["is_session_close"]) == [False, False, True, False, True]


def test_prepare_vwap_data_does_not_mutate_caller_frame():
    df = _two_session_intraday_df()
    prepared = prepare_vwap_data(df)
    assert "vwap" in prepared.columns
    assert "vwap" not in df.columns
    assert "vwap_deviation" not in df.columns
    assert "is_session_close" not in df.columns


def test_prepare_vwap_data_zero_volume_session_yields_nan_not_crash():
    index = pd.date_range("2024-01-02 09:30", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open":   [100, 100],
        "high":   [102, 102],
        "low":    [98, 98],
        "close":  [100, 100],
        "volume": [0, 0],
    }, index=index)
    prepared = prepare_vwap_data(df)
    assert prepared["vwap"].isna().all()
    assert prepared["vwap_deviation"].isna().all()


def test_vwap_strategy_buys_when_stretched_below_vwap():
    row = pd.Series({"close": 99, "vwap": 100, "vwap_deviation": -0.01,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "BUY"


def test_vwap_strategy_holds_when_deviation_inside_threshold():
    row = pd.Series({"close": 99.9, "vwap": 100, "vwap_deviation": -0.001,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_does_not_buy_when_stretched_above_vwap():
    row = pd.Series({"close": 101, "vwap": 100, "vwap_deviation": 0.01,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_respects_custom_entry_deviation():
    row = pd.Series({"close": 99.9, "vwap": 100, "vwap_deviation": -0.001,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={},
                            params={"vwap_entry_deviation": 0.0005})
    assert signals["SPY"] == "BUY"


def test_vwap_strategy_does_not_buy_if_already_open():
    row = pd.Series({"close": 99, "vwap": 100, "vwap_deviation": -0.01,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_exits_on_reversion_to_vwap():
    row = pd.Series({"close": 100, "vwap": 100, "vwap_deviation": 0.0,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "SELL"


def test_vwap_strategy_exits_when_price_pushes_above_vwap():
    row = pd.Series({"close": 101, "vwap": 100, "vwap_deviation": 0.01,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "SELL"


def test_vwap_strategy_holds_open_position_still_below_vwap():
    row = pd.Series({"close": 99, "vwap": 100, "vwap_deviation": -0.01,
                     "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_exits_flat_at_session_close_even_if_still_stretched():
    row = pd.Series({"close": 99, "vwap": 100, "vwap_deviation": -0.01,
                     "is_session_close": True})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "SELL"


def test_vwap_strategy_holds_on_nan_deviation():
    row = pd.Series({"close": 100, "vwap": float("nan"),
                     "vwap_deviation": float("nan"), "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_holds_open_position_on_nan_deviation_mid_session():
    row = pd.Series({"close": 100, "vwap": float("nan"),
                     "vwap_deviation": float("nan"), "is_session_close": False})
    signals = vwap_strategy({"SPY": row}, open_positions={"SPY": {}}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_holds_when_columns_missing():
    row = pd.Series({"close": 100})
    signals = vwap_strategy({"SPY": row}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_on_zero_volume_prepared_frame_holds():
    index = pd.date_range("2024-01-02 09:30", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "open":   [100, 100],
        "high":   [102, 102],
        "low":    [98, 98],
        "close":  [100, 100],
        "volume": [0, 0],
    }, index=index)
    prepared = prepare_vwap_data(df)
    signals = vwap_strategy({"SPY": prepared.iloc[0]}, open_positions={}, params={})
    assert signals["SPY"] == "HOLD"


def test_vwap_strategy_end_to_end_on_prepared_frame():
    prepared = prepare_vwap_data(_two_session_intraday_df())
    # Session 2, bar 5: close=91 vs vwap=90.5 -> stretched above, no entry.
    row5 = prepared.iloc[4]
    assert vwap_strategy({"SPY": row5}, open_positions={}, params={})["SPY"] == "HOLD"
    # Same bar is the session close, so an open position is flattened.
    assert vwap_strategy({"SPY": row5}, open_positions={"SPY": {}},
                         params={})["SPY"] == "SELL"


def test_get_strategy_returns_registered_vwap_pair():
    strategy_func, prepare_func = get_strategy("vwap")
    assert strategy_func is vwap_strategy
    assert prepare_func is prepare_vwap_data
