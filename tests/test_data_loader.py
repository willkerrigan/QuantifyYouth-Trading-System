"""Offline tests for DataLoader hardening.

No network: the yfinance module is replaced with a stub that returns whatever
frame the test wants, so every failure mode can be exercised deterministically.
"""

import logging
import sys
import types

import pandas as pd
import pytest

from backtester.data_loader import DataLoader, DataLoadError


def _config(source="yahoo", timeframe="1d"):
    return {
        "backtest": {"start_date": "2024-01-01", "end_date": "2024-01-11"},
        "data": {"source": source, "timeframe": timeframe},
    }


def _valid_frame(rows=5, columns=("Open", "High", "Low", "Close", "Volume")):
    index = pd.date_range("2024-01-01", periods=rows, freq="D")
    data = {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1000}
    return pd.DataFrame({c: data.get(c, 1.0) for c in columns}, index=index)


def _stub_yfinance(monkeypatch, frame, calls=None):
    """Install a fake `yfinance` module whose download() returns `frame`."""
    def download(symbol, **kwargs):
        if calls is not None:
            calls.append((symbol, kwargs))
        return frame.copy() if isinstance(frame, pd.DataFrame) else frame

    module = types.ModuleType("yfinance")
    module.download = download
    monkeypatch.setitem(sys.modules, "yfinance", module)


# --- empty / malformed frames fail loudly ---------------------------------

def test_empty_frame_raises_with_context(monkeypatch):
    _stub_yfinance(monkeypatch, pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"]))
    loader = DataLoader(_config())

    with pytest.raises(DataLoadError) as excinfo:
        loader.load("SPY")

    message = str(excinfo.value)
    assert "EMPTY" in message
    assert "SPY" in message and "yahoo" in message and "1d" in message
    assert "2024-01-01" in message and "2024-01-11" in message
    assert "SPY" not in loader.cache


def test_none_frame_raises(monkeypatch):
    _stub_yfinance(monkeypatch, None)
    with pytest.raises(DataLoadError):
        DataLoader(_config()).load("SPY")


def test_missing_column_raises_naming_present_columns(monkeypatch):
    _stub_yfinance(monkeypatch, _valid_frame(columns=("Open", "High", "Low", "Close")))
    loader = DataLoader(_config())

    with pytest.raises(DataLoadError) as excinfo:
        loader.load("QQQ")

    message = str(excinfo.value)
    assert "volume" in message                      # names what is missing
    assert "open" in message and "close" in message  # names what was present
    assert "QQQ" in message


def test_multiindex_without_ohlcv_raises(monkeypatch):
    frame = _valid_frame()
    frame.columns = pd.MultiIndex.from_product([["SPY"], ["a", "b", "c", "d", "e"]])
    _stub_yfinance(monkeypatch, frame)

    with pytest.raises(DataLoadError):
        DataLoader(_config()).load("SPY")


# --- MultiIndex flattening -------------------------------------------------

def test_multiindex_price_ticker_columns_flatten(monkeypatch):
    """The real incident: yfinance started returning (Price, Ticker) columns."""
    frame = _valid_frame()
    frame.columns = pd.MultiIndex.from_product(
        [["Open", "High", "Low", "Close", "Volume"], ["SPY"]], names=["Price", "Ticker"])
    _stub_yfinance(monkeypatch, frame)

    df = DataLoader(_config()).load("SPY")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 5


def test_multiindex_ticker_price_columns_flatten(monkeypatch):
    """group_by='ticker' inverts the levels; level 0 is then useless."""
    frame = _valid_frame()
    frame.columns = pd.MultiIndex.from_product(
        [["SPY"], ["Open", "High", "Low", "Close", "Volume"]], names=["Ticker", "Price"])
    _stub_yfinance(monkeypatch, frame)

    df = DataLoader(_config()).load("SPY")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_alpaca_style_multiindex_row_index_flattens():
    """Alpaca's .df carries a (symbol, timestamp) row MultiIndex; no alpaca install needed."""
    index = pd.MultiIndex.from_product(
        [["SPY", "QQQ"], pd.date_range("2024-01-01", periods=3, freq="D")],
        names=["symbol", "timestamp"])
    frame = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                          "close": 100.5, "volume": 1000}, index=index)

    flat = DataLoader._flatten_index(frame, "QQQ", "test-request")

    assert not isinstance(flat.index, pd.MultiIndex)
    assert len(flat) == 3
    assert isinstance(flat.index, pd.DatetimeIndex)


def test_alpaca_style_multiindex_unknown_symbol_raises():
    index = pd.MultiIndex.from_product(
        [["SPY"], pd.date_range("2024-01-01", periods=2, freq="D")],
        names=["symbol", "timestamp"])
    frame = pd.DataFrame({"open": 1.0}, index=index)

    with pytest.raises(DataLoadError) as excinfo:
        DataLoader._flatten_index(frame, "TSLA", "test-request")
    assert "SPY" in str(excinfo.value)


# --- quality warnings ------------------------------------------------------

def test_nan_in_ohlc_warns(monkeypatch, caplog):
    frame = _valid_frame()
    frame.iloc[2, frame.columns.get_loc("Close")] = float("nan")
    _stub_yfinance(monkeypatch, frame)

    with caplog.at_level(logging.WARNING, logger="backtester.data_loader"):
        df = DataLoader(_config()).load("SPY")

    assert len(df) == 5  # still returned; NaNs are a warning, not a hard failure
    assert any("NaN" in record.message for record in caplog.records)


def test_non_monotonic_index_warns(monkeypatch, caplog):
    frame = _valid_frame()
    frame = frame.iloc[[0, 3, 1, 2, 4]]
    _stub_yfinance(monkeypatch, frame)

    with caplog.at_level(logging.WARNING, logger="backtester.data_loader"):
        DataLoader(_config()).load("SPY")

    assert any("monotonically increasing" in record.message for record in caplog.records)


def test_clean_data_emits_no_warnings(monkeypatch, caplog):
    _stub_yfinance(monkeypatch, _valid_frame())

    with caplog.at_level(logging.WARNING, logger="backtester.data_loader"):
        DataLoader(_config()).load("SPY")

    assert caplog.records == []


def test_intraday_60_day_warning_preserved(monkeypatch, caplog):
    calls = []
    _stub_yfinance(monkeypatch, _valid_frame(), calls=calls)

    with caplog.at_level(logging.WARNING, logger="backtester.data_loader"):
        DataLoader(_config(timeframe="5m")).load("SPY")

    assert any("60 days" in record.message for record in caplog.records)
    assert calls[0][1].get("period") == "60d"
    assert "start" not in calls[0][1]


def test_yahoo_download_uses_normalized_timeframe(monkeypatch):
    calls = []
    _stub_yfinance(monkeypatch, _valid_frame(), calls=calls)

    DataLoader(_config(timeframe="15 minutes")).load("SPY")

    assert calls[0][1]["interval"] == "15m"
    assert calls[0][1]["period"] == "60d"


# --- happy path / caching --------------------------------------------------

def test_valid_data_loads_and_caches(monkeypatch):
    calls = []
    _stub_yfinance(monkeypatch, _valid_frame(rows=7), calls=calls)
    loader = DataLoader(_config())

    df = loader.load("SPY")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert len(df) == 7
    assert df["close"].iloc[0] == 100.5

    again = loader.load("SPY")
    assert again is df
    assert len(calls) == 1  # served from cache, no second download

    loader.load("SPY", force_refresh=True)
    assert len(calls) == 2


def test_extra_columns_are_dropped(monkeypatch):
    _stub_yfinance(monkeypatch, _valid_frame(
        columns=("Open", "High", "Low", "Close", "Volume", "Adj Close", "Dividends")))

    df = DataLoader(_config()).load("SPY")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


# --- csv source ------------------------------------------------------------

def test_csv_source_loads(monkeypatch, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "SPY.csv").write_text("Date,Open,High,Low,Close,Volume\n"
                                      "2024-01-01,100,101,99,100.5,1000\n"
                                      "2024-01-02,100,101,99,100.5,1000\n")
    monkeypatch.chdir(tmp_path)

    df = DataLoader(_config(source="csv")).load("SPY")

    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert isinstance(df.index, pd.DatetimeIndex)
    assert len(df) == 2


def test_csv_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(DataLoadError) as excinfo:
        DataLoader(_config(source="csv")).load("NOPE")
    assert "NOPE" in str(excinfo.value)
