import logging

import pytest
import pandas as pd
from datetime import datetime
from backtester.engine import BacktestEngine, Trade
from backtester.metrics import RiskMetrics, periods_per_year_for_timeframe


class _StubDataLoader:
    """Serves a fixed 10-day OHLCV frame so date-filtering can be tested without network/disk access."""

    def __init__(self, config):
        dates = pd.date_range("2024-01-01", periods=10, freq="D")
        self._df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                                 "close": 100.0, "volume": 1000}, index=dates)

    def load(self, symbol, force_refresh=False):
        return self._df


def _hold_strategy(daily_data, open_positions, params):
    return {symbol: "HOLD" for symbol in daily_data}


def test_run_restricts_to_date_window():
    config = {"backtest": {"initial_capital": 100000, "commission": 0.001, "slippage": 0.0}}
    engine = BacktestEngine(config, _hold_strategy, data_loader=_StubDataLoader(config))

    _, _, full_curve = engine.run(["SPY"], {})
    assert len(full_curve) == 10

    _, _, in_sample_curve = engine.run(["SPY"], {}, end_date="2024-01-05")
    assert len(in_sample_curve) == 5
    assert in_sample_curve["date"].max() == pd.Timestamp("2024-01-05")

    _, _, out_of_sample_curve = engine.run(["SPY"], {}, start_date="2024-01-06")
    assert len(out_of_sample_curve) == 5
    assert out_of_sample_curve["date"].min() == pd.Timestamp("2024-01-06")


class _StubIntradayDataLoader:
    """Serves tz-aware 15m bars, mirroring what yfinance actually returns intraday."""

    def __init__(self, config):
        dates = pd.date_range("2024-01-01 09:30", periods=8, freq="15min", tz="UTC")
        self._df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                                 "close": 100.0, "volume": 1000}, index=dates)

    def load(self, symbol, force_refresh=False):
        return self._df


def test_run_restricts_to_date_window_with_tz_aware_intraday_data():
    config = {"backtest": {"initial_capital": 100000, "commission": 0.001, "slippage": 0.0}}
    engine = BacktestEngine(config, _hold_strategy, data_loader=_StubIntradayDataLoader(config))

    # end_date is a bare date string; it must still include every bar that day, not just midnight.
    _, _, curve = engine.run(["SPY"], {}, end_date="2024-01-01")
    assert len(curve) == 8


class _StubOHLCDataLoader:
    """Serves a caller-supplied OHLC frame so stop-loss triggers can be tested exactly."""

    def __init__(self, rows):
        dates = pd.date_range("2024-01-01", periods=len(rows), freq="D")
        self._df = pd.DataFrame(rows, index=dates)

    def load(self, symbol, force_refresh=False):
        return self._df


def _buy_first_bar_then_hold(daily_data, open_positions, params):
    """Opens on the first bar and otherwise never trades, so any exit must come from the stop."""
    return {symbol: ("BUY" if symbol not in open_positions else "HOLD") for symbol in daily_data}


# Bar 0 opens the position at 100. Bar 1 dips to 97 (below a 2% stop at 98) but closes
# back at 100, so only a stop that looks at the low can catch it.
_DIP_ROWS = [
    {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
    {"open": 100.0, "high": 101.0, "low": 97.0, "close": 100.0, "volume": 1000},
    {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
]

_CONFIG = {"backtest": {"initial_capital": 100000, "commission": 0.001, "slippage": 0.0}}


def test_stop_loss_triggers_when_low_breaches_level():
    engine = BacktestEngine(_CONFIG, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    _, trades, _ = engine.run(["SPY"], {"stop_loss": 0.02})

    # The 97.0 low on bar 1 breaches the 98.0 stop even though that bar closed at 100.
    assert len(trades) == 1
    assert trades[0].entry_date == pd.Timestamp("2024-01-01")
    assert trades[0].exit_date == pd.Timestamp("2024-01-02")


def test_stop_loss_exit_price_is_the_stop_price_not_the_close():
    engine = BacktestEngine(_CONFIG, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    _, trades, _ = engine.run(["SPY"], {"stop_loss": 0.02})

    # entry 100 * (1 - 0.02) == 98.0, even though the bar closed at 100.
    assert trades[0].entry_price == 100.0
    assert trades[0].exit_price == pytest.approx(98.0)
    # Gross is the raw price move; realized is net of the commission actually paid
    # on both legs, since win rate and profit factor are scored off realized_pnl.
    assert trades[0].gross_pnl == pytest.approx(-2.0 * trades[0].size)
    assert trades[0].costs == pytest.approx(100 * 100 * 0.001 + 100 * 98 * 0.001)
    assert trades[0].realized_pnl == pytest.approx(trades[0].gross_pnl - trades[0].costs)


def test_stop_loss_does_not_trigger_when_low_stays_above_level():
    engine = BacktestEngine(_CONFIG, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    # A 5% stop sits at 95.0; the 97.0 low never reaches it.
    _, trades, _ = engine.run(["SPY"], {"stop_loss": 0.05})

    assert trades == []
    assert "SPY" in engine.open_positions


@pytest.mark.parametrize("params", [{}, {"stop_loss": None}, {"stop_loss": 0}])
def test_no_stop_loss_param_leaves_behavior_unchanged(params):
    engine = BacktestEngine(_CONFIG, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    final_capital, trades, curve = engine.run(["SPY"], params)

    baseline = BacktestEngine(_CONFIG, _buy_first_bar_then_hold,
                              data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    baseline_capital, baseline_trades, baseline_curve = baseline.run(["SPY"], {})

    assert trades == [] and baseline_trades == []
    assert final_capital == baseline_capital
    assert len(curve) == len(baseline_curve) == 3
    assert engine.open_positions["SPY"]["entry_price"] == 100.0


def test_stopped_out_position_ignores_the_strategys_signal_that_bar():
    def always_buy(daily_data, open_positions, params):
        return {symbol: "BUY" for symbol in daily_data}

    engine = BacktestEngine(_CONFIG, always_buy, data_loader=_StubOHLCDataLoader(_DIP_ROWS))
    _, trades, _ = engine.run(["SPY"], {"stop_loss": 0.02})

    # Bar 1 stops out; the BUY on that same bar must not re-open the position.
    # Only bar 2's BUY re-enters, so exactly one trade is closed.
    assert len(trades) == 1
    assert trades[0].exit_date == pd.Timestamp("2024-01-02")
    assert engine.open_positions["SPY"]["entry_date"] == pd.Timestamp("2024-01-03")


_RISING_ROWS = [
    {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
    {"open": 150.0, "high": 151.0, "low": 149.0, "close": 150.0, "volume": 1000},
    {"open": 200.0, "high": 201.0, "low": 199.0, "close": 200.0, "volume": 1000},
]

_FRICTIONLESS = {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.0}}


def test_equity_curve_marks_open_positions_to_market():
    """Buying must not look like a loss: cash leaves, position value replaces it."""
    engine = BacktestEngine(_FRICTIONLESS, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_RISING_ROWS))
    _, _, curve = engine.run(["SPY"], {})

    # Bar 0 opens the position at 100; equity must stay at the starting capital
    # rather than dropping by the cash spent.
    assert curve["equity"].iloc[0] == pytest.approx(100000.0)
    # Price doubles by bar 2 and the position is still open, so equity must rise.
    assert curve["equity"].iloc[-1] > curve["equity"].iloc[0]


def test_open_position_is_not_valued_at_zero_at_end_of_run():
    """A run that ends holding a winner must report a gain, not a phantom loss."""
    engine = BacktestEngine(_FRICTIONLESS, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_RISING_ROWS))
    final_equity, trades, _ = engine.run(["SPY"], {})

    assert trades == []                      # never closed
    assert "SPY" in engine.open_positions    # still held
    size = engine.open_positions["SPY"]["size"]
    # 100 -> 200 on `size` shares, with the rest of the capital still in cash.
    assert final_equity == pytest.approx(100000.0 + size * 100.0)


def test_slippage_is_applied_to_fills():
    config = {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.01}}
    engine = BacktestEngine(config, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_RISING_ROWS))
    engine.run(["SPY"], {})
    # Buying fills above the quoted close of 100, never at it.
    assert engine.open_positions["SPY"]["entry_price"] == pytest.approx(101.0)


def test_zero_share_position_is_not_opened():
    """A price above 10% of capital sizes to 0 shares; that must not book a position."""
    rows = [{"open": 1e6, "high": 1.1e6, "low": 0.9e6, "close": 1e6, "volume": 1}] * 2
    engine = BacktestEngine(_FRICTIONLESS, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(rows))
    _, trades, _ = engine.run(["SPY"], {})

    assert engine.open_positions == {}
    assert trades == []


def test_total_data_load_failure_raises_instead_of_reporting_zero_trades():
    class _FailingLoader:
        def load(self, symbol, force_refresh=False):
            raise ValueError("no data for you")

    engine = BacktestEngine(_FRICTIONLESS, _hold_strategy, data_loader=_FailingLoader())
    with pytest.raises(RuntimeError, match="No data could be loaded"):
        engine.run(["SPY"], {})


# --- risk: config block ------------------------------------------------------

# A flat 100.00 tape, so every position size below is exact arithmetic rather
# than a rounding artifact.
_FLAT_ROWS = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000}] * 3


def _risk_config(**risk):
    return {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.0},
            "risk": risk}


def test_configured_max_position_size_is_used_for_sizing():
    engine = BacktestEngine(_risk_config(max_position_size=0.25), _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    engine.run(["SPY"], {})

    # 25% of 100,000 equity at 100.00 a share.
    assert engine.max_position_size == 0.25
    assert engine.open_positions["SPY"]["size"] == 250


@pytest.mark.parametrize("config", [
    {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.0}},   # no risk block
    {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.0}, "risk": {}},
    {"backtest": {"initial_capital": 100000, "commission": 0.0, "slippage": 0.0}, "risk": None},
])
def test_max_position_size_defaults_to_ten_percent_when_unset(config):
    engine = BacktestEngine(config, _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    engine.run(["SPY"], {})

    assert engine.max_position_size == 0.1
    assert engine.open_positions["SPY"]["size"] == 100


@pytest.mark.parametrize("bad", ["abc", None, 0, -0.5, float("nan"), float("inf"), True, [0.2]])
def test_invalid_max_position_size_falls_back_to_default_with_a_warning(bad, caplog):
    # `None` is the one "invalid" value that means unset, so it must not warn.
    with caplog.at_level(logging.WARNING, logger="backtester.engine"):
        engine = BacktestEngine(_risk_config(max_position_size=bad), _buy_first_bar_then_hold,
                                data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    engine.run(["SPY"], {})

    assert engine.max_position_size == 0.1
    assert engine.open_positions["SPY"]["size"] == 100
    warned = any("max_position_size" in r.message for r in caplog.records
                 if r.levelno >= logging.WARNING)
    assert warned is (bad is not None)


def test_max_position_size_above_one_warns_but_is_honoured(caplog):
    with caplog.at_level(logging.WARNING, logger="backtester.engine"):
        engine = BacktestEngine(_risk_config(max_position_size=1.5, max_leverage=3.0),
                                _buy_first_bar_then_hold,
                                data_loader=_StubOHLCDataLoader(_FLAT_ROWS))

    assert engine.max_position_size == 1.5
    assert any("max_position_size" in r.message and "100%" in r.message
               for r in caplog.records if r.levelno >= logging.WARNING)


def test_max_leverage_blocks_an_entry_that_would_breach_the_cap(caplog):
    # 30% per position against a 0.5x gross cap: the first entry fits (0.3x), the
    # second would take gross exposure to 0.6x and must be skipped.
    engine = BacktestEngine(_risk_config(max_position_size=0.3, max_leverage=0.5),
                            _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    with caplog.at_level(logging.INFO, logger="backtester.engine"):
        engine.run(["SPY", "QQQ"], {})

    assert set(engine.open_positions) == {"SPY"}
    assert engine.open_positions["SPY"]["size"] == 300
    assert any("max_leverage" in r.message for r in caplog.records)
    # Cash was never the binding constraint: 70,000 was still uncommitted.
    assert engine.capital == pytest.approx(70000.0)


def test_second_entry_is_allowed_when_max_leverage_has_room():
    engine = BacktestEngine(_risk_config(max_position_size=0.3, max_leverage=1.0),
                            _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    engine.run(["SPY", "QQQ"], {})

    assert set(engine.open_positions) == {"SPY", "QQQ"}


def test_max_leverage_defaults_to_one_and_is_validated(caplog):
    engine = BacktestEngine(_risk_config(max_position_size=0.1), _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    assert engine.max_leverage == 1.0

    with caplog.at_level(logging.WARNING, logger="backtester.engine"):
        bad = BacktestEngine(_risk_config(max_leverage="lots"), _buy_first_bar_then_hold,
                             data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    assert bad.max_leverage == 1.0
    assert any("max_leverage" in r.message for r in caplog.records
               if r.levelno >= logging.WARNING)


def test_sizing_uses_total_equity_not_just_remaining_cash():
    """Deploying cash must not shrink later positions: equity is unchanged by a buy."""
    engine = BacktestEngine(_risk_config(max_position_size=0.1), _buy_first_bar_then_hold,
                            data_loader=_StubOHLCDataLoader(_FLAT_ROWS))
    engine.run(["SPY", "QQQ", "IWM"], {})

    sizes = [engine.open_positions[s]["size"] for s in ("SPY", "QQQ", "IWM")]
    # Sizing off cash alone would give 100, then 90, then 81 as capital drains.
    assert sizes == [100, 100, 100]
    assert engine.capital == pytest.approx(70000.0)


def test_equity_sizing_grows_positions_as_the_book_appreciates():
    """Equity marked to market, not cost basis: a winner funds a larger next entry."""
    rows = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000},
        {"open": 200.0, "high": 201.0, "low": 199.0, "close": 200.0, "volume": 1000},
    ]

    def buy_spy_then_qqq(daily_data, open_positions, params):
        if not open_positions:
            return {"SPY": "BUY"}
        return {"QQQ": "BUY"} if "QQQ" not in open_positions else {}

    engine = BacktestEngine(_risk_config(max_position_size=0.1, max_leverage=2.0),
                            buy_spy_then_qqq, data_loader=_StubOHLCDataLoader(rows))
    engine.run(["SPY", "QQQ"], {})

    # Bar 0: 10% of 100,000 at 100 -> 100 shares, 90,000 cash left.
    # Bar 1: SPY is now worth 20,000, so equity is 110,000; 10% of that at 200 -> 55 shares.
    assert engine.open_positions["SPY"]["size"] == 100
    assert engine.open_positions["QQQ"]["size"] == 55


def test_periods_per_year_for_daily_timeframe():
    assert periods_per_year_for_timeframe("1d") == 252


def test_periods_per_year_for_15m_timeframe():
    # 390 regular-session minutes / 15 = 26 bars/day * 252 trading days
    assert periods_per_year_for_timeframe("15m") == 26 * 252


def test_trade_creation():
    trade = Trade(asset="SPY", entry_date=datetime(2023, 1, 1, 10, 0, 0), entry_price=450.0,
                 exit_date=datetime(2023, 1, 2, 10, 0, 0), exit_price=455.0, size=100,
                 strategy_params={"ma_short": 20, "ma_long": 50})
    assert trade.asset == "SPY"
    assert trade.realized_pnl == 500.0

def test_sharpe_ratio():
    returns = [0.01, 0.02, -0.01, 0.03, 0.02]
    sharpe = RiskMetrics.calculate_sharpe_ratio(returns)
    assert isinstance(sharpe, float)
    assert sharpe > 0
