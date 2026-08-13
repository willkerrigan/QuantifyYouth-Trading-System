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
