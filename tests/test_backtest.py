import pytest
from datetime import datetime
from backtester.engine import BacktestEngine, Trade
from backtester.metrics import RiskMetrics

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
