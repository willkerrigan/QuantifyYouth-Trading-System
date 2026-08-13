"""Tests for the live execution path.

Everything here runs against in-process fakes: no network, no alpaca-py, no
real broker client is ever constructed.
"""

from datetime import datetime

import pytest

from execution import live_trader as live_trader_module
from execution.broker_adapter import BrokerAdapter
from execution.live_trader import DEFAULT_MAX_POSITION_PCT, LiveTrader
from execution.signal_handler import Signal, SignalType


class FakeBroker:
    """Stand-in for BrokerAdapter that records submitted orders."""

    def __init__(self, buying_power=100_000.0, positions=None):
        self.paper_trading = True
        self.account = {"buying_power": buying_power, "cash": buying_power,
                        "portfolio_value": buying_power, "equity": buying_power}
        self.positions = positions or {}
        self.orders = []
        self.closed = []

    def get_account(self):
        return dict(self.account)

    def get_positions(self):
        return dict(self.positions)

    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        self.orders.append({"symbol": symbol, "qty": qty, "side": side})
        return {"order_id": f"fake-{len(self.orders)}", "symbol": symbol,
                "qty": float(qty), "side": side, "status": "accepted"}

    def close_position(self, symbol):
        self.closed.append(symbol)
        return {"order_id": "fake-close", "symbol": symbol, "qty": 1.0, "side": "sell"}


def make_trader(monkeypatch, config=None, broker=None):
    """Build a LiveTrader whose broker is a fake (never touches Alpaca)."""
    broker = broker or FakeBroker()
    monkeypatch.setattr(live_trader_module, "BrokerAdapter", lambda cfg: broker)
    trader = LiveTrader(config or {}, strategy_name="test")
    return trader, broker


def buy_signal(symbol="AAPL", price=100.0, **metadata):
    if price is not None:
        metadata["current_price"] = price
    return Signal(symbol, SignalType.BUY, datetime.now(), confidence=1.0, metadata=metadata)


# --- position sizing is config driven -------------------------------------

def test_default_position_pct_when_unconfigured(monkeypatch):
    trader, broker = make_trader(monkeypatch, {})
    assert trader.max_position_pct == DEFAULT_MAX_POSITION_PCT

    trader._execute_signal(buy_signal(price=100.0))

    # 100_000 * 0.1 / 100 = 100 shares
    assert broker.orders == [{"symbol": "AAPL", "qty": 100, "side": "buy"}]


def test_position_size_respects_configured_pct(monkeypatch):
    config = {"position_management": {"max_position_pct": 0.25}}
    trader, broker = make_trader(monkeypatch, config)
    assert trader.max_position_pct == 0.25

    trader._execute_signal(buy_signal(price=50.0))

    # 100_000 * 0.25 / 50 = 500 shares
    assert broker.orders[0]["qty"] == 500


@pytest.mark.parametrize("bad_pct", [0, -0.5, "abc", None])
def test_invalid_position_pct_falls_back_to_default(monkeypatch, bad_pct):
    config = {"position_management": {"max_position_pct": bad_pct}}
    trader, _ = make_trader(monkeypatch, config)
    assert trader.max_position_pct == DEFAULT_MAX_POSITION_PCT


# --- a missing / bad price must refuse to trade ---------------------------

def test_missing_current_price_refuses_to_trade(monkeypatch):
    trader, broker = make_trader(monkeypatch, {})

    signal = Signal("AAPL", SignalType.BUY, datetime.now(), confidence=1.0, metadata={})
    trader._execute_signal(signal)

    # The old code divided buying power by a default price of 1, sizing a
    # 100_000-share position. Nothing may be submitted now.
    assert broker.orders == []
    assert trader.trades_executed == []


@pytest.mark.parametrize("bad_price", [0, -10.0, "not-a-number", float("nan"), float("inf"), None])
def test_invalid_current_price_refuses_to_trade(monkeypatch, bad_price):
    trader, broker = make_trader(monkeypatch, {})

    signal = Signal("AAPL", SignalType.BUY, datetime.now(), confidence=1.0,
                    metadata={"current_price": bad_price})
    trader._execute_signal(signal)

    assert broker.orders == []


# --- notional must fit inside buying power --------------------------------

def test_order_exceeding_buying_power_is_rejected(monkeypatch):
    # An over-leveraged pct would size a position worth more than the account
    # can pay for; it must be refused rather than sent to the broker.
    config = {"position_management": {"max_position_pct": 1.5}}
    broker = FakeBroker(buying_power=10_000.0)
    trader, broker = make_trader(monkeypatch, config, broker)

    trader._execute_signal(buy_signal(price=100.0))

    assert broker.orders == []


def test_order_within_buying_power_is_submitted(monkeypatch):
    config = {"position_management": {"max_position_pct": 1.0}}
    broker = FakeBroker(buying_power=10_000.0)
    trader, broker = make_trader(monkeypatch, config, broker)

    trader._execute_signal(buy_signal(price=100.0))

    assert broker.orders[0]["qty"] == 100
    assert broker.orders[0]["qty"] * 100.0 <= broker.account["buying_power"]


@pytest.mark.parametrize("buying_power", [0, -100.0])
def test_no_buying_power_refuses_to_trade(monkeypatch, buying_power):
    trader, broker = make_trader(monkeypatch, {}, FakeBroker(buying_power=buying_power))

    trader._execute_signal(buy_signal(price=100.0))

    assert broker.orders == []


def test_price_too_high_for_allocation_submits_nothing(monkeypatch):
    # 10_000 * 0.1 = 1_000 of allocation against a 5_000 share price -> 0 shares.
    broker = FakeBroker(buying_power=10_000.0)
    trader, broker = make_trader(monkeypatch, {}, broker)

    trader._execute_signal(buy_signal(price=5_000.0))

    assert broker.orders == []


# --- portfolio limits ------------------------------------------------------

def test_max_open_positions_enforced(monkeypatch):
    config = {"position_management": {"max_open_positions": 2}}
    positions = {"MSFT": {"qty": 1}, "TSLA": {"qty": 1}}
    broker = FakeBroker(positions=positions)
    trader, broker = make_trader(monkeypatch, config, broker)

    trader._execute_signal(buy_signal(symbol="AAPL", price=100.0))

    assert broker.orders == []


def test_buy_allowed_below_max_open_positions(monkeypatch):
    config = {"position_management": {"max_open_positions": 2}}
    broker = FakeBroker(positions={"MSFT": {"qty": 1}})
    trader, broker = make_trader(monkeypatch, config, broker)

    trader._execute_signal(buy_signal(symbol="AAPL", price=100.0))

    assert len(broker.orders) == 1


def test_existing_position_is_not_doubled_up(monkeypatch):
    broker = FakeBroker(positions={"AAPL": {"qty": 10}})
    trader, broker = make_trader(monkeypatch, {}, broker)

    trader._execute_signal(buy_signal(symbol="AAPL", price=100.0))

    assert broker.orders == []


def test_default_max_open_positions(monkeypatch):
    trader, _ = make_trader(monkeypatch, {})
    assert trader.max_open_positions == 10


# --- sell path -------------------------------------------------------------

def test_sell_closes_open_position(monkeypatch):
    broker = FakeBroker(positions={"AAPL": {"qty": 10}})
    trader, broker = make_trader(monkeypatch, {}, broker)

    signal = Signal("AAPL", SignalType.SELL, datetime.now(), confidence=1.0)
    trader._execute_signal(signal)

    assert broker.closed == ["AAPL"]
    assert len(trader.trades_executed) == 1


def test_sell_without_position_is_a_noop(monkeypatch):
    trader, broker = make_trader(monkeypatch, {})

    signal = Signal("AAPL", SignalType.SELL, datetime.now(), confidence=1.0)
    trader._execute_signal(signal)

    assert broker.closed == []


# --- broker adapter validation --------------------------------------------

class FakeClient:
    """Client stub that fails loudly if a bad order ever reaches it."""

    def __init__(self):
        self.submitted = []

    def submit_order(self, request):  # pragma: no cover - must not be called
        self.submitted.append(request)
        raise AssertionError("submit_order should not have reached the broker client")


def make_adapter():
    """BrokerAdapter without _initialize_client (alpaca-py is not installed)."""
    adapter = object.__new__(BrokerAdapter)
    adapter.config = {}
    adapter.alpaca_config = {}
    adapter.paper_trading = True
    adapter.client = FakeClient()
    return adapter


@pytest.mark.parametrize("qty", [0, 0.0, -1, -0.5, "abc", None, float("nan")])
def test_submit_order_rejects_non_positive_qty(qty):
    adapter = make_adapter()

    assert adapter.submit_order("AAPL", qty, "buy") is None
    assert adapter.client.submitted == []


@pytest.mark.parametrize("side", ["", "hold", "BUYY", None, 1])
def test_submit_order_rejects_invalid_side(side):
    adapter = make_adapter()

    assert adapter.submit_order("AAPL", 10, side) is None
    assert adapter.client.submitted == []


@pytest.mark.parametrize("symbol", ["", None, 123])
def test_submit_order_rejects_invalid_symbol(symbol):
    adapter = make_adapter()

    assert adapter.submit_order(symbol, 10, "buy") is None
    assert adapter.client.submitted == []


def test_live_trader_never_submits_non_positive_qty(monkeypatch):
    """End-to-end: whatever the trader sizes, it is always a positive qty."""
    trader, broker = make_trader(monkeypatch, {"position_management": {"max_position_pct": 0.5}})

    for price in (1.0, 33.33, 999.0):
        trader._execute_signal(buy_signal(symbol=f"S{price}", price=price))

    assert broker.orders
    assert all(order["qty"] > 0 for order in broker.orders)
