"""Tests for the pre-trade safety rails.

Everything is deterministic: the clock is injected (never read from the wall
clock), the broker is a fake, and no network call, alpaca import, or sleep
happens anywhere in this file.
"""

from datetime import datetime, time as dtime, timezone
from zoneinfo import ZoneInfo

import pytest

from execution import live_trader as live_trader_module
from execution.live_trader import LiveTrader
from execution.risk_guard import (
    REASON_DAILY_LOSS,
    REASON_EQUITY_UNKNOWN,
    REASON_KILL_SWITCH,
    REASON_MARKET_CLOSED,
    DailyLossGuard,
    KillSwitch,
    MarketHoursGate,
    RiskGuard,
)
from execution.signal_handler import Signal, SignalType

ET = ZoneInfo("America/New_York")

# Fixed reference instants. Never `datetime.now()`.
WED_MIDDAY = datetime(2024, 3, 13, 12, 0, tzinfo=ET)      # Wednesday, session open
WED_PREMARKET = datetime(2024, 3, 13, 9, 0, tzinfo=ET)    # Wednesday, before 09:30
WED_AFTERHOURS = datetime(2024, 3, 13, 16, 30, tzinfo=ET)  # Wednesday, after 16:00
SATURDAY = datetime(2024, 3, 16, 12, 0, tzinfo=ET)        # Weekend
THU_MIDDAY = datetime(2024, 3, 14, 12, 0, tzinfo=ET)      # Next trading day


class FakeBroker:
    """Stand-in for BrokerAdapter. Records orders; never talks to a broker."""

    def __init__(self, equity=100_000.0, buying_power=None, positions=None, account=None):
        self.paper_trading = True
        bp = buying_power if buying_power is not None else equity
        self.account = account if account is not None else {
            "buying_power": bp, "cash": bp, "portfolio_value": equity, "equity": equity}
        self.positions = dict(positions or {})
        self.orders = []
        self.closed = []

    def set_equity(self, equity):
        self.account["equity"] = equity
        self.account["portfolio_value"] = equity

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
        self.positions.pop(symbol, None)
        return {"order_id": "fake-close", "symbol": symbol, "qty": 1.0, "side": "sell"}


def make_trader(monkeypatch, config=None, broker=None, now=WED_MIDDAY):
    """LiveTrader backed by a fake broker and a frozen clock."""
    broker = broker or FakeBroker()
    monkeypatch.setattr(live_trader_module, "BrokerAdapter", lambda cfg: broker)
    trader = LiveTrader(config or {}, strategy_name="test", now_provider=lambda: now)
    return trader, broker


def buy_signal(symbol="AAPL", price=100.0):
    return Signal(symbol, SignalType.BUY, datetime.now(), confidence=1.0,
                  metadata={"current_price": price})


# =========================================================================== #
# market hours gate
# =========================================================================== #

@pytest.mark.parametrize("moment, expected", [
    (WED_MIDDAY, True),
    (datetime(2024, 3, 13, 9, 30, tzinfo=ET), True),    # exactly at the open
    (datetime(2024, 3, 13, 15, 59, tzinfo=ET), True),
    (datetime(2024, 3, 13, 16, 0, tzinfo=ET), False),   # the close is exclusive
    (WED_PREMARKET, False),
    (WED_AFTERHOURS, False),
    (SATURDAY, False),
    (datetime(2024, 3, 17, 12, 0, tzinfo=ET), False),   # Sunday
])
def test_market_hours_weekday_and_time(moment, expected):
    assert MarketHoursGate().is_open(moment) is expected


def test_market_hours_uses_zoneinfo_not_a_fixed_utc_offset():
    """The same UTC instant is in-session in March (EDT) and out in January (EST)."""
    gate = MarketHoursGate()
    # 20:30 UTC == 16:30 EDT (closed) in summer time, but 15:30 EST (open) in winter.
    assert gate.is_open(datetime(2024, 1, 10, 20, 30, tzinfo=timezone.utc)) is True
    assert gate.is_open(datetime(2024, 7, 10, 20, 30, tzinfo=timezone.utc)) is False


def test_market_hours_converts_aware_utc_input():
    # 2024-03-13 17:00 UTC == 13:00 EDT -> open.
    assert MarketHoursGate().is_open(datetime(2024, 3, 13, 17, 0, tzinfo=timezone.utc)) is True


def test_market_hours_treats_naive_input_as_market_local():
    assert MarketHoursGate().is_open(datetime(2024, 3, 13, 12, 0)) is True
    assert MarketHoursGate().is_open(datetime(2024, 3, 13, 20, 0)) is False


def test_market_hours_gate_is_on_by_default():
    gate = MarketHoursGate()
    assert gate.enabled is True
    assert gate.allow_outside_hours is False
    assert gate.active is True
    decision = gate.check(SATURDAY)
    assert not decision
    assert decision.code == REASON_MARKET_CLOSED


def test_declaring_a_risk_section_arms_the_market_gate():
    """The gate is the one rail you get without asking: an empty `risk:` arms it."""
    assert RiskGuard.from_config({"risk": {}}).market_hours.active is True
    assert RiskGuard.from_config({"risk": {"max_daily_loss_pct": 0.01}}).market_hours.active is True
    assert not RiskGuard.from_config({"risk": {}}).check_cycle({"equity": 1.0}, now=SATURDAY)


def test_no_risk_section_at_all_keeps_legacy_behaviour():
    """Configs written before risk_guard existed must not start halting silently."""
    guard = RiskGuard.from_config({})
    assert guard.market_hours.active is False
    assert guard.check_cycle({"equity": 1.0}, now=SATURDAY)


def test_allow_outside_hours_override_defaults_off_and_can_be_enabled():
    assert RiskGuard.from_config({"risk": {}}).market_hours.allow_outside_hours is False
    guard = RiskGuard.from_config({"risk": {"market_hours": {"allow_outside_hours": True}}})
    assert guard.market_hours.active is False
    assert guard.check_cycle({"equity": 1.0}, now=SATURDAY)


def test_market_hours_can_be_disabled_entirely():
    guard = RiskGuard.from_config({"risk": {"market_hours": {"enabled": False}}})
    assert guard.check_cycle({"equity": 1.0}, now=SATURDAY)


def test_custom_session_times_and_bad_timezone_fallback():
    guard = RiskGuard.from_config({"risk": {"market_hours": {"open": "10:00", "close": "15:00"}}})
    assert guard.market_hours.open_time == dtime(10, 0)
    assert guard.market_hours.close_time == dtime(15, 0)
    assert guard.market_hours.is_open(datetime(2024, 3, 13, 9, 45, tzinfo=ET)) is False

    bad = MarketHoursGate(tz_name="Not/AZone")
    assert bad.tz_name == "America/New_York"


def test_market_hours_docstring_disclaims_holiday_support():
    doc = MarketHoursGate.__doc__ or ""
    assert "HOLIDAY" in doc.upper()


# =========================================================================== #
# kill switch
# =========================================================================== #

def test_kill_switch_programmatic_trip_latches():
    switch = KillSwitch()
    assert switch.check()
    switch.trip("operator halt")
    assert not switch.check()
    assert switch.check().code == REASON_KILL_SWITCH
    # Repeated polls must not clear it.
    assert switch.poll() is True
    assert switch.tripped is True


def test_kill_switch_file_trips_and_stays_tripped_after_file_removal(tmp_path):
    path = tmp_path / "KILL"
    switch = KillSwitch(str(path))
    assert switch.check()

    path.write_text("halt")
    assert not switch.check()

    path.unlink()
    # The whole point: removing the file must NOT silently re-arm trading.
    assert switch.poll() is True
    assert not switch.check()


def test_kill_switch_reset_requires_the_file_to_be_gone(tmp_path):
    path = tmp_path / "KILL"
    path.write_text("halt")
    switch = KillSwitch(str(path))
    switch.poll()

    assert switch.reset() is False
    assert switch.tripped is True

    path.unlink()
    assert switch.reset() is True
    assert switch.tripped is False
    assert switch.check()


def test_no_kill_switch_file_configured_never_trips_by_itself():
    switch = KillSwitch(None)
    for _ in range(3):
        assert switch.poll() is False


# =========================================================================== #
# max daily loss
# =========================================================================== #

def test_daily_loss_disabled_when_unconfigured():
    guard = DailyLossGuard()
    assert guard.enabled is False
    guard.start_session({"equity": 100_000.0})
    assert guard.update({"equity": 1.0})  # a 99.999% drawdown still allowed
    assert guard.breached is False


def test_daily_loss_pct_breach():
    guard = DailyLossGuard(max_daily_loss_pct=0.02)
    guard.start_session({"equity": 100_000.0})
    assert guard.starting_equity == 100_000.0
    assert guard.loss_limit() == 2_000.0

    assert guard.update({"equity": 98_500.0})       # -1,500, under the limit
    decision = guard.update({"equity": 98_000.0})   # -2,000, at the limit
    assert not decision
    assert decision.code == REASON_DAILY_LOSS


def test_daily_loss_absolute_amount_breach():
    guard = DailyLossGuard(max_daily_loss_amount=500.0)
    guard.start_session({"equity": 10_000.0})
    assert guard.update({"equity": 9_600.0})
    assert not guard.update({"equity": 9_400.0})


def test_tighter_of_pct_and_amount_binds():
    # 1% of 100k = 1,000 vs an absolute 250 -> 250 binds.
    guard = DailyLossGuard(max_daily_loss_pct=0.01, max_daily_loss_amount=250.0)
    guard.start_session({"equity": 100_000.0})
    assert guard.loss_limit() == 250.0
    assert not guard.update({"equity": 99_700.0})


def test_daily_loss_breach_latches_even_if_equity_recovers():
    guard = DailyLossGuard(max_daily_loss_pct=0.02)
    guard.start_session({"equity": 100_000.0})
    assert not guard.update({"equity": 97_000.0})

    # Bounce back above the starting equity - the day is still over.
    assert not guard.update({"equity": 105_000.0})
    assert guard.breached is True


def test_unrealized_loss_counts_because_equity_is_marked_to_market():
    guard = DailyLossGuard(max_daily_loss_pct=0.05)
    guard.start_session({"equity": 50_000.0, "cash": 50_000.0})
    # Cash unchanged (nothing realized) but equity is down on open positions.
    assert not guard.update({"equity": 47_000.0, "cash": 50_000.0})


def test_unreadable_equity_blocks_entries_without_latching():
    guard = DailyLossGuard(max_daily_loss_pct=0.02)
    guard.start_session({"equity": 100_000.0})

    decision = guard.update({})  # broker returned {} on an API error
    assert not decision
    assert decision.code == REASON_EQUITY_UNKNOWN
    assert guard.breached is False

    # Once the account reads again, trading resumes: a transient blip must not
    # disable the session.
    assert guard.update({"equity": 99_500.0})


def test_start_session_clears_a_previous_breach():
    guard = DailyLossGuard(max_daily_loss_amount=100.0)
    guard.start_session({"equity": 10_000.0})
    assert not guard.update({"equity": 9_800.0})

    guard.start_session({"equity": 9_800.0})
    assert guard.breached is False
    assert guard.starting_equity == 9_800.0
    assert guard.update({"equity": 9_750.0})


@pytest.mark.parametrize("bad", [0, -0.5, "abc", float("nan"), float("inf")])
def test_invalid_daily_loss_config_is_ignored_not_fatal(bad):
    guard = RiskGuard.from_config({"risk": {"max_daily_loss_pct": bad}})
    assert guard.daily_loss.max_daily_loss_pct is None
    assert guard.daily_loss.enabled is False


def test_liquidate_on_daily_loss_defaults_off():
    guard = RiskGuard.from_config({"risk": {"max_daily_loss_pct": 0.01}})
    assert guard.daily_loss.liquidate_on_breach is False
    guard.start_session({"equity": 1_000.0}, now=WED_MIDDAY)
    guard.check_cycle({"equity": 900.0}, now=WED_MIDDAY)
    assert guard.daily_loss.breached is True
    assert guard.should_liquidate is False


# =========================================================================== #
# RiskGuard aggregation
# =========================================================================== #

def test_absent_risk_config_blocks_nothing():
    guard = RiskGuard.from_config({})
    assert guard.daily_loss.enabled is False
    assert guard.kill_switch.file_path is None
    assert guard.check_cycle({"equity": 100.0}, now=WED_MIDDAY)


def test_check_cycle_order_kill_switch_wins_over_market_hours():
    guard = RiskGuard.from_config({"risk": {}})
    guard.trip("manual")
    assert guard.check_cycle({"equity": 100.0}, now=SATURDAY).code == REASON_KILL_SWITCH


def test_check_new_entry_is_clock_free():
    """The entry gate must not depend on wall-clock time; only latched halts."""
    guard = RiskGuard.from_config({"risk": {"max_daily_loss_pct": 0.01}})
    guard.start_session({"equity": 10_000.0}, now=WED_MIDDAY)
    assert guard.check_new_entry({"equity": 9_950.0})
    assert not guard.check_new_entry({"equity": 9_800.0})
    assert guard.check_new_entry({"equity": 9_800.0}).code == REASON_DAILY_LOSS


def test_new_trading_day_rebaselines_loss_but_not_the_kill_switch():
    guard = RiskGuard.from_config({"risk": {"max_daily_loss_pct": 0.02}})
    guard.start_session({"equity": 100_000.0}, now=WED_MIDDAY)
    assert not guard.check_cycle({"equity": 97_000.0}, now=WED_MIDDAY)
    guard.trip("operator halt")

    # Next trading day: the loss baseline resets, the kill switch does not.
    decision = guard.check_cycle({"equity": 97_000.0}, now=THU_MIDDAY)
    assert decision.code == REASON_KILL_SWITCH
    guard.reset_kill_switch()
    assert guard.check_cycle({"equity": 97_000.0}, now=THU_MIDDAY)
    assert guard.daily_loss.breached is False
    assert guard.daily_loss.starting_equity == 97_000.0


def test_describe_reports_rail_state():
    guard = RiskGuard.from_config({"risk": {"max_daily_loss_pct": 0.02,
                                            "kill_switch_file": "run/KILL"}})
    snapshot = guard.describe()
    assert snapshot["daily_loss_enabled"] is True
    assert snapshot["kill_switch_file"] == "run/KILL"
    assert snapshot["market_hours_enforced"] is True
    assert snapshot["market_timezone"] == "America/New_York"


# =========================================================================== #
# LiveTrader wiring
# =========================================================================== #

def test_trader_builds_rails_from_config(monkeypatch):
    config = {"risk": {"max_daily_loss_pct": 0.03, "kill_switch_file": "run/K"}}
    trader, _ = make_trader(monkeypatch, config)
    assert trader.risk_guard.daily_loss.max_daily_loss_pct == 0.03
    assert trader.risk_guard.kill_switch.file_path == "run/K"


def test_cycle_blocked_outside_market_hours(monkeypatch):
    trader, broker = make_trader(monkeypatch, {"risk": {}}, now=SATURDAY)
    assert trader._risk_check_passed() is False
    assert broker.orders == []


def test_cycle_blocked_before_the_open_and_after_the_close(monkeypatch):
    for moment in (WED_PREMARKET, WED_AFTERHOURS):
        trader, _ = make_trader(monkeypatch, {"risk": {}}, now=moment)
        assert trader._risk_check_passed() is False


def test_cycle_allowed_during_market_hours(monkeypatch):
    trader, _ = make_trader(monkeypatch, {"risk": {}}, now=WED_MIDDAY)
    assert trader._risk_check_passed() is True


def test_halt_blocks_new_buys_and_latches(monkeypatch):
    trader, broker = make_trader(monkeypatch, {})
    trader.halt("risk officer said stop")

    trader._execute_signal(buy_signal())
    assert broker.orders == []
    assert trader._risk_check_passed() is False

    assert trader.reset_kill_switch() is True
    trader._execute_signal(buy_signal())
    assert len(broker.orders) == 1


def test_kill_switch_file_blocks_buys_on_the_next_check(monkeypatch, tmp_path):
    path = tmp_path / "KILL"
    trader, broker = make_trader(monkeypatch, {"risk": {"kill_switch_file": str(path)}})

    trader._execute_signal(buy_signal(symbol="AAPL"))
    assert len(broker.orders) == 1

    path.write_text("stop")
    trader._execute_signal(buy_signal(symbol="MSFT"))
    assert len(broker.orders) == 1  # nothing new

    path.unlink()
    trader._execute_signal(buy_signal(symbol="TSLA"))
    assert len(broker.orders) == 1  # still latched


def test_daily_loss_halts_new_entries_but_keeps_positions(monkeypatch):
    broker = FakeBroker(equity=100_000.0, positions={"MSFT": {"qty": 5}})
    trader, broker = make_trader(monkeypatch, {"risk": {"max_daily_loss_pct": 0.02}}, broker)
    trader.risk_guard.start_session(broker.get_account(), now=WED_MIDDAY)

    broker.set_equity(97_000.0)
    trader._execute_signal(buy_signal(symbol="AAPL"))

    assert broker.orders == []
    assert broker.closed == []          # default is halt-only, never auto-flatten
    assert broker.positions == {"MSFT": {"qty": 5}}


def test_daily_loss_auto_liquidates_only_when_opted_in(monkeypatch):
    config = {"risk": {"max_daily_loss_pct": 0.02, "liquidate_on_daily_loss": True}}
    broker = FakeBroker(equity=100_000.0, positions={"MSFT": {"qty": 5}, "AAPL": {"qty": 2}})
    trader, broker = make_trader(monkeypatch, config, broker)
    trader.risk_guard.start_session({"equity": 100_000.0}, now=WED_MIDDAY)

    broker.set_equity(97_000.0)
    assert trader._risk_check_passed() is False

    assert sorted(broker.closed) == ["AAPL", "MSFT"]
    assert len(trader.trades_executed) == 2

    # Liquidation runs at most once, even across repeated blocked cycles.
    trader._risk_check_passed()
    assert sorted(broker.closed) == ["AAPL", "MSFT"]


def test_unconfigured_trader_behaviour_is_unchanged(monkeypatch):
    """Absent risk config must not change today's behaviour - at any hour."""
    for moment in (WED_MIDDAY, SATURDAY, WED_AFTERHOURS):
        trader, broker = make_trader(monkeypatch, {}, now=moment)
        assert trader._risk_check_passed() is True
        trader._execute_signal(buy_signal(price=100.0))
        assert broker.orders == [{"symbol": "AAPL", "qty": 100, "side": "buy"}]


def test_sell_is_not_blocked_by_the_entry_gate(monkeypatch):
    """A latched halt stops new risk, not the ability to exit an open position."""
    broker = FakeBroker(positions={"AAPL": {"qty": 10}})
    trader, broker = make_trader(monkeypatch, {}, broker)
    trader.halt("stop new risk")

    trader._execute_signal(Signal("AAPL", SignalType.SELL, datetime.now(), confidence=1.0))
    assert broker.closed == ["AAPL"]


def test_status_reports_halt_state(monkeypatch):
    trader, _ = make_trader(monkeypatch, {})
    assert trader.get_status()["halted"] is False
    trader.halt("test")
    status = trader.get_status()
    assert status["halted"] is True
    assert status["risk"]["kill_switch_tripped"] is True
