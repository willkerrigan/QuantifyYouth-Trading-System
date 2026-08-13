"""Tests for the backtest-strategy -> live-signal bridge.

Everything here is offline: a stub data loader stands in for market data and a fake
broker stands in for Alpaca. No network calls and no alpaca imports.
"""

import pandas as pd
import pytest

from execution.live_trader import LiveTrader
from execution.signal_handler import SignalHandler, SignalType
from execution.strategy_runner import StrategyRunner


def _frame(closes, start="2024-01-01"):
    index = pd.date_range(start, periods=len(closes), freq="D")
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c + 1 for c in closes],
            "low": [c - 1 for c in closes],
            "close": closes,
            "volume": [1000] * len(closes),
        },
        index=index,
    )


class _StubDataLoader:
    """Serves canned frames per symbol; mimics DataLoader.load's signature."""

    def __init__(self, frames):
        self.frames = frames
        self.load_calls = []

    def load(self, symbol, force_refresh=False):
        self.load_calls.append((symbol, force_refresh))
        if symbol not in self.frames:
            raise ValueError(f"no data for {symbol}")
        return self.frames[symbol]


class _FakeBroker:
    """Stands in for BrokerAdapter. Records orders instead of sending them anywhere."""

    def __init__(self, config=None):
        self.paper_trading = True
        self.positions = {}
        self.orders = []

    def get_account(self):
        return {"buying_power": 100000.0, "cash": 100000.0,
                "portfolio_value": 100000.0, "equity": 100000.0}

    def get_positions(self):
        return dict(self.positions)

    def submit_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        self.orders.append({"symbol": symbol, "qty": qty, "side": side})
        return {"order_id": f"fake-{len(self.orders)}", "symbol": symbol, "qty": qty, "side": side}

    def close_position(self, symbol):
        self.orders.append({"symbol": symbol, "qty": 0, "side": "sell"})
        return {"order_id": "fake-close", "symbol": symbol}


def _always(action):
    def strategy(daily_data, open_positions, params):
        return {symbol: action for symbol in daily_data}
    return strategy


def _make_runner(frames, strategy_func, symbols=("SPY",), config_extra=None, **kwargs):
    config = {"data": {"symbols": list(symbols)}, "strategy": {"name": "stub"}}
    if config_extra:
        config.update(config_extra)
    handler = SignalHandler({})
    runner = StrategyRunner(
        config,
        signal_handler=handler,
        data_loader=_StubDataLoader(frames),
        strategy_name="stub",
        strategy_func=strategy_func,
        **kwargs,
    )
    return runner, handler


def test_buy_signal_is_submitted_with_symbol_and_current_price():
    frames = {"SPY": _frame([100.0, 101.0, 123.45])}
    runner, handler = _make_runner(frames, _always("BUY"))

    submitted = runner.poll()

    assert len(submitted) == 1
    signal = submitted[0]
    assert signal.symbol == "SPY"
    assert signal.signal_type == SignalType.BUY
    # live_trader._execute_signal sizes positions off this key.
    assert signal.metadata["current_price"] == pytest.approx(123.45)
    assert len(handler.signal_queue) == 1
    assert handler.signal_queue[0] is signal
    # Live polls must bypass the loader cache or they'd re-read a stale bar forever.
    assert runner.data_loader.load_calls == [("SPY", True)]


def test_hold_produces_no_signal():
    frames = {"SPY": _frame([100.0, 101.0, 102.0])}
    runner, handler = _make_runner(frames, _always("HOLD"))

    assert runner.poll() == []
    assert handler.signal_queue == []


def test_missing_and_empty_data_do_not_crash_the_runner():
    frames = {
        "SPY": _frame([100.0, 101.0, 102.0]),
        "IWM": _frame([]),          # empty frame
        # QQQ absent entirely -> stub loader raises, mimicking a fetch failure
    }
    runner, handler = _make_runner(frames, _always("BUY"), symbols=("SPY", "QQQ", "IWM"))

    submitted = runner.poll()

    assert [s.symbol for s in submitted] == ["SPY"]
    assert len(handler.signal_queue) == 1


def test_poll_with_no_usable_data_at_all_returns_no_signals():
    runner, handler = _make_runner({}, _always("BUY"), symbols=("SPY", "QQQ"))

    assert runner.poll() == []
    assert handler.signal_queue == []


def test_same_bar_does_not_emit_duplicate_signals_across_polls():
    frames = {"SPY": _frame([100.0, 101.0, 102.0])}
    runner, handler = _make_runner(frames, _always("BUY"))

    assert len(runner.poll()) == 1
    assert runner.poll() == []          # same bar, already acted on
    assert len(handler.signal_queue) == 1

    frames["SPY"] = _frame([100.0, 101.0, 102.0, 103.0])
    assert len(runner.poll()) == 1      # new bar -> new signal
    assert handler.signal_queue[-1].metadata["current_price"] == pytest.approx(103.0)


def test_open_positions_from_broker_are_passed_to_the_strategy():
    frames = {"SPY": _frame([100.0, 101.0, 102.0])}
    broker = _FakeBroker()
    broker.positions = {"SPY": {"qty": 10}}
    seen = {}

    def strategy(daily_data, open_positions, params):
        seen.update(open_positions)
        return {symbol: ("SELL" if symbol in open_positions else "HOLD") for symbol in daily_data}

    runner, handler = _make_runner(frames, strategy, position_provider=broker.get_positions)
    submitted = runner.poll()

    assert seen == {"SPY": {"qty": 10}}
    assert [s.signal_type for s in submitted] == [SignalType.SELL]


def test_prepare_data_func_runs_before_the_strategy():
    frames = {"SPY": _frame([100.0, 101.0, 102.0])}

    def prepare(df):
        df = df.copy()
        df["flag"] = 42
        return df

    def strategy(daily_data, open_positions, params):
        return {sym: ("BUY" if row.get("flag") == 42 else "HOLD") for sym, row in daily_data.items()}

    runner, _ = _make_runner(frames, strategy, prepare_data_func=prepare)
    assert [s.symbol for s in runner.poll()] == ["SPY"]


def test_optimizer_style_param_grids_collapse_to_single_values():
    frames = {"SPY": _frame([100.0, 101.0, 102.0])}
    captured = {}

    def strategy(daily_data, open_positions, params):
        captured.update(params)
        return {sym: "HOLD" for sym in daily_data}

    runner, _ = _make_runner(
        frames, strategy,
        config_extra={"strategy": {"name": "stub",
                                   "parameters": {"rsi_buy_threshold": [5, 10], "rsi_sell_threshold": 70}}},
    )
    runner.poll()

    assert captured == {"rsi_buy_threshold": 5, "rsi_sell_threshold": 70}


def test_real_backtest_strategy_drives_live_signals():
    """The same rsi2 function used in backtests, resolved by name, must produce live signals."""
    falling = [100.0 - i for i in range(10)]   # monotonic decline -> RSI(2) == 0 -> BUY
    frames = {"SPY": _frame(falling)}
    handler = SignalHandler({})
    runner = StrategyRunner(
        {"data": {"symbols": ["SPY"]},
         "strategy": {"name": "rsi2", "parameters": {"rsi_buy_threshold": [5], "rsi_sell_threshold": [70]}}},
        signal_handler=handler,
        data_loader=_StubDataLoader(frames),
    )

    submitted = runner.poll()

    assert [(s.symbol, s.signal_type) for s in submitted] == [("SPY", SignalType.BUY)]
    assert submitted[0].metadata["current_price"] == pytest.approx(falling[-1])
    assert submitted[0].metadata["strategy"] == "rsi2"


def test_unknown_strategy_name_is_rejected_at_construction():
    with pytest.raises(ValueError):
        StrategyRunner({"data": {"symbols": ["SPY"]}, "strategy": {"name": "not_a_strategy"}},
                       signal_handler=SignalHandler({}), data_loader=_StubDataLoader({}))


def test_live_trader_loop_executes_a_strategy_signal_through_the_broker(monkeypatch):
    """End-to-end through LiveTrader.start with a fake broker: one poll, one order."""
    monkeypatch.setattr("execution.live_trader.BrokerAdapter", _FakeBroker)

    frames = {"SPY": _frame([100.0, 101.0, 102.0])}
    trader = LiveTrader({}, strategy_name="stub", poll_interval=0)
    runner = StrategyRunner(
        {"data": {"symbols": ["SPY"]}, "strategy": {"name": "stub"}},
        signal_handler=trader.signal_handler,
        data_loader=_StubDataLoader(frames),
        strategy_name="stub",
        strategy_func=_always("BUY"),
        position_provider=trader.broker.get_positions,
    )

    class _OneShotRunner:
        def poll(self):
            trader.running = False     # stop after this single cycle
            return runner.poll()

    trader.strategy_runner = _OneShotRunner()
    trader.start()

    assert [(o["symbol"], o["side"]) for o in trader.broker.orders] == [("SPY", "buy")]
    assert len(trader.trades_executed) == 1
    assert trader.trades_executed[0]["signal"].metadata["current_price"] == pytest.approx(102.0)


def test_live_trader_defaults_to_a_sane_poll_interval_and_reads_config(monkeypatch):
    monkeypatch.setattr("execution.live_trader.BrokerAdapter", _FakeBroker)

    assert LiveTrader({}).poll_interval == 60
    assert LiveTrader({"live": {"poll_interval_seconds": 5}}).poll_interval == 5
