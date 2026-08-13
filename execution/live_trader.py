import logging
import math
import time
from datetime import datetime
from typing import Dict, Optional

from .broker_adapter import BrokerAdapter
from .risk_guard import RiskGuard
from .signal_handler import SignalHandler, Signal, SignalType

logger = logging.getLogger(__name__)

# Fraction of buying power allocated to a single new position when
# position_management.max_position_pct is not configured.
DEFAULT_MAX_POSITION_PCT = 0.1
DEFAULT_MAX_OPEN_POSITIONS = 10


DEFAULT_POLL_INTERVAL_SECONDS = 60

# Minimum pause between blocked cycles. A halted or out-of-hours trader with
# poll_interval=0 would otherwise spin on get_account() and burn the rate limit.
BLOCKED_CYCLE_SLEEP_SECONDS = 1.0


class LiveTrader:
    def __init__(self, broker_config: Dict, strategy_name: str = "ma_crossover",
                 strategy_runner=None, poll_interval: Optional[float] = None,
                 risk_guard: Optional[RiskGuard] = None, now_provider=None):
        self.broker_config = broker_config
        self.strategy_name = strategy_name
        self.broker = BrokerAdapter(broker_config)
        self.signal_handler = SignalHandler(broker_config)
        self.strategy_runner = strategy_runner
        # Kill switch / max-daily-loss / market-hours rails. Injectable so tests
        # can supply a fixed clock instead of reading the wall clock.
        self.risk_guard = risk_guard if risk_guard is not None else RiskGuard.from_config(
            broker_config, now_provider=now_provider)
        self._liquidated_for_daily_loss = False
        self.running = False
        self.trades_executed = []
        position_management = broker_config.get("position_management") or {}
        self.max_open_positions = position_management.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS)
        self.max_position_pct = self._resolve_max_position_pct(position_management)
        if poll_interval is None:
            poll_interval = (broker_config.get("live") or {}).get(
                "poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
        self.poll_interval = max(float(poll_interval), 0.0)

    @staticmethod
    def _resolve_max_position_pct(position_management: Dict) -> float:
        """Fraction of buying power to deploy per position, from config.

        Falls back to DEFAULT_MAX_POSITION_PCT when unset or unusable so that
        an unconfigured deployment keeps its historical behaviour.
        """
        raw = position_management.get("max_position_pct", DEFAULT_MAX_POSITION_PCT)
        try:
            pct = float(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"Invalid position_management.max_position_pct={raw!r}; "
                f"falling back to {DEFAULT_MAX_POSITION_PCT}"
            )
            return DEFAULT_MAX_POSITION_PCT
        if not math.isfinite(pct) or pct <= 0:
            logger.warning(
                f"position_management.max_position_pct must be > 0 (got {raw!r}); "
                f"falling back to {DEFAULT_MAX_POSITION_PCT}"
            )
            return DEFAULT_MAX_POSITION_PCT
        return pct

    def start(self) -> None:
        self.running = True
        logger.info("="*60)
        logger.info(f"Live Trading Started: {self.strategy_name}")
        logger.info(f"Paper Trading: {self.broker.paper_trading}")
        if not self.broker.paper_trading:
            logger.warning("!!! LIVE (NON-PAPER) TRADING IS ACTIVE - REAL MONEY IS AT RISK !!!")
        logger.info(f"Poll interval: {self.poll_interval}s")
        logger.info(f"Risk rails: {self.risk_guard.describe()}")
        logger.info("="*60)
        account = self.broker.get_account()
        logger.info(f"Initial Account: {account}")
        self.risk_guard.start_session(account)

        try:
            while self.running:
                if not self._risk_check_passed():
                    time.sleep(max(self.poll_interval, BLOCKED_CYCLE_SLEEP_SECONDS))
                    continue
                if self.strategy_runner is not None:
                    try:
                        self.strategy_runner.poll()
                    except Exception as e:
                        # A bad poll must not kill the trade loop or strand open positions.
                        logger.error(f"Strategy poll failed: {e}")
                signal = self.signal_handler.get_next_signal()
                while signal:
                    self._execute_signal(signal)
                    signal = self.signal_handler.get_next_signal()
                positions = self.broker.get_positions()
                logger.debug(f"Open positions: {len(positions)}")
                time.sleep(self.poll_interval)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Trading error: {e}")
            self.stop()

    def stop(self) -> None:
        self.running = False
        logger.info(f"Trading stopped. Trades executed: {len(self.trades_executed)}")

    # --- risk rails --------------------------------------------------------

    def halt(self, reason: str = "halt() called") -> None:
        """Trip the kill switch. Trading stops at the next check; the process and
        any in-flight order are left alone. Stays tripped until reset_kill_switch()."""
        self.risk_guard.trip(reason)

    def reset_kill_switch(self) -> bool:
        return self.risk_guard.reset_kill_switch()

    def _risk_check_passed(self) -> bool:
        """One pre-cycle gate: kill switch, market hours, max daily loss."""
        decision = self.risk_guard.check_cycle(self.broker.get_account())
        if decision:
            return True
        logger.error("Trading paused this cycle [%s]: %s", decision.code, decision.reason)
        self._maybe_liquidate()
        return False

    def _maybe_liquidate(self) -> None:
        """Flatten open positions after a daily-loss breach - opt-in only.

        Force-closing is itself risky (slippage, closing into a gap), so it never
        happens unless risk.liquidate_on_daily_loss is explicitly true, and it
        runs at most once per breach.
        """
        if self._liquidated_for_daily_loss or not self.risk_guard.should_liquidate:
            return
        self._liquidated_for_daily_loss = True
        positions = self.broker.get_positions()
        logger.error("AUTO-LIQUIDATION: flattening %d open position(s) after daily-loss breach.",
                     len(positions))
        for symbol in list(positions):
            try:
                order = self.broker.close_position(symbol)
            except Exception as e:
                logger.error("Auto-liquidation failed for %s: %s", symbol, e)
                continue
            if order:
                self.trades_executed.append(
                    {"timestamp": datetime.now(), "signal": None, "order": order})
            else:
                logger.error("Auto-liquidation of %s was not accepted by the broker.", symbol)

    def submit_signal(self, signal: Signal) -> bool:
        return self.signal_handler.add_signal(signal)

    def _execute_signal(self, signal: Signal) -> None:
        if not self.signal_handler.validate_signal(signal):
            return
        symbol = signal.symbol
        positions = self.broker.get_positions()
        account = self.broker.get_account()

        if signal.signal_type == SignalType.BUY:
            # Last gate before any new position is sized. Clock-free by design
            # (kill switch + daily loss); the market-hours gate runs once per
            # cycle in _risk_check_passed() so the order path stays deterministic.
            entry = self.risk_guard.check_new_entry(account)
            if not entry:
                logger.error("Refusing BUY %s [%s]: %s", symbol, entry.code, entry.reason)
                self._maybe_liquidate()
                return
            if symbol in positions:
                logger.debug(f"Skipping BUY {symbol}: position already open")
                return
            if len(positions) >= self.max_open_positions:
                logger.info(
                    f"Skipping BUY {symbol}: max_open_positions reached "
                    f"({len(positions)}/{self.max_open_positions})"
                )
                return

            current_price = self._resolve_current_price(signal)
            if current_price is None:
                return

            buying_power = self._resolve_buying_power(account)
            if buying_power is None:
                return

            position_size = int((buying_power * self.max_position_pct) / current_price)
            if position_size <= 0:
                logger.info(
                    f"Skipping BUY {symbol}: computed position size is 0 "
                    f"(buying_power={buying_power}, price={current_price}, pct={self.max_position_pct})"
                )
                return

            notional = position_size * current_price
            if notional > buying_power:
                logger.warning(
                    f"Refusing BUY {symbol}: order notional {notional:.2f} exceeds "
                    f"available buying power {buying_power:.2f}"
                )
                return

            order = self.broker.submit_order(symbol, position_size, "buy")
            if order:
                self.trades_executed.append({"timestamp": datetime.now(), "signal": signal, "order": order})
                self.signal_handler.process_signal(signal)

        elif signal.signal_type == SignalType.SELL:
            if symbol not in positions:
                return
            order = self.broker.close_position(symbol)
            if order:
                self.trades_executed.append({"timestamp": datetime.now(), "signal": signal, "order": order})
                self.signal_handler.process_signal(signal)

    @staticmethod
    def _resolve_current_price(signal: Signal) -> Optional[float]:
        """Price used for sizing, or None if the signal cannot be sized safely.

        A missing price must never fall back to a placeholder: dividing buying
        power by a stand-in price of 1 would size an enormous position.
        """
        raw = signal.metadata.get("current_price")
        if raw is None:
            logger.warning(f"Refusing to trade {signal.symbol}: signal metadata has no current_price")
            return None
        try:
            price = float(raw)
        except (TypeError, ValueError):
            logger.warning(f"Refusing to trade {signal.symbol}: non-numeric current_price {raw!r}")
            return None
        if not math.isfinite(price) or price <= 0:
            logger.warning(f"Refusing to trade {signal.symbol}: invalid current_price {raw!r}")
            return None
        return price

    @staticmethod
    def _resolve_buying_power(account: Dict) -> Optional[float]:
        raw = account.get("buying_power", 0)
        try:
            buying_power = float(raw)
        except (TypeError, ValueError):
            logger.warning(f"Refusing to trade: non-numeric buying_power {raw!r}")
            return None
        if not math.isfinite(buying_power) or buying_power <= 0:
            logger.warning(f"Refusing to trade: no buying power available ({raw!r})")
            return None
        return buying_power

    def get_status(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        return {"running": self.running, "strategy": self.strategy_name, "account": account,
               "open_positions": len(positions), "trades_executed": len(self.trades_executed),
               "halted": self.risk_guard.halted, "risk": self.risk_guard.describe(),
               "timestamp": datetime.now().isoformat()}
