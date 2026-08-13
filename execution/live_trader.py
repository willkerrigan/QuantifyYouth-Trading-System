import logging
import math
import time
from datetime import datetime
from typing import Dict, Optional

from .broker_adapter import BrokerAdapter
from .signal_handler import SignalHandler, Signal, SignalType

logger = logging.getLogger(__name__)

# Fraction of buying power allocated to a single new position when
# position_management.max_position_pct is not configured.
DEFAULT_MAX_POSITION_PCT = 0.1
DEFAULT_MAX_OPEN_POSITIONS = 10


class LiveTrader:
    def __init__(self, broker_config: Dict, strategy_name: str = "ma_crossover"):
        self.broker_config = broker_config
        self.strategy_name = strategy_name
        self.broker = BrokerAdapter(broker_config)
        self.signal_handler = SignalHandler(broker_config)
        self.running = False
        self.trades_executed = []
        position_management = broker_config.get("position_management") or {}
        self.max_open_positions = position_management.get("max_open_positions", DEFAULT_MAX_OPEN_POSITIONS)
        self.max_position_pct = self._resolve_max_position_pct(position_management)

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
        logger.info("="*60)
        account = self.broker.get_account()
        logger.info(f"Initial Account: {account}")

        try:
            while self.running:
                signal = self.signal_handler.get_next_signal()
                if signal:
                    self._execute_signal(signal)
                positions = self.broker.get_positions()
                logger.debug(f"Open positions: {len(positions)}")
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()
        except Exception as e:
            logger.error(f"Trading error: {e}")
            self.stop()

    def stop(self) -> None:
        self.running = False
        logger.info(f"Trading stopped. Trades executed: {len(self.trades_executed)}")

    def submit_signal(self, signal: Signal) -> bool:
        return self.signal_handler.add_signal(signal)

    def _execute_signal(self, signal: Signal) -> None:
        if not self.signal_handler.validate_signal(signal):
            return
        symbol = signal.symbol
        positions = self.broker.get_positions()
        account = self.broker.get_account()

        if signal.signal_type == SignalType.BUY:
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
               "timestamp": datetime.now().isoformat()}
