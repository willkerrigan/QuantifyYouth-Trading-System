import logging
import time
from datetime import datetime
from typing import Dict, Optional

from .broker_adapter import BrokerAdapter
from .signal_handler import SignalHandler, Signal, SignalType

logger = logging.getLogger(__name__)


class LiveTrader:
    def __init__(self, broker_config: Dict, strategy_name: str = "ma_crossover"):
        self.broker_config = broker_config
        self.strategy_name = strategy_name
        self.broker = BrokerAdapter(broker_config)
        self.signal_handler = SignalHandler(broker_config)
        self.running = False
        self.trades_executed = []
        self.max_open_positions = broker_config.get("position_management", {}).get("max_open_positions", 10)

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
            if symbol in positions or len(positions) >= self.max_open_positions:
                return
            position_size = int((account.get("buying_power", 0) * 0.1) / signal.metadata.get("current_price", 1))
            if position_size > 0:
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

    def get_status(self) -> Dict:
        account = self.broker.get_account()
        positions = self.broker.get_positions()
        return {"running": self.running, "strategy": self.strategy_name, "account": account,
               "open_positions": len(positions), "trades_executed": len(self.trades_executed),
               "timestamp": datetime.now().isoformat()}
