import logging
from datetime import datetime
from typing import Dict, Optional, List
from enum import Enum

logger = logging.getLogger(__name__)


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class Signal:
    def __init__(self, symbol: str, signal_type: SignalType, timestamp: datetime, confidence: float = 1.0, metadata: Optional[Dict] = None):
        self.symbol = symbol
        self.signal_type = signal_type
        self.timestamp = timestamp
        self.confidence = confidence
        self.metadata = metadata or {}
        self.processed = False

    def is_expired(self, ttl_seconds: int = 60) -> bool:
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > ttl_seconds


class SignalHandler:
    def __init__(self, config: Dict):
        self.config = config
        self.signal_queue = []
        self.processed_signals = []
        self.ttl_seconds = config.get("signal_processing", {}).get("signal_ttl_seconds", 60)
        self.confidence_threshold = config.get("signal_processing", {}).get("confidence_threshold", 0.5)

    def add_signal(self, signal: Signal) -> bool:
        if signal.is_expired(self.ttl_seconds):
            logger.warning(f"Signal rejected (expired): {signal.symbol}")
            return False
        if signal.confidence < self.confidence_threshold:
            logger.warning(f"Signal rejected (low confidence): {signal.symbol}")
            return False
        self.signal_queue.append(signal)
        return True

    def get_next_signal(self) -> Optional[Signal]:
        while self.signal_queue:
            signal = self.signal_queue.pop(0)
            if not signal.is_expired(self.ttl_seconds):
                return signal
        return None

    def validate_signal(self, signal: Signal) -> bool:
        if signal.is_expired(self.ttl_seconds) or signal.confidence < self.confidence_threshold:
            return False
        return bool(signal.symbol and isinstance(signal.symbol, str))

    def process_signal(self, signal: Signal) -> bool:
        if not signal.processed:
            signal.processed = True
            self.processed_signals.append(signal)
            logger.info(f"Signal processed: {signal.symbol}")
            return True
        return False
