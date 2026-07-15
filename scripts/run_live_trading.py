#!/usr/bin/env python
import argparse
import logging
from datetime import datetime
import yaml
from execution.live_trader import LiveTrader
from execution.signal_handler import Signal, SignalType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--strategy", default="ma_crossover", help="Strategy name")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    trader = LiveTrader(config, strategy_name=args.strategy)
    status = trader.get_status()
    logger.info("="*60)
    logger.info("LIVE TRADING STATUS")
    logger.info("="*60)
    logger.info(f"Strategy: {status['strategy']}")
    logger.info(f"Running: {status['running']}")
    logger.info(f"Timestamp: {status['timestamp']}")
    logger.info("="*60)

    test_signal = Signal(symbol="SPY", signal_type=SignalType.HOLD, timestamp=datetime.now(),
                        confidence=0.8, metadata={"current_price": 450.00})
    trader.submit_signal(test_signal)

    try:
        trader.start()
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logger.error(f"Error: {e}")
        trader.stop()

if __name__ == "__main__":
    main()
