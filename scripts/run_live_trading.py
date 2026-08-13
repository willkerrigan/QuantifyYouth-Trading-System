#!/usr/bin/env python
import argparse
import logging
import sys
import yaml
from execution.live_trader import LiveTrader
from execution.strategy_runner import StrategyRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _check_trading_mode(broker_config):
    """Paper trading is the default and stays the default. Going live requires
    BOTH trading.paper_trading: false AND an explicit trading.confirm_live_trading: true."""
    trading = broker_config.get("trading", {}) or {}
    paper_trading = trading.get("paper_trading", True)
    if paper_trading:
        logger.info("Paper trading mode (no real money at risk).")
        return True
    if not trading.get("confirm_live_trading", False):
        logger.error(
            "Config sets trading.paper_trading: false but trading.confirm_live_trading is not true. "
            "Refusing to start. Set paper_trading back to true, or explicitly opt in to LIVE trading."
        )
        return False
    logger.warning("!" * 60)
    logger.warning("!!! LIVE TRADING ENABLED - ORDERS WILL USE REAL MONEY !!!")
    logger.warning("!!! paper_trading=false and confirm_live_trading=true in config       !!!")
    logger.warning("!" * 60)
    return True


def main():
    parser = argparse.ArgumentParser(description="Run live trading")
    parser.add_argument("--config", required=True, help="Path to broker config file")
    parser.add_argument("--strategy-config", default=None,
                        help="Path to the strategy/data config (defaults to --config). "
                             "Supplies strategy.name, strategy.parameters and data.symbols.")
    parser.add_argument("--strategy", default=None,
                        help="Override strategy.name from the strategy config")
    parser.add_argument("--poll-interval", type=float, default=None,
                        help="Seconds between strategy polls (overrides live.poll_interval_seconds)")
    args = parser.parse_args()

    broker_config = _load_yaml(args.config)
    strategy_config = _load_yaml(args.strategy_config) if args.strategy_config else broker_config

    if not _check_trading_mode(broker_config):
        sys.exit(1)

    strategy_name = args.strategy or (strategy_config.get("strategy", {}) or {}).get("name")
    if not strategy_name:
        logger.error("No strategy configured. Set strategy.name in the strategy config or pass --strategy.")
        sys.exit(1)

    symbols = (strategy_config.get("data", {}) or {}).get("symbols", [])
    if not symbols:
        logger.error("No symbols configured. Set data.symbols in the strategy config.")
        sys.exit(1)

    trader = LiveTrader(broker_config, strategy_name=strategy_name, poll_interval=args.poll_interval)
    runner = StrategyRunner(
        strategy_config,
        signal_handler=trader.signal_handler,
        strategy_name=strategy_name,
        position_provider=trader.broker.get_positions,
    )
    trader.strategy_runner = runner
    if args.poll_interval is None:
        trader.poll_interval = runner.poll_interval_seconds

    status = trader.get_status()
    logger.info("="*60)
    logger.info("LIVE TRADING STATUS")
    logger.info("="*60)
    logger.info(f"Strategy: {status['strategy']}")
    logger.info(f"Parameters: {runner.params}")
    logger.info(f"Symbols: {runner.symbols}")
    logger.info(f"Poll interval: {trader.poll_interval}s")
    logger.info(f"Running: {status['running']}")
    logger.info(f"Timestamp: {status['timestamp']}")
    logger.info("="*60)

    try:
        trader.start()
    except KeyboardInterrupt:
        trader.stop()
    except Exception as e:
        logger.error(f"Error: {e}")
        trader.stop()


if __name__ == "__main__":
    main()
