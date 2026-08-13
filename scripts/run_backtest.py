#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import yaml
from backtester.engine import BacktestEngine
from backtester.metrics import RiskMetrics
from backtester.strategies import rsi2_strategy, prepare_rsi2_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run backtest")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output", default="output/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    symbols = config["data"]["symbols"]
    params = {k: v[0] if isinstance(v, list) else v for k, v in config["strategy"]["parameters"].items()}
    engine = BacktestEngine(config, rsi2_strategy, prepare_data_func=prepare_rsi2_data)
    final_equity, trades, equity_curve = engine.run(symbols, params)
    metrics = RiskMetrics.calculate_metrics_summary(trades, equity_curve, config["backtest"]["initial_capital"])

    logger.info("="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    for key, value in metrics.items():
        logger.info(f"{key:.<40} {value:>15.2f}" if isinstance(value, float) else f"{key:.<40} {value:>15}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.export_trade_log(str(output_dir / "trade_log.csv"))
    engine.export_equity_curve(str(output_dir / "equity_curve.csv"))
    logger.info(f"Results exported to {output_dir}")

if __name__ == "__main__":
    main()
