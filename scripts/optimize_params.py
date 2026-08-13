#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import yaml
from optimizer.param_optimizer import ParameterOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def example_ma_crossover_strategy(daily_data: dict, open_positions: dict, params: dict):
    return {symbol: "HOLD" for symbol in daily_data}

def main():
    parser = argparse.ArgumentParser(description="Run parameter optimization")
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    symbols = config["data"]["symbols"]
    metric = config["optimization"].get("metric", "sharpe_ratio")
    direction = config["optimization"].get("score_direction", "maximize")
    top_n = config["optimization"].get("export_top_n", 10)

    optimizer = ParameterOptimizer(config, example_ma_crossover_strategy, metric=metric, direction=direction, workers=args.workers)
    best_params, all_results = optimizer.optimize(symbols)

    logger.info("="*60)
    logger.info(f"TOP {top_n} RESULTS ({metric})")
    logger.info("="*60)
    for i, result in enumerate(all_results[:top_n]):
        logger.info(f"Rank {i+1}: {metric}={result[metric]:.4f}")
        logger.info(f"  Parameters: {result['parameters']}")
        logger.info(f"  Total Return: {result['total_return_pct']:.2f}%")
        logger.info(f"  Num Trades: {result['num_trades']}")
        logger.info("-" * 60)

    optimizer.export_results(args.output, top_n=top_n)
    logger.info(f"Results exported to {args.output}")

if __name__ == "__main__":
    main()
