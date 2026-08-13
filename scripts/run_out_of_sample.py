#!/usr/bin/env python
import argparse
import json
import logging
from pathlib import Path
import yaml
from backtester.engine import BacktestEngine
from backtester.metrics import RiskMetrics, periods_per_year_for_timeframe
from backtester.strategies import get_strategy
from config.validation import validate_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Run a single backtest over the held-out out-of-sample window "
                     "(config.backtest.in_sample_end_date through config.backtest.end_date), "
                     "using parameters the optimizer already chose from the in-sample window."
    )
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--params", required=True,
                        help="Path to an optimization_summary_*.json produced by scripts/optimize_params.py, "
                             "or a JSON object of parameters directly")
    parser.add_argument("--output", default="output/", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fail fast on a malformed config: a bad config has repeatedly produced
    # plausible-looking but invalid results rather than an obvious crash.
    for warning in validate_config(config):
        logger.warning("config: %s", warning)

    in_sample_end_date = config["backtest"].get("in_sample_end_date")
    if not in_sample_end_date:
        raise ValueError("config.backtest.in_sample_end_date must be set to define the held-out window")

    params_path = Path(args.params)
    if params_path.exists():
        with open(params_path) as f:
            payload = json.load(f)
        params = payload["top_results"][0]["parameters"] if "top_results" in payload else payload
    else:
        params = json.loads(args.params)

    symbols = config["data"]["symbols"]
    logger.info(f"Running OUT-OF-SAMPLE backtest ({in_sample_end_date} -> {config['backtest']['end_date']}) "
               f"with params: {params}")

    strategy_func, prepare_data_func = get_strategy(config["strategy"]["name"])
    engine = BacktestEngine(config, strategy_func, prepare_data_func=prepare_data_func)
    final_equity, trades, equity_curve = engine.run(symbols, params, start_date=in_sample_end_date)
    periods_per_year = periods_per_year_for_timeframe(config["data"].get("timeframe", "1d"))
    metrics = RiskMetrics.calculate_metrics_summary(trades, equity_curve, config["backtest"]["initial_capital"],
                                                     periods_per_year=periods_per_year)

    logger.info("="*60)
    logger.info("OUT-OF-SAMPLE RESULTS")
    logger.info("="*60)
    for key, value in metrics.items():
        logger.info(f"{key:.<40} {value:>15.2f}" if isinstance(value, float) else f"{key:.<40} {value:>15}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    engine.export_trade_log(str(output_dir / "out_of_sample_trade_log.csv"))
    engine.export_equity_curve(str(output_dir / "out_of_sample_equity_curve.csv"))
    logger.info(f"Results exported to {output_dir}")

if __name__ == "__main__":
    main()
