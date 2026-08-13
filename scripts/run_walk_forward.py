#!/usr/bin/env python
import argparse
import logging
from pathlib import Path
import yaml
from backtester.strategies import get_strategy
from config.validation import validate_config
from optimizer.walk_forward import WalkForwardValidator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(
        description="Run walk-forward validation: repeatedly optimize on a training window, then "
                    "score the winning parameters on the immediately following unseen window, and "
                    "roll forward. Large train->test degradation, or winning parameters that change "
                    "wildly fold to fold, means the strategy is curve-fit rather than edge-driven."
    )
    parser.add_argument("--config", required=True, help="Path to config file")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--folds", type=int, default=None,
                        help="Number of walk-forward folds (default: config.walk_forward.n_folds, else 4)")
    parser.add_argument("--train-days", type=int, default=None,
                        help="Fixed training window length in days (requires --test-days)")
    parser.add_argument("--test-days", type=int, default=None,
                        help="Fixed test window length in days (requires --train-days)")
    parser.add_argument("--rolling", action="store_true",
                        help="Use a fixed-length rolling training window instead of an anchored/expanding one")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Fail fast on a malformed config: a bad config has repeatedly produced
    # plausible-looking but invalid results rather than an obvious crash.
    for warning in validate_config(config):
        logger.warning("config: %s", warning)

    symbols = config["data"]["symbols"]
    strategy_func, prepare_data_func = get_strategy(config["strategy"]["name"])

    validator = WalkForwardValidator(
        config, strategy_func, prepare_data_func=prepare_data_func,
        n_folds=args.folds, train_period_days=args.train_days, test_period_days=args.test_days,
        anchored=False if args.rolling else None, workers=args.workers,
    )

    logger.info("="*60)
    logger.info(f"WALK-FORWARD PLAN ({len(validator.folds)} folds, "
                f"{'rolling' if not validator.anchored else 'anchored'} training window)")
    logger.info("="*60)
    for fold in validator.folds:
        logger.info(f"Fold {fold.index}: train {fold.train_start.date()} -> {fold.train_end.date()} "
                   f"({fold.train_days}d) | test {fold.test_start.date()} -> {fold.test_end.date()} "
                   f"({fold.test_days}d)")

    report = validator.run(symbols)
    metric = report["metric"]

    logger.info("="*60)
    logger.info(f"WALK-FORWARD RESULTS ({metric})")
    logger.info("="*60)
    for fold in report["folds"]:
        degradation = fold.get("degradation") or {}
        logger.info(f"Fold {fold['fold']} ({fold['test_start']} -> {fold['test_end']})")
        logger.info(f"  Parameters: {fold['parameters']}")
        if degradation:
            pct = degradation.get("degradation_pct")
            # Positive == the test window did worse, whichever way the metric is optimized.
            pct_text = f"  (degradation {pct:+.1f}%)" if isinstance(pct, float) else ""
            logger.info(f"  Train {metric}: {degradation['train']:.4f}  ->  "
                       f"Test {metric}: {degradation['test']:.4f}{pct_text}")
        else:
            logger.info(f"  No comparable metrics (status: {fold.get('status')})")
        logger.info("-" * 60)

    summary = report["summary"]
    logger.info("SUMMARY")
    for key, value in summary.items():
        if key in ("parameter_stability", "interpretation"):
            continue
        logger.info(f"{key:.<40} {value:>15.4f}" if isinstance(value, float) else f"{key:.<40} {value:>15}")
    for name, info in (summary.get("parameter_stability") or {}).items():
        logger.info(f"  param '{name}': {info['values_by_fold']} "
                   f"({info['unique_values']} distinct, {info['consistency_pct']:.0f}% consistent)")
    if summary.get("interpretation"):
        logger.info(f"  {summary['interpretation']}")

    output_dir = Path(args.output)
    exported = validator.export_results(str(output_dir))
    logger.info(f"Results exported to {exported}")

if __name__ == "__main__":
    main()
