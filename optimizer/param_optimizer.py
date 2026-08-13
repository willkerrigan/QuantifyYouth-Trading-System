import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtester.engine import BacktestEngine
from backtester.metrics import RiskMetrics, periods_per_year_for_timeframe
from .grid_search import GridSearchGenerator

logger = logging.getLogger(__name__)


class ParameterOptimizer:
    def __init__(self, config: Dict, strategy_func, metric: str = "sharpe_ratio", direction: str = "maximize",
                 workers: int = 4, prepare_data_func: Optional[callable] = None,
                 in_sample_start_date: Optional[str] = None, in_sample_end_date: Optional[str] = None):
        """in_sample_start_date/in_sample_end_date override config.backtest.in_sample_end_date
        for callers that optimize over a specific sub-window (e.g. one walk-forward
        training fold). When omitted, behaviour is unchanged: the in-sample window runs
        from the start of the loaded data through config.backtest.in_sample_end_date,
        which remains required."""
        self.config = config
        self.strategy_func = strategy_func
        self.metric = metric
        self.direction = direction
        self.workers = workers
        self.prepare_data_func = prepare_data_func
        self.in_sample_start_date = in_sample_start_date
        self.in_sample_end_date = in_sample_end_date
        self.results = []

    def optimize(self, symbols: List[str]) -> Tuple[Dict, List[Dict]]:
        param_ranges = self.config["strategy"].get("parameters", {})
        total_combos = GridSearchGenerator.count_combinations(param_ranges)

        in_sample_start_date = self.in_sample_start_date
        in_sample_end_date = self.in_sample_end_date or self.config["backtest"].get("in_sample_end_date")
        if not in_sample_end_date:
            raise ValueError(
                "config.backtest.in_sample_end_date is required: the optimizer must not be "
                "allowed to see data past the in-sample window, or its 'best' parameters will "
                "be tuned on the out-of-sample period and look better than they really are."
            )
        logger.info(f"Starting parameter optimization: {total_combos} combinations "
                   f"(in-sample data {in_sample_start_date or 'from start'} through {in_sample_end_date})")

        combinations = list(GridSearchGenerator.generate_combinations(param_ranges))
        self.results = []

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self._run_backtest_wrapper, symbols, params, in_sample_end_date,
                                       in_sample_start_date): (i, params)
                      for i, params in enumerate(combinations)}
            for future in as_completed(futures):
                idx, params = futures[future]
                try:
                    result = future.result()
                    if result:
                        self.results.append(result)
                except Exception as e:
                    logger.error(f"Backtest failed for params {params}: {e}")

        self.results.sort(key=lambda x: x[self.metric], reverse=(self.direction == "maximize"))
        if not self.results:
            return {}, []
        best_result = self.results[0]
        logger.info(f"Optimization complete. Best {self.metric}: {best_result[self.metric]:.4f}")
        return best_result["parameters"], self.results

    def _run_backtest_wrapper(self, symbols: List[str], params: Dict, in_sample_end_date: str,
                              in_sample_start_date: Optional[str] = None) -> Optional[Dict]:
        try:
            engine = BacktestEngine(self.config, self.strategy_func, prepare_data_func=self.prepare_data_func)
            final_equity, trades, equity_curve = engine.run(symbols, params, start_date=in_sample_start_date,
                                                            end_date=in_sample_end_date)
            periods_per_year = periods_per_year_for_timeframe(self.config["data"].get("timeframe", "1d"))
            metrics = RiskMetrics.calculate_metrics_summary(trades, equity_curve, self.config["backtest"]["initial_capital"],
                                                             periods_per_year=periods_per_year)
            return {"parameters": params, "final_equity": final_equity, "num_trades": len(trades),
                   "trades": trades, "equity_curve": equity_curve, **metrics}
        except Exception as e:
            logger.error(f"Backtest failed for params {params}: {e}")
            return None

    def export_results(self, output_dir: str, top_n: int = 10) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        summary = {"optimization_date": datetime.now().isoformat(), "metric": self.metric,
                  "direction": self.direction, "total_combinations": len(self.results),
                  "top_results": [{"rank": i + 1, "parameters": result["parameters"],
                                  **{k: v for k, v in result.items() if k not in ["parameters", "trades", "equity_curve"]}}
                                 for i, result in enumerate(self.results[:top_n])]}
        with open(output_path / f"optimization_summary_{timestamp}.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)

        for i, result in enumerate(self.results[:top_n]):
            if result["trades"]:
                pd.DataFrame([trade.to_dict() for trade in result["trades"]]).to_csv(
                    output_path / f"trade_log_rank{i+1}_{timestamp}.csv", index=False)
            if not result["equity_curve"].empty:
                result["equity_curve"].to_csv(output_path / f"equity_curve_rank{i+1}_{timestamp}.csv", index=False)
