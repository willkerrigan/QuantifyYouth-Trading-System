import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
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
        # max_drawdown is reported as a negative fraction (-0.25 == a 25% drawdown),
        # so "minimize" selects -0.40 over -0.02 — the worst run, not the best.
        # The intuitive-sounding setting is the wrong one, so say so loudly.
        if metric == "max_drawdown" and direction == "minimize":
            logger.warning(
                "optimization.metric=max_drawdown with score_direction=minimize selects the "
                "LARGEST drawdown, because drawdowns are negative numbers. Use "
                "score_direction: maximize to prefer shallower drawdowns.")

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

        self._sort_results()
        if not self.results:
            return {}, []
        best_result = self.results[0]
        logger.info(f"Optimization complete. Best {self.metric}: {best_result[self.metric]:.4f}")
        return best_result["parameters"], self.results

    def _sort_results(self) -> None:
        """Rank results best-first, keeping degenerate scores from winning.

        A NaN score used as a sort key makes ordering depend on input order
        (every NaN comparison is False), so a single degenerate combination can
        silently scramble the whole ranking. Non-finite scores are pushed to the
        end instead. An infinite profit_factor is the common case: a combination
        with zero losing trades scores inf and would otherwise always rank first
        no matter how few trades produced it.
        """
        maximize = self.direction == "maximize"

        def sort_key(result):
            value = result.get(self.metric)
            try:
                value = float(value)
            except (TypeError, ValueError):
                return (1, 0.0)
            if not np.isfinite(value):
                return (1, 0.0)
            return (0, -value if maximize else value)

        self.results.sort(key=sort_key)
        dropped = sum(1 for r in self.results if sort_key(r)[0] == 1)
        if dropped:
            logger.warning(f"{dropped} combination(s) scored a non-finite {self.metric} "
                           f"and were ranked last rather than allowed to win.")

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
    def sensitivity_analysis(self, symbols: List[str], best_params: Dict, steps: int = 2) -> Dict:
        """For each parameter in best_params, nudge it up and down by `steps` 
        increments and record the metric. A real edge should be stable across 
        nearby values. A sharp spike means the optimizer got lucky on one number."""
        in_sample_end_date = self.in_sample_end_date or self.config["backtest"].get("in_sample_end_date")
        param_ranges = self.config["strategy"].get("parameters", {})
        sensitivity = {}

        for param_name, best_value in best_params.items():
            grid = param_ranges.get(param_name)
            if not isinstance(grid, list) or len(grid) < 2:
                logger.info(f"Skipping sensitivity for '{param_name}': not enough grid values to nudge")
                continue

            sorted_grid = sorted(grid)
            if best_value not in sorted_grid:
                continue

            idx = sorted_grid.index(best_value)
            candidates = []
            for offset in range(-steps, steps + 1):
                candidate_idx = idx + offset
                if 0 <= candidate_idx < len(sorted_grid):
                    candidates.append(sorted_grid[candidate_idx])

            scores = {}
            for candidate in candidates:
                test_params = dict(best_params)
                test_params[param_name] = candidate
                result = self._run_backtest_wrapper(symbols, test_params, in_sample_end_date)
                if result:
                    scores[candidate] = result.get(self.metric, 0.0)
                else:
                    scores[candidate] = None

            sensitivity[param_name] = {
                "best_value": best_value,
                "best_score": scores.get(best_value),
                "scores_by_value": scores,
                "stable": self._is_stable(scores, best_value),
            }
            logger.info(f"Sensitivity '{param_name}': {scores}")
            if not sensitivity[param_name]["stable"]:
                logger.warning(
                    f"Parameter '{param_name}' looks unstable -- performance collapses "
                    f"when nudged away from {best_value}. This may be curve-fitting."
                )

        return sensitivity

    @staticmethod
    def _is_stable(scores: Dict, best_value) -> bool:
        """Returns True if nearby values perform within 50% of the best value's score.
        If moving one step away causes a collapse, it's not a real edge."""
        best_score = scores.get(best_value)
        if best_score is None or best_score == 0:
            return False
        for value, score in scores.items():
            if score is None:
                continue
            if abs(score - best_score) / abs(best_score) > 0.5:
                return False
        return True