"""Walk-forward validation.

A single in-sample optimization followed by one out-of-sample test (see
scripts/run_out_of_sample.py) answers "did these parameters survive one unseen
period?".  Walk-forward asks the harder question: "do freshly optimized
parameters survive *repeatedly*, across many unseen periods?".

The date range is cut into sequential folds.  For each fold we grid-search on a
TRAIN window and then score the winning parameters on the TEST window that
immediately follows it.  Rolling forward and comparing train metrics against
test metrics exposes two distinct symptoms of curve fitting:

  1. Large train -> test degradation: the parameters only worked on the data
     they were fitted to.
  2. Unstable winning parameters: if the "best" parameter set jumps around from
     fold to fold, there is no stable edge to find, only noise being fitted.
     `summary["parameter_stability"]` surfaces this.

Leakage rules enforced here (this is the whole point of the exercise):

  * A fold's TRAIN window ends at least one full day before its own TEST window
    starts, and before the TEST window of every later fold.  Windows are cut on
    day boundaries and BacktestEngine treats end_date as inclusive-of-that-whole-day,
    so a one-day gap is required for the windows to be genuinely disjoint --
    adjacent-but-equal boundary dates would double-count a day.
  * The optimizer for fold i is handed an explicit
    (in_sample_start_date, in_sample_end_date) pair covering only that fold's
    train window, so its BacktestEngine.run() calls are date-clamped on both
    ends.  It cannot see its own test window or any later fold's data.
  * `validate_no_leakage()` re-checks every train/test pair before any backtest
    runs, and raises rather than silently producing optimistic numbers.

Known caveat, inherited from BacktestEngine: `prepare_data_func` runs over the
whole loaded frame before date filtering.  That is safe for causal indicators
(rolling means, RSI, ATR) but a non-causal transform -- a centered window, a
full-sample z-score, a forward fill from the future -- would leak across the
boundary no matter how the windows are cut.  Keep data preparation causal.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from backtester.engine import BacktestEngine
from backtester.metrics import RiskMetrics, periods_per_year_for_timeframe

from .param_optimizer import ParameterOptimizer

logger = logging.getLogger(__name__)

ONE_DAY = pd.Timedelta(days=1)

DEFAULT_N_FOLDS = 4

# Metrics carried into the fold report; `trades` / `equity_curve` are dropped so
# the summary stays JSON-serializable and small.
_HEAVY_RESULT_KEYS = ("trades", "equity_curve", "parameters")


def _as_date_str(ts: pd.Timestamp) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class Fold:
    """One train/test split.  All four bounds are inclusive day boundaries."""

    index: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @property
    def train_days(self) -> int:
        return (self.train_end - self.train_start).days + 1

    @property
    def test_days(self) -> int:
        return (self.test_end - self.test_start).days + 1

    def to_dict(self) -> Dict:
        return {
            "fold": self.index,
            "train_start": _as_date_str(self.train_start),
            "train_end": _as_date_str(self.train_end),
            "test_start": _as_date_str(self.test_start),
            "test_end": _as_date_str(self.test_end),
            "train_days": self.train_days,
            "test_days": self.test_days,
        }


def generate_folds(start_date, end_date, n_folds: int = DEFAULT_N_FOLDS,
                   train_period_days: Optional[int] = None,
                   test_period_days: Optional[int] = None,
                   anchored: bool = True) -> List[Fold]:
    """Cut [start_date, end_date] into sequential, non-overlapping folds.

    Two modes:

    * Explicit windows -- pass both `train_period_days` and `test_period_days`
      and the folds roll forward one test window at a time until the range runs
      out (`n_folds` then acts as a maximum, not a target).
    * Equal blocks (default) -- the range is split into `n_folds + 1` blocks.
      Block 0 is the initial training data; blocks 1..n are the test windows, so
      the test windows are contiguous, equal length, mutually disjoint, and
      together cover everything from the end of block 0 to `end_date`.

    `anchored=True` trains on all data from `start_date` up to the fold's train
    end (expanding window); `anchored=False` trains on a fixed-length window
    that rolls forward with the test window.
    """
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if end <= start:
        raise ValueError(f"end_date ({_as_date_str(end)}) must be after start_date ({_as_date_str(start)})")
    if n_folds is not None and n_folds < 1:
        raise ValueError(f"n_folds must be >= 1, got {n_folds}")

    total_days = (end - start).days + 1

    if (train_period_days is None) != (test_period_days is None):
        raise ValueError("train_period_days and test_period_days must be given together, or neither")

    if train_period_days is not None:
        return _rolling_folds(start, end, train_period_days, test_period_days, n_folds, anchored)

    n_blocks = n_folds + 1
    if total_days < n_blocks:
        raise ValueError(
            f"date range is only {total_days} days, too short for {n_folds} folds "
            f"({n_blocks} blocks needed). Widen the range or lower n_folds."
        )

    block_len = total_days // n_blocks
    remainder = total_days % n_blocks
    # The leftover days go to the initial training block so every TEST window is
    # exactly the same length and folds stay comparable to one another.
    offsets = [0, block_len + remainder]
    for _ in range(n_blocks - 1):
        offsets.append(offsets[-1] + block_len)

    folds = []
    for i in range(n_folds):
        test_start = start + pd.Timedelta(days=offsets[i + 1])
        test_end = start + pd.Timedelta(days=offsets[i + 2]) - ONE_DAY
        train_end = test_start - ONE_DAY
        train_start = start if anchored else start + pd.Timedelta(days=offsets[i])
        folds.append(Fold(index=i + 1, train_start=train_start, train_end=train_end,
                          test_start=test_start, test_end=test_end))
    return folds


def _rolling_folds(start: pd.Timestamp, end: pd.Timestamp, train_period_days: int,
                   test_period_days: int, max_folds: Optional[int], anchored: bool) -> List[Fold]:
    if train_period_days < 1 or test_period_days < 1:
        raise ValueError("train_period_days and test_period_days must both be >= 1")

    folds: List[Fold] = []
    step = pd.Timedelta(days=test_period_days)
    window_start = start
    while True:
        train_end = window_start + pd.Timedelta(days=train_period_days) - ONE_DAY
        test_start = train_end + ONE_DAY
        test_end = test_start + pd.Timedelta(days=test_period_days) - ONE_DAY
        if test_end > end:
            break
        train_start = start if anchored else window_start
        folds.append(Fold(index=len(folds) + 1, train_start=train_start, train_end=train_end,
                          test_start=test_start, test_end=test_end))
        if max_folds is not None and len(folds) >= max_folds:
            break
        window_start = window_start + step

    if not folds:
        raise ValueError(
            f"no folds fit in [{_as_date_str(start)}, {_as_date_str(end)}] with "
            f"train_period_days={train_period_days} and test_period_days={test_period_days}"
        )
    return folds


def validate_no_leakage(folds: List[Fold]) -> None:
    """Raise unless every train window ends strictly before its own test window
    and before every later fold's test window.

    This is the guarantee the whole module exists to provide, so it is checked
    explicitly rather than assumed from the fold arithmetic.
    """
    if not folds:
        raise ValueError("no folds to validate")

    for i, fold in enumerate(folds):
        if fold.train_start > fold.train_end:
            raise ValueError(f"fold {fold.index}: empty train window")
        if fold.test_start > fold.test_end:
            raise ValueError(f"fold {fold.index}: empty test window")
        for later in folds[i:]:
            if fold.train_end >= later.test_start:
                raise ValueError(
                    f"LOOK-AHEAD LEAK: fold {fold.index} trains through "
                    f"{_as_date_str(fold.train_end)}, which reaches into the test window of "
                    f"fold {later.index} ({_as_date_str(later.test_start)} -> "
                    f"{_as_date_str(later.test_end)}). Training must end at least one full day "
                    f"before any test window it will later be scored against."
                )

    for previous, current in zip(folds, folds[1:]):
        if current.test_start <= previous.test_end:
            raise ValueError(
                f"fold {current.index} test window overlaps fold {previous.index}: "
                f"{_as_date_str(current.test_start)} <= {_as_date_str(previous.test_end)}"
            )


def default_optimizer_factory(config: Dict, strategy_func, prepare_data_func, metric: str,
                              direction: str, workers: int, train_start: pd.Timestamp,
                              train_end: pd.Timestamp) -> ParameterOptimizer:
    """Build a ParameterOptimizer clamped to a single fold's train window."""
    return ParameterOptimizer(
        config, strategy_func, metric=metric, direction=direction, workers=workers,
        prepare_data_func=prepare_data_func,
        in_sample_start_date=_as_date_str(train_start),
        in_sample_end_date=_as_date_str(train_end),
    )


def default_engine_factory(config: Dict, strategy_func, prepare_data_func) -> BacktestEngine:
    return BacktestEngine(config, strategy_func, prepare_data_func=prepare_data_func)


class WalkForwardValidator:
    """Optimize on a train window, score on the next unseen window, roll forward.

    `optimizer_factory` / `engine_factory` are injectable so the fold plumbing can
    be exercised in tests without running a real grid search.
    """

    def __init__(self, config: Dict, strategy_func, prepare_data_func: Optional[Callable] = None,
                 n_folds: Optional[int] = None, train_period_days: Optional[int] = None,
                 test_period_days: Optional[int] = None, anchored: Optional[bool] = None,
                 metric: Optional[str] = None, direction: Optional[str] = None,
                 workers: int = 4, start_date: Optional[str] = None, end_date: Optional[str] = None,
                 optimizer_factory: Optional[Callable] = None,
                 engine_factory: Optional[Callable] = None):
        self.config = config
        self.strategy_func = strategy_func
        self.prepare_data_func = prepare_data_func
        self.workers = workers

        wf_config = config.get("walk_forward", {}) or {}
        opt_config = config.get("optimization", {}) or {}
        backtest_config = config.get("backtest", {}) or {}

        self.n_folds = n_folds if n_folds is not None else wf_config.get("n_folds", DEFAULT_N_FOLDS)
        self.train_period_days = (train_period_days if train_period_days is not None
                                  else wf_config.get("train_period_days"))
        self.test_period_days = (test_period_days if test_period_days is not None
                                 else wf_config.get("test_period_days"))
        self.anchored = anchored if anchored is not None else wf_config.get("anchored", True)
        self.metric = metric or opt_config.get("metric", "sharpe_ratio")
        self.direction = direction or opt_config.get("score_direction", "maximize")

        # Walk-forward legitimately spans the full range: each fold's optimizer is
        # walled off from its own future, so there is no single in-sample cut-off
        # to respect here. in_sample_end_date is used only as a fallback when the
        # config has no explicit end_date.
        self.start_date = pd.Timestamp(
            start_date or wf_config.get("start_date") or backtest_config["start_date"]).normalize()
        self.end_date = pd.Timestamp(
            end_date or wf_config.get("end_date") or backtest_config.get("end_date")
            or backtest_config["in_sample_end_date"]).normalize()

        self.optimizer_factory = optimizer_factory or default_optimizer_factory
        self.engine_factory = engine_factory or default_engine_factory

        self.folds: List[Fold] = generate_folds(
            self.start_date, self.end_date, n_folds=self.n_folds,
            train_period_days=self.train_period_days, test_period_days=self.test_period_days,
            anchored=self.anchored,
        )
        validate_no_leakage(self.folds)

        self.fold_results: List[Dict] = []

    def run(self, symbols: List[str]) -> Dict:
        """Run every fold and return the aggregated walk-forward report."""
        # Re-checked here as well: a caller that mutated `self.folds` after
        # construction must not be able to slip a leaking window past us.
        validate_no_leakage(self.folds)

        # Recorded so the final interpretation can distinguish "the search found
        # overfit parameters" from "there was nothing to search".
        from .grid_search import GridSearchGenerator
        self._combinations_per_fold = GridSearchGenerator.count_combinations(
            self.config["strategy"].get("parameters", {}))
        if self._combinations_per_fold <= 1:
            logger.warning("Only one parameter combination configured: walk-forward will measure "
                           "period-to-period variance, not optimization bias.")

        logger.info(f"Walk-forward validation: {len(self.folds)} folds over "
                    f"{_as_date_str(self.start_date)} -> {_as_date_str(self.end_date)} "
                    f"({'anchored' if self.anchored else 'rolling'} train window, metric={self.metric})")

        self.fold_results = []
        for fold in self.folds:
            logger.info(
                f"Fold {fold.index}/{len(self.folds)}: train "
                f"{_as_date_str(fold.train_start)} -> {_as_date_str(fold.train_end)} "
                f"| test {_as_date_str(fold.test_start)} -> {_as_date_str(fold.test_end)}"
            )
            self.fold_results.append(self._run_fold(fold, symbols))

        return self.build_report()

    def _run_fold(self, fold: Fold, symbols: List[str]) -> Dict:
        optimizer = self.optimizer_factory(
            self.config, self.strategy_func, self.prepare_data_func, self.metric,
            self.direction, self.workers, fold.train_start, fold.train_end,
        )
        best_params, results = optimizer.optimize(symbols)

        if not results:
            logger.warning(f"Fold {fold.index}: optimization produced no results; skipping test window")
            return {**fold.to_dict(), "parameters": None, "train_metrics": {}, "test_metrics": {},
                    "degradation": {}, "status": "no_optimization_results"}

        train_metrics = {k: v for k, v in results[0].items() if k not in _HEAVY_RESULT_KEYS}
        test_metrics = self._evaluate_on_test_window(fold, symbols, best_params)

        logger.info(f"Fold {fold.index}: best params {best_params} | "
                    f"train {self.metric}={train_metrics.get(self.metric)} "
                    f"-> test {self.metric}={test_metrics.get(self.metric)}")

        return {
            **fold.to_dict(),
            "parameters": best_params,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "degradation": self._degradation(train_metrics, test_metrics),
            "status": "ok",
        }

    def _evaluate_on_test_window(self, fold: Fold, symbols: List[str], params: Dict) -> Dict:
        """Score `params` on the fold's test window only.

        Both bounds are passed explicitly so the run cannot spill into a
        neighbouring fold in either direction.
        """
        engine = self.engine_factory(self.config, self.strategy_func, self.prepare_data_func)
        _, trades, equity_curve = engine.run(
            symbols, params,
            start_date=_as_date_str(fold.test_start),
            end_date=_as_date_str(fold.test_end),
        )
        periods_per_year = periods_per_year_for_timeframe(self.config.get("data", {}).get("timeframe", "1d"))
        metrics = RiskMetrics.calculate_metrics_summary(
            trades, equity_curve, self.config["backtest"]["initial_capital"],
            periods_per_year=periods_per_year,
        )
        return {"num_trades": len(trades), **metrics}

    def _degradation(self, train_metrics: Dict, test_metrics: Dict) -> Dict:
        train_value = train_metrics.get(self.metric)
        test_value = test_metrics.get(self.metric)
        if not isinstance(train_value, (int, float)) or not isinstance(test_value, (int, float)):
            return {}

        # "Degradation" is always signed so that a positive number means the test
        # window did worse, whichever direction the metric is optimized in.
        raw = train_value - test_value if self.direction == "maximize" else test_value - train_value
        degradation_pct = (raw / abs(train_value) * 100.0) if train_value else None
        return {
            "metric": self.metric,
            "train": train_value,
            "test": test_value,
            "degradation": raw,
            "degradation_pct": degradation_pct,
            "degraded": raw > 0,
        }

    def build_report(self) -> Dict:
        return {
            "walk_forward_date": datetime.now().isoformat(),
            "metric": self.metric,
            "direction": self.direction,
            "start_date": _as_date_str(self.start_date),
            "end_date": _as_date_str(self.end_date),
            "n_folds": len(self.folds),
            "train_window": "anchored" if self.anchored else "rolling",
            "folds": self.fold_results,
            "summary": self.summarize(),
        }

    def summarize(self) -> Dict:
        scored = [f for f in self.fold_results if f.get("degradation")]
        if not scored:
            return {"folds_scored": 0, "note": "no fold produced comparable train/test metrics"}

        train_values = [f["degradation"]["train"] for f in scored]
        test_values = [f["degradation"]["test"] for f in scored]
        degradations = [f["degradation"]["degradation"] for f in scored]
        mean_train = sum(train_values) / len(train_values)
        mean_test = sum(test_values) / len(test_values)

        return {
            "folds_scored": len(scored),
            "metric": self.metric,
            "mean_train_metric": mean_train,
            "mean_test_metric": mean_test,
            "mean_degradation": sum(degradations) / len(degradations),
            "mean_degradation_pct": ((mean_train - mean_test) / abs(mean_train) * 100.0) if mean_train else None,
            "worst_fold_degradation": max(degradations),
            "folds_degraded": sum(1 for d in degradations if d > 0),
            "efficiency": (mean_test / mean_train) if mean_train else None,
            "parameter_stability": self.parameter_stability(),
            "interpretation": self._interpret(mean_train, mean_test, degradations),
        }

    def parameter_stability(self) -> Dict:
        """How often the grid search picked the same value fold to fold.

        Winning parameters that never settle are themselves evidence of
        overfitting: the search is chasing noise, not a persistent edge.
        """
        param_sets = [f["parameters"] for f in self.fold_results if f.get("parameters")]
        if not param_sets:
            return {}

        stability = {}
        for name in sorted({key for params in param_sets for key in params}):
            values = [params.get(name) for params in param_sets]
            counts = Counter(repr(v) for v in values)
            most_common_repr, most_common_count = counts.most_common(1)[0]
            stability[name] = {
                "values_by_fold": values,
                "unique_values": len(counts),
                "most_common": next(v for v in values if repr(v) == most_common_repr),
                "consistency_pct": most_common_count / len(values) * 100.0,
            }
        return stability

    def _interpret(self, mean_train: float, mean_test: float, degradations: List[float]) -> str:
        stability = self.parameter_stability()
        unstable = [name for name, info in stability.items() if info["consistency_pct"] < 50.0]

        # With a single-point grid there is nothing to search, so no amount of
        # degradation can be evidence of curve-fitting — say so instead of
        # reporting a fit that never happened.
        single_point_grid = all(info["unique_values"] == 1 for info in stability.values()) and bool(stability)
        searched_combinations = getattr(self, "_combinations_per_fold", None)
        if single_point_grid and (searched_combinations is None or searched_combinations <= 1):
            return ("Only one parameter combination was searched, so this run cannot detect "
                    "curve-fitting: it measures period-to-period variance of fixed parameters, "
                    "not optimization bias. Widen strategy.parameters to test for overfitting.")

        # The ratio test is only meaningful when in-sample performance is positive.
        # With a negative mean_train, mean_test/mean_train < 0.5 is satisfied
        # precisely when test does BETTER, which inverts the verdict.
        if mean_train > 0 and mean_test / mean_train < 0.5:
            verdict = ("Out-of-sample performance is less than half of in-sample: strong evidence "
                       "the parameters are curve-fit to each training window.")
        elif mean_train <= 0:
            verdict = (f"In-sample performance was not positive (mean train {mean_train:.4f}), so "
                       f"train-vs-test ratios are not meaningful here; judge this on the raw "
                       f"per-fold numbers rather than on degradation.")
        elif sum(1 for d in degradations if d > 0) == len(degradations):
            verdict = ("Every fold degraded from train to test, though not catastrophically: some "
                       "optimization bias is present, as expected.")
        else:
            verdict = "Test performance broadly tracks train performance across folds."

        if unstable:
            verdict += (f" Winning parameters were unstable across folds ({', '.join(unstable)}), "
                        f"which is itself a sign the search is fitting noise.")
        return verdict

    def export_results(self, output_dir: str, filename: Optional[str] = None) -> Path:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = output_path / (filename or f"walk_forward_summary_{timestamp}.json")
        with open(target, "w") as f:
            json.dump(self.build_report(), f, indent=2, default=str)
        return target
