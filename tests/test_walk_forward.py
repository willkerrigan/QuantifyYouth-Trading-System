import pytest
import pandas as pd

from backtester.engine import BacktestEngine
from optimizer.walk_forward import (
    Fold,
    WalkForwardValidator,
    generate_folds,
    validate_no_leakage,
)


class _StubDataLoader:
    """Serves a deterministic two-year daily frame; no network, no disk."""

    def __init__(self, config=None):
        dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
        # A gentle ramp so equity curves and metrics are well-defined but trivial.
        closes = [100.0 + i * 0.1 for i in range(len(dates))]
        self._df = pd.DataFrame({"open": closes, "high": closes, "low": closes,
                                 "close": closes, "volume": 1000}, index=dates)

    def load(self, symbol, force_refresh=False):
        return self._df


def _alternating_strategy(daily_data, open_positions, params):
    """Deterministic: buy when flat, sell when held. Params only shift nothing,
    so every parameter combination is equally 'good' — the point of these tests
    is the fold plumbing, not the search."""
    return {symbol: ("SELL" if symbol in open_positions else "BUY") for symbol in daily_data}


def _config(start="2024-01-01", end="2025-12-31", parameters=None):
    return {
        "backtest": {"start_date": start, "end_date": end, "in_sample_end_date": "2025-06-30",
                     "initial_capital": 100000, "commission": 0.0, "slippage": 0.0},
        "data": {"symbols": ["SPY"], "timeframe": "1d"},
        "strategy": {"name": "stub", "parameters": parameters or {"threshold": [1, 2]}},
        "optimization": {"metric": "sharpe_ratio", "score_direction": "maximize"},
    }


class _RecordingOptimizer:
    """Stands in for ParameterOptimizer: records the train window it was handed
    and returns a canned result instead of running a grid search."""

    def __init__(self, train_start, train_end, log, params=None, metric_value=1.0):
        self.train_start = train_start
        self.train_end = train_end
        self._log = log
        self._params = params if params is not None else {"threshold": 1}
        self._metric_value = metric_value

    def optimize(self, symbols):
        self._log.append(("train", pd.Timestamp(self.train_start), pd.Timestamp(self.train_end)))
        result = {"parameters": self._params, "sharpe_ratio": self._metric_value,
                  "total_return_pct": 10.0, "num_trades": 5, "trades": [], "equity_curve": pd.DataFrame()}
        return self._params, [result]


class _RecordingEngine:
    """Stands in for BacktestEngine: records the test window it was asked to run."""

    def __init__(self, log, initial_capital=100000.0):
        self._log = log
        self._initial_capital = initial_capital

    def run(self, symbols, params, start_date=None, end_date=None):
        self._log.append(("test", pd.Timestamp(start_date), pd.Timestamp(end_date)))
        dates = pd.date_range(start_date, end_date, freq="D")
        # A gently varying curve: a perfectly flat one makes Sharpe a 0/0 form.
        equity = [self._initial_capital * (1.0 + 0.0001 * (i % 3)) for i in range(len(dates))]
        return equity[-1], [], pd.DataFrame({"date": dates, "equity": equity})


def _validator_with_recorders(config, log, fold_params=None, metric_values=None, **kwargs):
    calls = {"n": 0}

    def optimizer_factory(cfg, strategy_func, prepare_data_func, metric, direction, workers,
                          train_start, train_end):
        i = calls["n"]
        calls["n"] += 1
        params = fold_params[i] if fold_params else None
        metric_value = metric_values[i] if metric_values else 1.0
        return _RecordingOptimizer(train_start, train_end, log, params=params, metric_value=metric_value)

    def engine_factory(cfg, strategy_func, prepare_data_func):
        return _RecordingEngine(log)

    return WalkForwardValidator(config, _alternating_strategy, optimizer_factory=optimizer_factory,
                                engine_factory=engine_factory, **kwargs)


# --------------------------------------------------------------------------
# The central guarantee: training never touches its own or any later test data.
# --------------------------------------------------------------------------

def test_no_train_window_overlaps_its_own_or_any_later_test_window():
    """This is the whole reason walk-forward exists. If a training fold can see
    even one bar of a test window it will later be scored on, the out-of-sample
    numbers are fiction."""
    for n_folds in (1, 2, 3, 4, 7, 12):
        for anchored in (True, False):
            folds = generate_folds("2020-01-01", "2025-12-31", n_folds=n_folds, anchored=anchored)
            assert len(folds) == n_folds
            for i, fold in enumerate(folds):
                for later in folds[i:]:
                    assert fold.train_end < later.test_start, (
                        f"n_folds={n_folds} anchored={anchored}: fold {fold.index} trains through "
                        f"{fold.train_end} which reaches fold {later.index}'s test window "
                        f"starting {later.test_start}"
                    )
                # A full day of separation, not just a different timestamp: the engine
                # treats end_date as inclusive of that entire day.
                assert (fold.test_start - fold.train_end) >= pd.Timedelta(days=1)


def test_recorded_train_and_test_windows_never_overlap_during_a_real_run():
    """Same guarantee, but checked against the windows actually handed to the
    optimizer and the engine rather than against the fold arithmetic."""
    log = []
    validator = _validator_with_recorders(_config(), log, n_folds=4)
    validator.run(["SPY"])

    train_windows = [(s, e) for kind, s, e in log if kind == "train"]
    test_windows = [(s, e) for kind, s, e in log if kind == "test"]
    assert len(train_windows) == 4 and len(test_windows) == 4

    for i, (train_start, train_end) in enumerate(train_windows):
        for test_start, test_end in test_windows[i:]:
            assert train_end < test_start
            # No day is a member of both windows.
            train_days = set(pd.date_range(train_start, train_end, freq="D"))
            test_days = set(pd.date_range(test_start, test_end, freq="D"))
            assert not (train_days & test_days)


def test_train_windows_are_bounded_on_both_ends():
    """The optimizer must receive an explicit start AND end, otherwise a rolling
    train window would silently include earlier folds' data it was meant to drop."""
    log = []
    validator = _validator_with_recorders(_config(), log, n_folds=3, anchored=False)
    validator.run(["SPY"])

    train_windows = [(s, e) for kind, s, e in log if kind == "train"]
    assert all(start is not pd.NaT and end is not pd.NaT for start, end in train_windows)
    # Rolling: each successive window starts later than the previous one.
    starts = [start for start, _ in train_windows]
    assert starts == sorted(starts) and len(set(starts)) == len(starts)


def test_validate_no_leakage_rejects_a_leaking_fold():
    leaking = [
        Fold(1, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-30"),
             pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31")),
        # Fold 2 trains through 2024-07-15, which is inside fold 1's test window
        # and, worse, inside its own test window's predecessor period.
        Fold(2, pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-15"),
             pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30")),
    ]
    with pytest.raises(ValueError, match="LOOK-AHEAD LEAK"):
        validate_no_leakage(leaking)


def test_validate_no_leakage_rejects_shared_boundary_day():
    """train_end == test_start is a leak, not a contiguity nicety: BacktestEngine
    includes the whole of end_date, so that day would be simulated in both."""
    folds = [Fold(1, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-07-01"),
                  pd.Timestamp("2024-07-01"), pd.Timestamp("2024-12-31"))]
    with pytest.raises(ValueError, match="LOOK-AHEAD LEAK"):
        validate_no_leakage(folds)


def test_validator_refuses_to_construct_with_leaking_folds(monkeypatch):
    import optimizer.walk_forward as wf

    def leaking_folds(*args, **kwargs):
        return [Fold(1, pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31"),
                     pd.Timestamp("2024-06-01"), pd.Timestamp("2024-12-31"))]

    monkeypatch.setattr(wf, "generate_folds", leaking_folds)
    with pytest.raises(ValueError, match="LOOK-AHEAD LEAK"):
        WalkForwardValidator(_config(), _alternating_strategy)


# --------------------------------------------------------------------------
# Fold boundary math
# --------------------------------------------------------------------------

def test_test_windows_are_contiguous_and_cover_the_tail_of_the_range():
    folds = generate_folds("2024-01-01", "2025-12-31", n_folds=4)

    for previous, current in zip(folds, folds[1:]):
        assert current.test_start == previous.test_end + pd.Timedelta(days=1)

    assert folds[0].test_start == folds[0].train_end + pd.Timedelta(days=1)
    assert folds[-1].test_end == pd.Timestamp("2025-12-31")
    # Together the folds cover every day from the first test day to the range end.
    covered = sum(fold.test_days for fold in folds)
    assert covered == (folds[-1].test_end - folds[0].test_start).days + 1


def test_test_windows_are_equal_length_and_do_not_overlap():
    folds = generate_folds("2024-01-01", "2025-12-31", n_folds=5)
    lengths = {fold.test_days for fold in folds}
    assert len(lengths) == 1, f"test windows should be equal length, got {lengths}"

    seen = set()
    for fold in folds:
        days = set(pd.date_range(fold.test_start, fold.test_end, freq="D"))
        assert not (days & seen)
        seen |= days


def test_folds_stay_inside_the_requested_range():
    folds = generate_folds("2024-03-05", "2025-08-20", n_folds=3)
    assert folds[0].train_start == pd.Timestamp("2024-03-05")
    assert all(fold.train_start >= pd.Timestamp("2024-03-05") for fold in folds)
    assert all(fold.test_end <= pd.Timestamp("2025-08-20") for fold in folds)
    assert folds[-1].test_end == pd.Timestamp("2025-08-20")


def test_anchored_folds_expand_and_rolling_folds_slide():
    anchored = generate_folds("2024-01-01", "2025-12-31", n_folds=4, anchored=True)
    assert {fold.train_start for fold in anchored} == {pd.Timestamp("2024-01-01")}
    assert [fold.train_days for fold in anchored] == sorted(fold.train_days for fold in anchored)

    rolling = generate_folds("2024-01-01", "2025-12-31", n_folds=4, anchored=False)
    assert len({fold.train_start for fold in rolling}) == 4
    assert rolling[0].train_start == pd.Timestamp("2024-01-01")


def test_explicit_train_and_test_periods_roll_forward_by_the_test_length():
    folds = generate_folds("2024-01-01", "2024-12-31", train_period_days=90,
                           test_period_days=30, anchored=False, n_folds=None)
    assert len(folds) >= 2
    assert folds[0].train_start == pd.Timestamp("2024-01-01")
    assert folds[0].train_days == 90 and folds[0].test_days == 30
    assert folds[1].train_start == folds[0].train_start + pd.Timedelta(days=30)
    assert all(fold.test_end <= pd.Timestamp("2024-12-31") for fold in folds)
    validate_no_leakage(folds)


def test_explicit_periods_respect_the_fold_cap():
    folds = generate_folds("2024-01-01", "2025-12-31", train_period_days=90,
                           test_period_days=30, n_folds=3)
    assert len(folds) == 3


def test_invalid_fold_configurations_are_rejected():
    with pytest.raises(ValueError, match="too short"):
        generate_folds("2024-01-01", "2024-01-03", n_folds=10)
    with pytest.raises(ValueError, match="must be after"):
        generate_folds("2024-06-01", "2024-01-01", n_folds=2)
    with pytest.raises(ValueError, match="must be given together"):
        generate_folds("2024-01-01", "2024-12-31", train_period_days=90)
    with pytest.raises(ValueError, match="no folds fit"):
        generate_folds("2024-01-01", "2024-02-01", train_period_days=90, test_period_days=30)


# --------------------------------------------------------------------------
# Aggregation / reporting
# --------------------------------------------------------------------------

def test_report_contains_per_fold_params_train_test_metrics_and_degradation():
    log = []
    validator = _validator_with_recorders(
        _config(), log, n_folds=3,
        fold_params=[{"threshold": 1}, {"threshold": 2}, {"threshold": 3}],
        metric_values=[2.0, 2.0, 2.0],
    )
    report = validator.run(["SPY"])

    assert report["n_folds"] == 3
    assert len(report["folds"]) == 3
    for fold in report["folds"]:
        assert fold["parameters"] is not None
        assert "sharpe_ratio" in fold["train_metrics"]
        assert "sharpe_ratio" in fold["test_metrics"]
        assert fold["degradation"]["train"] == 2.0
        # Test metrics come from the fold's own out-of-sample run, not the optimizer.
        assert fold["degradation"]["test"] == pytest.approx(fold["test_metrics"]["sharpe_ratio"])
        assert fold["degradation"]["degradation"] == pytest.approx(
            fold["degradation"]["train"] - fold["degradation"]["test"])
        assert fold["degradation"]["degraded"] is (fold["degradation"]["degradation"] > 0)

    summary = report["summary"]
    assert summary["folds_scored"] == 3
    assert summary["mean_train_metric"] == pytest.approx(2.0)
    expected_mean_degradation = sum(f["degradation"]["degradation"] for f in report["folds"]) / 3
    assert summary["mean_degradation"] == pytest.approx(expected_mean_degradation)
    assert summary["folds_degraded"] == sum(1 for f in report["folds"] if f["degradation"]["degraded"])


def test_parameter_stability_flags_params_that_change_every_fold():
    log = []
    validator = _validator_with_recorders(
        _config(), log, n_folds=3,
        fold_params=[{"threshold": 1, "stop": 5}, {"threshold": 2, "stop": 5},
                     {"threshold": 3, "stop": 5}],
    )
    report = validator.run(["SPY"])
    stability = report["summary"]["parameter_stability"]

    assert stability["threshold"]["unique_values"] == 3
    assert stability["threshold"]["consistency_pct"] == pytest.approx(100 / 3)
    assert stability["stop"]["unique_values"] == 1
    assert stability["stop"]["consistency_pct"] == 100.0
    assert "unstable" in report["summary"]["interpretation"]


def test_degradation_sign_follows_the_optimization_direction():
    """Positive degradation must always mean "the test window did worse", whether
    the metric is being maximized (Sharpe) or minimized (drawdown)."""
    train = {"max_drawdown": 5.0}
    test = {"max_drawdown": 12.0}

    maximizing = _validator_with_recorders(_config(), [], n_folds=2, metric="max_drawdown",
                                           direction="maximize")
    minimizing = _validator_with_recorders(_config(), [], n_folds=2, metric="max_drawdown",
                                           direction="minimize")

    # Maximizing: 12 > 5 is an improvement, so degradation is negative.
    assert maximizing._degradation(train, test)["degradation"] == pytest.approx(-7.0)
    assert maximizing._degradation(train, test)["degraded"] is False
    # Minimizing (drawdown): 12 > 5 is worse, so degradation is positive.
    assert minimizing._degradation(train, test)["degradation"] == pytest.approx(7.0)
    assert minimizing._degradation(train, test)["degraded"] is True
    # A missing metric yields no degradation block rather than a bogus number.
    assert minimizing._degradation(train, {}) == {}


def test_export_results_writes_json(tmp_path):
    import json

    log = []
    validator = _validator_with_recorders(_config(), log, n_folds=2)
    validator.run(["SPY"])
    target = validator.export_results(str(tmp_path))

    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["n_folds"] == 2
    assert len(payload["folds"]) == 2
    assert "summary" in payload and "parameter_stability" in payload["summary"]


# --------------------------------------------------------------------------
# Integration with the real BacktestEngine (stub data loader, no network)
# --------------------------------------------------------------------------

def test_test_window_evaluation_uses_only_the_fold_test_window():
    """Runs the real BacktestEngine against a stub loader and confirms the equity
    curve it produces is confined to the fold's test window."""
    seen_windows = []

    def optimizer_factory(cfg, strategy_func, prepare_data_func, metric, direction, workers,
                          train_start, train_end):
        seen_windows.append((train_start, train_end))
        return _RecordingOptimizer(train_start, train_end, [], params={"threshold": 1})

    def engine_factory(cfg, strategy_func, prepare_data_func):
        return BacktestEngine(cfg, strategy_func, data_loader=_StubDataLoader(cfg),
                              prepare_data_func=prepare_data_func)

    config = _config()
    validator = WalkForwardValidator(config, _alternating_strategy, n_folds=3,
                                     optimizer_factory=optimizer_factory,
                                     engine_factory=engine_factory)

    captured = {}
    original = validator._evaluate_on_test_window

    def spy(fold, symbols, params):
        engine = engine_factory(config, _alternating_strategy, None)
        _, _, curve = engine.run(symbols, params, start_date=str(fold.test_start.date()),
                                 end_date=str(fold.test_end.date()))
        captured[fold.index] = curve
        return original(fold, symbols, params)

    validator._evaluate_on_test_window = spy
    report = validator.run(["SPY"])

    assert report["n_folds"] == 3
    for fold in validator.folds:
        curve = captured[fold.index]
        assert not curve.empty
        assert curve["date"].min() >= fold.test_start
        assert curve["date"].max() <= fold.test_end
        # And nothing from the training period bled in.
        assert curve["date"].min() > fold.train_end
