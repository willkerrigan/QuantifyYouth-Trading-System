import logging

from optimizer.param_optimizer import ParameterOptimizer


def _optimizer(metric="sharpe_ratio", direction="maximize"):
    config = {"backtest": {"initial_capital": 100000, "in_sample_end_date": "2025-01-01"},
              "data": {"timeframe": "1d"}, "strategy": {"parameters": {}}}
    return ParameterOptimizer(config, lambda *a: {}, metric=metric, direction=direction)


def test_maximize_ranks_highest_first():
    opt = _optimizer()
    opt.results = [{"sharpe_ratio": 0.5}, {"sharpe_ratio": 2.0}, {"sharpe_ratio": 1.0}]
    opt._sort_results()
    assert [r["sharpe_ratio"] for r in opt.results] == [2.0, 1.0, 0.5]


def test_minimize_ranks_lowest_first():
    opt = _optimizer(direction="minimize")
    opt.results = [{"sharpe_ratio": 0.5}, {"sharpe_ratio": 2.0}, {"sharpe_ratio": 1.0}]
    opt._sort_results()
    assert [r["sharpe_ratio"] for r in opt.results] == [0.5, 1.0, 2.0]


def test_nan_score_never_wins():
    opt = _optimizer()
    opt.results = [{"sharpe_ratio": float("nan")}, {"sharpe_ratio": 0.3}]
    opt._sort_results()
    # The real score must rank first; a NaN comparison would otherwise decide by input order.
    assert opt.results[0]["sharpe_ratio"] == 0.3


def test_infinite_profit_factor_never_wins():
    """A combination with zero losing trades scores inf; one lucky trade must not top the ranking."""
    opt = _optimizer(metric="profit_factor")
    opt.results = [{"profit_factor": float("inf")}, {"profit_factor": 2.5}]
    opt._sort_results()
    assert opt.results[0]["profit_factor"] == 2.5


def test_warns_when_minimizing_max_drawdown(caplog):
    with caplog.at_level(logging.WARNING):
        _optimizer(metric="max_drawdown", direction="minimize")
    assert "LARGEST drawdown" in caplog.text


def test_no_warning_when_maximizing_max_drawdown(caplog):
    with caplog.at_level(logging.WARNING):
        _optimizer(metric="max_drawdown", direction="maximize")
    assert "LARGEST drawdown" not in caplog.text
