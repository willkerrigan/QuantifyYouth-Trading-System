"""Unit tests for the risk metrics in backtester/metrics.py.

Expected values are hand-computed (shown inline as the arithmetic that produces
them) so a regression in the formulas fails loudly rather than silently drifting.
No network or data files are touched.
"""

import math

import pandas as pd
import pytest

from backtester.metrics import RiskMetrics


class FakeTrade:
    """Minimal stand-in for backtester.engine.Trade for metric arithmetic."""

    def __init__(self, realized_pnl, entry_date=None, exit_date=None):
        self.realized_pnl = realized_pnl
        self.entry_date = pd.Timestamp(entry_date) if entry_date is not None else None
        self.exit_date = pd.Timestamp(exit_date) if exit_date is not None else None


def make_equity_curve(equities, start="2024-01-01", freq="D"):
    dates = pd.date_range(start=start, periods=len(equities), freq=freq)
    return pd.DataFrame({"date": dates, "equity": equities})


# ---------------------------------------------------------------- sortino ----

class TestSortinoRatio:
    def test_hand_computed_value(self):
        returns = [0.02, -0.01, 0.03, -0.02]
        # mean excess (rf=0) = 0.02/4 = 0.005
        # downside squares = [0, 0.0001, 0, 0.0004] -> mean = 0.000125
        expected = 0.005 / math.sqrt(0.000125)
        result = RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
        assert result == pytest.approx(expected)
        assert result == pytest.approx(0.4472135955)

    def test_single_losing_return(self):
        # mean = -0.01, downside deviation = 0.01 -> ratio = -1.0
        result = RiskMetrics.calculate_sortino_ratio([-0.01], risk_free_rate=0.0, periods_per_year=1)
        assert result == pytest.approx(-1.0)

    def test_annualization_scales_by_sqrt_periods(self):
        returns = [0.02, -0.01, 0.03, -0.02]
        base = RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
        annual = RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
        # risk_free_rate is 0, so only the sqrt(periods) factor differs
        assert annual == pytest.approx(base * math.sqrt(252))

    def test_ignores_upside_volatility_unlike_sharpe(self):
        # Big upside, small downside: Sortino must be materially higher than Sharpe.
        returns = [0.05, -0.01, 0.05, -0.01]
        sharpe = RiskMetrics.calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
        sortino = RiskMetrics.calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=1)
        assert sharpe == pytest.approx(0.02 / 0.03)              # std dev = 0.03
        assert sortino == pytest.approx(0.02 / math.sqrt(0.0001 / 2))  # downside dev
        assert sortino > sharpe

    def test_no_downside_returns_zero(self):
        assert RiskMetrics.calculate_sortino_ratio([0.01, 0.02, 0.03], risk_free_rate=0.0) == 0.0

    def test_all_zero_returns_is_zero(self):
        assert RiskMetrics.calculate_sortino_ratio([0.0, 0.0], risk_free_rate=0.0) == 0.0

    def test_empty_returns_zero(self):
        assert RiskMetrics.calculate_sortino_ratio([]) == 0.0

    def test_single_element_no_downside(self):
        assert RiskMetrics.calculate_sortino_ratio([0.05], risk_free_rate=0.0) == 0.0


# ----------------------------------------------------------------- calmar ----

class TestCalmarRatio:
    def test_hand_computed_value(self):
        # +30% return against a 15% max drawdown -> 30 / 15 = 2.0
        assert RiskMetrics.calculate_calmar_ratio(30.0, -0.15) == pytest.approx(2.0)

    def test_negative_return_gives_negative_ratio(self):
        assert RiskMetrics.calculate_calmar_ratio(-10.0, -0.20) == pytest.approx(-0.5)

    def test_drawdown_sign_does_not_matter(self):
        assert RiskMetrics.calculate_calmar_ratio(30.0, 0.15) == pytest.approx(2.0)

    def test_zero_drawdown_returns_zero(self):
        assert RiskMetrics.calculate_calmar_ratio(30.0, 0.0) == 0.0

    def test_zero_return_and_zero_drawdown(self):
        assert RiskMetrics.calculate_calmar_ratio(0.0, 0.0) == 0.0

    def test_zero_return_with_drawdown(self):
        assert RiskMetrics.calculate_calmar_ratio(0.0, -0.4) == 0.0


# ----------------------------------------------------------- avg win/loss ----

class TestAvgWinLoss:
    def test_hand_computed_values(self):
        trades = [FakeTrade(100), FakeTrade(-50), FakeTrade(200), FakeTrade(-100), FakeTrade(0)]
        result = RiskMetrics.calculate_avg_win_loss(trades)
        assert result["avg_win"] == pytest.approx(150.0)    # (100 + 200) / 2
        assert result["avg_loss"] == pytest.approx(-75.0)   # (-50 + -100) / 2
        assert result["win_loss_ratio"] == pytest.approx(2.0)

    def test_returns_all_three_keys(self):
        result = RiskMetrics.calculate_avg_win_loss([])
        assert set(result) == {"avg_win", "avg_loss", "win_loss_ratio"}

    def test_empty_trades(self):
        assert RiskMetrics.calculate_avg_win_loss([]) == {
            "avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0}

    def test_all_winners_guards_division(self):
        result = RiskMetrics.calculate_avg_win_loss([FakeTrade(10), FakeTrade(20)])
        assert result["avg_win"] == pytest.approx(15.0)
        assert result["avg_loss"] == 0.0
        assert result["win_loss_ratio"] == 0.0

    def test_all_losers(self):
        result = RiskMetrics.calculate_avg_win_loss([FakeTrade(-10), FakeTrade(-20)])
        assert result["avg_win"] == 0.0
        assert result["avg_loss"] == pytest.approx(-15.0)
        assert result["win_loss_ratio"] == 0.0

    def test_only_breakeven_trades(self):
        result = RiskMetrics.calculate_avg_win_loss([FakeTrade(0), FakeTrade(0)])
        assert result == {"avg_win": 0.0, "avg_loss": 0.0, "win_loss_ratio": 0.0}

    def test_single_trade(self):
        result = RiskMetrics.calculate_avg_win_loss([FakeTrade(42.5)])
        assert result["avg_win"] == pytest.approx(42.5)
        assert result["avg_loss"] == 0.0


# ------------------------------------------------ max consecutive losses ----

class TestMaxConsecutiveLosses:
    def test_hand_computed_streak(self):
        pnls = [-1, -2, 5, -1, -1, -1, 2, -1]  # streaks: 2, 3, 1 -> longest 3
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(p) for p in pnls]) == 3

    def test_streak_at_end_of_list(self):
        pnls = [5, -1, -1, -1, -1]
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(p) for p in pnls]) == 4

    def test_breakeven_trade_breaks_streak(self):
        pnls = [-1, 0, -1]
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(p) for p in pnls]) == 1

    def test_all_losses(self):
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(-1)] * 6) == 6

    def test_no_losses(self):
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(1), FakeTrade(2)]) == 0

    def test_empty_trades(self):
        assert RiskMetrics.calculate_max_consecutive_losses([]) == 0

    def test_single_losing_trade(self):
        assert RiskMetrics.calculate_max_consecutive_losses([FakeTrade(-5)]) == 1

    def test_returns_int(self):
        assert isinstance(RiskMetrics.calculate_max_consecutive_losses([FakeTrade(-5)]), int)


# ----------------------------------------------------------- exposure time ----

class TestExposureTime:
    def test_hand_computed_percentage(self):
        curve = make_equity_curve([100.0] * 10)  # 2024-01-01 .. 2024-01-10
        trades = [
            FakeTrade(1, "2024-01-02", "2024-01-05"),  # bars 02, 03, 04 -> 3
            FakeTrade(-1, "2024-01-08", "2024-01-09"),  # bar 08 -> 1
        ]
        assert RiskMetrics.calculate_exposure_time(curve, trades) == pytest.approx(40.0)

    def test_overlapping_trades_counted_once(self):
        curve = make_equity_curve([100.0] * 10)
        trades = [
            FakeTrade(1, "2024-01-02", "2024-01-05"),  # bars 02, 03, 04
            FakeTrade(1, "2024-01-03", "2024-01-06"),  # bars 03, 04, 05
        ]  # union = 02, 03, 04, 05 -> 4 of 10
        assert RiskMetrics.calculate_exposure_time(curve, trades) == pytest.approx(40.0)

    def test_always_in_market(self):
        curve = make_equity_curve([100.0] * 4)  # 01-01 .. 01-04
        trades = [FakeTrade(1, "2024-01-01", "2024-01-05")]
        assert RiskMetrics.calculate_exposure_time(curve, trades) == pytest.approx(100.0)

    def test_single_bar_curve(self):
        curve = make_equity_curve([100.0])
        trades = [FakeTrade(1, "2024-01-01", "2024-01-02")]
        assert RiskMetrics.calculate_exposure_time(curve, trades) == pytest.approx(100.0)

    def test_exit_bar_not_counted(self):
        curve = make_equity_curve([100.0] * 4)
        trades = [FakeTrade(1, "2024-01-01", "2024-01-01")]  # never held over a bar
        assert RiskMetrics.calculate_exposure_time(curve, trades) == 0.0

    def test_intraday_bars(self):
        curve = make_equity_curve([100.0] * 8, start="2024-01-02 09:30", freq="15min")
        # bars: 09:30, 09:45, 10:00, 10:15, 10:30, 10:45, 11:00, 11:15
        trades = [FakeTrade(1, "2024-01-02 10:00", "2024-01-02 10:45")]  # 10:00, 10:15, 10:30
        assert RiskMetrics.calculate_exposure_time(curve, trades) == pytest.approx(3 / 8 * 100)

    def test_no_trades(self):
        assert RiskMetrics.calculate_exposure_time(make_equity_curve([100.0] * 5), []) == 0.0

    def test_empty_equity_curve(self):
        trades = [FakeTrade(1, "2024-01-02", "2024-01-05")]
        assert RiskMetrics.calculate_exposure_time(pd.DataFrame(), trades) == 0.0

    def test_equity_curve_without_date_column(self):
        curve = pd.DataFrame({"equity": [100.0, 101.0]})
        trades = [FakeTrade(1, "2024-01-02", "2024-01-05")]
        assert RiskMetrics.calculate_exposure_time(curve, trades) == 0.0

    def test_trade_without_dates_is_skipped(self):
        curve = make_equity_curve([100.0] * 5)
        assert RiskMetrics.calculate_exposure_time(curve, [FakeTrade(-10)]) == 0.0


# ---------------------------------------------------------------- summary ----

EXISTING_KEYS = {
    "total_trades", "total_return_pct", "sharpe_ratio", "max_drawdown",
    "win_rate_pct", "profit_factor", "final_equity", "total_pnl",
}
NEW_KEYS = {
    "sortino_ratio", "calmar_ratio", "avg_win", "avg_loss", "win_loss_ratio",
    "max_consecutive_losses", "exposure_time_pct",
}


class TestMetricsSummary:
    @staticmethod
    def build():
        curve = make_equity_curve([10000.0, 10200.0, 9900.0, 10500.0, 11000.0])
        trades = [
            FakeTrade(200.0, "2024-01-01", "2024-01-02"),
            FakeTrade(-300.0, "2024-01-02", "2024-01-03"),
            FakeTrade(600.0, "2024-01-03", "2024-01-04"),
        ]
        return trades, curve

    def test_existing_keys_preserved(self):
        trades, curve = self.build()
        summary = RiskMetrics.calculate_metrics_summary(trades, curve, 10000.0)
        assert EXISTING_KEYS.issubset(summary)

    def test_existing_values_unchanged(self):
        trades, curve = self.build()
        summary = RiskMetrics.calculate_metrics_summary(trades, curve, 10000.0)
        assert summary["total_trades"] == 3
        assert summary["total_return_pct"] == pytest.approx(10.0)   # 11000 vs 10000
        assert summary["final_equity"] == pytest.approx(11000.0)
        assert summary["total_pnl"] == pytest.approx(1000.0)
        assert summary["win_rate_pct"] == pytest.approx(2 / 3 * 100)
        assert summary["profit_factor"] == pytest.approx(800 / 300)
        # worst drawdown: 9900 off a 10200 peak
        assert summary["max_drawdown"] == pytest.approx((9900 - 10200) / 10200)

    def test_new_keys_present(self):
        trades, curve = self.build()
        summary = RiskMetrics.calculate_metrics_summary(trades, curve, 10000.0)
        assert NEW_KEYS.issubset(summary)

    def test_new_values_hand_computed(self):
        trades, curve = self.build()
        summary = RiskMetrics.calculate_metrics_summary(trades, curve, 10000.0)
        assert summary["avg_win"] == pytest.approx(400.0)    # (200 + 600) / 2
        assert summary["avg_loss"] == pytest.approx(-300.0)
        assert summary["win_loss_ratio"] == pytest.approx(400 / 300)
        assert summary["max_consecutive_losses"] == 1
        # trades span bars 01-01 .. 01-03 (exit bars excluded) of 5 bars
        assert summary["exposure_time_pct"] == pytest.approx(60.0)
        expected_calmar = summary["total_return_pct"] / (abs(summary["max_drawdown"]) * 100)
        assert summary["calmar_ratio"] == pytest.approx(expected_calmar)
        returns = RiskMetrics.calculate_returns(curve)
        assert summary["sortino_ratio"] == pytest.approx(
            RiskMetrics.calculate_sortino_ratio(returns))

    def test_empty_inputs_do_not_raise(self):
        summary = RiskMetrics.calculate_metrics_summary([], pd.DataFrame(), 10000.0)
        assert EXISTING_KEYS.issubset(summary)
        assert NEW_KEYS.issubset(summary)
        assert summary["sortino_ratio"] == 0.0
        assert summary["calmar_ratio"] == 0.0
        assert summary["avg_win"] == 0.0
        assert summary["avg_loss"] == 0.0
        assert summary["win_loss_ratio"] == 0.0
        assert summary["max_consecutive_losses"] == 0
        assert summary["exposure_time_pct"] == 0.0

    def test_single_bar_curve_does_not_raise(self):
        summary = RiskMetrics.calculate_metrics_summary([], make_equity_curve([10000.0]), 10000.0)
        assert summary["sortino_ratio"] == 0.0
        assert summary["exposure_time_pct"] == 0.0
        assert summary["total_return_pct"] == pytest.approx(0.0)

    # An all-zero equity curve makes the pre-existing calculate_returns /
    # calculate_max_drawdown produce NaN (0/0); the filter keeps that expected
    # numpy warning out of the suite output.
    @pytest.mark.filterwarnings("ignore:invalid value encountered in divide")
    def test_zero_initial_capital_does_not_raise(self):
        summary = RiskMetrics.calculate_metrics_summary([], make_equity_curve([0.0, 0.0]), 0.0)
        # NaN inputs must not leak into the new ratios (they are optimizer sort keys)
        assert summary["calmar_ratio"] == 0.0
        assert summary["sortino_ratio"] == 0.0
