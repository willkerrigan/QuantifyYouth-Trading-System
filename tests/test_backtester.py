from __future__ import annotations

import unittest

import pandas as pd

from backtester import BacktestConfig, run_backtest
from strategies import buy_and_hold, moving_average_crossover


def sample_prices() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=80, freq="B")
    close = pd.Series(range(100, 180), index=dates, dtype="float64")
    return pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": 1_000_000,
        },
        index=dates,
    )


class BacktesterTests(unittest.TestCase):
    def test_buy_and_hold_grows_with_rising_market(self) -> None:
        data = sample_prices()
        result = run_backtest(
            data,
            buy_and_hold(data),
            BacktestConfig(initial_cash=10_000, transaction_cost=0),
        )

        self.assertGreater(result.metrics["final_equity"], 10_000)
        self.assertAlmostEqual(
            result.metrics["total_return"],
            result.metrics["buy_hold_return"],
            places=12,
        )

    def test_transaction_cost_reduces_return_on_entry(self) -> None:
        data = sample_prices()
        signal = buy_and_hold(data)

        no_cost = run_backtest(
            data,
            signal,
            BacktestConfig(initial_cash=10_000, transaction_cost=0),
        )
        with_cost = run_backtest(
            data,
            signal,
            BacktestConfig(initial_cash=10_000, transaction_cost=0.01),
        )

        self.assertLess(with_cost.metrics["final_equity"], no_cost.metrics["final_equity"])

    def test_moving_average_crossover_respects_long_only_flag(self) -> None:
        data = sample_prices().iloc[::-1].copy()

        long_only_signal = moving_average_crossover(data, fast_window=5, slow_window=20)
        long_short_signal = moving_average_crossover(
            data,
            fast_window=5,
            slow_window=20,
            long_only=False,
        )

        self.assertGreaterEqual(long_only_signal.min(), 0)
        self.assertEqual(long_short_signal.min(), -1)


if __name__ == "__main__":
    unittest.main()
