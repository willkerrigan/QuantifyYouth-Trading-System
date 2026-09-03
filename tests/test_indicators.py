from __future__ import annotations

import unittest

import pandas as pd

from indicators import add_technical_indicators, rsi, sma


class IndicatorTests(unittest.TestCase):
    def test_sma_uses_full_window_before_output(self) -> None:
        values = pd.Series([1, 2, 3, 4], dtype="float64")

        average = sma(values, 3)

        self.assertTrue(pd.isna(average.iloc[1]))
        self.assertEqual(average.iloc[2], 2)

    def test_rsi_stays_in_expected_range(self) -> None:
        values = pd.Series([1, 2, 3, 2, 4, 5, 4, 6, 7, 6, 8, 9, 10, 9, 11], dtype="float64")

        values_rsi = rsi(values, 5).dropna()

        self.assertTrue(((values_rsi >= 0) & (values_rsi <= 100)).all())

    def test_add_technical_indicators_adds_named_columns(self) -> None:
        close = pd.Series(range(1, 61), dtype="float64")
        data = pd.DataFrame({"Close": close})

        enriched = add_technical_indicators(data, fast_window=5, slow_window=20, rsi_window=5)

        self.assertIn("SMA_5", enriched.columns)
        self.assertIn("SMA_20", enriched.columns)
        self.assertIn("EMA_5", enriched.columns)
        self.assertIn("RSI_5", enriched.columns)


if __name__ == "__main__":
    unittest.main()
