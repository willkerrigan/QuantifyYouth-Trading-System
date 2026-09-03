from __future__ import annotations

import unittest

import pandas as pd

from data_loader import clean_price_data


class DataLoaderTests(unittest.TestCase):
    def test_clean_price_data_sorts_and_drops_invalid_rows(self) -> None:
        raw = pd.DataFrame(
            {
                "Open": [101, None, 100],
                "High": [102, 103, 101],
                "Low": [99, 100, 98],
                "Close": [101, 102, 100],
                "Volume": [1000, 2000, 1500],
            },
            index=pd.to_datetime(["2024-01-03", "2024-01-02", "2024-01-01"]),
        )

        cleaned = clean_price_data(raw)

        self.assertEqual(list(cleaned.index), sorted(cleaned.index))
        self.assertEqual(len(cleaned), 2)
        self.assertEqual(cleaned["Close"].iloc[0], 100)


if __name__ == "__main__":
    unittest.main()
