"""Market data loading utilities for the trading system."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf


REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def load_historical_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    *,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Load historical OHLCV data for a ticker from Yahoo Finance.

    Parameters
    ----------
    ticker:
        Public market symbol, for example ``AAPL`` or ``SPY``.
    period:
        yfinance period string such as ``6mo``, ``1y``, ``2y``, or ``5y``.
    interval:
        yfinance interval string such as ``1d``, ``1wk``, or ``1mo``.
    auto_adjust:
        Whether yfinance should adjust OHLC values for splits and dividends.
    """

    if not ticker or not ticker.strip():
        raise ValueError("ticker is required")

    stock = yf.Ticker(ticker.upper().strip())
    df = stock.history(period=period, interval=interval, auto_adjust=auto_adjust)

    if df.empty:
        raise ValueError(f"No data for '{ticker}'. Check the symbol or date range.")

    return clean_price_data(df)


def load_csv_data(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV file with a date column or date index."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    date_column = next((col for col in df.columns if col.lower() in {"date", "datetime"}), None)
    if date_column:
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.set_index(date_column)
    else:
        first_column = df.columns[0]
        parsed_dates = pd.to_datetime(df[first_column], errors="coerce")
        if parsed_dates.notna().mean() < 0.8:
            raise ValueError("CSV data must include a Date or Datetime column.")
        df = df.drop(columns=first_column)
        df.index = parsed_dates

    return clean_price_data(df)


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV data for indicator and backtest calculations."""

    missing = [column for column in REQUIRED_PRICE_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required price columns: {', '.join(missing)}")

    cleaned = df.copy()
    cleaned = cleaned.loc[:, [*REQUIRED_PRICE_COLUMNS, *extra_columns(cleaned)]]
    cleaned.index = pd.to_datetime(cleaned.index)
    cleaned = cleaned.sort_index()
    cleaned = cleaned[~cleaned.index.duplicated(keep="last")]

    numeric_columns = [*REQUIRED_PRICE_COLUMNS, *extra_columns(cleaned)]
    cleaned[numeric_columns] = cleaned[numeric_columns].apply(pd.to_numeric, errors="coerce")
    cleaned = cleaned.dropna(subset=REQUIRED_PRICE_COLUMNS)

    if cleaned.empty:
        raise ValueError("No usable rows remain after cleaning price data.")

    return cleaned


def extra_columns(df: pd.DataFrame) -> list[str]:
    """Return non-required columns while preserving input order."""

    return [column for column in df.columns if column not in REQUIRED_PRICE_COLUMNS]


if __name__ == "__main__":
    data = load_historical_data("AAPL")
    print(data.head())
    print(f"\nGot {len(data)} rows for AAPL")
