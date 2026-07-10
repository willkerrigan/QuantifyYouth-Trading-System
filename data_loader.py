import yfinance as yf
import pandas as pd


def load_historical_data(ticker: str) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    df = stock.history(period="2y", interval="1d")

    if df.empty:
        raise ValueError(f"No data for '{ticker}' — check the symbol.")

    return df


if __name__ == "__main__":
    data = load_historical_data("AAPL")
    print(data.head())
    print(f"\nGot {len(data)} rows for AAPL")
