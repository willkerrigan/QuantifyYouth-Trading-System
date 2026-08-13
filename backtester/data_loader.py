import logging
from typing import Dict
import pandas as pd

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, config: Dict):
        self.config = config
        self.source = config.get("data", {}).get("source", "yahoo")
        self.cache = {}

    def load(self, symbol: str, force_refresh: bool = False) -> pd.DataFrame:
        if symbol in self.cache and not force_refresh:
            return self.cache[symbol]

        start_date = self.config["backtest"].get("start_date", "2023-01-01")
        end_date = self.config["backtest"].get("end_date", "2026-07-14")
        logger.info(f"Loading {symbol} data from {self.source}")

        if self.source == "yahoo":
            import yfinance as yf
            df = yf.download(symbol, start=start_date, end=end_date, progress=False)
        elif self.source == "alpaca":
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            client = StockHistoricalDataClient()
            request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_date, end=end_date)
            df = client.get_stock_bars(request_params).df
        else:
            df = pd.read_csv(f"data/{symbol}.csv", index_col="Date", parse_dates=True)

        df.columns = [col.lower() for col in df.columns]
        self.cache[symbol] = df[["open", "high", "low", "close", "volume"]]
        return self.cache[symbol]
