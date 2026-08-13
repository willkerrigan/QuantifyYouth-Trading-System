import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

from .data_loader import DataLoader
from .metrics import RiskMetrics

logger = logging.getLogger(__name__)


class Trade:
    """Represents a single completed trade."""

    def __init__(self, asset: str, entry_date: datetime, entry_price: float,
                 exit_date: datetime, exit_price: float, size: float, strategy_params: Dict):
        self.asset = asset
        self.entry_date = entry_date
        self.entry_price = entry_price
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.size = size
        self.strategy_params = strategy_params
        self.realized_pnl = (exit_price - entry_price) * size
        self.pnl_percent = ((exit_price - entry_price) / entry_price) * 100

    def to_dict(self) -> Dict:
        return {
            "Date": self.entry_date.strftime("%Y-%m-%d"),
            "Entry_Time": self.entry_date.strftime("%H:%M:%S"),
            "Asset": self.asset,
            "Entry_Price": f"{self.entry_price:.2f}",
            "Exit_Price": f"{self.exit_price:.2f}",
            "Size": f"{self.size:.0f}",
            "Realized_PnL": f"{self.realized_pnl:.2f}",
            "PnL_Percent": f"{self.pnl_percent:.2f}%",
        }


class BacktestEngine:
    def __init__(self, config: Dict, strategy_func, data_loader: Optional[DataLoader] = None):
        self.config = config
        self.strategy_func = strategy_func
        self.data_loader = data_loader or DataLoader(config)
        self.initial_capital = config["backtest"]["initial_capital"]
        self.commission = config["backtest"].get("commission", 0.001)
        self.slippage = config["backtest"].get("slippage", 0.0)
        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital
        self.open_positions = {}

    def run(self, symbols: List[str], params: Dict) -> Tuple[float, List[Trade], pd.DataFrame]:
        logger.info(f"Starting backtest with params: {params}")
        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital
        self.open_positions = {}

        data = {}
        for symbol in symbols:
            try:
                df = self.data_loader.load(symbol)
                data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                continue

        if not data:
            logger.error("No data loaded")
            return self.capital, [], pd.DataFrame()

        dates = self._get_aligned_dates(data)
        for date in dates:
            daily_data = {sym: df.loc[date] for sym, df in data.items() if date in df.index}
            signals = self.strategy_func(daily_data, self.open_positions, params)

            for symbol, signal in signals.items():
                if signal == "BUY" and symbol not in self.open_positions:
                    self._enter_position(symbol, daily_data[symbol]["close"], date, params)
                elif signal == "SELL" and symbol in self.open_positions:
                    self._exit_position(symbol, daily_data[symbol]["close"], date)

            self.equity_curve.append({"date": date, "equity": self.capital})

        logger.info(f"Backtest complete. Generated {len(self.trades)} trades.")
        return self.capital, self.trades, self._equity_curve_to_df()

    def _enter_position(self, symbol: str, price: float, date: datetime, params: Dict) -> None:
        position_size = int((self.capital * 0.1) / price)
        entry_cost = position_size * price * (1 + self.commission)
        if entry_cost <= self.capital:
            self.open_positions[symbol] = {"entry_price": price, "entry_date": date, "size": position_size, "params": params}
            self.capital -= entry_cost

    def _exit_position(self, symbol: str, price: float, date: datetime) -> None:
        if symbol not in self.open_positions:
            return
        pos = self.open_positions.pop(symbol)
        exit_proceeds = pos["size"] * price * (1 - self.commission)
        self.capital += exit_proceeds
        trade = Trade(asset=symbol, entry_date=pos["entry_date"], entry_price=pos["entry_price"],
                      exit_date=date, exit_price=price, size=pos["size"], strategy_params=pos["params"])
        self.trades.append(trade)

    def _get_aligned_dates(self, data: Dict) -> List[datetime]:
        all_dates = set()
        for df in data.values():
            all_dates.update(df.index)
        return sorted(list(all_dates))

    def _equity_curve_to_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.equity_curve) if self.equity_curve else pd.DataFrame()

    def export_trade_log(self, filepath: str) -> None:
        if not self.trades:
            logger.warning("No trades to export.")
            return
        trade_dicts = [trade.to_dict() for trade in self.trades]
        pd.DataFrame(trade_dicts).to_csv(filepath, index=False)
        logger.info(f"Trade log exported to {filepath}")

    def export_equity_curve(self, filepath: str) -> None:
        if not self.equity_curve:
            logger.warning("No equity curve data to export.")
            return
        self._equity_curve_to_df().to_csv(filepath, index=False)
        logger.info(f"Equity curve exported to {filepath}")
