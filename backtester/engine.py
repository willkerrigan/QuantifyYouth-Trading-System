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
    def __init__(self, config: Dict, strategy_func, data_loader: Optional[DataLoader] = None,
                 prepare_data_func: Optional[callable] = None):
        self.config = config
        self.strategy_func = strategy_func
        self.data_loader = data_loader or DataLoader(config)
        self.prepare_data_func = prepare_data_func
        self.initial_capital = config["backtest"]["initial_capital"]
        self.commission = config["backtest"].get("commission", 0.001)
        self.slippage = config["backtest"].get("slippage", 0.0)
        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital
        self.open_positions = {}

    def run(self, symbols: List[str], params: Dict, start_date: Optional[str] = None,
            end_date: Optional[str] = None) -> Tuple[float, List[Trade], pd.DataFrame]:
        """Run the backtest. start_date/end_date (inclusive) restrict which loaded
        bars are simulated, e.g. to confine an optimization run to an in-sample
        window while leaving later data untouched for out-of-sample testing."""
        logger.info(f"Starting backtest with params: {params}")
        self.trades = []
        self.equity_curve = []
        self.capital = self.initial_capital
        self.open_positions = {}

        data = {}
        for symbol in symbols:
            try:
                df = self.data_loader.load(symbol)
                if self.prepare_data_func is not None:
                    df = self.prepare_data_func(df)
                data[symbol] = df
            except Exception as e:
                logger.warning(f"Failed to load data for {symbol}: {e}")
                continue

        if not data:
            logger.error("No data loaded")
            return self.capital, [], pd.DataFrame()

        dates = self._get_aligned_dates(data)
        tz = dates[0].tzinfo if dates else None
        if start_date is not None:
            start_ts = pd.Timestamp(start_date)
            start_ts = start_ts.tz_localize(tz) if tz is not None else start_ts
            dates = [d for d in dates if d >= start_ts]
        if end_date is not None:
            # +1 day so an intraday end_date includes the whole day, not just its midnight instant
            end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1)
            end_ts = end_ts.tz_localize(tz) if tz is not None else end_ts
            dates = [d for d in dates if d < end_ts]
        for date in dates:
            daily_data = {sym: df.loc[date] for sym, df in data.items() if date in df.index}

            # Stops are checked against the bar's low before the strategy is consulted,
            # so a position that gapped/traded through its stop is already flat when the
            # strategy sees open_positions.
            stopped_out = self._apply_stop_losses(daily_data, date)

            signals = self.strategy_func(daily_data, self.open_positions, params)

            for symbol, signal in signals.items():
                # A position stopped out on this bar is done for the bar: no re-entry
                # and no second exit from the strategy's own signal.
                if symbol in stopped_out:
                    continue
                if signal == "BUY" and symbol not in self.open_positions:
                    self._enter_position(symbol, daily_data[symbol]["close"], date, params)
                elif signal == "SELL" and symbol in self.open_positions:
                    self._exit_position(symbol, daily_data[symbol]["close"], date)

            self.equity_curve.append({"date": date, "equity": self.capital})

        logger.info(f"Backtest complete. Generated {len(self.trades)} trades.")
        return self.capital, self.trades, self._equity_curve_to_df()

    @staticmethod
    def _stop_loss_fraction(params: Dict) -> Optional[float]:
        """Return the stop-loss fraction from params, or None when no stop is configured.

        A missing key, None, 0 or any non-usable value means "no stop" so that the
        engine behaves exactly as it did before stops existed.
        """
        if not params:
            return None
        raw = params.get("stop_loss")
        if raw is None:
            return None
        try:
            stop_loss = float(raw)
        except (TypeError, ValueError):
            logger.warning(f"Ignoring non-numeric stop_loss: {raw!r}")
            return None
        if not np.isfinite(stop_loss) or stop_loss <= 0:
            return None
        if stop_loss >= 1:
            logger.warning(f"Ignoring stop_loss {stop_loss}: must be a fraction below 1 (e.g. 0.02 for 2%).")
            return None
        return stop_loss

    def _apply_stop_losses(self, daily_data: Dict, date: datetime) -> set:
        """Exit any open position whose bar low reached its stop. Returns the symbols stopped out."""
        stopped_out = set()
        for symbol in list(self.open_positions.keys()):
            pos = self.open_positions[symbol]
            stop_price = pos.get("stop_price")
            if stop_price is None or symbol not in daily_data:
                continue
            bar = daily_data[symbol]
            low = bar["low"] if "low" in bar else bar["close"]
            if low is None or (isinstance(low, float) and np.isnan(low)):
                continue
            if low <= stop_price:
                # Fill at the stop, not the close: the stop triggers intrabar.
                self._exit_position(symbol, stop_price, date)
                stopped_out.add(symbol)
        return stopped_out

    def _enter_position(self, symbol: str, price: float, date: datetime, params: Dict) -> None:
        position_size = int((self.capital * 0.1) / price)
        entry_cost = position_size * price * (1 + self.commission)
        if entry_cost <= self.capital:
            stop_loss = self._stop_loss_fraction(params)
            stop_price = price * (1 - stop_loss) if stop_loss is not None else None
            self.open_positions[symbol] = {"entry_price": price, "entry_date": date, "size": position_size,
                                           "params": params, "stop_price": stop_price}
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
