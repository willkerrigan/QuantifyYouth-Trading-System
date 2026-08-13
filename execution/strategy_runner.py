"""Bridge between the backtester's strategy functions and live signal submission.

The whole point of this module is that a strategy which backtests is the *same*
callable that trades live: it reuses ``backtester.strategies.get_strategy`` for
the (strategy_func, prepare_data_func) pair and ``backtester.data_loader.DataLoader``
for bars, so there is exactly one data path and one strategy implementation.

On each :meth:`StrategyRunner.poll` the runner:
  1. loads recent bars per configured symbol (force_refresh, so live polls see new data),
  2. runs ``prepare_data_func`` over the frame exactly as the backtest engine does,
  3. hands the *latest* bar of each symbol to ``strategy_func`` together with the
     broker's current open positions,
  4. converts BUY/SELL into :class:`Signal` objects (HOLD is dropped) and submits
     them through the existing :class:`SignalHandler`.
"""

import logging
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional

from .signal_handler import Signal, SignalHandler, SignalType

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 120
DEFAULT_POLL_INTERVAL_SECONDS = 60


class StrategyRunner:
    """Runs a backtester strategy against live-ish data and emits Signals."""

    def __init__(
        self,
        config: Dict,
        signal_handler: SignalHandler,
        data_loader=None,
        strategy_name: Optional[str] = None,
        position_provider: Optional[Callable[[], Dict]] = None,
        strategy_func: Optional[Callable] = None,
        prepare_data_func: Optional[Callable] = None,
    ):
        self.config = config or {}
        self.signal_handler = signal_handler
        self.position_provider = position_provider

        strategy_config = self.config.get("strategy", {}) or {}
        self.strategy_name = strategy_name or strategy_config.get("name")
        if strategy_func is None:
            if not self.strategy_name:
                raise ValueError(
                    "No strategy configured: set strategy.name in the config or pass strategy_name."
                )
            # Imported lazily so unit tests can inject a strategy without importing backtester.
            from backtester.strategies import get_strategy

            strategy_func, prepare_data_func = get_strategy(self.strategy_name)
        self.strategy_func = strategy_func
        self.prepare_data_func = prepare_data_func

        self.params = self._normalize_params(strategy_config.get("parameters", {}) or {})
        self.symbols = list((self.config.get("data", {}) or {}).get("symbols", []) or [])
        if not self.symbols:
            logger.warning("StrategyRunner has no symbols configured (data.symbols is empty).")

        live_config = self.config.get("live", {}) or {}
        self.lookback_days = int(live_config.get("lookback_days", DEFAULT_LOOKBACK_DAYS))
        self.signal_confidence = float(live_config.get("signal_confidence", 1.0))
        self.poll_interval_seconds = float(
            live_config.get("poll_interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
        )

        self._owns_loader = data_loader is None
        self.data_loader = data_loader if data_loader is not None else self._build_data_loader()
        # (symbol -> (action, bar timestamp)) so one bar can't spawn a signal per poll.
        self._last_emitted: Dict[str, tuple] = {}

    # ------------------------------------------------------------------ setup

    def _normalize_params(self, parameters: Dict) -> Dict:
        """Backtest configs express parameters as optimizer grids (lists). Live
        trading needs a single value, so collapse each list to its first entry."""
        params = {}
        for key, value in parameters.items():
            if isinstance(value, (list, tuple)):
                if not value:
                    logger.warning("Strategy parameter '%s' is an empty list; skipping.", key)
                    continue
                if len(value) > 1:
                    logger.warning(
                        "Strategy parameter '%s' has %d candidate values %s; live trading needs one "
                        "value, using %s. Lock parameters in before trading.",
                        key, len(value), list(value), value[0],
                    )
                params[key] = value[0]
            else:
                params[key] = value
        return params

    def _build_data_loader(self):
        from backtester.data_loader import DataLoader

        loader_config = dict(self.config)
        loader_config["backtest"] = dict(self.config.get("backtest", {}) or {})
        loader = DataLoader(loader_config)
        self._refresh_window(loader)
        return loader

    def _refresh_window(self, loader=None) -> None:
        """Point the loader at a trailing window ending today, recomputed each poll
        so a process running across midnight doesn't go stale."""
        loader = loader if loader is not None else self.data_loader
        if not self._owns_loader or not hasattr(loader, "config"):
            return
        today = datetime.now().date()
        backtest_config = loader.config.setdefault("backtest", {})
        backtest_config["start_date"] = str(today - timedelta(days=self.lookback_days))
        backtest_config["end_date"] = str(today + timedelta(days=1))

    # ------------------------------------------------------------------- data

    def _latest_bar(self, symbol: str):
        """Latest prepared bar for ``symbol``, or None if data is missing/empty/unusable."""
        try:
            df = self.data_loader.load(symbol, force_refresh=True)
        except Exception as e:
            logger.warning("Failed to load live data for %s: %s", symbol, e)
            return None

        if df is None or len(df) == 0:
            logger.warning("No bars returned for %s; skipping this poll.", symbol)
            return None

        try:
            if self.prepare_data_func is not None:
                df = self.prepare_data_func(df)
        except Exception as e:
            logger.warning("prepare_data_func failed for %s: %s", symbol, e)
            return None

        if df is None or len(df) == 0:
            logger.warning("No bars left for %s after data preparation; skipping.", symbol)
            return None

        row = df.iloc[-1]
        if "close" not in row.index:
            logger.warning("Latest bar for %s has no 'close' column; skipping.", symbol)
            return None
        return row

    def _open_positions(self) -> Dict:
        if self.position_provider is None:
            return {}
        try:
            return self.position_provider() or {}
        except Exception as e:
            logger.error("Failed to read open positions: %s", e)
            return {}

    # ------------------------------------------------------------------- poll

    def poll(self) -> List[Signal]:
        """One live cycle: fetch bars, run the strategy, submit BUY/SELL signals.

        Returns the signals accepted by the SignalHandler (empty list on HOLD-only,
        missing data, or a strategy error - polling never raises into the trade loop)."""
        self._refresh_window()

        latest_bars = {}
        for symbol in self.symbols:
            row = self._latest_bar(symbol)
            if row is not None:
                latest_bars[symbol] = row

        if not latest_bars:
            logger.warning("No usable market data this poll; no signals generated.")
            return []

        open_positions = self._open_positions()
        try:
            raw_signals = self.strategy_func(latest_bars, open_positions, self.params) or {}
        except Exception as e:
            logger.error("Strategy '%s' raised during live evaluation: %s", self.strategy_name, e)
            return []

        submitted = []
        for symbol, action in raw_signals.items():
            action = str(action).upper()
            if action not in ("BUY", "SELL"):
                continue
            row = latest_bars.get(symbol)
            if row is None:
                logger.warning("Strategy emitted %s for %s with no bar data; ignoring.", action, symbol)
                continue

            try:
                current_price = float(row["close"])
            except (KeyError, TypeError, ValueError):
                logger.warning("Unusable close price for %s; dropping %s signal.", symbol, action)
                continue
            if current_price <= 0:
                logger.warning("Non-positive close price (%s) for %s; dropping %s signal.",
                               current_price, symbol, action)
                continue

            bar_timestamp = getattr(row, "name", None)
            if self._last_emitted.get(symbol) == (action, bar_timestamp):
                logger.debug("Already emitted %s for %s on bar %s; skipping duplicate.",
                             action, symbol, bar_timestamp)
                continue

            signal = Signal(
                symbol=symbol,
                signal_type=SignalType[action],
                timestamp=datetime.now(),
                confidence=self.signal_confidence,
                metadata={
                    "current_price": current_price,
                    "strategy": self.strategy_name,
                    "bar_timestamp": str(bar_timestamp) if bar_timestamp is not None else None,
                },
            )
            if self.signal_handler.add_signal(signal):
                self._last_emitted[symbol] = (action, bar_timestamp)
                submitted.append(signal)
                logger.info("Live signal: %s %s @ %.4f (bar %s)",
                            action, symbol, current_price, bar_timestamp)
            else:
                logger.warning("SignalHandler rejected %s signal for %s.", action, symbol)

        return submitted
