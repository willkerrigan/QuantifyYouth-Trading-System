import logging
from typing import Dict
import pandas as pd
from .metrics import normalize_timeframe

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
OHLC_COLUMNS = ("open", "high", "low", "close")
_DATE_COLUMN_CANDIDATES = ("date", "datetime", "timestamp", "time")


class DataLoadError(ValueError):
    """Raised when market data is empty, malformed, or structurally unusable.

    A bad frame must stop the run loudly: silently handing an empty or
    column-less frame downstream produces a "0 trades / 0% return" backtest
    that looks like a legitimate result. Subclasses ValueError so callers
    that already catch ValueError keep working.
    """


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
        interval = normalize_timeframe(self.config.get("data", {}).get("timeframe", "1d"))
        logger.info(f"Loading {symbol} data from {self.source} (interval={interval})")

        # Human-readable description of the request, attached to every error so a
        # failure names the symbol, source, interval and window that produced it.
        request = (f"symbol={symbol!r} source={self.source!r} interval={interval!r} "
                   f"range={start_date} -> {end_date}")

        if self.source == "yahoo":
            import yfinance as yf
            if interval == "1d":
                df = yf.download(symbol, start=start_date, end=end_date, interval=interval, progress=False)
            else:
                # Yahoo only retains intraday bars (any interval below 1d) for the
                # trailing 60 days; configured start_date/end_date can't be honored.
                logger.warning(f"Yahoo intraday data ({interval}) only covers the trailing 60 days; "
                               f"ignoring configured start_date/end_date ({start_date} -> {end_date})")
                df = yf.download(symbol, period="60d", interval=interval, progress=False)
            df = self._require_frame(df, request)
            df = self._flatten_columns(df, request)
        elif self.source == "alpaca":
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
            client = StockHistoricalDataClient()
            request_params = StockBarsRequest(symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=start_date, end=end_date)
            df = client.get_stock_bars(request_params).df
            df = self._require_frame(df, request)
            # Alpaca returns a (symbol, timestamp) MultiIndex on the *rows*, and can
            # return MultiIndex columns when several symbols are requested at once.
            df = self._flatten_index(df, symbol, request)
            df = self._flatten_columns(df, request)
        else:
            df = self._read_csv(symbol, request)

        df = self._require_frame(df, request)
        df.columns = [str(col).lower() for col in df.columns]
        self._require_columns(df, request)
        self._warn_on_suspect_data(df, symbol, request)

        self.cache[symbol] = df[list(REQUIRED_COLUMNS)]
        return self.cache[symbol]

    # ---- fetching helpers -------------------------------------------------

    def _read_csv(self, symbol: str, request: str) -> pd.DataFrame:
        path = f"data/{symbol}.csv"
        try:
            df = pd.read_csv(path)
        except FileNotFoundError as exc:
            raise DataLoadError(f"No CSV data file at {path!r} ({request})") from exc
        if df.empty:
            raise DataLoadError(f"CSV file {path!r} contains no rows ({request})")
        # Accept any casing / common naming for the timestamp column instead of
        # hard-requiring a column literally named "Date".
        date_col = next((c for c in df.columns
                         if str(c).strip().lower() in _DATE_COLUMN_CANDIDATES), None)
        if date_col is None:
            raise DataLoadError(
                f"CSV file {path!r} has no recognizable date column "
                f"(looked for {list(_DATE_COLUMN_CANDIDATES)}); columns present: "
                f"{[str(c) for c in df.columns]} ({request})")
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        return df.set_index(date_col)

    # ---- validation / normalization --------------------------------------

    @staticmethod
    def _require_frame(df, request: str) -> pd.DataFrame:
        """Fail loudly on a missing/empty frame rather than letting it flow downstream."""
        if df is None:
            raise DataLoadError(f"Data source returned no DataFrame at all ({request})")
        if not isinstance(df, pd.DataFrame):
            raise DataLoadError(
                f"Data source returned {type(df).__name__}, expected DataFrame ({request})")
        if len(df.index) == 0:
            raise DataLoadError(
                f"Downloaded data is EMPTY (0 rows) ({request}). Check the symbol spelling, "
                f"the date range, and whether the source supports this interval.")
        return df

    @staticmethod
    def _flatten_columns(df: pd.DataFrame, request: str) -> pd.DataFrame:
        """Collapse MultiIndex columns to a single level of OHLCV names.

        yfinance returns (Price, Ticker) columns by default and (Ticker, Price)
        when grouped by ticker, so pick the level that actually holds the OHLCV
        names instead of blindly taking level 0 (which is what broke before).
        """
        if not isinstance(df.columns, pd.MultiIndex):
            return df
        df = df.copy()
        best_level, best_hits = 0, -1
        for level in range(df.columns.nlevels):
            values = [str(v).lower() for v in df.columns.get_level_values(level)]
            hits = sum(1 for v in values if v in REQUIRED_COLUMNS)
            if hits > best_hits:
                best_level, best_hits = level, hits
        if best_hits <= 0:
            raise DataLoadError(
                f"MultiIndex columns contain no OHLCV names on any level; "
                f"columns present: {[tuple(str(p) for p in c) for c in df.columns]} ({request})")
        df.columns = df.columns.get_level_values(best_level)
        # A ticker level can leave duplicate OHLCV names behind (multi-symbol frames).
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()]
        return df

    @staticmethod
    def _flatten_index(df: pd.DataFrame, symbol: str, request: str) -> pd.DataFrame:
        """Reduce a MultiIndex row index (e.g. alpaca's (symbol, timestamp)) to timestamps."""
        if not isinstance(df.index, pd.MultiIndex):
            return df
        names = [str(n).lower() if n is not None else None for n in df.index.names]
        if "symbol" in names:
            level = names.index("symbol")
            try:
                df = df.xs(symbol, level=level)
            except KeyError as exc:
                available = sorted({str(v) for v in df.index.get_level_values(level)})
                raise DataLoadError(
                    f"Requested symbol not present in returned data; symbols present: "
                    f"{available} ({request})") from exc
        # Drop any remaining non-datetime levels, keeping the timestamp level.
        while isinstance(df.index, pd.MultiIndex) and df.index.nlevels > 1:
            droppable = [i for i in range(df.index.nlevels)
                         if not pd.api.types.is_datetime64_any_dtype(df.index.get_level_values(i))]
            if not droppable:
                droppable = [0]
            df = df.droplevel(droppable[0])
        return df

    @staticmethod
    def _require_columns(df: pd.DataFrame, request: str) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataLoadError(
                f"Downloaded data is missing required column(s) {missing} after normalization; "
                f"columns actually present: {[str(c) for c in df.columns]} ({request})")

    @staticmethod
    def _warn_on_suspect_data(df: pd.DataFrame, symbol: str, request: str) -> None:
        """Flag data that silently corrupts a backtest without making it fail."""
        nan_counts = {c: int(df[c].isna().sum()) for c in OHLC_COLUMNS if c in df.columns}
        bad = {c: n for c, n in nan_counts.items() if n > 0}
        if bad:
            logger.warning(f"{symbol}: OHLC data contains NaN values {bad} out of {len(df)} rows "
                           f"({request}); these bars will silently distort signals and returns")

        index = df.index
        if not index.is_monotonic_increasing:
            logger.warning(f"{symbol}: data is NOT monotonically increasing in time "
                           f"({request}); out-of-order bars corrupt backtest sequencing")
        if index.has_duplicates:
            dupes = int(index.duplicated().sum())
            logger.warning(f"{symbol}: data contains {dupes} duplicate timestamp(s) ({request})")
