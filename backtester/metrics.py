import numpy as np
import pandas as pd
import re
from typing import Dict, List

# Regular US equity session length, used to convert an intraday bar count into
# an equivalent number of trading days for Sharpe-ratio annualization.
_SESSION_MINUTES = 390
_TIMEFRAME_PATTERN = re.compile(r"^\s*(\d+)\s*([a-zA-Z]+)\s*$")


def normalize_timeframe(timeframe: str) -> str:
    """Convert common timeframe spellings to compact yfinance-style strings.

    Accepts forms such as "1D", "15min", "15 minutes", "1hr" and "1 hour",
    returning "1d", "15m" or "1h". Keeping this normalization in one place
    prevents config validation, downloads and metrics from accepting different
    sets of strings.
    """
    if not isinstance(timeframe, str):
        raise ValueError(f"Timeframe must be a string, got {type(timeframe).__name__}")

    match = _TIMEFRAME_PATTERN.match(timeframe)
    if not match:
        raise ValueError(f"Unsupported timeframe: {timeframe}")

    value = int(match.group(1))
    unit = match.group(2).lower()
    if value <= 0:
        raise ValueError(f"Timeframe value must be positive: {timeframe}")

    aliases = {
        "d": "d",
        "day": "d",
        "days": "d",
        "m": "m",
        "min": "m",
        "mins": "m",
        "minute": "m",
        "minutes": "m",
        "h": "h",
        "hr": "h",
        "hrs": "h",
        "hour": "h",
        "hours": "h",
    }
    if unit not in aliases:
        raise ValueError(f"Unsupported timeframe unit: {timeframe}")
    return f"{value}{aliases[unit]}"


def periods_per_year_for_timeframe(timeframe: str) -> int:
    """Bars per year for a given bar size, so calculate_sharpe_ratio annualizes
    correctly regardless of whether the equity curve is built from daily or
    intraday bars. '1d' -> 252; 'Nm'/'Nh' -> 252 * (bars per regular session)."""
    timeframe = normalize_timeframe(timeframe)
    if timeframe == "1d":
        return 252
    unit, value = timeframe[-1], int(timeframe[:-1])
    if unit == "m":
        bars_per_day = _SESSION_MINUTES / value
    elif unit == "h":
        bars_per_day = (_SESSION_MINUTES / 60) / value
    else:
        raise ValueError(f"Unsupported timeframe for annualization: {timeframe}")
    return round(252 * bars_per_day)


class RiskMetrics:
    @staticmethod
    def calculate_sharpe_ratio(returns: List[float], risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
        returns_array = np.array(returns)
        if len(returns_array) == 0:
            return 0.0
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        std_dev = np.std(excess_returns)
        return float(np.mean(excess_returns) / std_dev * np.sqrt(periods_per_year)) if std_dev != 0 else 0.0

    @staticmethod
    def calculate_max_drawdown(equity_curve: pd.DataFrame) -> float:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return 0.0
        running_max = equity_curve["equity"].expanding().max()
        drawdown = (equity_curve["equity"] - running_max) / running_max
        return float(drawdown.min())

    @staticmethod
    def calculate_returns(equity_curve: pd.DataFrame) -> List[float]:
        if equity_curve.empty or "equity" not in equity_curve.columns:
            return []
        equity = equity_curve["equity"].values
        return (np.diff(equity) / equity[:-1]).tolist()

    @staticmethod
    def calculate_total_return(initial: float, final: float) -> float:
        return ((final - initial) / initial) * 100 if initial != 0 else 0.0

    @staticmethod
    def calculate_win_rate(trades: List) -> float:
        if len(trades) == 0:
            return 0.0
        winners = sum(1 for trade in trades if trade.realized_pnl > 0)
        return (winners / len(trades)) * 100

    @staticmethod
    def calculate_profit_factor(trades: List) -> float:
        gross_profit = sum(trade.realized_pnl for trade in trades if trade.realized_pnl > 0)
        gross_loss = abs(sum(trade.realized_pnl for trade in trades if trade.realized_pnl < 0))
        return gross_profit / gross_loss if gross_loss != 0 else (0.0 if gross_profit == 0 else float('inf'))

    @staticmethod
    def calculate_sortino_ratio(returns: List[float], risk_free_rate: float = 0.02,
                                periods_per_year: int = 252) -> float:
        """Sharpe's downside-only cousin: excess return over the *downside* deviation.

        Upside volatility is not penalized, so a strategy is not punished for
        large winning bars. Downside deviation is the root-mean-square of the
        negative excess returns taken over the whole sample (target
        semideviation with the risk-free rate as target), i.e. periods with
        non-negative excess returns contribute zero to the sum but still count
        in the denominator. Returns 0.0 when there is no data or no downside
        deviation at all, mirroring the Sharpe guard."""
        returns_array = np.array(returns, dtype=float)
        if returns_array.size == 0:
            return 0.0
        excess_returns = returns_array - (risk_free_rate / periods_per_year)
        downside = np.minimum(excess_returns, 0.0)
        downside_deviation = float(np.sqrt(np.mean(downside ** 2)))
        mean_excess = float(np.mean(excess_returns))
        if downside_deviation == 0 or not np.isfinite(downside_deviation) or not np.isfinite(mean_excess):
            return 0.0
        return float(mean_excess / downside_deviation * np.sqrt(periods_per_year))

    @staticmethod
    def calculate_calmar_ratio(total_return_pct: float, max_drawdown: float) -> float:
        """Return per unit of worst peak-to-trough pain.

        ``total_return_pct`` is a percentage (e.g. 20.0 for +20%) and
        ``max_drawdown`` is the signed fraction produced by
        calculate_max_drawdown (e.g. -0.25 for a 25% drawdown). No further
        annualization is applied here: the caller supplies the return figure it
        wants measured, which keeps this metric immune to the sqrt(periods)
        inflation that distorts intraday Sharpe ratios. Returns 0.0 when there
        is no drawdown to divide by, or when either input is not finite (a
        degenerate equity curve can make max drawdown NaN, and a NaN leaking
        into an optimizer's sort key is worse than a zero)."""
        total_return_pct = float(total_return_pct)
        denominator = abs(float(max_drawdown)) * 100.0
        if denominator == 0 or not np.isfinite(denominator) or not np.isfinite(total_return_pct):
            return 0.0
        return total_return_pct / denominator

    @staticmethod
    def calculate_avg_win_loss(trades: List) -> Dict:
        """Average winning trade, average losing trade and their ratio.

        ``avg_loss`` keeps its natural negative sign; ``win_loss_ratio`` uses
        its magnitude. Any of the three is 0.0 when the corresponding sample is
        empty (no trades, no winners, or no losers), so an all-winners run
        reports a 0.0 ratio rather than dividing by zero."""
        wins = [trade.realized_pnl for trade in trades if trade.realized_pnl > 0]
        losses = [trade.realized_pnl for trade in trades if trade.realized_pnl < 0]
        avg_win = float(np.mean(wins)) if wins else 0.0
        avg_loss = float(np.mean(losses)) if losses else 0.0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
        return {"avg_win": avg_win, "avg_loss": avg_loss, "win_loss_ratio": win_loss_ratio}

    @staticmethod
    def calculate_max_consecutive_losses(trades: List) -> int:
        """Longest run of consecutive losing trades, in list order.

        Break-even trades (realized_pnl == 0) are not losses and therefore break
        a losing streak. Returns 0 for an empty list or a run with no losers."""
        longest = 0
        current = 0
        for trade in trades:
            if trade.realized_pnl < 0:
                current += 1
                longest = max(longest, current)
            else:
                current = 0
        return int(longest)

    @staticmethod
    def calculate_exposure_time(equity_curve: pd.DataFrame, trades: List) -> float:
        """Percentage of equity-curve bars during which at least one position was open.

        A trade is held over the bars in [entry_date, exit_date): the position is
        opened on the close of its entry bar and closed on the close of its exit
        bar, so the exit bar itself is not counted as held. Overlapping trades in
        different symbols are counted once (this is time in the market, not a sum
        of per-position exposures). Returns 0.0 with no bars, no trades, or an
        equity curve lacking the 'date' column."""
        if equity_curve is None or equity_curve.empty or "date" not in equity_curve.columns:
            return 0.0
        bar_dates = pd.to_datetime(pd.Series(equity_curve["date"].to_numpy()))
        if len(bar_dates) == 0 or not trades:
            return 0.0
        exposed = np.zeros(len(bar_dates), dtype=bool)
        for trade in trades:
            entry_date = getattr(trade, "entry_date", None)
            exit_date = getattr(trade, "exit_date", None)
            if entry_date is None or exit_date is None:
                continue
            exposed |= ((bar_dates >= entry_date) & (bar_dates < exit_date)).to_numpy()
        return float(exposed.sum()) / len(bar_dates) * 100.0

    @staticmethod
    def calculate_metrics_summary(trades: List, equity_curve: pd.DataFrame, initial_capital: float,
                                  periods_per_year: int = 252) -> Dict:
        final_equity = equity_curve["equity"].iloc[-1] if not equity_curve.empty else initial_capital
        returns = RiskMetrics.calculate_returns(equity_curve)
        total_return_pct = RiskMetrics.calculate_total_return(initial_capital, final_equity)
        max_drawdown = RiskMetrics.calculate_max_drawdown(equity_curve)
        win_loss = RiskMetrics.calculate_avg_win_loss(trades)
        return {
            "total_trades": len(trades),
            "total_return_pct": total_return_pct,
            "sharpe_ratio": RiskMetrics.calculate_sharpe_ratio(returns, periods_per_year=periods_per_year),
            "max_drawdown": max_drawdown,
            "win_rate_pct": RiskMetrics.calculate_win_rate(trades),
            "profit_factor": RiskMetrics.calculate_profit_factor(trades),
            "final_equity": final_equity,
            "total_pnl": final_equity - initial_capital,
            "sortino_ratio": RiskMetrics.calculate_sortino_ratio(returns, periods_per_year=periods_per_year),
            "calmar_ratio": RiskMetrics.calculate_calmar_ratio(total_return_pct, max_drawdown),
            "avg_win": win_loss["avg_win"],
            "avg_loss": win_loss["avg_loss"],
            "win_loss_ratio": win_loss["win_loss_ratio"],
            "max_consecutive_losses": RiskMetrics.calculate_max_consecutive_losses(trades),
            "exposure_time_pct": RiskMetrics.calculate_exposure_time(equity_curve, trades),
        }
