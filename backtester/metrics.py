import numpy as np
import pandas as pd
from typing import Dict, List


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
    def calculate_metrics_summary(trades: List, equity_curve: pd.DataFrame, initial_capital: float) -> Dict:
        final_equity = equity_curve["equity"].iloc[-1] if not equity_curve.empty else initial_capital
        returns = RiskMetrics.calculate_returns(equity_curve)
        return {
            "total_trades": len(trades),
            "total_return_pct": RiskMetrics.calculate_total_return(initial_capital, final_equity),
            "sharpe_ratio": RiskMetrics.calculate_sharpe_ratio(returns),
            "max_drawdown": RiskMetrics.calculate_max_drawdown(equity_curve),
            "win_rate_pct": RiskMetrics.calculate_win_rate(trades),
            "profit_factor": RiskMetrics.calculate_profit_factor(trades),
            "final_equity": final_equity,
            "total_pnl": final_equity - initial_capital,
        }
