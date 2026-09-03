"""Formatting and charting helpers for backtest results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PERCENT_METRICS = {
    "total_return",
    "buy_hold_return",
    "annual_return",
    "annual_volatility",
    "max_drawdown",
    "win_rate",
    "exposure",
}


def format_metrics(metrics: dict[str, float]) -> str:
    """Return a terminal-friendly performance summary."""

    labels = {
        "total_return": "Total return",
        "buy_hold_return": "Buy/hold return",
        "annual_return": "Annualized return",
        "annual_volatility": "Annualized volatility",
        "sharpe_ratio": "Sharpe ratio",
        "max_drawdown": "Max drawdown",
        "win_rate": "Win rate",
        "exposure": "Market exposure",
        "number_of_trades": "Trades",
        "final_equity": "Final equity",
        "final_buy_hold_equity": "Final buy/hold equity",
        "market_correlation": "Market correlation",
    }

    lines = ["Backtest Summary", "----------------"]
    for key, label in labels.items():
        if key not in metrics:
            continue
        value = metrics[key]
        if key in PERCENT_METRICS:
            lines.append(f"{label:24} {value:>10.2%}")
        elif key.startswith("final_"):
            lines.append(f"{label:24} ${value:>10,.2f}")
        elif key == "number_of_trades":
            lines.append(f"{label:24} {value:>10.0f}")
        else:
            lines.append(f"{label:24} {value:>10.2f}")
    return "\n".join(lines)


def save_equity_curve(equity_curve: pd.DataFrame, path: str | Path) -> Path:
    """Save the full backtest equity curve to CSV."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    equity_curve.to_csv(output_path, index_label="Date")
    return output_path


def plot_equity_curve(equity_curve: pd.DataFrame, path: str | Path) -> Path:
    """Save a PNG chart comparing strategy equity to buy-and-hold equity."""

    import matplotlib.pyplot as plt

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    equity_curve[["equity", "buy_hold_equity"]].plot(ax=ax)
    ax.set_title("Strategy vs Buy-and-Hold")
    ax.set_ylabel("Portfolio value")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.25)
    ax.legend(["Strategy", "Buy and hold"])
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
