"""Command-line interface for QuantifyYouth Trading System."""

from __future__ import annotations

import argparse

from backtester import BacktestConfig, run_backtest
from data_loader import load_csv_data, load_historical_data
from reporting import format_metrics, plot_equity_curve, save_equity_curve
from strategies import moving_average_crossover


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Backtest a moving-average crossover strategy on market data."
    )
    parser.add_argument("--ticker", default="AAPL", help="Ticker symbol to download with yfinance.")
    parser.add_argument("--csv", help="Optional local CSV file with OHLCV data.")
    parser.add_argument("--period", default="2y", help="Download period, such as 6mo, 1y, 2y, or 5y.")
    parser.add_argument("--interval", default="1d", help="Download interval, such as 1d, 1wk, or 1mo.")
    parser.add_argument("--fast", type=int, default=20, help="Fast moving average window.")
    parser.add_argument("--slow", type=int, default=50, help="Slow moving average window.")
    parser.add_argument("--cash", type=float, default=10_000.0, help="Initial portfolio cash.")
    parser.add_argument(
        "--transaction-cost",
        type=float,
        default=0.001,
        help="Cost per position change. 0.001 means 0.1%%.",
    )
    parser.add_argument("--allow-short", action="store_true", help="Allow short positions.")
    parser.add_argument("--save-csv", help="Optional path for the equity-curve CSV.")
    parser.add_argument("--plot", help="Optional path for a PNG equity-curve chart.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    data = load_csv_data(args.csv) if args.csv else load_historical_data(
        args.ticker,
        period=args.period,
        interval=args.interval,
    )
    signal = moving_average_crossover(
        data,
        fast_window=args.fast,
        slow_window=args.slow,
        long_only=not args.allow_short,
    )
    result = run_backtest(
        data,
        signal,
        BacktestConfig(initial_cash=args.cash, transaction_cost=args.transaction_cost),
    )

    print(format_metrics(result.metrics))

    if args.save_csv:
        csv_path = save_equity_curve(result.equity_curve, args.save_csv)
        print(f"\nSaved equity curve to {csv_path}")

    if args.plot:
        plot_path = plot_equity_curve(result.equity_curve, args.plot)
        print(f"Saved chart to {plot_path}")


if __name__ == "__main__":
    main()
