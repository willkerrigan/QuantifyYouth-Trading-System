# QuantifyYouth Trading System

A beginner-friendly quant trading project that downloads price data, creates technical-analysis signals, runs a vectorized backtest, and reports strategy performance against a buy-and-hold benchmark.

This project is for education and research only. It is not financial advice and it does not place live trades.

## Features

- Yahoo Finance data loading through `yfinance`
- Local CSV loading for offline backtests
- Simple moving averages, exponential moving averages, and RSI
- Moving-average crossover strategy
- Close-to-close vectorized backtester with transaction costs
- Performance metrics including return, volatility, Sharpe ratio, drawdown, win rate, exposure, and trade count
- Optional CSV and PNG report outputs
- Unit tests that run without internet access

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Quick Start

Run a two-year Apple backtest with the default 20/50 moving-average crossover:

```bash
python cli.py --ticker AAPL
```

Save the equity curve and chart:

```bash
python cli.py --ticker SPY --period 5y --fast 50 --slow 200 --save-csv outputs/spy_equity.csv --plot outputs/spy_equity.png
```

Use a local CSV instead of downloading data:

```bash
python cli.py --csv data/prices.csv --fast 10 --slow 30
```

CSV files must include `Date` or `Datetime`, plus `Open`, `High`, `Low`, `Close`, and `Volume` columns. A saved dataframe index column with dates also works.

## Project Structure

```text
.
├── backtester.py      # Portfolio accounting and performance metrics
├── cli.py             # Command-line interface
├── data_loader.py     # yfinance and CSV loading
├── indicators.py      # SMA, EMA, RSI
├── reporting.py       # Text, CSV, and chart outputs
├── strategies.py      # Trading signals
└── tests/             # Offline unit tests
```

## Testing

```bash
python -m unittest discover
```

## Notes

The backtester shifts positions by one bar, so signals generated from today's close are applied to the next bar's return. That keeps the example from accidentally using future information.
