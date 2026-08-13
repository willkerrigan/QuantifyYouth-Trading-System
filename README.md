# QuantifyYouth Trading System

A production-grade algorithmic trading platform with parameter optimization, live execution, and comprehensive risk analysis.

## Features

- **Backtesting Engine**: Historical performance analysis with precise trade simulation
- **Parameter Optimizer**: Grid search with Sharpe ratio scoring to find optimal strategy parameters
- **Raw Trade Logs**: Detailed CSV export of all simulated trades for Risk Officer review
- **Live Execution**: Alpaca broker integration for paper/live trading with signal-driven execution
- **Risk Metrics**: Equity curve tracking, drawdown analysis, risk-adjusted returns

## Quick Start

### Installation

```bash
git clone https://github.com/willkerrigan/QuantifyYouth-Trading-System.git
cd QuantifyYouth-Trading-System
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configuration

1. Copy config templates and edit:
```bash
cp config/backtest_config.example.yaml config/backtest_config.yaml
cp config/broker_config.example.yaml config/broker_config.yaml
```

## Usage

### Run Parameter Optimization

```bash
python scripts/optimize_params.py --config config/backtest_config.yaml --output output/
```

This tests all parameter combinations and exports:
- `trade_log_rank1.csv` - Raw trade log with Date, Asset, Entry/Exit Price, PnL
- `equity_curve_rank1.csv` - Equity curve over time
- `optimization_summary.json` - Best parameter sets

### Run Single Backtest

```bash
python scripts/run_backtest.py --config config/backtest_config.yaml
```

### Start Live Trading (Paper)

```bash
python scripts/run_live_trading.py --config config/broker_config.yaml
```

## Project Structure

- **backtester/** - Core backtest engine with trade logging
- **optimizer/** - Parameter grid search with parallel processing
- **execution/** - Live trading with Alpaca integration
- **config/** - Example configuration templates
- **scripts/** - Entry points for backtest, optimization, and live trading
- **output/** - Generated CSV files for Risk Officer

## Key Components

### 1. Parameter Optimization
- Grid search over all parameter combinations
- Scores each by Sharpe ratio, total return, or max drawdown
- Runs backtests in parallel for speed
- Exports top-N parameter sets

### 2. Trade Log Export
- Every simulated trade captured:
  - Date, Entry Time, Asset
  - Entry Price, Exit Price, Position Size
  - Realized PnL, PnL %
- Automatically exported to CSV for Risk review

### 3. Live Execution
- Connects to Alpaca broker
- Processes BUY/SELL signals
- Paper trading enabled by default
- Position management and risk limits

## Development

```bash
pytest tests/ -v --cov
```
