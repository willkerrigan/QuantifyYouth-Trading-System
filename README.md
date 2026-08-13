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

**Never commit `config/broker_config.yaml` (or any non-`.example` file under `config/`) once it
contains real Alpaca `api_key`/`secret_key` values.** These files are excluded via `.gitignore`
(`config/*.yaml` / `config/*.yml`, with the `.example` templates explicitly re-included), so a
plain `git add -A` will not stage them. Still, double-check with `git status` before committing,
and prefer sourcing keys from environment variables or a secrets manager over checking in any
filled-in YAML at all. If real keys are ever committed, treat them as compromised: rotate them in
the Alpaca dashboard immediately, even after removing them from history.

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

### Run Out-of-Sample Test

`config.backtest.in_sample_end_date` walls off a held-out window: the optimizer
above only ever sees data up through that date. Once parameters are chosen,
run them once against the untouched remainder to check for overfitting:

```bash
python scripts/run_out_of_sample.py --config config/backtest_config.yaml \
  --params output/optimization_summary_<timestamp>.json --output output/
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

### 4. Pre-Trade Risk Rails (`execution/risk_guard.py`)

Checked before every order, configured under `risk:` in the broker config:

- **Max daily loss** (`max_daily_loss_pct` / `max_daily_loss_amount`) — measured
  against equity at session start, so realized and unrealized P&L both count.
  A breach halts new entries and **leaves open positions alone**; set
  `liquidate_on_daily_loss: true` to opt into auto-flattening. Omit the keys to
  disable.
- **Kill switch** — programmatic (`LiveTrader.halt()`) or operational (create the
  file at `risk.kill_switch_file`, polled each cycle). It **latches**: remove the
  file *and* call `LiveTrader.reset_kill_switch()` to resume.
- **Market-hours gate — ON BY DEFAULT once a `risk:` section exists.** This is
  the one rail you get without asking for it: declaring `risk:` (even empty)
  arms it, and outside weekdays 09:30–16:00 `America/New_York` the trader skips
  the whole poll cycle. Opt out with `risk.market_hours.enabled: false` or
  `allow_outside_hours: true`. A config with **no** `risk:` section keeps the
  old always-on behaviour so pre-existing deployments don't start halting
  silently. The timezone is resolved with `zoneinfo`, so US DST is handled —
  never replace it with a fixed UTC offset. **Market holidays are NOT tracked:**
  the gate is weekday + time-of-day only, so Thanksgiving, Good Friday and 13:00
  early closes all read as "open". A real calendar needs an exchange data source.

## Development

### Install dev dependencies

Install the pinned runtime dependencies, then install the project itself in
editable mode. The editable install is what makes `backtester`, `optimizer`,
and `execution` importable from anywhere — no more `PYTHONPATH=.` prefixes on
test or script invocations.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # pinned runtime dependencies
pip install -e ".[dev]"           # editable install + pytest / pre-commit
```

### Run tests

```bash
pytest tests/ -v
pytest tests/ -v --cov       # with coverage
```

The test suite is hermetic: it needs no network access and no broker API keys.
Keep it that way — stub out data sources rather than calling live endpoints.
CI (`.github/workflows/ci.yml`) runs the same command on every push and pull
request.

### Enable pre-commit hooks

The hooks exist mainly to keep Alpaca/exchange credentials out of this public
repo (`gitleaks` secret scanning plus `detect-private-key`), along with basic
whitespace and YAML/TOML validity checks.

```bash
pip install pre-commit
pre-commit install              # run automatically on every `git commit`
pre-commit run --all-files      # optional: scan the whole tree once
```
