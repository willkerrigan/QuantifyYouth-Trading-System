"""Fail-fast validation for backtest / optimizer / walk-forward YAML configs.

A malformed config used to surface as a ``KeyError`` deep inside a run, or --
worse -- as a run that completed and produced plausible-looking but invalid
numbers. Two real incidents motivated this module:

* a config with no ``backtest.in_sample_end_date`` let the optimizer search the
  entire date range, so every "out-of-sample" result was contaminated by
  look-ahead bias;
* a ``data.timeframe`` string the code cannot parse was silently ignored, so
  Sharpe ratios were annualized with the wrong bars-per-year.

``validate_config`` therefore splits problems in two:

* **hard errors** -- anything that makes the run invalid or impossible. These
  raise :class:`ConfigValidationError`, which lists *every* problem found
  (rather than only the first) so a broken config can be fixed in one pass.
* **warnings** -- returned as a list of strings for the caller to log. These are
  things that are probably a mistake but still produce a well-defined run.

Values are always echoed back in the message so the offending key *and* the
value found are both obvious.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List, Optional

__all__ = ["ConfigValidationError", "validate_config"]

# Sections every entry point dereferences directly.
_REQUIRED_SECTIONS = ("backtest", "data", "strategy")

_DATE_FORMATS = ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S")

# Yahoo only serves a short trailing window of intraday bars; see
# backtester/data_loader.py for the download path this affects.
_YAHOO_INTRADAY_RETENTION_DAYS = 60


class ConfigValidationError(ValueError):
    """Raised when a config contains errors that would invalidate a run."""

    def __init__(self, errors):
        self.errors: List[str] = list(errors)
        if len(self.errors) == 1:
            message = f"Invalid config: {self.errors[0]}"
        else:
            body = "\n".join(f"  - {err}" for err in self.errors)
            message = f"Invalid config ({len(self.errors)} problems found):\n{body}"
        super().__init__(message)


def _describe(value: Any) -> str:
    """Render a value for an error message, with its type when that's the point."""
    return f"{value!r} (type {type(value).__name__})"


def _is_number(value: Any) -> bool:
    # bool is an int subclass in Python; `commission: true` is not a number.
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (int, float, str, bool))


def _parse_date(value: Any) -> Optional[dt.datetime]:
    """Parse a config date, or return None if it cannot be parsed.

    PyYAML turns an unquoted ``2024-01-01`` into a ``datetime.date``, so both the
    quoted-string and native-date spellings have to be accepted.
    """
    if isinstance(value, dt.datetime):
        return value
    if isinstance(value, dt.date):
        return dt.datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        text = value.strip()
        for fmt in _DATE_FORMATS:
            try:
                return dt.datetime.strptime(text, fmt)
            except ValueError:
                continue
    return None


def _timeframe_problem(timeframe: Any) -> Optional[str]:
    """Return a description of why `timeframe` is unusable, or None if it's fine.

    The set of accepted strings is not hardcoded here: it is whatever
    ``backtester.metrics.periods_per_year_for_timeframe`` can actually turn into
    a bars-per-year figure. Anything it rejects would either crash a run or be
    silently annualized wrong.
    """
    from backtester.metrics import periods_per_year_for_timeframe

    if not isinstance(timeframe, str):
        return "must be a string such as '1d', '15m' or '1h'"
    try:
        periods = periods_per_year_for_timeframe(timeframe)
    except (ValueError, IndexError, TypeError, ZeroDivisionError):
        return "is not a timeframe this code can parse; supported forms include '1d', '15m', '15min', '1h' and '1hour'"
    if periods <= 0:
        return "is too coarse to annualize (it works out to zero bars per year)"
    return None


def _is_intraday(timeframe: Any) -> bool:
    if not isinstance(timeframe, str):
        return False
    from backtester.metrics import normalize_timeframe

    try:
        return normalize_timeframe(timeframe).endswith(("m", "h"))
    except ValueError:
        return False


def _check_backtest(backtest: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    capital = backtest.get("initial_capital")
    if "initial_capital" not in backtest:
        errors.append("backtest.initial_capital is required but missing")
    elif not _is_number(capital):
        errors.append(f"backtest.initial_capital must be a positive number, found {_describe(capital)}")
    elif capital <= 0:
        errors.append(f"backtest.initial_capital must be a positive number, found {capital!r}")

    parsed: Dict[str, dt.datetime] = {}
    for key in ("start_date", "end_date", "in_sample_end_date"):
        raw = backtest.get(key)
        if key not in backtest or raw is None:
            if key == "in_sample_end_date":
                errors.append(
                    "backtest.in_sample_end_date is required but missing; without it the optimizer "
                    "searches the entire date range and every out-of-sample result is contaminated "
                    "by look-ahead bias"
                )
            else:
                errors.append(f"backtest.{key} is required but missing")
            continue
        date = _parse_date(raw)
        if date is None:
            errors.append(f"backtest.{key} is not a parseable date, found {_describe(raw)}; expected 'YYYY-MM-DD'")
        else:
            parsed[key] = date

    start, end = parsed.get("start_date"), parsed.get("end_date")
    in_sample = parsed.get("in_sample_end_date")

    if start is not None and end is not None and start >= end:
        errors.append(
            f"backtest.start_date ({backtest.get('start_date')!r}) must be strictly before "
            f"backtest.end_date ({backtest.get('end_date')!r})"
        )

    if in_sample is not None:
        if start is not None and in_sample <= start:
            errors.append(
                f"backtest.in_sample_end_date ({backtest.get('in_sample_end_date')!r}) must be strictly after "
                f"backtest.start_date ({backtest.get('start_date')!r}); there is no in-sample window to optimize on"
            )
        if end is not None and in_sample >= end:
            errors.append(
                f"backtest.in_sample_end_date ({backtest.get('in_sample_end_date')!r}) must be strictly before "
                f"backtest.end_date ({backtest.get('end_date')!r}); as configured there is no out-of-sample "
                f"window at all, so the optimizer would see every bar it is later validated on (look-ahead bias)"
            )

    for key, warn_above in (("commission", 0.01), ("slippage", None)):
        if key not in backtest or backtest.get(key) is None:
            continue
        value = backtest[key]
        if not _is_number(value):
            errors.append(f"backtest.{key} must be a number >= 0, found {_describe(value)}")
        elif value < 0:
            errors.append(f"backtest.{key} must be a number >= 0, found {value!r}")
        elif warn_above is not None and value > warn_above:
            warnings.append(
                f"backtest.{key} is {value!r}, i.e. {value * 100:.4g}% per side, which is implausibly high; "
                f"this is usually a units mistake (write 0.01 for 1%, not 1)"
            )


def _check_data(data: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    symbols = data.get("symbols")
    if "symbols" not in data:
        errors.append("data.symbols is required but missing")
    elif not isinstance(symbols, (list, tuple)):
        errors.append(f"data.symbols must be a non-empty list of ticker strings, found {_describe(symbols)}")
    elif len(symbols) == 0:
        errors.append("data.symbols must be a non-empty list of ticker strings, found an empty list")
    else:
        bad = [s for s in symbols if not isinstance(s, str) or not s.strip()]
        if bad:
            errors.append(f"data.symbols must contain only non-empty ticker strings, found {bad!r} in {symbols!r}")

    timeframe = data.get("timeframe")
    if timeframe is not None:
        problem = _timeframe_problem(timeframe)
        if problem:
            errors.append(f"data.timeframe {problem}, found {_describe(timeframe)}")

    source = data.get("source")
    if source is not None and not isinstance(source, str):
        errors.append(f"data.source must be a string, found {_describe(source)}")
    elif isinstance(source, str) and source.lower() == "yahoo" and _is_intraday(timeframe):
        warnings.append(
            f"data.source is 'yahoo' with intraday data.timeframe {timeframe!r}: Yahoo only retains about "
            f"{_YAHOO_INTRADAY_RETENTION_DAYS} days of intraday history, so the configured "
            f"backtest.start_date cannot be honored and the run will silently cover a much shorter window"
        )


def _check_strategy(strategy: Dict[str, Any], errors: List[str], warnings: List[str]) -> None:
    from backtester.strategies import STRATEGIES

    name = strategy.get("name")
    if "name" not in strategy:
        errors.append(f"strategy.name is required but missing; available strategies: {sorted(STRATEGIES)}")
    elif not isinstance(name, str):
        errors.append(f"strategy.name must be a string, found {_describe(name)}")
    elif name not in STRATEGIES:
        errors.append(f"strategy.name {name!r} is not a known strategy; available strategies: {sorted(STRATEGIES)}")

    if "parameters" not in strategy:
        errors.append("strategy.parameters is required but missing (use an empty mapping {} for a strategy with no knobs)")
        return
    parameters = strategy["parameters"]
    if not isinstance(parameters, dict):
        errors.append(f"strategy.parameters must be a mapping of parameter name to value or list of values, "
                      f"found {_describe(parameters)}")
        return

    for param, value in parameters.items():
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                warnings.append(
                    f"strategy.parameters.{param} is an empty grid ([]), so this parameter contributes nothing "
                    f"to the search and the strategy default will be used"
                )
            else:
                bad = [v for v in value if not _is_scalar(v)]
                if bad:
                    errors.append(
                        f"strategy.parameters.{param} grid must contain only scalar values, found {bad!r} in {value!r}"
                    )
        elif not _is_scalar(value):
            errors.append(
                f"strategy.parameters.{param} must be a scalar or a list of scalars (a grid), found {_describe(value)}"
            )


def validate_config(config: Dict[str, Any]) -> List[str]:
    """Validate a loaded config dict.

    Returns a list of human-readable warning strings for soft problems (the
    caller should log them). Raises :class:`ConfigValidationError` -- listing
    every hard problem found -- for anything that would make the run invalid.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError([f"config must be a mapping of sections, found {_describe(config)}"])

    errors: List[str] = []
    warnings: List[str] = []

    sections: Dict[str, Dict[str, Any]] = {}
    for name in _REQUIRED_SECTIONS:
        if name not in config:
            errors.append(f"'{name}' section is required but missing from the config")
        elif not isinstance(config[name], dict):
            errors.append(f"'{name}' section must be a mapping, found {_describe(config[name])}")
        else:
            sections[name] = config[name]

    if "backtest" in sections:
        _check_backtest(sections["backtest"], errors, warnings)
    if "data" in sections:
        _check_data(sections["data"], errors, warnings)
    if "strategy" in sections:
        _check_strategy(sections["strategy"], errors, warnings)

    # risk.max_position_size and risk.max_leverage are enforced by BacktestEngine;
    # validate their shape here so a typo is caught before a run silently falls
    # back to the defaults mid-backtest.
    risk = config.get("risk")
    if risk is not None:
        if not isinstance(risk, dict):
            errors.append(f"'risk' must be a mapping, got {type(risk).__name__}: {risk!r}")
        else:
            for key in ("max_position_size", "max_leverage"):
                value = risk.get(key)
                if value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    warnings.append(f"risk.{key} is {value!r} (not a number); the engine will "
                                    f"ignore it and use its default")
                elif value <= 0:
                    warnings.append(f"risk.{key} is {value}, which is not a positive fraction; "
                                    f"the engine will ignore it and use its default")

    if errors:
        raise ConfigValidationError(errors)
    return warnings
