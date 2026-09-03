"""Tests for config/validation.py.

Every hard error and every warning documented in the module gets a case here,
plus a fully valid config that must pass cleanly with no warnings at all.
No network access is involved: validation is pure dict inspection.
"""

import copy
import datetime

import pytest

from config.validation import ConfigValidationError, validate_config


def valid_config():
    """A minimal config that must validate cleanly with zero warnings."""
    return {
        "backtest": {
            "start_date": "2023-01-01",
            "in_sample_end_date": "2025-06-30",
            "end_date": "2026-07-14",
            "initial_capital": 100000,
            "commission": 0.001,
            "slippage": 0.0002,
        },
        "data": {
            "source": "alpaca",
            "symbols": ["SPY", "QQQ"],
            "timeframe": "1d",
        },
        "strategy": {
            "name": "rsi2",
            "parameters": {"rsi_buy_threshold": [5, 10], "rsi_sell_threshold": 70},
        },
    }


def with_backtest(**overrides):
    config = valid_config()
    config["backtest"].update(overrides)
    return config


def errors_from(config):
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    return excinfo.value.errors


def only_error(config):
    errors = errors_from(config)
    assert len(errors) == 1, f"expected exactly one error, got {errors}"
    return errors[0]


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------

def test_valid_config_passes_with_no_warnings():
    assert validate_config(valid_config()) == []


def test_valid_config_accepts_native_yaml_dates():
    # Unquoted YAML dates arrive as datetime.date, not str.
    config = with_backtest(
        start_date=datetime.date(2023, 1, 1),
        in_sample_end_date=datetime.date(2025, 6, 30),
        end_date=datetime.date(2026, 7, 14),
    )
    assert validate_config(config) == []


def test_valid_config_accepts_intraday_timeframes():
    for timeframe in ("5m", "15m", "1h", "4h"):
        config = valid_config()
        config["data"]["timeframe"] = timeframe
        assert validate_config(config) == [], timeframe


def test_repo_example_config_is_valid():
    """The checked-in example config must not be rejected by our own rules."""
    import pathlib

    import yaml

    path = pathlib.Path(__file__).resolve().parents[1] / "config" / "backtest_config.example.yaml"
    with open(path) as handle:
        config = yaml.safe_load(handle)
    validate_config(config)  # must not raise


# --------------------------------------------------------------------------
# structural hard errors
# --------------------------------------------------------------------------

def test_non_mapping_config_is_an_error():
    assert "must be a mapping" in only_error(["not", "a", "dict"])


@pytest.mark.parametrize("section", ["backtest", "data", "strategy"])
def test_missing_required_section_is_an_error(section):
    config = valid_config()
    del config[section]
    assert f"'{section}' section is required" in only_error(config)


def test_section_of_wrong_type_is_an_error():
    config = valid_config()
    config["data"] = "SPY"
    assert "'data' section must be a mapping" in only_error(config)


def test_all_errors_are_reported_at_once():
    config = valid_config()
    config["backtest"]["initial_capital"] = -5
    config["data"]["symbols"] = []
    config["strategy"]["name"] = "nope"
    errors = errors_from(config)
    assert len(errors) == 3
    joined = "\n".join(errors)
    assert "initial_capital" in joined and "symbols" in joined and "nope" in joined


# --------------------------------------------------------------------------
# backtest.initial_capital
# --------------------------------------------------------------------------

def test_missing_initial_capital_is_an_error():
    config = valid_config()
    del config["backtest"]["initial_capital"]
    assert "backtest.initial_capital is required" in only_error(config)


@pytest.mark.parametrize("value", [0, -1000, "100000", None, True])
def test_non_positive_or_non_numeric_initial_capital_is_an_error(value):
    message = only_error(with_backtest(initial_capital=value))
    assert "backtest.initial_capital" in message
    assert repr(value) in message


# --------------------------------------------------------------------------
# backtest dates
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["start_date", "end_date"])
def test_missing_date_is_an_error(key):
    config = valid_config()
    del config["backtest"][key]
    assert f"backtest.{key} is required" in only_error(config)


@pytest.mark.parametrize("key", ["start_date", "end_date", "in_sample_end_date"])
def test_unparseable_date_is_an_error(key):
    message = only_error(with_backtest(**{key: "07/14/2026"}))
    assert f"backtest.{key} is not a parseable date" in message
    assert "'07/14/2026'" in message


def test_start_date_after_end_date_is_an_error():
    config = with_backtest(start_date="2026-01-01", end_date="2024-01-01", in_sample_end_date="2025-01-01")
    joined = "\n".join(errors_from(config))
    assert "backtest.start_date ('2026-01-01') must be strictly before backtest.end_date ('2024-01-01')" in joined


def test_start_date_equal_to_end_date_is_an_error():
    config = with_backtest(start_date="2024-01-01", end_date="2024-01-01", in_sample_end_date="2024-01-01")
    assert any("must be strictly before backtest.end_date" in err for err in errors_from(config))


# --- the look-ahead guard ---------------------------------------------------

def test_missing_in_sample_end_date_is_a_hard_error():
    config = valid_config()
    del config["backtest"]["in_sample_end_date"]
    message = only_error(config)
    assert "backtest.in_sample_end_date is required" in message
    assert "look-ahead" in message


def test_null_in_sample_end_date_is_a_hard_error():
    assert "backtest.in_sample_end_date is required" in only_error(with_backtest(in_sample_end_date=None))


def test_in_sample_end_date_equal_to_end_date_is_a_hard_error():
    message = only_error(with_backtest(in_sample_end_date="2026-07-14"))
    assert "backtest.in_sample_end_date ('2026-07-14')" in message
    assert "no out-of-sample window" in message


def test_in_sample_end_date_after_end_date_is_a_hard_error():
    message = only_error(with_backtest(in_sample_end_date="2027-01-01"))
    assert "must be strictly before backtest.end_date ('2026-07-14')" in message


def test_in_sample_end_date_at_or_before_start_date_is_a_hard_error():
    message = only_error(with_backtest(in_sample_end_date="2023-01-01"))
    assert "must be strictly after backtest.start_date ('2023-01-01')" in message
    assert "no in-sample window" in message


def test_in_sample_end_date_strictly_between_is_accepted():
    assert validate_config(with_backtest(in_sample_end_date="2023-01-02")) == []


# --------------------------------------------------------------------------
# commission / slippage
# --------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["commission", "slippage"])
@pytest.mark.parametrize("value", [-0.001, "0.001", True])
def test_negative_or_non_numeric_costs_are_errors(key, value):
    message = only_error(with_backtest(**{key: value}))
    assert f"backtest.{key} must be a number >= 0" in message
    assert repr(value) in message


def test_absent_costs_are_allowed():
    config = valid_config()
    del config["backtest"]["commission"]
    del config["backtest"]["slippage"]
    assert validate_config(config) == []


def test_zero_costs_are_allowed():
    assert validate_config(with_backtest(commission=0, slippage=0)) == []


def test_implausibly_high_commission_only_warns():
    warnings = validate_config(with_backtest(commission=1))
    assert len(warnings) == 1
    assert "backtest.commission is 1" in warnings[0]
    assert "units mistake" in warnings[0]


def test_commission_at_the_one_percent_boundary_does_not_warn():
    assert validate_config(with_backtest(commission=0.01)) == []


def test_high_slippage_does_not_warn():
    assert validate_config(with_backtest(slippage=0.5)) == []


# --------------------------------------------------------------------------
# data.symbols
# --------------------------------------------------------------------------

def test_missing_symbols_is_an_error():
    config = valid_config()
    del config["data"]["symbols"]
    assert "data.symbols is required" in only_error(config)


def test_empty_symbols_list_is_an_error():
    config = valid_config()
    config["data"]["symbols"] = []
    assert "found an empty list" in only_error(config)


def test_symbols_not_a_list_is_an_error():
    config = valid_config()
    config["data"]["symbols"] = "SPY"
    message = only_error(config)
    assert "data.symbols must be a non-empty list" in message
    assert "'SPY'" in message


def test_non_string_symbol_is_an_error():
    config = valid_config()
    config["data"]["symbols"] = ["SPY", 42, ""]
    message = only_error(config)
    assert "data.symbols must contain only non-empty ticker strings" in message
    assert "42" in message


# --------------------------------------------------------------------------
# data.timeframe -- validated against periods_per_year_for_timeframe
# --------------------------------------------------------------------------

@pytest.mark.parametrize("timeframe", ["1w", "daily", "", "15", "m", "abcm", 15])
def test_unsupported_timeframe_is_an_error(timeframe):
    config = valid_config()
    config["data"]["timeframe"] = timeframe
    message = only_error(config)
    assert "data.timeframe" in message
    assert repr(timeframe) in message


def test_zero_length_timeframe_is_an_error():
    config = valid_config()
    config["data"]["timeframe"] = "0m"
    assert "data.timeframe" in only_error(config)


def test_absent_timeframe_is_allowed():
    config = valid_config()
    del config["data"]["timeframe"]
    assert validate_config(config) == []


def test_every_accepted_timeframe_actually_annualizes():
    """The accepted set is exactly what backtester.metrics can parse."""
    from backtester.metrics import periods_per_year_for_timeframe

    for timeframe in ("1d", "1D", "1m", "5m", "30m", "15min", "1h", "1hour", "2h"):
        config = valid_config()
        config["data"]["timeframe"] = timeframe
        assert validate_config(config) == []
        assert periods_per_year_for_timeframe(timeframe) > 0


def test_yahoo_with_normalized_intraday_timeframe_warns_about_retention():
    config = valid_config()
    config["data"]["source"] = "yahoo"
    config["data"]["timeframe"] = "15 minutes"
    warnings = validate_config(config)
    assert len(warnings) == 1
    assert "60 days of intraday history" in warnings[0]


# --------------------------------------------------------------------------
# data.source
# --------------------------------------------------------------------------

def test_yahoo_with_intraday_timeframe_warns_about_retention():
    config = valid_config()
    config["data"]["source"] = "yahoo"
    config["data"]["timeframe"] = "15m"
    warnings = validate_config(config)
    assert len(warnings) == 1
    assert "yahoo" in warnings[0]
    assert "60 days of intraday history" in warnings[0]
    assert "start_date" in warnings[0]


def test_yahoo_with_daily_timeframe_does_not_warn():
    config = valid_config()
    config["data"]["source"] = "yahoo"
    config["data"]["timeframe"] = "1d"
    assert validate_config(config) == []


def test_non_yahoo_intraday_does_not_warn():
    config = valid_config()
    config["data"]["source"] = "alpaca"
    config["data"]["timeframe"] = "15m"
    assert validate_config(config) == []


def test_non_string_source_is_an_error():
    config = valid_config()
    config["data"]["source"] = ["yahoo"]
    assert "data.source must be a string" in only_error(config)


# --------------------------------------------------------------------------
# strategy
# --------------------------------------------------------------------------

def test_unknown_strategy_name_is_an_error_and_lists_known_ones():
    from backtester.strategies import STRATEGIES

    config = valid_config()
    config["strategy"]["name"] = "mean_reversion_9000"
    message = only_error(config)
    assert "'mean_reversion_9000' is not a known strategy" in message
    for known in STRATEGIES:
        assert known in message


def test_every_registered_strategy_name_is_accepted():
    from backtester.strategies import STRATEGIES

    for name in STRATEGIES:
        config = valid_config()
        config["strategy"]["name"] = name
        assert validate_config(config) == [], name


def test_missing_strategy_name_is_an_error():
    config = valid_config()
    del config["strategy"]["name"]
    assert "strategy.name is required" in only_error(config)


def test_non_string_strategy_name_is_an_error():
    config = valid_config()
    config["strategy"]["name"] = 2
    assert "strategy.name must be a string" in only_error(config)


def test_missing_parameters_is_an_error():
    config = valid_config()
    del config["strategy"]["parameters"]
    assert "strategy.parameters is required" in only_error(config)


def test_parameters_not_a_mapping_is_an_error():
    config = valid_config()
    config["strategy"]["parameters"] = ["rsi_buy_threshold"]
    message = only_error(config)
    assert "strategy.parameters must be a mapping" in message
    assert "['rsi_buy_threshold']" in message


def test_empty_parameters_mapping_is_allowed():
    config = valid_config()
    config["strategy"]["parameters"] = {}
    assert validate_config(config) == []


def test_empty_parameter_grid_only_warns():
    config = valid_config()
    config["strategy"]["parameters"] = {"rsi_buy_threshold": []}
    warnings = validate_config(config)
    assert len(warnings) == 1
    assert "strategy.parameters.rsi_buy_threshold is an empty grid" in warnings[0]


def test_non_scalar_parameter_value_is_an_error():
    config = valid_config()
    config["strategy"]["parameters"] = {"rsi_buy_threshold": {"low": 5}}
    message = only_error(config)
    assert "strategy.parameters.rsi_buy_threshold must be a scalar or a list of scalars" in message


def test_non_scalar_inside_a_grid_is_an_error():
    config = valid_config()
    config["strategy"]["parameters"] = {"rsi_buy_threshold": [5, [10]]}
    message = only_error(config)
    assert "strategy.parameters.rsi_buy_threshold grid must contain only scalar values" in message
    assert "[10]" in message


def test_scalar_parameters_are_allowed():
    config = valid_config()
    config["strategy"]["parameters"] = {"a": 5, "b": 1.5, "c": "wide", "d": True}
    assert validate_config(config) == []


# --------------------------------------------------------------------------
# the unread risk block
# --------------------------------------------------------------------------

def test_valid_risk_section_produces_no_warnings():
    """risk.max_position_size / max_leverage are enforced by the engine now."""
    config = valid_config()
    config["risk"] = {"max_position_size": 0.1, "max_leverage": 1.0}
    assert validate_config(config) == []


def test_risk_section_warns_on_unusable_values():
    config = valid_config()
    config["risk"] = {"max_position_size": "lots", "max_leverage": -2}
    warnings = validate_config(config)
    assert len(warnings) == 2
    assert any("max_position_size" in w for w in warnings)
    assert any("max_leverage" in w for w in warnings)


def test_non_mapping_risk_section_is_a_hard_error():
    config = valid_config()
    config["risk"] = "yes please"
    with pytest.raises(ConfigValidationError, match="must be a mapping"):
        validate_config(config)


# --------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------

def test_validate_config_does_not_mutate_the_input():
    config = valid_config()
    snapshot = copy.deepcopy(config)
    validate_config(config)
    assert config == snapshot


def test_error_message_lists_every_problem():
    config = valid_config()
    config["backtest"]["initial_capital"] = 0
    config["data"]["symbols"] = []
    with pytest.raises(ConfigValidationError) as excinfo:
        validate_config(config)
    text = str(excinfo.value)
    assert "2 problems found" in text
    assert "backtest.initial_capital" in text
    assert "data.symbols" in text


def test_config_validation_error_is_a_value_error():
    assert issubclass(ConfigValidationError, ValueError)


def test_multiple_warnings_are_all_returned():
    config = valid_config()
    config["backtest"]["commission"] = 1
    config["strategy"]["parameters"] = {"rsi_buy_threshold": []}
    config["risk"] = {"max_leverage": "not-a-number"}
    assert len(validate_config(config)) == 3
