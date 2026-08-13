"""Pre-trade safety rails: kill switch, max daily loss, and a market-hours gate.

These are the checks a fund cannot run live without. They are deliberately
independent of the broker: :class:`RiskGuard` only ever reads an account dict
(the shape :meth:`execution.broker_adapter.BrokerAdapter.get_account` returns)
and a caller-supplied ``now``. Nothing here touches the network, and no clock is
read unless the caller declines to supply one, so every rail is unit-testable.

Three rails:

1. :class:`DailyLossGuard` - latches once the session's loss exceeds a configured
   fraction of starting equity and/or an absolute amount. Latched means latched:
   an equity rebound within the same session does NOT re-arm trading, because a
   limit that un-trips is not a limit.
2. :class:`KillSwitch` - programmatic (:meth:`KillSwitch.trip`) and operational
   (a file path polled each cycle). Also latching; only an explicit
   :meth:`KillSwitch.reset` clears it.
3. :class:`MarketHoursGate` - weekday + time-of-day gate for US equity regular
   hours. See its docstring for what it does NOT cover (holidays).

Configuration lives under the ``risk`` key of the broker config. Every limit is
opt-in *except* the market-hours gate: declaring a ``risk:`` section at all turns
it on, because trading into a closed market is never intentional. A config with
no ``risk:`` section keeps the legacy always-on behaviour, so an existing
deployment does not silently change the moment this module ships::

    risk:
      max_daily_loss_pct: 0.02        # omit to disable
      max_daily_loss_amount: 5000     # omit to disable
      liquidate_on_daily_loss: false  # opt-in; default is halt-new-entries only
      kill_switch_file: "run/KILL"    # omit to disable the file switch
      market_hours:
        enabled: true                 # default true
        allow_outside_hours: false    # default false
        timezone: "America/New_York"
        open: "09:30"
        close: "16:00"
"""

import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, time as dtime, timezone
from typing import Dict, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

DEFAULT_MARKET_TIMEZONE = "America/New_York"
DEFAULT_MARKET_OPEN = dtime(9, 30)
DEFAULT_MARKET_CLOSE = dtime(16, 0)

# Reason codes carried on a GuardDecision so callers can branch without
# string-matching log messages.
REASON_KILL_SWITCH = "kill_switch"
REASON_DAILY_LOSS = "max_daily_loss"
REASON_EQUITY_UNKNOWN = "equity_unknown"
REASON_MARKET_CLOSED = "market_closed"


@dataclass(frozen=True)
class GuardDecision:
    """Outcome of a risk check. Falsy when trading must not proceed."""

    allowed: bool
    code: Optional[str] = None
    reason: Optional[str] = None

    def __bool__(self) -> bool:
        return self.allowed


ALLOWED = GuardDecision(True)


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #

def _optional_positive_float(raw, name: str) -> Optional[float]:
    """Parse an optional positive limit. Returns None when unset or unusable.

    An unusable value is logged and treated as *absent* rather than raising:
    a typo in a risk limit must not stop the whole trading process, but it must
    be loud enough to notice.
    """
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("Ignoring risk.%s=%r: not a number", name, raw)
        return None
    if not math.isfinite(value) or value <= 0:
        logger.warning("Ignoring risk.%s=%r: must be a finite value > 0", name, raw)
        return None
    return value


def _parse_time(raw, default: dtime, name: str) -> dtime:
    """Parse an ``HH:MM`` string (or a datetime.time) into a time-of-day."""
    if raw is None:
        return default
    if isinstance(raw, dtime):
        return raw
    try:
        parts = str(raw).strip().split(":")
        hour, minute = int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
        return dtime(hour, minute)
    except (ValueError, IndexError):
        logger.warning("Ignoring risk.market_hours.%s=%r: expected HH:MM", name, raw)
        return default


def _equity_from_account(account: Optional[Dict]) -> Optional[float]:
    """Mark-to-market equity from an account dict, or None if unreadable.

    ``equity`` already folds in realized and unrealized P&L, so a single reading
    covers both halves of "realized+unrealized loss for the day".
    """
    if not account:
        return None
    for key in ("equity", "portfolio_value"):
        if key not in account:
            continue
        try:
            value = float(account[key])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


# --------------------------------------------------------------------------- #
# kill switch
# --------------------------------------------------------------------------- #

class KillSwitch:
    """Latching halt, trippable in code or by creating a file on disk.

    Tripping never interrupts an in-flight order: callers poll it between
    cycles and before submitting a new one, so the worst case is that an
    already-submitted order completes. Once tripped it stays tripped until
    :meth:`reset` is called explicitly - a switch that silently re-arms itself
    is worse than no switch at all.
    """

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = str(file_path) if file_path else None
        self._tripped = False
        self._reason: Optional[str] = None

    @property
    def tripped(self) -> bool:
        return self._tripped

    @property
    def reason(self) -> Optional[str]:
        return self._reason

    def trip(self, reason: str = "tripped programmatically") -> None:
        if self._tripped:
            return
        self._tripped = True
        self._reason = reason
        logger.error("!" * 60)
        logger.error("!!! KILL SWITCH TRIPPED: %s", reason)
        logger.error("!!! No new positions will be opened until it is reset.")
        logger.error("!" * 60)

    def poll(self) -> bool:
        """Re-check the file switch. Returns True when tripped (latched)."""
        if not self._tripped and self.file_path and os.path.exists(self.file_path):
            self.trip(f"kill switch file present at {self.file_path}")
        return self._tripped

    def reset(self) -> bool:
        """Explicitly re-arm trading. Refuses while the file switch is present.

        Returns True when the switch was cleared.
        """
        if self.file_path and os.path.exists(self.file_path):
            logger.error(
                "Refusing to reset kill switch: %s still exists. Remove the file first.",
                self.file_path,
            )
            return False
        if self._tripped:
            logger.warning("Kill switch reset (was: %s). Trading may resume.", self._reason)
        self._tripped = False
        self._reason = None
        return True

    def check(self) -> GuardDecision:
        if self.poll():
            return GuardDecision(False, REASON_KILL_SWITCH,
                                 f"kill switch is tripped ({self._reason})")
        return ALLOWED


# --------------------------------------------------------------------------- #
# daily loss
# --------------------------------------------------------------------------- #

class DailyLossGuard:
    """Halts new entries once the session is down more than the configured limit.

    The baseline is the account's equity the first time :meth:`update` sees a
    readable account for the session; loss is ``baseline - current_equity``,
    which covers realized and unrealized P&L together. When both a percentage
    and an absolute limit are configured the *tighter* of the two binds.

    Breach is latching for the session. :meth:`start_session` (called on a new
    trading day) re-baselines and clears it.
    """

    def __init__(self, max_daily_loss_pct: Optional[float] = None,
                 max_daily_loss_amount: Optional[float] = None,
                 liquidate_on_breach: bool = False):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_daily_loss_amount = max_daily_loss_amount
        self.liquidate_on_breach = bool(liquidate_on_breach)
        self.starting_equity: Optional[float] = None
        self.last_equity: Optional[float] = None
        self._breached = False
        self._breach_reason: Optional[str] = None

    @property
    def enabled(self) -> bool:
        return self.max_daily_loss_pct is not None or self.max_daily_loss_amount is not None

    @property
    def breached(self) -> bool:
        return self._breached

    @property
    def breach_reason(self) -> Optional[str]:
        return self._breach_reason

    def start_session(self, account: Optional[Dict] = None) -> None:
        """Begin a new trading session: clear the breach and re-baseline equity."""
        self._breached = False
        self._breach_reason = None
        self.starting_equity = None
        self.last_equity = None
        if account is not None:
            self.update(account)

    def loss_limit(self) -> Optional[float]:
        """Currency loss that trips the guard, or None if not configured yet."""
        limits = []
        if self.max_daily_loss_amount is not None:
            limits.append(self.max_daily_loss_amount)
        if self.max_daily_loss_pct is not None and self.starting_equity is not None:
            limits.append(self.max_daily_loss_pct * self.starting_equity)
        return min(limits) if limits else None

    def update(self, account: Optional[Dict]) -> GuardDecision:
        """Fold in the latest account reading and report whether entries are allowed."""
        if not self.enabled:
            return ALLOWED

        equity = _equity_from_account(account)
        if equity is None:
            # Fail closed but do NOT latch: a transient get_account() failure
            # should pause new entries, not disable trading for the session.
            logger.warning(
                "Cannot read account equity; blocking new entries while the "
                "max-daily-loss limit is unverifiable."
            )
            return GuardDecision(False, REASON_EQUITY_UNKNOWN,
                                 "account equity unavailable; daily-loss limit unverifiable")

        self.last_equity = equity
        if self.starting_equity is None:
            self.starting_equity = equity
            logger.info("Session starting equity recorded: %.2f (daily loss limit: %s)",
                        equity, self._format_limit())

        if self._breached:
            return self._breach_decision()

        limit = self.loss_limit()
        if limit is None:
            return ALLOWED

        loss = self.starting_equity - equity
        if loss >= limit:
            self._breached = True
            self._breach_reason = (
                f"daily loss {loss:.2f} reached limit {limit:.2f} "
                f"(equity {equity:.2f} vs session start {self.starting_equity:.2f})"
            )
            logger.error("!" * 60)
            logger.error("!!! MAX DAILY LOSS BREACHED: %s", self._breach_reason)
            if self.liquidate_on_breach:
                logger.error("!!! liquidate_on_daily_loss is ON: open positions will be flattened.")
            else:
                logger.error("!!! Halting new entries. Existing positions are left alone "
                             "(set risk.liquidate_on_daily_loss: true to auto-flatten).")
            logger.error("!" * 60)
            return self._breach_decision()

        return ALLOWED

    def check(self) -> GuardDecision:
        """Current state without folding in a new account reading."""
        if self._breached:
            return self._breach_decision()
        return ALLOWED

    def _breach_decision(self) -> GuardDecision:
        return GuardDecision(False, REASON_DAILY_LOSS,
                             f"max daily loss breached ({self._breach_reason})")

    def _format_limit(self) -> str:
        limit = self.loss_limit()
        return f"{limit:.2f}" if limit is not None else "not configured"


# --------------------------------------------------------------------------- #
# market hours
# --------------------------------------------------------------------------- #

class MarketHoursGate:
    """Regular-session gate for US equities: weekdays, 09:30-16:00 America/New_York.

    The timezone comes from :mod:`zoneinfo` (stdlib) rather than a hardcoded UTC
    offset, so US daylight-saving transitions are handled automatically.

    LIMITATION - THIS GATE DOES NOT KNOW ABOUT MARKET HOLIDAYS. It checks the
    day of week and the time of day only. Thanksgiving, Good Friday, Juneteenth,
    federal-holiday closures, and 13:00 early closes all read as "open" here.
    A real holiday calendar needs an exchange calendar data source; do not
    assume this class provides one.
    """

    def __init__(self, enabled: bool = True, allow_outside_hours: bool = False,
                 tz_name: str = DEFAULT_MARKET_TIMEZONE,
                 open_time: dtime = DEFAULT_MARKET_OPEN,
                 close_time: dtime = DEFAULT_MARKET_CLOSE):
        self.enabled = bool(enabled)
        self.allow_outside_hours = bool(allow_outside_hours)
        self.tz_name = tz_name
        self.open_time = open_time
        self.close_time = close_time
        try:
            self.tz = ZoneInfo(tz_name)
        except Exception:
            logger.warning("Unknown risk.market_hours.timezone=%r; falling back to %s",
                           tz_name, DEFAULT_MARKET_TIMEZONE)
            self.tz_name = DEFAULT_MARKET_TIMEZONE
            self.tz = ZoneInfo(DEFAULT_MARKET_TIMEZONE)

    @property
    def active(self) -> bool:
        """True when this gate can actually block anything."""
        return self.enabled and not self.allow_outside_hours

    def to_market_time(self, now: datetime) -> datetime:
        """Convert ``now`` to market-local time.

        A naive datetime is taken to already be in the market timezone; an aware
        one is converted. Nothing here reads the wall clock.
        """
        if now.tzinfo is None:
            return now.replace(tzinfo=self.tz)
        return now.astimezone(self.tz)

    def is_open(self, now: datetime) -> bool:
        """Weekday + time-of-day check. Holidays are NOT considered (see class docstring)."""
        local = self.to_market_time(now)
        if local.weekday() >= 5:  # 5=Saturday, 6=Sunday
            return False
        return self.open_time <= local.time() < self.close_time

    def check(self, now: datetime) -> GuardDecision:
        if not self.enabled or self.allow_outside_hours:
            return ALLOWED
        if self.is_open(now):
            return ALLOWED
        local = self.to_market_time(now)
        return GuardDecision(
            False, REASON_MARKET_CLOSED,
            f"US equity regular session is closed at {local.isoformat()} "
            f"({self.open_time.strftime('%H:%M')}-{self.close_time.strftime('%H:%M')} {self.tz_name}, "
            f"weekdays only; holidays not tracked)",
        )


# --------------------------------------------------------------------------- #
# aggregate guard
# --------------------------------------------------------------------------- #

class RiskGuard:
    """The three rails behind one object, wired into the live trading loop.

    Split into two entry points on purpose:

    * :meth:`check_cycle` - "should this poll happen at all?" Includes the
      market-hours gate, so it needs a clock.
    * :meth:`check_new_entry` - "may this specific BUY be sized and sent?"
      Deliberately clock-free (kill switch + daily loss only), so the order path
      stays deterministic and defensible in tests. It is reached only from
      inside a cycle that already passed :meth:`check_cycle`.
    """

    def __init__(self, kill_switch: Optional[KillSwitch] = None,
                 daily_loss: Optional[DailyLossGuard] = None,
                 market_hours: Optional[MarketHoursGate] = None,
                 now_provider=None):
        self.kill_switch = kill_switch or KillSwitch()
        self.daily_loss = daily_loss or DailyLossGuard()
        self.market_hours = market_hours or MarketHoursGate()
        self._now_provider = now_provider or (lambda: datetime.now(timezone.utc))
        self._session_date = None

    # -- construction -------------------------------------------------------

    @classmethod
    def from_config(cls, config: Optional[Dict], now_provider=None) -> "RiskGuard":
        """Build from a broker config's ``risk`` section.

        Kill switch and daily-loss limits are strictly opt-in: with no keys set
        they never block anything, so an unconfigured deployment behaves exactly
        as it did before.

        The market-hours gate is the exception. It defaults to ON as soon as a
        ``risk:`` section exists (even an empty one) - you have to write
        ``market_hours: {enabled: false}`` to trade outside the session. With no
        ``risk:`` section at all it stays off, so configs written before this
        module existed keep their old behaviour instead of silently halting.
        """
        config = config or {}
        risk = (config.get("risk") or {})
        market = (risk.get("market_hours") or {})
        # Presence of the key, not its contents: `risk: {}` still arms the gate.
        gate_default = "risk" in config

        daily_loss = DailyLossGuard(
            max_daily_loss_pct=_optional_positive_float(
                risk.get("max_daily_loss_pct"), "max_daily_loss_pct"),
            max_daily_loss_amount=_optional_positive_float(
                risk.get("max_daily_loss_amount"), "max_daily_loss_amount"),
            liquidate_on_breach=bool(risk.get("liquidate_on_daily_loss", False)),
        )
        gate = MarketHoursGate(
            enabled=bool(market.get("enabled", gate_default)),
            allow_outside_hours=bool(market.get("allow_outside_hours", False)),
            tz_name=market.get("timezone") or DEFAULT_MARKET_TIMEZONE,
            open_time=_parse_time(market.get("open"), DEFAULT_MARKET_OPEN, "open"),
            close_time=_parse_time(market.get("close"), DEFAULT_MARKET_CLOSE, "close"),
        )
        return cls(
            kill_switch=KillSwitch(risk.get("kill_switch_file")),
            daily_loss=daily_loss,
            market_hours=gate,
            now_provider=now_provider,
        )

    # -- clock --------------------------------------------------------------

    def now(self, now: Optional[datetime] = None) -> datetime:
        return now if now is not None else self._now_provider()

    # -- session ------------------------------------------------------------

    def start_session(self, account: Optional[Dict] = None,
                      now: Optional[datetime] = None) -> None:
        """Record the session baseline. The kill switch is intentionally untouched."""
        self._session_date = self.market_hours.to_market_time(self.now(now)).date()
        self.daily_loss.start_session(account)

    def _roll_session_if_new_day(self, now: datetime) -> None:
        today = self.market_hours.to_market_time(now).date()
        if self._session_date is None:
            self._session_date = today
            return
        if today != self._session_date:
            logger.info("New trading day (%s): resetting daily-loss baseline. "
                        "The kill switch is NOT reset by a date rollover.", today)
            self._session_date = today
            self.daily_loss.start_session(None)

    # -- checks -------------------------------------------------------------

    def check_cycle(self, account: Optional[Dict] = None,
                    now: Optional[datetime] = None) -> GuardDecision:
        """Full pre-cycle check: kill switch, market hours, then daily loss."""
        current = self.now(now)
        self._roll_session_if_new_day(current)

        decision = self.kill_switch.check()
        if not decision:
            return decision

        decision = self.market_hours.check(current)
        if not decision:
            return decision

        return self.daily_loss.update(account)

    def check_new_entry(self, account: Optional[Dict] = None) -> GuardDecision:
        """Clock-free check run immediately before sizing/submitting a new position."""
        decision = self.kill_switch.check()
        if not decision:
            return decision
        if account is None:
            return self.daily_loss.check()
        return self.daily_loss.update(account)

    # -- operations ---------------------------------------------------------

    def trip(self, reason: str = "tripped programmatically") -> None:
        self.kill_switch.trip(reason)

    def reset_kill_switch(self) -> bool:
        return self.kill_switch.reset()

    @property
    def halted(self) -> bool:
        return self.kill_switch.tripped or self.daily_loss.breached

    @property
    def should_liquidate(self) -> bool:
        """True only when the daily loss breached AND auto-liquidation is opted in."""
        return self.daily_loss.breached and self.daily_loss.liquidate_on_breach

    def describe(self) -> Dict:
        """Snapshot for logging / status endpoints."""
        return {
            "kill_switch_tripped": self.kill_switch.tripped,
            "kill_switch_file": self.kill_switch.file_path,
            "daily_loss_enabled": self.daily_loss.enabled,
            "daily_loss_breached": self.daily_loss.breached,
            "starting_equity": self.daily_loss.starting_equity,
            "daily_loss_limit": self.daily_loss.loss_limit(),
            "liquidate_on_daily_loss": self.daily_loss.liquidate_on_breach,
            "market_hours_enforced": self.market_hours.active,
            "market_timezone": self.market_hours.tz_name,
        }
