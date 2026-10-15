"""EXEC-CLIENT: the exchange order client's settings, its start-check guard and its rate limiter (Advisor DRILL-VENUE-2
and HEAD3 (1); spec /mnt/project-files/sleeve-fund/pe3-plans/exec-client-spec.md).

It ships OFF: with EXEC_MODE unset (or anything but demo or live) paper keeps the simulated client and nothing here
reaches the exchange. One client for demo and live: the framework's Bybit execution adapter, built by one function
from the host and keys in the environment, with no demo-only path. ON is gated in code by two journaled records of the
PM's (EXEC-SPEC MUST 1): drill-go lets demo ON for the drills, drills-passed lets the fund's paper Strategies onto the
demo client; live is refused until G2. Only a trade-only key passes. Every line here names hosts and settings, never a
key's value."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sleeve_fund import exec_target

# Where the client sends orders is read by exec_target, the one reader the start check's "Order client" line shares
# (PE4, head 3): EXEC_MODE, EXEC_VENUE, EXEC_HOST and the venue's own key pair for that mode (BYBIT_DEMO_API_KEY and
# _SECRET in demo). This module adds the gates on top: the PM's records, the G2 lock, the key's permissions.
MODE_ENV, HOST_ENV = exec_target.MODE_ENV, exec_target.HOST_ENV
OFF, DEMO, LIVE = "off", "demo", "live"
ENVIRONMENT = {DEMO: "DEMO", LIVE: "MAINNET"}  # the adapter's BybitEnvironment, by mode
LIVE_BEFORE_G2 = False  # the live lock (store's live accounts): nothing trades real money before G2
DRILL_GO = "exec_drill_go"  # the PM's journaled go for the demo drills: lets demo ON
DRILLS_PASSED = "exec_drills_passed"  # the PM's journaled record that they passed: the fund's paper Strategies may use it
QUEUE_ENV = "EXEC_ENTRY_QUEUE_MAX_S"  # how long an entry over the limit may queue (EXEC-SPEC Q1): default 5 s, 0-10
QUEUE_DEFAULT_S, QUEUE_MAX_S = 5.0, 10.0
# The venue's orderLinkId: at most 36 letters, digits, dashes and underscores (EXEC-SPEC MUST 5).
LINK_ID = re.compile(r"[A-Za-z0-9_-]{1,36}")
LINK_ID_SOURCE = "https://bybit-exchange.github.io/docs/v5/order/create-order"
# Every venue figure this module holds names its documented source (HEAD3 (1) c); recorded responses join it when the
# ON path is built (drill prep).

# Production per-UID limits for linear perpetuals, used in demo as well: a demo's own looser limits are never trusted
# (DRILL-VENUE-2 (b)). endpoint -> (requests, per seconds, source). Checked against the source before QA.
_RATE_DOC = "https://bybit-exchange.github.io/docs/v5/rate-limit"
LIMITS = {
    "order/create": (10, 1.0, _RATE_DOC),
    "order/amend": (10, 1.0, _RATE_DOC),
    "order/cancel": (10, 1.0, _RATE_DOC),
    "order/realtime": (50, 1.0, _RATE_DOC),
    "position/list": (50, 1.0, _RATE_DOC),
}
SOURCES = {"rate limits": _RATE_DOC, "orderLinkId": LINK_ID_SOURCE}


@dataclass(frozen=True)
class ExecSettings:
    mode: str  # off, demo or live; any other value is ON and fails closed (exec_target N-KH1)
    host: str
    key_set: bool
    secret_set: bool
    key_var: str | None = None  # the key and secret variables for this venue and mode (exec_target.KEYS)
    secret_var: str | None = None
    problems: tuple[str, ...] = ()  # exec_target.problems: the host and key mismatches, named

    @property
    def on(self) -> bool:
        return self.mode != OFF or bool(self.problems)  # "OFF" (not exact "off") reads as ON with an unknown mode


def exec_settings(environ) -> ExecSettings:
    """The client's settings, read through exec_target (the reader the start check shares). Unset, "off" or "none" is
    OFF; any other mode is ON, so an unknown one fails closed rather than quietly running the simulated client."""
    t = exec_target.read(environ)
    if t is None:
        return ExecSettings(mode=OFF, host="", key_set=False, secret_set=False)
    return ExecSettings(mode=t.mode, host=t.host, key_set=t.key_set, secret_set=t.secret_set, key_var=t.key_var,
                        secret_var=t.secret_var, problems=tuple(exec_target.problems(t)))


def entry_queue_max(environ) -> timedelta:
    """EXEC-SPEC Q1: how long an entry over the limit may wait before it is refused and journaled. Outside 0-10 s it is
    refused at load, never clamped."""
    raw = (environ.get(QUEUE_ENV) or "").strip()
    try:
        secs = float(raw) if raw else QUEUE_DEFAULT_S
    except ValueError:
        secs = -1.0
    if not 0.0 <= secs <= QUEUE_MAX_S:
        raise ValueError(f"{QUEUE_ENV}={raw!r}: an entry may queue 0 to 10 seconds")
    return timedelta(seconds=secs)


def can_trade(key_permissions: dict) -> bool:
    return bool(key_permissions.get("trade")) and not key_permissions.get("read_only")


def cannot_protect(key_permissions: dict | None, *, venue_holds: bool) -> str | None:
    """EXEC-SPEC Q2: a key that can't trade (read-only or revoked, at start or mid-session) while the venue holds a
    position or an order for us can't move or close it: CRITICAL, alerted, never a quiet FAIL."""
    if key_permissions is None or can_trade(key_permissions) or not venue_holds:
        return None
    return ("CRITICAL: cannot protect: the key can't trade while the exchange holds our position or order; restore a "
            "trade-only key now")


def link_id_ok(cid: str) -> bool:
    """EXEC-SPEC MUST 5: whether an order id fits the venue's orderLinkId. The client refuses any other before it is
    sent."""
    return bool(LINK_ID.fullmatch(cid or ""))


def gates(s: ExecSettings, *, drill_go: bool, key_permissions: dict | None,
          venue_holds: bool = False) -> list[tuple[str, str]]:
    """The client's own gates, as FAIL lines (none when OFF or all pass): live before G2, demo with no PM drill-go
    record, and the key's permissions once the venue has said them ({"trade", "withdraw", "read_only"}; {} when
    revoked; None before the first connect). venue_holds: the venue holds a position or an order for us. The start
    check adds these to exec_target's "Order client" section, so the host and key lines show once."""
    if not s.on:
        return []
    out = []
    if s.mode == LIVE and not LIVE_BEFORE_G2:
        out.append(("FAIL", "live mode before G2: nothing trades real money until the PM approves G2"))
    if s.mode == DEMO and not drill_go:
        out.append(("FAIL", "demo mode with no drill-go record: the client goes ON only on the PM's go for the drills"))
    if key_permissions is not None:
        if key_permissions.get("withdraw"):
            out.append(("FAIL", "the key can withdraw: only a trade-only key is allowed"))
        if not can_trade(key_permissions):
            out.append(("FAIL", cannot_protect(key_permissions, venue_holds=venue_holds)
                        or "the key can't trade (read-only or revoked): a trade-only key is needed"))
    return out


def guard(s: ExecSettings, *, drill_go: bool, key_permissions: dict | None,
          venue_holds: bool = False) -> list[tuple[str, str]]:
    """Every line for the client, as (PASS | FAIL | NOTE, what): exec_target's host and key mismatches, then the gates.
    A FAIL refuses entries only (the A7 rule); exits and stops always run."""
    if not s.on:
        return [("NOTE", "exchange order client OFF: paper runs on the simulated client")]
    out = [("FAIL", p) for p in s.problems]
    out += gates(s, drill_go=drill_go, key_permissions=key_permissions, venue_holds=venue_holds)
    if not out:
        scope = "trade-only key" if key_permissions is not None else "key scope not yet checked"
        out.append(("PASS", f"exchange order client ON, {s.mode} mode on {s.host}, {scope}"))
    return out


def adapter_kwargs(s: ExecSettings, environ) -> dict:
    """The execution adapter's settings, the same function for demo and live: the environment from the mode, the URL
    from the host, the keys from the mode's own pair (exec_target.KEYS). Refused when OFF, and on any host or key
    mismatch: nothing builds the client then, and it builds its URL from nothing else."""
    if not s.on:
        raise ValueError("the exchange order client is OFF: paper runs on the simulated client")
    if s.problems:
        raise exec_target.ExecTargetRefused("order client refused: " + "; ".join(s.problems))
    return {"environment": ENVIRONMENT[s.mode], "base_url_http": f"https://{s.host}",
            "api_key": environ.get(s.key_var), "api_secret": environ.get(s.secret_var),
            "margin_mode": "ISOLATED_MARGIN"}


class Limiter:
    """Production limits per endpoint, as token buckets refilled continuously. take() answers send, wait or refuse:
    an exit, a stop or a cancel (opening=False) always goes, ahead of any entry; an entry over the limit waits, and
    once it has waited longer than queue_max (EXEC-SPEC Q1) it is refused, to be journaled as such. An entry that is
    sent after waiting re-runs T1-T5 and the order rules on the current book at submit (the caller's gate)."""

    def __init__(self, limits: dict | None = None, now=None, queue_max: timedelta | None = None) -> None:
        self.queue_max = queue_max if queue_max is not None else timedelta(seconds=QUEUE_DEFAULT_S)
        self.limits = {k: v[:2] for k, v in (limits if limits is not None else LIMITS).items()}
        self._now = now or (lambda: datetime.now(UTC))
        self._tokens: dict[str, float] = {k: float(n) for k, (n, _) in self.limits.items()}
        self._at: dict[str, datetime] = {}

    def _refill(self, endpoint: str) -> None:
        n, per = self.limits[endpoint]
        now = self._now()
        last = self._at.get(endpoint, now)
        self._tokens[endpoint] = min(float(n), self._tokens[endpoint] + (now - last).total_seconds() * n / per)
        self._at[endpoint] = now

    def take(self, endpoint: str, *, opening: bool, since: datetime | None = None) -> str:
        self._refill(endpoint)
        if not opening:  # never held back: it may go over the limit, and the venue's answer is counted as any other
            self._tokens[endpoint] -= 1.0
            return "send"
        if self._tokens[endpoint] >= 1.0:
            self._tokens[endpoint] -= 1.0
            return "send"
        if since is not None and self._now() - since > self.queue_max:
            return "refuse"
        return "wait"


def recorded(engine, action: str) -> bool:
    """Whether the PM has journaled the record (DRILL_GO or DRILLS_PASSED); anyone else's is ignored."""
    from sqlalchemy import select

    from sleeve_fund.store import decisions_t

    with engine.connect() as c:
        return c.execute(select(decisions_t.c.id).where(decisions_t.c.action == action,
                                                        decisions_t.c.actor == "pm").limit(1)).first() is not None


def assert_off(environ) -> None:
    """A paper node runs on the simulated client only, until the client's drills (spec section 8): ON is refused at the
    node, whatever the start check or the drills-passed record says, because the Strategy's instrument and market data
    on the exchange aren't built yet. OFF changes nothing."""
    from sleeve_fund.paper.safety import PaperSafetyError

    s = exec_settings(environ)
    if s.on:
        raise PaperSafetyError(f"{MODE_ENV}={s.mode}: the exchange order client can't run a paper node yet (its "
                               "market data on the exchange is on the drill list); unset it to run on the simulated "
                               "client")
