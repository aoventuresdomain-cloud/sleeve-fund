"""EXEC-CLIENT: the exchange order client's settings, its start-check guard and its rate limiter (Advisor DRILL-VENUE-2
and HEAD3 (1); spec /mnt/project-files/sleeve-fund/pe3-plans/exec-client-spec.md).

It ships OFF: with EXEC_MODE unset (or anything but demo or live) paper keeps the simulated client and nothing here
reaches the exchange. One client for demo and live: the framework's execution adapter for the venue, built by one function
from the host and keys in the environment, with no demo-only path. ON is gated in code by two journaled records of the
PM's (EXEC-SPEC MUST 1): drill-go lets demo ON for the drills, drills-passed lets the fund's paper Strategies onto the
demo client; live is refused until G2. Only a trade-only key passes. Every line here names hosts and settings, never a
key's value."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from sleeve_fund import exec_target

# Where the client sends orders is read by exec_target, the one reader the start check's "Order client" line shares
# (PE4, head 3): EXEC_MODE, EXEC_VENUE, EXEC_HOST and the venue's own key pair for that mode (exec_target.KEYS). Hosts,
# key variables and documentation links live there, never here (test_venues). This module adds the gates on top: the PM's records, the G2 lock, the key's permissions.
MODE_ENV, HOST_ENV = exec_target.MODE_ENV, exec_target.HOST_ENV
OFF, DEMO, LIVE = "off", "demo", "live"
ENVIRONMENT = {DEMO: "DEMO", LIVE: "MAINNET"}  # the adapter's environment, by mode
LIVE_BEFORE_G2 = False  # the live lock (store's live accounts): nothing trades real money before G2
DRILL_GO = "exec_drill_go"  # the PM's journaled go for the demo drills: lets demo ON
DRILLS_PASSED = "exec_drills_passed"  # the PM's journaled record that they passed: the fund's paper Strategies may use it
QUEUE_ENV = "EXEC_ENTRY_QUEUE_MAX_S"  # how long an entry over the limit may queue (EXEC-SPEC Q1): default 5 s, 0-10
QUEUE_DEFAULT_S, QUEUE_MAX_S = 5.0, 10.0
# The venue's orderLinkId: at most 36 letters, digits, dashes and underscores (EXEC-SPEC MUST 5).
LINK_ID = re.compile(r"[A-Za-z0-9_-]{1,36}")
LINK_ID_SOURCE = exec_target.DOC_ORDER_LINK_ID
# Every venue figure this module holds names its documented source (HEAD3 (1) c); recorded responses join it when the
# ON path is built (drill prep).

# Production per-UID limits for linear perpetuals, used in demo as well: a demo's own looser limits are never trusted
# (DRILL-VENUE-2 (b)). endpoint -> (requests, per seconds, source). Checked against the source before QA.
_RATE_DOC = exec_target.DOC_RATE_LIMITS
LIMITS = {
    "order/create": (10, 1.0, _RATE_DOC),
    "order/amend": (10, 1.0, _RATE_DOC),
    "order/cancel": (10, 1.0, _RATE_DOC),
    "order/realtime": (50, 1.0, _RATE_DOC),
    "order/history": (50, 1.0, _RATE_DOC),
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


def node_configs(s: ExecSettings, environ, *, urls: dict | None = None, http_timeout_secs: int | None = None) -> tuple:
    """The Bybit data and execution client configs a paper node adds with the client ON (B-05): one function for
    demo and live (adapter_kwargs), linear perpetuals, isolated margin. The client never retries an order request on
    its own (max_retries=0): a create that timed out is looked up by its id before anything is sent again (MUST 2).
    urls: the stand-in venue's addresses in cells (base_url_http and the WebSocket ones); never set in paper."""
    from nautilus_trader.adapters import bybit

    kw = adapter_kwargs(s, environ)
    urls = dict(urls or {})
    common = {"product_types": [bybit.BybitProductType.LINEAR],
              "environment": getattr(bybit.BybitEnvironment, kw["environment"]), "api_key": kw["api_key"],
              "api_secret": kw["api_secret"], "base_url_http": urls.pop("base_url_http", kw["base_url_http"]),
              "max_retries": 0, "http_timeout_secs": http_timeout_secs}
    ws_public, ws_private, ws_trade = (urls.pop(k, None) for k in ("base_url_ws_public", "base_url_ws_private",
                                                                     "base_url_ws_trade"))
    if urls:
        raise ValueError(f"unknown client addresses: {sorted(urls)}")
    data = bybit.BybitDataClientConfig(**common, base_url_ws_public=ws_public, base_url_ws_private=ws_private)
    # No margin_mode: with one, the client posts the account's margin mode at connect, an account-wide write on a
    # shared account (B-05 finding, Advisor MUST 3: never auto-fix). The connect check reads it instead.
    execution = bybit.BybitExecutionClientConfig(**common, base_url_ws_private=ws_private, base_url_ws_trade=ws_trade)
    return data, execution


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
        if since is None:  # HoQA (drill-prep line 14): an entry with no first-asked time would wait for ever
            raise ValueError(f"an entry over the limit on {endpoint} must carry the time it was first asked for")
        if self._now() - since > self.queue_max:
            return "refuse"
        return "wait"


def recorded(engine, action: str) -> bool:
    """Whether the PM has journaled the record (DRILL_GO or DRILLS_PASSED); anyone else's is ignored."""
    from sqlalchemy import select

    from sleeve_fund.store import decisions_t

    with engine.connect() as c:
        return c.execute(select(decisions_t.c.id).where(decisions_t.c.action == action,
                                                        decisions_t.c.actor == "pm").limit(1)).first() is not None


# Drill-prep MUST 9 (HoE ruling, 20 Oct): the exchange's venue profile, as data only. It is not registered
# (venues.register), so no form, study or history store offers it. A second venue's data hub is B-25 ground, which
# waits the PM's word: until one exists ON stays refused at the node (assert_off), naming why.
NO_HUB = "no data hub for this venue"
EXCHANGE_PROFILE = {"name": "BYBIT", "category": "linear", "perpetual": True, "position_mode": "one-way",
                    "margin_mode": "isolated", "funding_hours": (0, 8, 16), "link_id_max": 36, "hub": False}


def assert_off(environ) -> None:
    """A paper node runs on the simulated client only, until the client's drills (spec section 8): ON is refused at the
    node, whatever the start check or the drills-passed record says, because the Strategy's market data on the
    exchange has no hub yet (MUST 9). OFF changes nothing."""
    from sleeve_fund.paper.safety import PaperSafetyError

    s = exec_settings(environ)
    if s.on:
        why = NO_HUB if not EXCHANGE_PROFILE["hub"] else "the drills haven't passed"
        raise PaperSafetyError(f"{MODE_ENV}={s.mode}: the exchange order client can't run a paper node yet: {why} "
                               "(drill-prep MUST 9); unset it to run on the simulated client")


# Order ids ON (drill-prep MUST 6, EXEC-SPEC MUST 5 finding; B-05). Paper's ids (O-date-time-<trader>-<strategy>-n)
# and the A2 close ids (orphan-close-<entry>-n) pass 36 characters, the venue's orderLinkId limit. ON, every order the
# fund sends is in its namespace (stop_mover.ID_PREFIX, Advisor STOP-CLAIM): an entry, exit or stop is "sf-" and a
# UUID's 32 hex digits; a late fill's close is "sf-z" and 32 hex digits of its entry id and number, so two writers
# racing journal one row, as the OFF id does. "z" is no hex digit, so a close is told from any other order of ours.
CLOSE_ID_PREFIX = "sf-z"


def order_id() -> str:
    """A new order id for the client ON: 35 characters, in the fund's namespace."""
    import uuid

    return f"sf-{uuid.uuid4().hex}"


def close_id(entry_id: str, n: int = 0) -> str:
    """The A2 close id ON for the (n+1)th close of `entry_id`'s late fills: deterministic, 36 characters."""
    import hashlib

    return CLOSE_ID_PREFIX + hashlib.sha1(f"{entry_id}#{n}".encode()).hexdigest()[:32]


NOT_FOUND = "not found"  # the venue answered and has no such order (an unanswered lookup is None, never this)
# How long a just-created order may take to show in the venue's lookups. Unconfirmed (the docs host is unreachable
# from build boxes): on MUST 17's confirm list, with whether a filled order shows in open orders or only in history.
VISIBILITY_WINDOW = timedelta(seconds=10)


def may_resend(row) -> bool:
    """MUST 2 (spec section 5): after a create whose answer never came, the order is looked up by its own id before
    anything is sent again. Only the venue's word that it has no such order (NOT_FOUND, in open orders AND history,
    past the visibility window: resend_check), or that it refused it with nothing filled, lets a new one go; a
    working, filled or part-filled order, or no answer at all (None), never does."""
    if row == NOT_FOUND:
        return True
    if not isinstance(row, dict):
        return False  # None: no answer is not evidence of anything
    try:
        filled = Decimal(str(row.get("cumExecQty") or 0))
    except (ArithmeticError, ValueError):
        return False
    return row.get("orderStatus") == "Rejected" and filled == 0


def resend_check(stops, oid: str, *, sent_at: datetime, now: datetime) -> str:
    """Whether an order whose create went unanswered may be sent again (QD M1-M3 review, HoE MUST): "wait" inside
    the venue's visibility window; "no answer" when either lookup went unanswered; "keep" when the venue has it
    (working, filled or part-filled, in open orders or in history); "resend" only when neither lookup finds it, or
    the venue refused it with nothing filled. stops: venue_stops.VenueStops (lookup = open orders, history)."""
    if now - sent_at < VISIBILITY_WINDOW:
        return "wait"
    rows = []
    for read in (stops.lookup, stops.history):
        try:
            rows.append(read(oid))
        except ConnectionError:
            return "no answer"
    found = [r for r in rows if r is not None]
    if not found:
        return "resend" if may_resend(NOT_FOUND) else "keep"
    return "resend" if all(may_resend(r) for r in found) else "keep"


# The connect check (B-05 M4; drill-prep MUST 3, 8, 9 clock, 10, 12, 13). Run by the owner when the client connects,
# on the venue's own answers; its FAIL is journaled per Strategy and refuses entries at runtime (runtime.
# exec_target_block), exits and stops still run, until a later check passes. Nothing here writes to the account:
# a wrong margin mode, position mode or leverage is named, never fixed (MUST 3: never auto-fix open).
CONNECT_FAIL, CONNECT_PASS = "exec_connect_fail", "exec_connect_pass"
MAX_SKEW_MS = 1000  # the venue's clock against ours; the signed recv window is 5 s (venue_stops.RECV_WINDOW)
SET_IT = "Set it on the account; the fund never changes an account or instrument setting"  # Advisor B05-NO-ACCOUNT-WRITES
AUTH_CODES = frozenset({10003, 10004, 10005, 10007, 33004})  # invalid key, sign, permission denied, auth, key expired


def key_permissions(answer: dict | None) -> dict | None:
    """The key's permissions as the venue's key query reports them (user/query-api), as gates() reads them:
    {"trade", "withdraw", "read_only"}. No answer: None (not yet checked, never shown as verified)."""
    if not answer or answer.get("retCode") != 0:
        return None
    r = answer.get("result") or {}
    perms = r.get("permissions") or {}
    return {"trade": "Order" in (perms.get("ContractTrade") or []),
            "withdraw": any("Withdraw" in (v or []) for v in perms.values()),
            "read_only": str(r.get("readOnly", "0")) == "1"}


def connect_check(s: ExecSettings, *, drill_go: bool, key: dict | None, account: dict | None,
                  positions: dict[str, dict | None], caps: dict[str, float], server_ms: int | None,
                  local_ms: int) -> list[tuple[str, str]]:
    """Every line for the client at connect, as (PASS | FAIL | NOTE, what), on the venue's answers: the key's scope
    (guard), the account's margin mode (account/info), each instrument's position mode and leverage (position/list)
    against the engine's cap, and the clock. positions: symbol -> the venue's position row (None: unreadable)."""
    perms = key_permissions(key)
    out = [x for x in guard(s, drill_go=drill_go, key_permissions=perms) if x[0] == "FAIL"]
    if perms is None:
        out.append(("FAIL", "the key's permissions couldn't be read from the venue: a trade-only key can't be shown"))
    mode = ((account or {}).get("result") or {}).get("marginMode")
    if mode != "ISOLATED_MARGIN":
        out.append(("FAIL", (f"the account's margin mode is {mode or 'unreadable'}, not isolated: it must be Isolated. "
                             f"{SET_IT}")))
    for symbol, row in sorted(positions.items()):
        if row is None:
            out.append(("FAIL", f"{symbol}: its position settings couldn't be read"))
            continue
        if int(row.get("positionIdx", 0)) != 0:
            out.append(("FAIL", f"{symbol}: the position mode is hedge, not one-way: it must be One-way. {SET_IT}"))
        cap = caps.get(symbol)
        try:
            lev = float(row.get("leverage"))
        except (TypeError, ValueError):
            lev = None
        if cap is None or lev is None or abs(lev - cap) > 1e-9:
            need = f"{cap:g}x, the Strategy's cap" if cap is not None else "the Strategy's cap, which is unknown"
            out.append(("FAIL", (f"{symbol}: the leverage is {row.get('leverage') or 'unreadable'}: it must be {need}. "
                                 f"{SET_IT}, and never with a position open")))
    if server_ms is None:
        out.append(("FAIL", "the venue's clock couldn't be read"))
    elif abs(server_ms - local_ms) > MAX_SKEW_MS:
        out.append(("FAIL", f"this machine's clock is {abs(server_ms - local_ms)} ms off the venue's"))
    if not out:
        out.append(("PASS", (f"order client connected, {s.mode} mode: trade-only key, isolated, one-way, leverage "
                             "at the cap, clock in step")))
    return out


def record_connect(store, name: str, lines: list[tuple[str, str]], now: datetime) -> None:
    """The check in the journal: a FAIL opens (or restates) the hold as an error, which the alerts send; a PASS after
    a FAIL clears it. A PASS with nothing held writes nothing."""
    fails = [t for k, t in lines if k == "FAIL"]
    last = store.last_event(name, (CONNECT_FAIL, CONNECT_PASS))
    held = last is not None and last["kind"] == CONNECT_FAIL
    if fails:
        store.event(name, "error", CONNECT_FAIL, "Order client check failed: " + "; ".join(fails) + ". No new entries "
                    "until it passes; exits and stops still go out.", ts=now)
    elif held:
        store.event(name, "info", CONNECT_PASS, "Order client check passed: entries are allowed again", ts=now)


def lost_permission(code: int | None, *, venue_holds: bool) -> str | None:
    """MUST 8 (Advisor Q2): the first trade-endpoint auth error mid-session is CRITICAL "cannot protect" while the
    venue holds a position or order for us (cannot_protect's words), and a FAIL either way."""
    if code not in AUTH_CODES:
        return None
    return cannot_protect({}, venue_holds=venue_holds) or (f"the venue refused the key ({code}) on a trade request: "
                                                           "a trade-only key is needed")


def holds(store, name: str) -> list[str]:
    """Why the client ON refuses `name` new entries, from the journal: a failed connect check (or a lost key) not
    since passed, a restart reconcile that didn't tie (exec_reconcile.hold), and a foreign order on its instrument
    (exec_start.foreign_hold, fund-wide). Unreadable: held (fail closed)."""
    from sleeve_fund.paper import exec_reconcile, exec_start

    out = []
    try:
        last = store.last_event(name, (CONNECT_FAIL, CONNECT_PASS))
    except Exception:  # noqa: BLE001 - fail closed, never an error out of the entry gate
        return ["the order client's last check couldn't be read from the journal"]
    if last is not None and last["kind"] == CONNECT_FAIL:
        out.append(last["message"])
    held = exec_reconcile.hold(store, name)
    if held is not None:
        out.append(held)
    try:  # MUST 20: a foreign order on its instrument holds it fund-wide
        instrument = store.sleeve(name).instrument
    except Exception:  # noqa: BLE001 - fail closed
        return [*out, "the Strategy's instrument couldn't be read from the journal"]
    held = exec_start.foreign_hold(store, instrument)
    if held is not None:
        out.append(held)
    return out


FEE_TOLERANCE = 0.01  # EXEC-SPEC MUST 3: a venue fee more than a cent off FEE-LIVE's on one fill is alerted


def fee_gap(*, qty: float, price: float, fee: float, rate: float | None) -> str | None:
    """EXEC-SPEC MUST 3 (drill-prep line 4): the fill is booked at the fee the venue reported; this names a gap of
    more than a cent from what FEE-LIVE's rate gives for it, or None. No rate known: nothing to compare (None)."""
    if rate is None:
        return None
    expected = abs(qty * price) * rate
    if abs(fee - expected) <= FEE_TOLERANCE:
        return None
    return (f"the venue charged {fee:.2f} on a fill where the fee rate in use gives {expected:.2f}: the venue's is "
            "booked; check the account's fee tier")


# Resting stops through the client (drill prep, QD; findings /mnt/project-files/sleeve-fund/qd/exec-client-stop-
# findings.md). The client's amend carries a new quantity but drops a new trigger price without a word, and it
# creates a stop with no trigger price type, so the venue's default applies. A level therefore only ever moves by
# cancel-and-replace, and after every place or replace the venue's order lookup is compared with the Strategy's stop.
STOP_TRIGGER_BY = "LastPrice"  # the stop triggers on the last trade price, as the backtest and the watched stop do
RESTING = frozenset({"Untriggered", "New"})  # a conditional order waiting for its trigger


def _dec(x):
    from decimal import Decimal, InvalidOperation

    try:
        return Decimal(str(x))
    except (InvalidOperation, ValueError):
        return None


def stop_change(old: tuple, new: tuple) -> str:
    """How a resting stop moves from (level, qty) `old` to `new`: "none", "amend" (the quantity alone) or "replace"
    (any level change: cancel and place again, because the client's amend would drop it)."""
    (old_level, old_qty), (new_level, new_qty) = old, new
    if _dec(old_level) != _dec(new_level):
        return "replace"
    return "none" if _dec(old_qty) == _dec(new_qty) else "amend"


def stop_drift(row: dict | None, *, side: str, qty, level) -> list[str]:
    """Every way the venue's order lookup row for a resting stop differs from the Strategy's stop (side BUY or SELL,
    its quantity and level), in words for the CRITICAL alert; [] when they match. None: no stop at the venue."""
    if row is None:
        return ["no resting stop at the venue"]
    out = []
    if row.get("orderStatus") not in RESTING:
        out.append(f"not resting ({row.get('orderStatus')})")
    if str(row.get("side", "")).upper() != side.upper():
        out.append(f"side {row.get('side')} instead of {side.title()}")
    if _dec(row.get("triggerPrice")) != _dec(level):
        out.append(f"level {row.get('triggerPrice')} instead of {level}")
    if row.get("triggerBy") != STOP_TRIGGER_BY:
        out.append(f"trigger type {row.get('triggerBy')} instead of {STOP_TRIGGER_BY}")
    if row.get("reduceOnly") is not True:
        out.append("not reduce-only")
    if _dec(row.get("qty")) != _dec(qty):
        out.append(f"quantity {row.get('qty')} instead of {qty}")
    return out


class TriggerAmendRefused(ValueError):
    """Advisor STOP-EXEC line 14: an amend through the exchange client that would move a trigger price. The client
    would send the quantity alone and drop the level without a word, so it is refused instead."""


def guard_modify(*, on: bool, trigger_price) -> None:
    """Raises TriggerAmendRefused for a trigger-price amend while the exchange client is on; the simulated client
    moves triggers itself, so OFF passes."""
    if on and trigger_price is not None:
        raise TriggerAmendRefused(f"a stop's level ({trigger_price}) moves by placing a new stop, then cancelling the "
                                  "old one: the exchange client's amend would drop the new level")


# A refused place by class (Advisor STOP-RETRY). The venue's codes, from its error-code page (to be checked against it
# before QA, as the stand-in venue's shapes are).
REJECT_CODES_SOURCE = "https://bybit-exchange.github.io/docs/v5/error"
TRANSIENT_CODES = frozenset({10006, 10016, 10018})  # rate limit, server error, IP rate limit
CROSSED_CODES = frozenset({110092, 110093})  # the trigger would fire at once: the price is already past it


def reject_class(code: int | None, message: str = "") -> str:
    """"transient" (retry once after the back-off), "crossed" (the stop is already hit) or "other" (no retry). No code
    at all (a timeout, a dropped connection) is transient."""
    if code is None or code in TRANSIENT_CODES:
        return "transient"
    return "crossed" if code in CROSSED_CODES else "other"
