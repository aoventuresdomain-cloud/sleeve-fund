"""A15 ops-health minimum (G2; Done when in hoe/bgl-plan.md under "### A15"): the alerts the PM needs to know the fund
is alive and protected, on top of what the forwarder already sends (sleeve_fund.alerts).

- HubStaleness: one warning per episode when a hub has had no trade or quote on an instrument for STALE_AFTER, and one
  line when it is back. Each Strategy already holds its own entries on a stale price (base.STALE_PRICE_WARN_MINUTES);
  this says which hub and instrument, once, instead of one note per Strategy.
- NoStopWatch: one error per episode when an open perp position has had no live stop (none journaled, or the price
  has gone through it, S2) for NO_STOP_AFTER (Advisor A15-OPS: absolute, not a bar, so a 1-hour Strategy is never an
  hour unprotected). A Model that places no stop is exempt only while it is held to 1x; above 1x it alerts. The safety
  stop and flatten that follow (S3, F-T-2) are the Strategy's, on the order path, not here.
- DailyDigest: once a day at DIGEST_AT UK, even on a quiet day, so a missing digest is itself the alarm.

Everything here only reads the journal and writes events; none of it places, cancels or holds an order."""

from __future__ import annotations

import os
import re
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

from sleeve_fund.wording import no_venues

UK = ZoneInfo("Europe/London")
STALE_AFTER = timedelta(minutes=5)  # matches the per-Strategy hold (strategies.base.STALE_PRICE_WARN_MINUTES)
NO_STOP_AFTER = timedelta(seconds=60)  # Advisor A15-OPS, 15 Oct
DIGEST_AT = time(8, 0)  # UK
FIRST_GRACE = timedelta(minutes=2)  # the forwarder steps about once a minute (supervisor.ALERT_EVERY)
_DAY = re.compile(r"daily digest for (\d{4}-\d{2}-\d{2})")


def _minutes(d: timedelta) -> str:
    m = int(d.total_seconds() // 60)
    return f"{m} minute{'s' if m != 1 else ''}"


class HubStaleness:
    """Per instrument a hub relays, when it last had a trade or quote. Fed by the hub (seen on each tick, want when an
    instrument is first asked for) and checked once a minute; the episode state is in memory, so a hub restart can say
    a quiet instrument once more, which is the safer way to be wrong."""

    def __init__(self, hub: str, stale_after: timedelta = STALE_AFTER) -> None:
        self.hub, self.stale_after = hub, stale_after
        self.last: dict[str, datetime] = {}
        self.out: set[str] = set()

    def want(self, instrument: str, at: datetime) -> None:
        """Asked for at `at` with nothing seen yet: stale if nothing comes within stale_after."""
        self.last.setdefault(instrument, at)

    def seen(self, instrument: str, at: datetime) -> None:
        self.last[instrument] = max(at, self.last.get(instrument, at))

    def check(self, store, now: datetime) -> None:
        hub = no_venues(self.hub)
        for inst in sorted(self.last):
            quiet = now - self.last[inst]
            if quiet >= self.stale_after and inst not in self.out:
                self.out.add(inst)
                store.event(None, "warning", "hub_data_stale", no_venues(
                    f"Market data hub {hub}: no trade or quote for {inst} in {_minutes(self.stale_after)}. Strategies "
                    "on it hold new entries and adds; exits and stops still run"), ts=now)
            elif quiet < self.stale_after and inst in self.out:
                self.out.discard(inst)
                store.event(None, "info", "hub_data_back", no_venues(
                    f"Market data hub {hub}: {inst} has trades or quotes again"), ts=now)


def stopless_at_1x(s) -> bool:
    """A Model that places no stop, margined at 1x or less: held there by design (strategies.check_perp_stop), so its
    having no stop is not an incident. The same Model above 1x is."""
    from sleeve_fund.risk import PROFILES, leverage_cap
    from sleeve_fund.strategies import STOPLESS_MAX_LEVERAGE, places_stop

    profile = PROFILES.get(s.risk_profile)
    return (not places_stop(s.strategy, s.params) and profile is not None
            and leverage_cap(profile, s.params) <= STOPLESS_MAX_LEVERAGE)


def _undersized(store, name: str) -> bool:
    """A15-F2: the stop the journal shows working covers less than the position held (a live stop is a journaled stop
    for the open quantity). None working at all is judged by the plan, as book_open_risk does."""
    from sleeve_fund.store import OPEN_ORDER_STATUSES

    working = store.orders(name, statuses=OPEN_ORDER_STATUSES, intents=("stop_loss",), limit=50)
    if not working:
        return False
    # held from the fills, not the last mark: a stop resized with a fill is judged against that same fill
    held = abs(float(store.journal_book(name, store.sleeve(name).starting_balance)["qty"]))
    return held > 0 and sum(float(o["qty"]) - float(o["filled_qty"] or 0) for o in working) < held - 1e-12


class NoStopWatch:
    """Open perp positions of stop-built Models with no live stop, from the same reader as the dashboard's Open risk
    (open_risk.book_open_risk), so the alert and the tile never disagree on which positions have no stop."""

    def __init__(self, after: timedelta = NO_STOP_AFTER) -> None:
        self.after = after
        self.since: dict[str, datetime] = {}  # Strategy -> first seen without a live stop
        self.out: set[str] = set()

    def check(self, store, now: datetime) -> None:
        from sleeve_fund import open_risk

        rows = {r["sleeve"]: r for r in open_risk.book_open_risk(store, lambda s: None)}
        bad = {}
        for name, r in rows.items():
            if r["basis"] == "stop" and not _undersized(store, name):
                continue
            s = store.sleeve(name)
            if not stopless_at_1x(s):
                bad[name] = r["basis"]
        for name in sorted(set(self.since) - set(bad)):
            del self.since[name]
            if name in self.out:
                self.out.discard(name)
                why = "has its stop again" if name in rows else "is flat"
                store.event(name, "info", "position_stop_back", f"The open position {why}", ts=now)
        for name, basis in sorted(bad.items()):
            first = self.since.setdefault(name, now)
            if name in self.out or now - first < self.after:
                continue
            self.out.add(name)
            what = ("the price has gone through its stop and it is still open" if basis == "gapped"
                    else "its working stop covers less than the position" if basis == "stop"
                    else "it has no stop in place")
            store.event(name, "error", "position_without_stop",
                        f"Open position without a live stop for {_minutes(now - first)}: {what}. Check it now", ts=now)


def _money(x) -> str:
    return f"{float(x):,.2f}"


def _uk(ts: datetime) -> str:
    return f"{_aware(ts).astimezone(UK):%a %d %b %H:%M} UK"


def t5_day(store, now: datetime) -> str:
    """The last completed T5 day (the UTC day the daily pause measures, 00:00 UTC = 01:00 UK in summer): the book at
    its last mark before the day began and before it ended, as the brake reads its day start (portfolio.gate)."""
    from datetime import UTC

    end = _aware(now).astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=1)
    a, b = store.book_equity_at_or_before(start), store.book_equity_at_or_before(end)
    head = f"Last completed trading day ({_uk(start)} to {_uk(end)}): "
    if not a or b is None:
        return head + "no book marks"
    return head + f"{_money(a)} to {_money(b)}, {_money(b - a)} ({float(b / a - 1):+.2%})"


def digest(store, now: datetime, backup_dir=None, since: datetime | None = None) -> str:
    """The daily digest's text: the fund, the last completed T5 day, each Strategy, open positions with their stops,
    the alerts since the last digest (`since`, else 24 hours), and the backup, recon and clock checks. Names no
    venue."""
    from sleeve_fund import backups, open_risk, supervisor
    from sleeve_fund.store import is_backtest

    day_ago = _aware(since) if since is not None else now - timedelta(days=1)
    lines = [f"Multi-Strategy Fund daily digest, {_uk(now)}, covering since {_uk(day_ago)}"]
    try:
        lines.append(f"Fund equity: {_money(supervisor.fund_equity(store))}")
    except Exception as exc:  # noqa: BLE001 - the digest still goes; the reason is the line
        lines.append(f"Fund equity: can't be marked ({type(exc).__name__})")
    lines.append(t5_day(store, now))
    strategies = [s for s in store.sleeves() if not is_backtest(s.name) and not supervisor.outside_fund_book(store, s)]
    lines.append("Strategies (equity at last mark, cash P&L since the last digest):")
    for s in strategies:
        last = store.last_equity(s.name)
        eq = _money(last["equity"]) if last else "not marked"
        lines.append(f"  {s.name}: {eq}, {_money(store.strategy_cash_pnl(s.name, since=day_ago, at=now))}")
    rows = open_risk.book_open_risk(store, lambda s: None)
    if rows:
        lines.append("Open positions:")
        for r in rows:
            last = store.last_equity(r["sleeve"])
            stop = {"stop": "stop in place", "gapped": "PRICE THROUGH ITS STOP", "stopless": "no stop"}[r["basis"]]
            lines.append(f"  {r['sleeve']}: {last['qty']:g} at {_money(last['price'])}, {stop}")
    else:
        lines.append("Open positions: none")
    n_warn, n_err = alert_counts(store, day_ago, now)
    lines.append(f"Alerts since the last digest: {n_warn} warning{'s' if n_warn != 1 else ''}, "
                 f"{n_err} error{'s' if n_err != 1 else ''}")
    issue = backups.problem(backup_dir, now) if backup_dir else None
    lines.append("Backup: " + ("not checked here (BACKUP_DIR)" if not backup_dir else (issue[1] if issue else "ok")))
    recon = store.last_event(None, ("recon_passed", "recon_failed"))
    lines.append("Recon: " + (recon["message"] if recon else "no nightly recon reported yet"))
    clk = store.last_event(None, ("clock_ok", "clock_drift", "clock_unchecked"))
    lines.append("Clock: " + (clk["message"] if clk else "not checked yet"))
    return no_venues("\n".join(lines))


def alert_counts(store, since: datetime, until: datetime) -> tuple[int, int]:
    """Warnings and errors journaled in (since, until], counted by the database with no cap (HoQA A15-M1: a capped
    read of the oldest rows reports none once the journal holds more than the cap). Saved backtests' never count."""
    from sqlalchemy import func, select

    from sleeve_fund.store import _not_backtest, events_t

    q = (select(events_t.c.level, func.count()).where(events_t.c.ts > since, events_t.c.ts <= until,
                                                     _not_backtest(events_t.c.sleeve),
                                                     events_t.c.level.in_(("warning", "error")))
         .group_by(events_t.c.level))
    with store.engine.connect() as c:
        n = dict(c.execute(q).all())
    return n.get("warning", 0), n.get("error", 0)


def _aware(ts: datetime) -> datetime:
    from datetime import UTC

    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


class DailyDigest:
    """Sends the digest once a UK day at DIGEST_AT or after. A failed send is journaled once (info, so it isn't itself
    forwarded) and retried every step; the day it went is read back from the journal, so a restart never sends it
    twice."""

    def __init__(self, store, environ=None, send=None, at: time = DIGEST_AT) -> None:
        env = os.environ if environ is None else environ
        self.store, self.at = store, at
        self.url = env.get("ALERT_WEBHOOK_URL") or None
        self.backup_dir = env.get("BACKUP_DIR") or None
        if send is None:
            from sleeve_fund.alerts import post as send
        self.send = send
        self._sent: str | None = None
        self._sent_at: datetime | None = None
        self._failing = False
        self._first = True

    def _sent_day(self) -> str | None:
        if self._sent is None:
            last = self.store.last_event(None, ("digest_sent",))
            m = _DAY.search(last["message"]) if last else None
            self._sent, self._sent_at = (m.group(1) if m else ""), (last["ts"] if last else None)
        return self._sent

    def step(self, now: datetime) -> None:
        if not self.url:
            return
        local = now.astimezone(UK)
        day = f"{local:%Y-%m-%d}"
        first, self._first = self._first, False
        due = local.replace(hour=self.at.hour, minute=self.at.minute, second=0, microsecond=0)
        if first and self._sent_day() == "" and local >= due + FIRST_GRACE:
            # None has ever gone and this is a first start well after 08:00 (a deploy): the first digest is the next
            # 08:00's, not one at deploy time. A supervisor already running steps within a minute of 08:00.
            self._sent = day
            return
        if local.time() < self.at or self._sent_day() == day:
            return
        from pathlib import Path

        from sleeve_fund.alerts import where

        try:
            self.send(self.url, digest(self.store, now, Path(self.backup_dir) if self.backup_dir else None,
                                       since=self._sent_at))
        except Exception as exc:  # noqa: BLE001 - kept and tried again next step
            if not self._failing:
                self._failing = True
                self.store.event(None, "info", "digest_send_failed",
                                 f"couldn't send the daily digest to {where(self.url)}: {type(exc).__name__}; "
                                 "retrying every minute", ts=now)
            return
        self._failing, self._sent, self._sent_at = False, day, now
        self.store.event(None, "info", "digest_sent", f"daily digest for {day} sent to {where(self.url)}", ts=now)


# F-RX-2 (Advisor, 15 Oct): repeated rejected exits -------------------------------------------------------------------
EXIT_REJECTS = 3  # closing orders in a row, per Strategy row, no time window
EXIT_REJECT_REALERT = timedelta(minutes=30)
EXIT_REJECTED = ("rejected", "denied", "unfilled_unknown")  # unfilled_unknown: confirmed no fill (DA 15 Oct)
EXIT_WENT = ("accepted", "partially_filled", "filled", "triggered")  # a close that went through ends the streak
REJECTS_OPEN, REJECTS_STILL, REJECTS_CLEARED = "exit_rejected_repeatedly", "exit_rejected_still", "exit_rejects_cleared"


def exit_reject_streak(rows: list[dict]) -> int:
    """How many closing orders in a row, from the newest, were rejected, denied or confirmed unfilled. One that went
    through ends it; one still in flight, cancelled or expired neither counts nor ends it."""
    n = 0
    for r in reversed(rows):
        if r["status"] in EXIT_REJECTED:
            n += 1
        elif r["status"] in EXIT_WENT:
            break
    return n


class ExitRejectWatch:
    """F-RX-2: closing orders rejected EXIT_REJECTS times in a row on one Strategy row that holds a position, read through
    QD's F-RX-1 reader
    (Store.closing_orders, which owns which intents close). The episode lives in the journal (the latest open or
    cleared event per row), so a restart neither opens it again nor re-alerts it, and the gate reads the same answer
    (paper.runtime.exit_reject_block). Alert only: nothing here changes an order."""

    def check(self, store, now: datetime) -> None:
        from sleeve_fund.store import is_backtest

        for s in store.sleeves():
            if is_backtest(s.name):
                continue
            last = store.last_event(s.name, (REJECTS_OPEN, REJECTS_CLEARED))
            is_open = last is not None and last["kind"] == REJECTS_OPEN
            cleared = store.last_event(s.name, (REJECTS_CLEARED,))
            streak = exit_reject_streak(store.closing_orders(s.name, cleared["ts"] if cleared else None))
            inst = no_venues(s.instrument)
            # Advisor F-RX-2: cleared by a closing fill or flat; a reduce-only reject on a flat row is benign (A1)
            flat = store.lifecycle_flat(s.name)[1]
            if is_open and flat:
                store.event(s.name, "info", REJECTS_CLEARED,
                            f"The position on {inst} is flat: entries on {inst} are allowed again", ts=now)
            elif not is_open and streak >= EXIT_REJECTS and not flat:
                store.event(s.name, "error", REJECTS_OPEN, no_venues(
                    f"Closing orders on {inst} rejected {streak} times in a row. No new entries on {inst} in the fund "
                    "until one goes through; exits and stops still go out. Check it now"), ts=now)
            elif is_open and streak < EXIT_REJECTS:
                store.event(s.name, "info", REJECTS_CLEARED,
                            f"A closing order on {inst} went through: entries on {inst} are allowed again", ts=now)
            elif is_open and now - _aware(last["ts"]) >= EXIT_REJECT_REALERT:
                still = store.last_event(s.name, (REJECTS_STILL,))
                if still is None or still["ts"] < last["ts"]:
                    store.event(s.name, "error", REJECTS_STILL, no_venues(
                        f"Closing orders on {inst} are still being rejected ({_minutes(now - _aware(last['ts']))} "
                        f"since the alert). No new entries on {inst}; exits and stops still go out"), ts=now)
