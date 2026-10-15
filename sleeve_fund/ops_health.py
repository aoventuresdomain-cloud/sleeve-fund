"""A15 ops-health minimum (G2; Done when in hoe/bgl-plan.md under "### A15"): the alerts the PM needs to know the fund
is alive and protected, on top of what the forwarder already sends (sleeve_fund.alerts).

- HubStaleness: one warning per episode when a hub has had no trade or quote on an instrument for STALE_AFTER, and one
  line when it is back. Each Strategy already holds its own entries on a stale price (base.STALE_PRICE_WARN_MINUTES);
  this says which hub and instrument, once, instead of one note per Strategy.
- NoStopWatch: one error per episode when an open perp position of a Model built with a stop has had no live stop
  (none journaled, or the price has gone through it) for one bar plus NO_STOP_GRACE. A Model built stopless is held to
  1x by design and is not watched.
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
NO_STOP_GRACE = timedelta(seconds=60)  # on top of one bar: a stop is set again at the next bar after a restart
DIGEST_AT = time(8, 0)  # UK
STOP_KEYS = ("stop_loss", "stop_atr", "stop_swing_bars")  # strategies.base._stop_designed
_BAR = re.compile(r"^(\d+)-(SECOND|MINUTE|HOUR|DAY)\b")
_UNIT = {"SECOND": 1, "MINUTE": 60, "HOUR": 3_600, "DAY": 86_400}
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


def stop_designed(params: dict | None) -> bool:
    return any((params or {}).get(k) for k in STOP_KEYS)


def bar_seconds(bar_spec: str) -> int:
    """The decision bar's length from a spec such as 1-MINUTE-LAST-INTERNAL; a minute when it can't be read."""
    m = _BAR.match(bar_spec or "")
    return int(m.group(1)) * _UNIT[m.group(2)] if m else 60


class NoStopWatch:
    """Open perp positions of stop-built Models with no live stop, from the same reader as the dashboard's Open risk
    (open_risk.book_open_risk), so the alert and the tile never disagree on which positions have no stop."""

    def __init__(self, grace: timedelta = NO_STOP_GRACE) -> None:
        self.grace = grace
        self.since: dict[str, datetime] = {}  # Strategy -> first seen without a live stop
        self.out: set[str] = set()

    def check(self, store, now: datetime) -> None:
        from sleeve_fund import open_risk

        rows = {r["sleeve"]: r for r in open_risk.book_open_risk(store, lambda s: None)}
        bad = {}
        for name, r in rows.items():
            if r["basis"] == "stop":
                continue
            s = store.sleeve(name)
            if stop_designed(s.params):
                bad[name] = (r["basis"], s)
        for name in sorted(set(self.since) - set(bad)):
            del self.since[name]
            if name in self.out:
                self.out.discard(name)
                why = "has its stop again" if name in rows else "is flat"
                store.event(name, "info", "position_stop_back", f"The open position {why}", ts=now)
        for name, (basis, s) in sorted(bad.items()):
            first = self.since.setdefault(name, now)
            if name in self.out or now - first < timedelta(seconds=bar_seconds(s.bar_spec)) + self.grace:
                continue
            self.out.add(name)
            what = ("the price has gone through its stop and it is still open" if basis == "gapped"
                    else "it has no stop in place")
            store.event(name, "error", "position_without_stop",
                        f"Open position without a live stop for {_minutes(now - first)}: {what}. Check it now", ts=now)


def _money(x) -> str:
    return f"{float(x):,.2f}"


def digest(store, now: datetime, backup_dir=None) -> str:
    """The daily digest's text: the fund, each Strategy, open positions with their stops, the last 24 hours' alerts,
    and the backup, recon and clock checks. Names no venue."""
    from sleeve_fund import backups, open_risk, supervisor
    from sleeve_fund.store import is_backtest

    day_ago = now - timedelta(days=1)
    lines = [f"Multi-Strategy Fund daily digest, {now.astimezone(UK):%a %d %b %H:%M} UK"]
    try:
        lines.append(f"Fund equity: {_money(supervisor.fund_equity(store))}")
    except Exception as exc:  # noqa: BLE001 - the digest still goes; the reason is the line
        lines.append(f"Fund equity: can't be marked ({type(exc).__name__})")
    strategies = [s for s in store.sleeves() if not is_backtest(s.name) and not supervisor.outside_fund_book(store, s)]
    lines.append("Strategies (equity at last mark, cash P&L last 24h):")
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
    events = [e for e in store.events_after(0, limit=100_000) if _aware(e["ts"]) > day_ago]
    n_warn = sum(e["level"] == "warning" for e in events)
    n_err = sum(e["level"] in ("error", "critical") for e in events)
    lines.append(f"Alerts in the last 24h: {n_warn} warning{'s' if n_warn != 1 else ''}, "
                 f"{n_err} error{'s' if n_err != 1 else ''}")
    issue = backups.problem(backup_dir, now) if backup_dir else None
    lines.append("Backup: " + ("not checked here (BACKUP_DIR)" if not backup_dir else (issue[1] if issue else "ok")))
    recon = store.last_event(None, ("recon_passed", "recon_failed"))
    lines.append("Recon: " + (recon["message"] if recon else "no nightly recon reported yet"))
    clk = store.last_event(None, ("clock_ok", "clock_drift", "clock_unchecked"))
    lines.append("Clock: " + (clk["message"] if clk else "not checked yet"))
    return no_venues("\n".join(lines))


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
        self._failing = False

    def _sent_day(self) -> str | None:
        if self._sent is None:
            last = self.store.last_event(None, ("digest_sent",))
            m = _DAY.search(last["message"]) if last else None
            self._sent = m.group(1) if m else ""
        return self._sent

    def step(self, now: datetime) -> None:
        if not self.url:
            return
        local = now.astimezone(UK)
        day = f"{local:%Y-%m-%d}"
        if local.time() < self.at or self._sent_day() == day:
            return
        from pathlib import Path

        from sleeve_fund.alerts import where

        try:
            self.send(self.url, digest(self.store, now, Path(self.backup_dir) if self.backup_dir else None))
        except Exception as exc:  # noqa: BLE001 - kept and tried again next step
            if not self._failing:
                self._failing = True
                self.store.event(None, "info", "digest_send_failed",
                                 f"couldn't send the daily digest to {where(self.url)}: {type(exc).__name__}; "
                                 "retrying every minute", ts=now)
            return
        self._failing, self._sent = False, day
        self.store.event(None, "info", "digest_sent", f"daily digest for {day} sent to {where(self.url)}", ts=now)
