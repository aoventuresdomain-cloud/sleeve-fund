"""Outbound alerts: the journal's warnings and errors, sent to a chat the PM reads, and a ping that
an outside uptime monitor watches, so a server that dies says so too.

Both are set on the server only, never in code:
    ALERT_WEBHOOK_URL     where alerts go. A JSON POST of {"text": ..., "content": ...}, which Slack and
                          Discord webhooks read; Telegram reads it too with the bot's sendMessage URL and
                          ?chat_id=... on the end. An ntfy.sh topic URL gets the plain text.
    HEALTHCHECK_PING_URL  pinged every minute (e.g. a healthchecks.io check); the monitor alerts when
                          the pings stop, which nothing on this server can do for itself.
The supervisor runs both, on a thread of their own, so neither the sleeves nor the supervisor's
loop ever waits on the network. It also raises an alert when the nightly database backup is late or
failed (BACKUP_DIR, read only), when the server clock drifts from a time server (sleeve_fund.clock), and
when the nightly recon finds a dashboard that disagrees with the journal (RECON_DIR, sleeve_fund.nightly_recon).

A send that times out after the far end got it is sent again next minute: a duplicate is the safer
way to be wrong.
"""

from __future__ import annotations

import json
import os
import re
import threading
import urllib.request
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from sleeve_fund import backups, clock, nightly_recon, ops_health
from sleeve_fund.store import Store, utcnow
from sleeve_fund.wording import leg_name, no_venues

MAX_LINES = 10  # alerts listed in one message; the rest are counted
TIMEOUT = 8  # per socket read
TOTAL = 20  # for a whole send: a far end that trickles bytes resets TIMEOUT on every read
BACKUP_CHECK = timedelta(hours=1)
CLOCK_CHECK = timedelta(minutes=5)
_THROUGH = re.compile(r"through event #(\d+)")


def post(url: str, text: str) -> None:
    if "ntfy" in (urlsplit(url).hostname or ""):
        req = urllib.request.Request(url, data=text.encode(), method="POST")
    else:
        req = urllib.request.Request(url, data=json.dumps({"text": text, "content": text}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    _within(TOTAL, _open, req)


def ping(url: str) -> None:
    _within(TOTAL, _open, url)


def _open(req) -> None:
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        r.read()


def _within(seconds: float, fn, *args) -> None:
    """fn(*args), or TimeoutError once `seconds` have passed. A send still going then finishes or
    fails on a thread of its own, bounded by its socket timeout; its result is dropped."""
    out: list[BaseException | None] = []

    def run():
        try:
            fn(*args)
            out.append(None)
        except BaseException as exc:  # noqa: BLE001 - handed back to the caller
            out.append(exc)

    t = threading.Thread(target=run, daemon=True, name="alert-send")
    t.start()
    t.join(seconds)
    if not out:
        raise TimeoutError(f"no answer within {seconds:g} s")
    if out[0] is not None:
        raise out[0]


def where(url: str | None) -> str:
    """The host an alert goes to, for the dashboard: never the path, which often holds the token."""
    return (urlsplit(url).hostname or "?") if url else ""


def message(events: list[dict]) -> str:
    lines = [f"Multi-Strategy Fund: {len(events)} alert{'s' if len(events) != 1 else ''}"]
    for e in events[:MAX_LINES]:
        lines.append(f"[{e['level']}] {leg_name(e['sleeve']) or 'system'}: {no_venues(e['message'])}")
    if len(events) > MAX_LINES:
        lines.append(f"and {len(events) - MAX_LINES} more on the Alerts page")
    return "\n".join(lines)


class Forwarder:
    """Sends warnings and errors journaled since the last send, in one message. Picks up where the
    last send stopped (each send journals how far it got), so alerts raised while the supervisor was
    down still go out after a restart; a failed send is retried next time."""

    def __init__(self, store: Store, environ=None, send=post, pinger=ping, now=utcnow,
                 clock_offset=clock.offset_ms) -> None:
        env = os.environ if environ is None else environ
        self.store, self.send, self.pinger, self.now = store, send, pinger, now
        self.url, self.ping_url = env.get("ALERT_WEBHOOK_URL") or None, env.get("HEALTHCHECK_PING_URL") or None
        self.backup_dir = Path(env["BACKUP_DIR"]) if env.get("BACKUP_DIR") else None
        self.recon_dir = Path(env["RECON_DIR"]) if env.get("RECON_DIR") else None
        self.recon_alert = env.get("NIGHTLY_ALERT") == "on"  # off (the default): a FAIL is journaled, not sent
        self._recon_tried = None  # the night last tried: a record that can't be written isn't retried every minute
        self.cursor = self._resume()
        self._failing = False
        self._backup_checked = None
        self._backup_said = None
        self.clock_server, self.clock_offset = clock.server(env), clock_offset
        self._clock_checked = None
        self._clock_said = None  # "ok", "drift" or "unreachable": what was last journaled
        # A15 (Drop 3a): the open positions with no live stop and the 08:00 UK digest, checked each step
        self.no_stop = ops_health.NoStopWatch()
        self.digest = ops_health.DailyDigest(store, env, send=send)
        self.exit_rejects = ops_health.ExitRejectWatch()  # F-RX-2
        self._ops_failing: set[str] = set()
        self._tables_said: set[str] = set()
        self._terms_said: set[tuple] = set()  # FEE-LIVE: (kind, instrument, state, published, day) already said  # VS-1-TABLE: old maintenance tables already warned about

    def _resume(self) -> int:
        last = self.store.last_event(None, ("alerts_sent",)) if self.url else None
        m = _THROUGH.search(last["message"]) if last else None
        newest = self.store.last_event_id()
        return min(int(m.group(1)), newest) if m else newest

    def describe(self) -> str:
        alerts = f"Alerts go to {where(self.url)}" if self.url else "Alerts are not set up (ALERT_WEBHOOK_URL)"
        up = (f"uptime pings go to {where(self.ping_url)}" if self.ping_url
              else "no uptime ping (HEALTHCHECK_PING_URL)")
        return f"{alerts}; {up}."

    def check_backup(self) -> None:
        """Hourly: a late or failed backup becomes a warning, said again only when the problem changes."""
        now = self.now()
        if self.backup_dir is None or (self._backup_checked and now - self._backup_checked < BACKUP_CHECK):
            return
        self._backup_checked = now
        issue = backups.problem(self.backup_dir, now)
        kind = issue[0] if issue else None
        if issue and kind != self._backup_said:
            self.store.event(None, "warning", "backup_problem", issue[1])
        self._backup_said = kind

    def check_recon(self) -> None:
        """Nightly (sleeve_fund.nightly_recon): every Strategy's dashboard against its journal, read-only, into a dated
        record. A FAIL is a warning, the level stale data alerts at, once NIGHTLY_ALERT is on; until then (the default)
        it is journaled as info and reaches no one. A PASS is journaled as info."""
        now = self.now()
        if self.recon_dir is None or self.store.url is None or self._recon_tried == now.date():
            return
        if not nightly_recon.due(self.recon_dir, now):
            return
        self._recon_tried = now.date()
        try:
            ok, path, fails = nightly_recon.run(self.recon_dir, self.store.url, now)
        except OSError as exc:  # the record couldn't be written: the night isn't proven, so it's a fail
            self.store.event(None, "warning" if self.recon_alert else "info", "recon_failed",
                             f"nightly recon couldn't write its record in {self.recon_dir}: {type(exc).__name__}")
            return
        if ok:
            self.store.event(None, "info", "recon_passed", f"nightly recon passed: every Strategy's dashboard "
                             f"matches its journal ({path.name})")
            return
        more = f"; and {len(fails) - 3} more" if len(fails) > 3 else ""
        quiet = "" if self.recon_alert else " (not sent: NIGHTLY_ALERT is off)"
        self.store.event(None, "warning" if self.recon_alert else "info", "recon_failed",
                         f"nightly recon FAILED ({path.name}): " + "; ".join(fails[:3]) + more + quiet)

    def check_clock(self) -> None:
        """Every CLOCK_CHECK: the server clock's offset from a time server. The first reading is journaled so
        it can be seen; after that only a change: past clock.MAX_OFFSET_MS is a warning, back within it is
        said once, and a time server that can't be reached is said once (the clock isn't known bad)."""
        now = self.now()
        if self.clock_server is None or (self._clock_checked and now - self._clock_checked < CLOCK_CHECK):
            return
        self._clock_checked = now
        try:
            ms = self.clock_offset(self.clock_server)
        except OSError as exc:
            if self._clock_said != "unreachable":
                self.store.event(None, "info", "clock_unchecked",
                                 f"can't check the server clock against {self.clock_server}: {exc!r}")
            self._clock_said = "unreachable"
            return
        state = "drift" if abs(ms) > clock.MAX_OFFSET_MS else "ok"
        if state == self._clock_said:
            return
        if state == "drift":
            self.store.event(None, "warning", "clock_drift",
                             f"the server clock is {abs(ms):,.0f} ms {'behind' if ms > 0 else 'ahead of'} "
                             f"{self.clock_server} (alert above {clock.MAX_OFFSET_MS} ms): order and fill times "
                             "no longer compare with the venue's")
        else:
            self.store.event(None, "info", "clock_ok", f"the server clock is within {abs(ms):,.0f} ms of "
                             f"{self.clock_server}")
        self._clock_said = state

    def check_tables(self, now) -> None:
        """VS-1-TABLE (Advisor: an alert): a live perp Strategy's maintenance table read more than 30 days ago is a
        warning, said once per table (the start check notes it too). Replaced by the daily fetch (FEE-LIVE) later."""
        from sleeve_fund import markets

        for line in markets.stale_maintenance(self.store.sleeves(), now.date()):
            if line.split(":")[0] not in self._tables_said:
                self._tables_said.add(line.split(":")[0])
                self.store.event(None, "warning", "maintenance_table_old", line)

    def check_venue_terms(self, now) -> None:
        """FEE-LIVE (Advisor FEE-LIVE-DW): an instrument's fee rates or tiers over 24 h old, refused (over 72 h or never
        fetched), or the published rate standing in for the account's, is journaled once a day while it lasts, so it
        goes out with the other alerts."""
        from sleeve_fund import venue_terms

        for t in venue_terms.live_terms(self.store, now):
            published = t.source == venue_terms.PUBLISHED_FEES
            if t.state == venue_terms.FRESH and not published:
                continue
            key = (t.kind, t.instrument, t.state, published, now.date())
            if key in self._terms_said:
                continue
            self._terms_said.add(key)
            level = "error" if t.state == venue_terms.REFUSED else "warning"
            self.store.event(None, level, f"venue_terms_{t.state if t.state != 'fresh' else 'published'}", t.words)

    def _ops_check(self, what: str, check) -> None:
        """One A15 check; a failure is one error until it works again (then one info line), and never stops the
        alerts or the uptime ping that follow it."""
        try:
            check()
        except Exception as exc:  # noqa: BLE001 - said once; tried again next step
            if what not in self._ops_failing:
                self._ops_failing.add(what)
                self.store.event(None, "error", "ops_check_failed",
                                 f"the {what} check failed: {type(exc).__name__}; retrying every minute")
            return
        if what in self._ops_failing:
            self._ops_failing.discard(what)
            self.store.event(None, "info", "ops_check_back", f"the {what} check works again")

    def step(self) -> None:
        self.check_backup()
        self.check_clock()
        self.check_recon()
        now = self.now()
        self._ops_check("no-stop", lambda: self.no_stop.check(self.store, now))
        self._ops_check("rejected-exit", lambda: self.exit_rejects.check(self.store, now))
        self._ops_check("daily digest", lambda: self.digest.step(now))
        self._ops_check("maintenance table age", lambda: self.check_tables(now))
        self._ops_check("venue fees and tiers", lambda: self.check_venue_terms(now))
        if self.ping_url:
            try:
                self.pinger(self.ping_url)
            except Exception:  # noqa: BLE001 - the monitor notices the missing ping; nothing to add here
                pass
        if not self.url:
            return
        events = self.store.events_after(self.cursor)
        if not events:
            return
        try:
            self.send(self.url, message(events))
        except Exception as exc:  # noqa: BLE001 - keep the cursor and try again next time
            if not self._failing:
                self._failing = True
                # Info, so the failure isn't itself forwarded in a loop.
                self.store.event(None, "info", "alert_send_failed", f"couldn't send alerts to {where(self.url)}: "
                                 f"{type(exc).__name__}; retrying every minute")
            return
        self._failing = False
        self.cursor = events[-1]["id"]
        self.store.event(None, "info", "alerts_sent", f"{len(events)} sent to {where(self.url)}, through event #{self.cursor}")
