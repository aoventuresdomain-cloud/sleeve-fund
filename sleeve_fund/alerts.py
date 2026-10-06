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
failed (BACKUP_DIR, read only), and when the server clock drifts from a time server (sleeve_fund.clock).

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

from sleeve_fund import backups, clock
from sleeve_fund.store import Store, utcnow
from sleeve_fund.wording import no_venues

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
        lines.append(f"[{e['level']}] {e['sleeve'] or 'system'}: {no_venues(e['message'])}")
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
        self.cursor = self._resume()
        self._failing = False
        self._backup_checked = None
        self._backup_said = None
        self.clock_server, self.clock_offset = clock.server(env), clock_offset
        self._clock_checked = None
        self._clock_said = None  # "ok", "drift" or "unreachable": what was last journaled

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

    def step(self) -> None:
        self.check_backup()
        self.check_clock()
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
