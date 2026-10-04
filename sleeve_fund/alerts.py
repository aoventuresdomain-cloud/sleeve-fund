"""Outbound alerts: the journal's warnings and errors, sent to a chat the PM reads, and a ping that
an outside uptime monitor watches, so a server that dies says so too.

Both are set on the server only, never in code:
    ALERT_WEBHOOK_URL     where alerts go. A JSON POST of {"text": ..., "content": ...}, which Slack and
                          Discord webhooks read; Telegram reads it too with the bot's sendMessage URL and
                          ?chat_id=... on the end. An ntfy.sh topic URL gets the plain text.
    HEALTHCHECK_PING_URL  pinged every minute (e.g. a healthchecks.io check); the monitor alerts when
                          the pings stop, which nothing on this server can do for itself.
The supervisor runs both: sleeve processes never wait on the network to journal an event.
"""

from __future__ import annotations

import json
import os
import urllib.request
from urllib.parse import urlsplit

from sleeve_fund.store import Store

MAX_LINES = 10  # alerts listed in one message; the rest are counted
TIMEOUT = 10


def post(url: str, text: str) -> None:
    if "ntfy" in (urlsplit(url).hostname or ""):
        req = urllib.request.Request(url, data=text.encode(), method="POST")
    else:
        req = urllib.request.Request(url, data=json.dumps({"text": text, "content": text}).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        r.read()


def ping(url: str) -> None:
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        r.read()


def where(url: str | None) -> str:
    """The host an alert goes to, for the dashboard: never the path, which often holds the token."""
    return (urlsplit(url).hostname or "?") if url else ""


def message(events: list[dict]) -> str:
    lines = [f"Multi-Strategy Fund: {len(events)} alert{'s' if len(events) != 1 else ''}"]
    for e in events[:MAX_LINES]:
        lines.append(f"[{e['level']}] {e['sleeve'] or 'system'}: {e['message']}")
    if len(events) > MAX_LINES:
        lines.append(f"and {len(events) - MAX_LINES} more on the Alerts page")
    return "\n".join(lines)


class Forwarder:
    """Sends warnings and errors journaled since the last send, in one message. Starts from the
    newest event, so a restart doesn't resend history; a failed send is retried next time."""

    def __init__(self, store: Store, environ=None, send=post, pinger=ping) -> None:
        env = os.environ if environ is None else environ
        self.store, self.send, self.pinger = store, send, pinger
        self.url, self.ping_url = env.get("ALERT_WEBHOOK_URL") or None, env.get("HEALTHCHECK_PING_URL") or None
        self.cursor = store.last_event_id()
        self._failing = False

    def describe(self) -> str:
        alerts = f"Alerts go to {where(self.url)}" if self.url else "Alerts are not set up (ALERT_WEBHOOK_URL)"
        up = (f"uptime pings go to {where(self.ping_url)}" if self.ping_url
              else "no uptime ping (HEALTHCHECK_PING_URL)")
        return f"{alerts}; {up}."

    def step(self) -> None:
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
