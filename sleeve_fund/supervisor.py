"""Keeps each sleeve's paper process matching what the PM asked for.

One OS process per sleeve (a crash or leak hits one sleeve only). Every few
seconds the supervisor compares the database's desired_state with what is
running, starts or stops processes, restarts crashed ones with backoff, and
restarts any whose heartbeat goes stale.

    python -m sleeve_fund.supervisor seed configs/sleeves/*.toml
    python -m sleeve_fund.supervisor run
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sleeve_fund import accounts
from sleeve_fund.alerts import Forwarder
from sleeve_fund.paper.safety import credential_var
from sleeve_fund.paper.config import load_sleeve, to_store_kwargs
from sleeve_fund.store import Sleeve, Store, utcnow

POLL_SECONDS = 5
KEY_CHECK_EVERY = 12  # polls between key-presence checks: about a minute
FEE_CHECK_EVERY = 720  # polls between fee-schedule reads from connected accounts: about an hour
ALERT_EVERY = 12  # polls between alert sends and uptime pings: about a minute
HEARTBEAT_STALE = timedelta(minutes=3)
STARTUP_GRACE = timedelta(minutes=3)
MAX_BACKOFF = 300


@dataclass
class Proc:
    popen: subprocess.Popen | None = None
    started_at: datetime | None = None
    crashes: int = 0
    next_start: datetime = field(default_factory=lambda: datetime.min.replace(tzinfo=utcnow().tzinfo))

    @property
    def alive(self) -> bool:
        return self.popen is not None and self.popen.poll() is None


def decide(sleeve: Sleeve, proc: Proc, now: datetime) -> str:
    """One of: start, stop, restart_stale, crashed, wait, none."""
    want = sleeve.desired_state == "running"
    if want and not proc.alive:
        if proc.popen is not None:
            return "crashed"
        return "start" if now >= proc.next_start else "wait"
    if not want and proc.alive:
        return "stop"
    if want and proc.alive and proc.started_at and now - proc.started_at > STARTUP_GRACE:
        if sleeve.heartbeat_at is None or now - sleeve.heartbeat_at > HEARTBEAT_STALE:
            return "restart_stale"
    return "none"


class Supervisor:
    def __init__(self, store: Store, python: str = sys.executable) -> None:
        self.store = store
        self.python = python
        self.procs: dict[str, Proc] = {}
        self._stopping = False

    def _start(self, name: str, proc: Proc) -> None:
        # Paper processes never need a venue key, so they don't inherit one.
        env = {k: v for k, v in os.environ.items() if not credential_var(k)}
        proc.popen = subprocess.Popen([self.python, "-m", "sleeve_fund.paper", "--db-sleeve", name], env=env)
        proc.started_at = utcnow()
        self.store.event(name, "info", "process_start", f"paper process started (pid {proc.popen.pid})")

    def _stop(self, name: str, proc: Proc, why: str) -> None:
        if proc.alive:
            proc.popen.send_signal(signal.SIGINT)
            try:
                proc.popen.wait(timeout=45)
            except subprocess.TimeoutExpired:
                proc.popen.kill()
                proc.popen.wait()
        proc.popen = None
        self.store.event(name, "info", "process_stop", why)

    def step(self) -> None:
        now = utcnow()
        for sleeve in self.store.sleeves():
            proc = self.procs.setdefault(sleeve.name, Proc())
            action = decide(sleeve, proc, now)
            if action == "start":
                self._start(sleeve.name, proc)
            elif action == "crashed":
                code = proc.popen.returncode
                proc.popen = None
                proc.crashes += 1
                delay = min(MAX_BACKOFF, 10 * 2 ** (proc.crashes - 1))
                proc.next_start = now + timedelta(seconds=delay)
                self.store.set_status(sleeve.name, "error", f"process exited with code {code}; restarting in {delay}s")
                self.store.event(sleeve.name, "error", "process_crash", f"exit code {code}; restart in {delay}s")
            elif action == "stop":
                self._stop(sleeve.name, proc, "stopped by PM")
                self.store.set_status(sleeve.name, "stopped", "stopped by PM")
            elif action == "restart_stale":
                self.store.event(sleeve.name, "error", "heartbeat_stale", "no heartbeat for 3 minutes; restarting")
                self._stop(sleeve.name, proc, "restart after stale heartbeat")
                self._start(sleeve.name, proc)
            elif action == "none" and proc.alive and proc.crashes and now - proc.started_at > STARTUP_GRACE:
                proc.crashes = 0  # healthy again

    def check_keys(self) -> None:
        """Tell the dashboard which live accounts have a Kraken key on this server (presence only)."""
        live = [a for a in self.store.accounts() if a["kind"] == "live"]
        if live:
            self.store.report_keys({a["name"]: accounts.key_present(a["name"], venue=a["venue"]) for a in live})

    def check_fees(self, environ=None) -> None:
        """Read each connected live account's fee schedule from its venue, so backtests and paper
        charge what the exchange actually charges that account. Query-only; the key never leaves
        this process."""
        from sleeve_fund.venues import venue as venue_profile

        for a in self.store.accounts():
            if a["kind"] != "live":
                continue
            creds = accounts.credentials(a["name"], a["venue"], environ)
            profile = venue_profile(a["venue"])
            if creds is None or profile.fetch_fees is None:
                continue
            try:
                fees = profile.fetch_fees(*creds)
                self.store.record_fees(profile.name, a["name"], float(fees.maker), float(fees.taker))
            except Exception as exc:  # noqa: BLE001 - keep the last good schedule; show why on the dashboard
                msg = str(exc).replace(creds[0], "***").replace(creds[1], "***")
                self.store.event(None, "warning", "fee_fetch_failed", f"{a['name']}: {msg}")

    def _send_alerts(self, alerts: Forwarder) -> None:
        try:
            alerts.step()
        except Exception as exc:  # noqa: BLE001 - the dashboard shows it; the next minute tries again
            self.store.event(None, "error", "supervisor_error", f"alerts: {exc!r}")

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stopping", True))
        alerts = Forwarder(self.store)
        self.store.event(None, "info", "supervisor_start", "supervisor started")
        self.store.event(None, "info", "alerts_config", alerts.describe())
        loops, sender = 0, None
        while not self._stopping:
            try:
                if loops % ALERT_EVERY == 0 and (sender is None or not sender.is_alive()):
                    # Its own thread: a slow webhook or monitor never holds up supervising the sleeves.
                    sender = threading.Thread(target=self._send_alerts, args=(alerts,), name="alerts", daemon=True)
                    sender.start()
                if loops % KEY_CHECK_EVERY == 0:
                    self.check_keys()
                if loops % FEE_CHECK_EVERY == 0:
                    self.check_fees()
                loops += 1
                self.step()
            except Exception as exc:  # keep supervising; the dashboard shows the error
                self.store.event(None, "error", "supervisor_error", repr(exc))
            time.sleep(POLL_SECONDS)
        for name, proc in self.procs.items():
            self._stop(name, proc, "supervisor shutting down")


def seed(store: Store, paths: list[str]) -> list[str]:
    """Insert sleeves from TOML files that aren't in the database yet. Never overwrites."""
    existing = {s.name for s in store.sleeves()}
    added = []
    for path in paths:
        cfg = load_sleeve(path)
        if cfg.name in existing:
            continue
        store.create_sleeve(**to_store_kwargs(cfg))
        store.decide("system", "create", f"seeded from {path}", cfg.name)
        added.append(cfg.name)
    return added


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.supervisor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sd = sub.add_parser("seed", help="add sleeves from TOML files if missing")
    sd.add_argument("paths", nargs="+")
    sub.add_parser("run", help="supervise sleeve processes until stopped")
    args = ap.parse_args(argv)
    store = Store()
    if args.cmd == "seed":
        print("added:", seed(store, args.paths) or "nothing new")
    else:
        Supervisor(store).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
