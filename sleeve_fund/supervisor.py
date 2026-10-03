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
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sleeve_fund.paper.config import load_sleeve, to_store_kwargs
from sleeve_fund.store import Sleeve, Store, utcnow

POLL_SECONDS = 5
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
        proc.popen = subprocess.Popen([self.python, "-m", "sleeve_fund.paper", "--db-sleeve", name])
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

    def run(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "_stopping", True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, "_stopping", True))
        self.store.event(None, "info", "supervisor_start", "supervisor started")
        while not self._stopping:
            try:
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
