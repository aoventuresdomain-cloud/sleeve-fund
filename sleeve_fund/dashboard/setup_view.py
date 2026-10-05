"""The Setup page: where the fund stands on the path to live, and the key facts of each setting area.

Read-only. It reports state; it approves nothing. G2 stays the PM's decision and live trading stays
locked in the shell until it is made.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from sleeve_fund import backups
from sleeve_fund.mirror import FLAG_ENV
from sleeve_fund.store import Store

# The path to live, in order: (key, label, what the step means).
STEPS = [
    ("research", "Research", "G1 studies"),
    ("paper", "Paper", "strategies trade simulated money"),
    ("demo", "Demo check", "the demo mirror copies paper fills to a venue's demo account"),
    ("g2", "G2 approval", "6+ weeks of paper, approved by you"),
    ("live", "Live", "keys installed and the server's live switch on"),
]
_ON = ("1", "true", "on", "yes")


def mirror_state(store: Store, sleeves: list, environ: dict | None = None) -> dict:
    """Whether the demo mirror is copying fills now. It runs in its own container, so the dashboard reads
    the evidence it leaves: a running strategy that asks for the mirror (params demo_mirror) and either a
    fill the mirror copied, a mirror start that named a demo account, or DEMO_MIRROR on here too."""
    env = os.environ if environ is None else environ
    asked = [s for s in sleeves if s.params.get("demo_mirror") and s.desired_state == "running"]
    on = []
    for s in asked:
        copied = any(r["status"] == "filled" for r in store.mirror_rows(s.name, limit=50))
        start = store.last_event(s.name, ("mirror_start",))
        named = bool(start) and "once that is set up" not in start["message"]
        if copied or named or env.get(FLAG_ENV, "").strip().lower() in _ON:
            on.append(s.name)
    return {"on": bool(on), "strategies": on, "asked": [s.name for s in asked],
            "words": "mirroring" if on else ("asked, not running" if asked else "off")}


def stage(shell: dict, sleeves: list, mirror: dict) -> int:
    """Index into STEPS of where the fund is now. Live mode is Live; G2 approved (live unlocked) leaves only
    the keys and the switch; with G2 not approved it is Demo check while the mirror is on, else Paper once a
    strategy is on paper, else Research."""
    if shell.get("mode") == "live" or not shell.get("live_locked", True):
        return 4
    if mirror["on"]:
        return 2
    return 1 if sleeves else 0


def path(here: int) -> list[dict]:
    return [{"key": k, "label": label, "note": note, "n": i + 1,
             "state": "done" if i < here else "here" if i == here else "todo"}
            for i, (k, label, note) in enumerate(STEPS)]


NEXT = {
    0: "Research. Next: put a strategy that passed G1 on paper.",
    1: "Paper trading. Next: the demo check, then your G2 approval.",
    2: "Demo check in progress: paper fills are copied to a demo account. Next: your G2 approval.",
    3: "Waiting for your G2 approval.",
    4: "G2 approved. Next: install the live keys and turn on the server's live switch.",
}


def outside_alerts(store: Store) -> dict:
    """Outside alerts as the supervisor last reported them (its alerts_config event)."""
    ev = store.last_event(None, ("alerts_config",))
    msg = ev["message"] if ev else ""
    on = msg.startswith("Alerts go to ")
    host = msg[len("Alerts go to "):].split(";")[0].strip() if on else ""
    return {"on": on, "host": host, "where": "Telegram" if "telegram" in host.lower() else host,
            "ping": bool(msg) and "no uptime ping" not in msg, "message": msg}


def backup_state(now: datetime) -> dict:
    b = backups.latest(Path(os.environ.get("BACKUP_DIR", "/data/backups")), now)
    check = (b or {}).get("check")
    restore = "not run yet" if not check else "passed" if check.get("ok") else "failed"
    good = bool(b and b.get("name") and not b.get("stale") and check and check.get("ok"))
    return {"b": b, "restore": restore, "good": good}
