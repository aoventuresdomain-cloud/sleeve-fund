"""The nightly database backups (deploy/backup.sh), read by the Operations page and the supervisor's
alerts: the newest dump, and whether the last night's dump restored."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

STALE = timedelta(hours=26)  # nightly, with room for a slow dump


def latest(folder: Path, now: datetime) -> dict | None:
    """The newest dump and the last run's result, or None if the folder has neither."""
    try:
        dumps = list(folder.glob("*.dump"))
    except OSError:
        return None
    check = _status(folder / "status.json")
    if not dumps and check is None:
        return None
    out = {"name": None, "bytes": 0, "age": None, "stale": True, "count": len(dumps), "check": check}
    if dumps:
        newest = max(dumps, key=lambda f: f.stat().st_mtime)
        st = newest.stat()
        age = now - datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
        out.update(name=newest.name, bytes=st.st_size, age=age, stale=age > STALE)
    return out


def _status(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def problem(folder: Path, now: datetime) -> str | None:
    """What is wrong with the backups, in a sentence for an alert, or None if the last one is sound."""
    b = latest(folder, now)
    if b is None:
        return "No database backup has been written yet"
    check = b["check"]
    if check is not None and not check.get("ok"):
        return f"The last database backup failed: {check.get('message') or 'no reason given'}"
    if b["stale"]:
        hours = b["age"].total_seconds() / 3600 if b["age"] is not None else None
        return (f"The newest database backup is {hours:.0f} hours old" if hours is not None
                else "No database dump is on disk")
    return None
