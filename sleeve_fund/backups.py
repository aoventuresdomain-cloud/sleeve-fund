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
    """The last run's result; None if there is none yet. A result that can't be read is a failure,
    never a pass: the backup that wrote it went wrong somehow (review round 7)."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"ok": False, "message": f"couldn't read the backup's status file ({exc.strerror or exc})"}
    try:
        out = json.loads(text)
    except ValueError:
        return {"ok": False, "message": "the backup's status file isn't valid, so the last run's result is unknown"}
    return out if isinstance(out, dict) else {"ok": False, "message": "the backup's status file isn't valid"}


def problem(folder: Path, now: datetime) -> tuple[str, str] | None:
    """What is wrong with the backups, as (kind, a sentence for an alert), or None if the last one is
    sound. The kind stays the same while the problem does, though the sentence may not (an age grows)."""
    b = latest(folder, now)
    if b is None:
        return "none", "No database backup has been written yet"
    check = b["check"]
    if check is not None and not check.get("ok"):
        return "failed", f"The last database backup failed: {check.get('message') or 'no reason given'}"
    if b["stale"]:
        if b["age"] is None:
            return "no_dump", "No database dump is on disk"
        return "stale", f"The newest database backup is {b['age'].total_seconds() / 3600:.0f} hours old"
    return None
