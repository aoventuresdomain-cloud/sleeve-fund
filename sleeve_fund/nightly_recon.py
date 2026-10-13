"""NIGHTLY-RECON (HoE 14 Oct): the start check's dashboard-against-the-journal comparison (start_check.check_books),
run every night during Strategy testing for every Strategy the dashboard shows, archived ones and reset runs aside.
Lifecycle isn't stored yet, so that is every Active Strategy and any winding down too.

Each night after RECON_HOUR UTC the supervisor's alert thread (alerts.Forwarder.check_recon) runs it once and
writes a dated record, RECON_DIR/recon-YYYY-MM-DD.txt, whose first line is PASS or FAIL. A FAIL, or a comparison
that couldn't run, is journaled as a warning, the level stale data uses, so it reaches the PM by the same alert
path. A record already there for the day means the night is done, so a restart doesn't run it twice.

The comparison is strictly read-only, like the start check: it opens the journal read-only (start_check.read_only_engine)
and imports nothing that places an order or opens a network connection. Its only writes are the record file and,
from the caller, the one journal event.

    python -m sleeve_fund.nightly_recon --dir DIR [--db URL]   # once now; exit 0 pass, 1 fail
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from sqlalchemy import make_url

from sleeve_fund.start_check import FAIL, Section, check_books, read_only_store

RECON_HOUR = 1  # UTC: 02:00 UK in summer time, after the day's last marks and clear of the evening's work


def record_path(folder: Path, now: datetime) -> Path:
    return folder / f"recon-{now:%Y-%m-%d}.txt"


def due(folder: Path, now: datetime) -> bool:
    """Tonight's comparison hasn't run yet and it's time it did."""
    return now.hour >= RECON_HOUR and not record_path(folder, now).exists()


def compare(url: str) -> Section:
    """check_books against the journal at url, opened read-only."""
    store = read_only_store(url)
    try:
        return check_books(store)
    finally:
        store.engine.dispose()


def run(folder: Path, url: str, now: datetime) -> tuple[bool, Path, list[str]]:
    """Tonight's comparison, written to its dated record. Returns (passed, the record, the FAIL lines). A comparison
    that can't run is a fail, never a pass."""
    try:
        sec = compare(url)
        lines = [f"  {st:<4}  {what}" for st, what in sec.lines]
        fails = [what for st, what in sec.lines if st == FAIL]
    except Exception as exc:  # noqa: BLE001 - recorded and alerted as a fail
        lines = [f"  FAIL  the comparison couldn't run: {type(exc).__name__}: {exc}"]
        fails = [f"the comparison couldn't run: {type(exc).__name__}"]
    ok = not fails
    where = make_url(url).render_as_string(hide_password=True)
    page = [f"{'PASS' if ok else 'FAIL'}", f"Nightly recon {now:%Y-%m-%d %H:%M} UTC, journal {where}, read-only", "",
            "Dashboard against the journal:", *lines]
    path = record_path(folder, now)
    folder.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(page) + "\n")
    os.replace(tmp, path)  # a half-written record never counts as the night done
    return ok, path, fails


def main(argv: list[str] | None = None) -> int:
    from sleeve_fund.store import DEFAULT_URL, utcnow

    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.nightly_recon", description=__doc__.split("\n\n")[0])
    ap.add_argument("--dir", required=True, help="where the dated records go (RECON_DIR)")
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL", DEFAULT_URL), help="journal URL (DATABASE_URL)")
    args = ap.parse_args(argv)
    ok, path, _ = run(Path(args.dir), args.db, utcnow())
    print(path.read_text(), end="")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
