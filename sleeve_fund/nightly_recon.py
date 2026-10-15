"""NIGHTLY-RECON (HoE 14 Oct): the start check's dashboard-against-the-journal comparison (start_check.check_books),
run every night during Strategy testing for every Strategy the dashboard shows, reset runs aside, to under a cent
(start_check.CENT): any gap of a cent or more fails. A Strategy withdrawn with dust is listed as known.

Once the fund holds the cash (Bridge Drop A), the night also runs the supervisor's per-poll book check (RECON-B; CAP-2
"identity check every poll and in NIGHTLY-RECON"), the same functions on the same journal: the fund's cash P&L read
two ways (identity_gap, past identity_tolerance fails), every position against its journal's fills (positions_breaks)
and any insurance credit inside the fund (insurance_breaks). A hold the poll already latched is listed as known.
Then every journal row's position by (Strategy, asset), rebuilt from its fills against what its orders record, exactly,
and every row written under another asset (sleeve_fund.reconcile).

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

from sleeve_fund.start_check import FAIL, NOTE, PASS, Section, check_books, journal_kind, read_only_store

RECON_HOUR = 1  # UTC: 02:00 UK in summer time, after the day's last marks and clear of the evening's work


def record_path(folder: Path, now: datetime) -> Path:
    return folder / f"recon-{now:%Y-%m-%d}.txt"


def due(folder: Path, now: datetime) -> bool:
    """Tonight's comparison hasn't run yet and it's time it did."""
    return now.hour >= RECON_HOUR and not record_path(folder, now).exists()


def identity(store) -> Section:
    """The supervisor's per-poll book check (RECON-B), read once on the night's journal: the same functions, so the
    night and the poll can't disagree on what a break is."""
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.supervisor import (_last_resume, bridge_start, identity_gap, identity_tolerance, insurance_breaks,
                                        positions_breaks)

    sec = Section("The fund's book check")
    start = bridge_start(store)
    if start is None:
        sec.add(NOTE, "the fund doesn't hold the cash yet (no re-base deposit applied): no book check to run")
        return sec
    if (held := DbLedger(store, lambda: ()).identity_hold()) is not None:
        sec.add(NOTE, f"known, new entries are already held by the book check until the PM resumes: {held}")
    gap, tol = identity_gap(store), identity_tolerance(store)
    if abs(gap) > tol:
        sec.add(FAIL, f"the fund's cash and Strategy P&L differ from the journal by {gap:,.2f}")
    else:
        sec.add(PASS, f"the fund's cash and Strategy P&L match the journal (gap {gap:,.2f}, within {tol:,.2f})")
    since = max((t for t in (start, _last_resume(store)) if t is not None), default=None)
    for what in positions_breaks(store) + insurance_breaks(store, since):
        sec.add(FAIL, what)
    return sec


def positions(store) -> Section:
    """Every row's position by (Strategy, asset), fills against orders, exactly (reconcile.nightly)."""
    from sleeve_fund import reconcile

    sec = Section("Positions by Strategy and asset")
    found = reconcile.nightly(store)
    for f in found:
        sec.add(FAIL, f.what)
    if not found:
        rows = reconcile.rebuild_positions(store)
        sec.add(PASS, f"{len(rows)} journal rows: every position its fills come to is what its orders record")
    for row, n in sorted(reconcile.unanchored(store).items()):
        sec.add(NOTE, f"{row}: {n} fills with no order row, counted as they are on both sides")
    return sec


def compare(url: str) -> list[Section]:
    """check_books, the book check and the positions by asset, against the journal at url, opened read-only."""
    store = read_only_store(url)
    try:
        return [check_books(store, cent=True), identity(store), positions(store)]
    finally:
        store.engine.dispose()


def run(folder: Path, url: str, now: datetime) -> tuple[bool, Path, list[str]]:
    """Tonight's comparison, written to its dated record. Returns (passed, the record, the FAIL lines). A comparison
    that can't run is a fail, never a pass."""
    try:
        lines, fails = [], []
        for sec in compare(url):
            lines += ["", f"{sec.title}:", *(f"  {st:<4}  {what}" for st, what in sec.lines)]
            fails += [what for st, what in sec.lines if st == FAIL]
    except Exception as exc:  # noqa: BLE001 - recorded and alerted as a fail
        lines = ["", f"  FAIL  the comparison couldn't run: {type(exc).__name__}: {exc}"]
        fails = [f"the comparison couldn't run: {type(exc).__name__}"]
    ok = not fails
    page = [f"{'PASS' if ok else 'FAIL'}", f"Nightly recon {now:%Y-%m-%d %H:%M} UTC, {journal_kind(url)} journal, "
            "read-only", *lines]
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
