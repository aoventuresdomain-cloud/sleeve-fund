"""DA-7, the restore drill (HoE, 15 Oct): proof that the newest nightly backup (deploy/backup.sh) brings the fund back
whole. deploy/restore_drill.sh restores the dump into a throwaway database on the server, on its own network and
volume, never the journal's; this then checks what came back, step by step, each timed:

- backup: the newest dump is the one the backup's own restore check last passed (status.json), and how old it is;
- migrate: the restored journal brought to the newest migration (schema.migrate), alembic_version at head;
- counts: the rows the backup's check counted (the trades and the money history, COUNTED) came back, exactly;
- tables: every table's rows, the history the fund's figures are rebuilt from;
- links: every row's parent is there (each foreign key in the schema, orphans counted);
- research: the archive beside the dump reads back whole (the idea ledger and the tear sheets);
- start check and books: start_check.run on the restored journal, whose book section rebuilds every Strategy's
  figures from its fills and compares them with the marks.

The log is public (restore-drill.yml): it prints step names, counts, timings and PASS, FAIL or NOTE only, never a
row, a name, a host or a path. Exit 0 when every step passes, else 1."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path

PASS, FAIL, NOTE = "PASS", "FAIL", "NOTE"
DUMP = re.compile(r"^sleeve_fund-(\d{8})-(\d{4})\.dump$")  # deploy/backup.sh's name: the UTC date and time it ran
# What the backup's own restore check counts (backup.sh's json_build_object): the trades and the money history
COUNTED = ("sleeves", "orders", "fills", "events", "decisions", "equity", "funding", "insurance", "fund_flows",
           "book_marks", "book_moves", "sleeve_archive", "gate_reservations", "trials")


class Drill:
    def __init__(self, out=print):
        self.out, self.failed = out, False

    def step(self, name: str, fn) -> None:
        """Runs fn, which returns (status, words); words hold counts only. An error is a FAIL by its type alone: its
        text can carry a host, a path or a row."""
        t = time.monotonic()
        try:
            status, words = fn()
        except Exception as exc:  # noqa: BLE001 - a step that can't run is a fail, never a pass
            status, words = FAIL, f"couldn't run: {type(exc).__name__}"
        self.failed |= status == FAIL
        self.out(f"{name}: {status} ({time.monotonic() - t:.1f}s){', ' + words if words else ''}")


def newest_dump(backups: Path) -> Path:
    dumps = [p for p in backups.glob("*.dump") if DUMP.match(p.name)]
    if not dumps:
        raise FileNotFoundError("no dump")
    return max(dumps, key=lambda p: p.name)


def taken_at(dump: Path) -> datetime:
    d, hm = DUMP.match(dump.name).groups()
    return datetime.strptime(d + hm, "%Y%m%d%H%M").replace(tzinfo=UTC)


def check_backup(backups: Path, dump: Path, now: datetime) -> tuple[str, str]:
    status = json.loads((backups / "status.json").read_text())
    at = taken_at(dump)
    words = f"taken {at:%Y-%m-%d %H:%M} UTC, {(now - at).total_seconds() / 3600:.1f}h old"
    if status.get("ok") is not True or status.get("file") != dump.name:
        return FAIL, words + ", not the dump the backup's own check last passed"
    return PASS, words


def check_counts(engine, backups: Path) -> tuple[str, str]:
    from sqlalchemy import text

    from sleeve_fund.store import metadata

    want = json.loads((backups / "status.json").read_text()).get("restored") or {}
    # Every table the backup counted (a name it counted that the schema hasn't is a FAIL, never a query); a COUNTED
    # table missing from it is a backup taken before backup.sh counted it: a note, not a fail.
    names = [t for t in COUNTED if t in want] + sorted(t for t in want if t not in COUNTED)
    with engine.connect() as c:
        got = {t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in names if t in metadata.tables}
    words = ", ".join(f"{t} {got[t]}" for t in names if t in got)
    apart = [t for t in names if want.get(t) != got.get(t)]
    unseen = [t for t in COUNTED if t not in want]
    note = f"; not in the backup's count: {', '.join(unseen)}" if unseen else ""
    if apart or not want:
        return FAIL, words + f"; apart from the backup's count: {', '.join(apart) or 'no count'}" + note
    return PASS, words + ", as the backup counted" + note


def check_tables(engine) -> tuple[str, str]:
    from sqlalchemy import func, select

    from sleeve_fund.store import metadata

    with engine.connect() as c:
        got = {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in metadata.sorted_tables}
    # Read whole, each counted: the backup's own count (counts) is the bar; a book just re-based can hold no trade yet
    return PASS, f"{len(got)} tables, {sum(got.values())} rows: " + ", ".join(f"{n} {k}" for n, k in sorted(got.items()))


def check_links(engine) -> tuple[str, str]:
    """Every foreign key in the schema, the rows whose parent is missing counted (none, as the journal enforces)."""
    from sqlalchemy import and_, exists, func, not_, select

    from sleeve_fund.store import metadata

    checked, orphans = 0, {}
    with engine.connect() as c:
        for t in metadata.sorted_tables:
            for fk in t.foreign_key_constraints:
                parent = fk.referred_table.alias("parent_row")  # an alias: a row's parent can be in its own table
                on = and_(*[parent.c[e.column.name] == e.parent for e in fk.elements])
                present = and_(*[e.parent.is_not(None) for e in fk.elements])
                n = c.execute(select(func.count()).select_from(t)
                              .where(present, not_(exists().where(on)))).scalar()
                checked += 1
                if n:
                    orphans[f"{t.name}->{fk.referred_table.name}"] = n
    if orphans:
        return FAIL, f"{checked} links, orphans: " + ", ".join(f"{k} {v}" for k, v in sorted(orphans.items()))
    return PASS, f"{checked} links, no orphan row"


def check_research(backups: Path, dump: Path) -> tuple[str, str]:
    """The archive the same night wrote (backup.sh: research-<date>-<time>.tar.gz), read back member by member."""
    tars = sorted(backups.glob("research-*.tar.gz"))
    if not tars:
        return NOTE, "no research archive beside the dump"
    stamp = dump.name[len("sleeve_fund-"):-len(".dump")]
    tar = next((p for p in tars if p.name == f"research-{stamp}.tar.gz"), tars[-1])
    files = ledger = sheets = 0
    with tarfile.open(tar, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            data = tf.extractfile(m).read()
            files += 1
            name = m.name.lstrip("./")
            if name == "idea_ledger.jsonl":
                ledger = sum(1 for line in data.splitlines() if line.strip())
            elif name.startswith("tearsheets/"):
                sheets += 1
    same = "the same night's" if tar.name == f"research-{stamp}.tar.gz" else "the newest"
    return PASS, f"{same} archive, {files} files, idea ledger {ledger} lines, {sheets} tear sheets"


def check_start(url: str) -> tuple[str, str]:
    """start_check's sections on the restored journal, as start_check.run takes them (its book section rebuilds each
    Strategy's figures from its fills against its marks). Their lines name Strategies, so only the count of lines by
    status is printed. The key section reads no environment here (none holds a key): it reads the journal's."""
    from sleeve_fund import start_check as sc

    store = sc.read_only_store(url)
    try:
        e = store.engine
        sections = [sc.check_store(e), sc.check_keys({}, e), sc.check_venues(e), sc.check_books(store),
                    sc.check_assets(store)]
    finally:
        store.engine.dispose()
    marks = [status for sec in sections for status, _ in sec.lines]
    words = ", ".join(f"{s} {marks.count(s)}" for s in (PASS, FAIL, NOTE))
    return (FAIL if any(sec.failed for sec in sections) else PASS), words


def main(argv: list[str] | None = None) -> int:
    import os

    from sqlalchemy import create_engine, text

    from sleeve_fund import schema

    ap = argparse.ArgumentParser(prog="python -m sleeve_fund.restore_drill", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backups", required=True, help="the backups folder (read only)")
    ap.add_argument("--db", default=os.environ.get("DATABASE_URL"), help="the throwaway database the dump went into")
    ap.add_argument("--restore-seconds", type=float, default=None, help="the restore's own time, from the script")
    args = ap.parse_args(argv)
    drill = Drill()
    backups = Path(args.backups)
    t0 = time.monotonic()
    holder: dict = {}

    def pick():
        holder["dump"] = newest_dump(backups)
        return check_backup(backups, holder["dump"], datetime.now(UTC))

    drill.step("backup", pick)
    if args.restore_seconds is not None:
        drill.out(f"restore: PASS ({args.restore_seconds:.1f}s)")
    if not args.db:
        drill.out("database: FAIL, no throwaway database named")
        return 1
    engine = create_engine(args.db.replace("postgres://", "postgresql://", 1))

    def migrate():
        schema.migrate(engine, log=lambda _: None)
        with engine.connect() as c:
            at = c.execute(text("SELECT version_num FROM alembic_version")).scalar()
        return (PASS, f"at head {at}") if at == schema.head() else (FAIL, f"at {at}, head is {schema.head()}")

    try:
        drill.step("migrate", migrate)
        drill.step("counts", lambda: check_counts(engine, backups))
        drill.step("tables", lambda: check_tables(engine))
        drill.step("links", lambda: check_links(engine))
        if "dump" in holder:
            drill.step("research", lambda: check_research(backups, holder["dump"]))
        drill.step("start check and books", lambda: check_start(args.db))
    finally:
        engine.dispose()
    total = time.monotonic() - t0 + (args.restore_seconds or 0)
    drill.out(f"restore drill: {FAIL if drill.failed else PASS} ({total:.1f}s in all)")
    return 1 if drill.failed else 0


if __name__ == "__main__":
    sys.exit(main())
