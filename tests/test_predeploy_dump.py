"""Advisor's pre-merge condition (15 Oct, HoE): before a deploy starts new code, deploy/predeploy_dump.sh (run inside
the backup container by remote_up.sh) dumps the journal as the running code left it and proves the copy by restoring
it into a scratch database. Red first:
1. on a real journal the dump restores at the same migration with every journal row, and the scratch database goes;
2. a copy that doesn't restore, or comes back short, fails (and so stops the deploy), the scratch database gone too;
3. only the newest 3 are kept, never named *.dump (backup.sh's rotation and the restore drill leave them alone);
4. remote_up.sh takes it before compose up starts the new code."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "predeploy_dump.sh"
URL = os.environ.get("TEST_DATABASE_URL", "")
PG_BIN = Path("/usr/lib/postgresql/16/bin")
needs_pg = pytest.mark.skipif(not URL.startswith("postgresql") or (shutil.which("pg_dump") is None
                                                                  and not (PG_BIN / "pg_dump").exists()),
                              reason="Postgres only: TEST_DATABASE_URL and pg_dump")
needs_bash = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
PASS = re.compile(r"^pre-deploy dump: PASS, restored at migration \w+(, [a-z_]+ \d+){6}$")


def _pg_env(db: str, check: str, backups: Path) -> dict:
    m = re.match(r"^postgresql(?:\+\w+)?://([^@/:]+)?(?::([^@/]*))?@([^:/]+)(?::(\d+))?/", URL)
    user, pw, host, port = m.groups()
    path = os.environ["PATH"] if shutil.which("pg_dump") else f"{PG_BIN}:{os.environ['PATH']}"
    env = {**os.environ, "PATH": path, "PGHOST": host, "PGPORT": port or "5432", "PGUSER": user or "postgres",
           "BACKUP_DB": db, "BACKUP_DIR": str(backups), "PREDEPLOY_CHECK_DB": check}
    if pw:
        env["PGPASSWORD"] = pw
    return env


def _journal(name: str) -> str:
    from sqlalchemy import create_engine, text

    from sleeve_fund import schema
    from sleeve_fund.store import Store

    admin = create_engine(URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    engine = create_engine(URL.rsplit("/", 1)[0] + "/" + name)
    schema.migrate(engine, log=lambda _: None)
    store = Store(engine=engine)
    store.create_sleeve(name="a-strategy", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, params={"market": "perp", "stop_loss": 0.02}, desired_state="stopped")
    store.event("a-strategy", "info", "note", "one")
    store.decide("pm", "note", "two", "a-strategy")
    engine.dispose()
    return schema.head()


def _databases() -> set[str]:
    from sqlalchemy import create_engine, text

    admin = create_engine(URL)
    with admin.connect() as c:
        names = {r[0] for r in c.execute(text("SELECT datname FROM pg_database"))}
    admin.dispose()
    return names


@needs_pg
@needs_bash
def test_a_real_journal_is_dumped_and_restores_whole_and_only_three_are_kept(tmp_path):
    head = _journal("predeploy_src")
    backups = tmp_path / "backups"
    backups.mkdir()
    for stamp in ("20261001-000000", "20261002-000000", "20261003-000000"):
        (backups / f"predeploy-{stamp}.pgdump").write_text("old")
        os.utime(backups / f"predeploy-{stamp}.pgdump", (1, int(stamp[6:8])))  # oldest first by time
    (backups / "sleeve_fund-20261015-0012.dump").write_text("a nightly dump")
    done = subprocess.run(["bash", str(SCRIPT)], env=_pg_env("predeploy_src", "predeploy_chk", backups),
                          capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stdout + done.stderr
    line = done.stdout.strip().splitlines()[-1]
    assert PASS.match(line) and f"migration {head}," in line and "orders 0" in line and "events 1" in line, line
    assert "decisions 1" in line
    kept = sorted(p.name for p in backups.glob("predeploy-*.pgdump"))
    assert len(kept) == 3 and "predeploy-20261001-000000.pgdump" not in kept
    assert (backups / "sleeve_fund-20261015-0012.dump").exists()  # the nightly dumps are not its to rotate
    assert not list(backups.glob("*.part"))
    assert "predeploy_chk" not in _databases()


# --- with stand-in Postgres tools ------------------------------------------------------------------------------------

def _tools(bin_dir: Path, log: Path, restore_ok: bool = True, copy: str = "0028 5 4 9 1 2 3") -> None:
    bin_dir.mkdir(exist_ok=True)
    stubs = {
        "pg_dump": 'while [ $# -gt 0 ]; do [ "$1" = -f ] && { echo dump > "$2"; }; shift; done',
        "createdb": "exit 0",
        "dropdb": "exit 0",
        "pg_restore": "exit 0" if restore_ok else "echo 'pg_restore: error: secret-host row' >&2; exit 1",
        # the live journal answers the same before and after; the scratch copy answers $copy
        "psql": f'case "$*" in *predeploy_check*) echo "{copy}";; *) echo "0028 5 4 9 1 2 3";; esac',
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(f'#!/usr/bin/env bash\necho "{name} $PGHOST:$PGUSER $*" >> {log}\n{body}\n')
        (bin_dir / name).chmod(0o755)


def _run(tmp_path: Path, **kw) -> tuple[subprocess.CompletedProcess, list[str]]:
    log = tmp_path / "calls.log"
    log.touch()
    _tools(tmp_path / "bin", log, **kw)
    (tmp_path / "b").mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k not in ("PGHOST", "PGUSER")}
    env |= {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b")}
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30, check=False)
    return done, log.read_text().splitlines()


@needs_bash
def test_a_copy_that_doesnt_restore_fails_and_drops_the_scratch_database(tmp_path):
    done, calls = _run(tmp_path, restore_ok=False)
    assert done.returncode == 1 and "pre-deploy dump: FAIL, it didn't restore" in done.stdout
    assert "secret-host" not in done.stdout + done.stderr
    assert calls[-1].startswith("dropdb") and "predeploy_check" in calls[-1]


@pytest.mark.parametrize("copy, words", [("0028 5 3 9 1 2 3", "fills short"), ("0027 5 4 9 1 2 3", "another migration"),
                                         ("0028 5 4 9 1 1 3", "book_moves short")])
@needs_bash
def test_a_copy_short_of_the_journal_fails(tmp_path, copy, words):
    done, _ = _run(tmp_path, copy=copy)
    assert done.returncode == 1 and done.stdout.startswith("pre-deploy dump: FAIL") and words in done.stdout, done.stdout


@needs_bash
def test_a_whole_copy_passes_and_is_never_named_like_a_nightly_dump(tmp_path):
    done, calls = _run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == ("pre-deploy dump: PASS, restored at migration 0028, orders 5, fills 4, events 9, "
                                   "fund_flows 1, book_moves 2, decisions 3")
    (dump,) = (tmp_path / "b").iterdir()
    assert re.fullmatch(r"predeploy-\d{8}-\d{6}\.pgdump", dump.name)
    assert any(c.startswith("pg_restore") and "--exit-on-error" in c for c in calls)
    # the backup container's environment holds only the password: the journal's host and role are backup.sh's
    tools = [c for c in calls if c.split(" ")[0] in ("pg_dump", "createdb", "dropdb", "pg_restore", "psql")]
    assert len(tools) >= 7 and all(c.split(" ")[1] == "db:sleeve" for c in tools), tools


def test_the_deploy_dumps_before_the_new_code_starts():
    lines = [ln.strip() for ln in (ROOT / "deploy" / "remote_up.sh").read_text().splitlines()]
    dump = next(i for i, ln in enumerate(lines) if "deploy/predeploy_dump.sh" in ln)
    assert lines[dump] == "docker compose exec -T backup bash -s < deploy/predeploy_dump.sh"
    assert dump < lines.index("docker compose up -d --build --remove-orphans")
    assert "set -euo pipefail" in lines  # a failed dump stops the deploy
