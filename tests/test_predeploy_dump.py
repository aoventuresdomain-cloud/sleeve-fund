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

def _tools(bin_dir: Path, log: Path, restore_ok: bool = True, copy: str = "0028 5 4 9 1 2 3",
           dump_ok: bool = True, read_ok: bool = True, make_ok: bool = True, copy_ok: bool = True) -> None:
    bin_dir.mkdir(exist_ok=True)
    stubs = {
        "pg_dump": ('while [ $# -gt 0 ]; do [ "$1" = -f ] && { echo dump > "$2"; }; shift; done' if dump_ok
                    else "echo 'pg_dump: error: secret-host' >&2; exit 1"),
        "createdb": "exit 0" if make_ok else "echo 'createdb: error: secret-host' >&2; exit 3",
        "dropdb": "exit 0",
        "pg_restore": "exit 0" if restore_ok else "echo 'pg_restore: error: secret-host row' >&2; exit 1",
        # the live journal answers the same before and after; the scratch copy answers $copy
        "psql": (f'case "$*" in *predeploy_check*) {f"echo {copy!r}" if copy_ok else "exit 4"};; '
                 '*) echo "0028 5 4 9 1 2 3";; esac' if read_ok
                 else "echo 'psql: error: secret-host' >&2; exit 2"),
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
    assert done.returncode == 1 and "pre-deploy dump: FAIL, it didn't restore (exit 1)" in done.stdout
    assert "secret-host" in (tmp_path / "b" / "predeploy-last.err").read_text()
    assert "secret-host" not in done.stdout + done.stderr
    assert calls[-1].startswith("dropdb") and "predeploy_check" in calls[-1]
    assert not list((tmp_path / "b").glob("*.pgdump*"))  # CR F-PDD-1: an unproved copy never stays under a good name


@pytest.mark.parametrize("copy, words", [("0028 5 3 9 1 2 3", "fills doesn't match the journal"),
                                         ("0027 5 4 9 1 2 3", "another migration"),
                                         ("0028 5 4 9 1 1 3", "book_moves doesn't match the journal"),
                                         ("0028 5 4 10 1 2 3", "events doesn't match the journal")])  # CR F-PDD-2
@needs_bash
def test_a_copy_unlike_the_journal_fails_and_leaves_no_copy(tmp_path, copy, words):
    done, _ = _run(tmp_path, copy=copy)
    assert done.returncode == 1 and done.stdout.startswith("pre-deploy dump: FAIL") and words in done.stdout, done.stdout
    assert not list((tmp_path / "b").glob("*.pgdump*"))


@pytest.mark.parametrize("check, journal", [("postgres", "sleeve_fund"), ("sleeve_fund", "sleeve_fund"),
                                            ("restore_check", "sleeve_fund"), ("predeploy", "sleeve_fund"),
                                            ("predeploy_x", "predeploy_x")])  # CR nit: never the journal itself
@needs_bash
def test_the_scratch_database_is_never_one_it_could_harm(tmp_path, check, journal):
    """It drops its scratch database on every exit: that name can only be its own (predeploy_*), never the journal's,
    the server's maintenance database or the nightly check's."""
    log = tmp_path / "calls.log"
    log.touch()
    _tools(tmp_path / "bin", log)
    (tmp_path / "b").mkdir()
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"),
           "PREDEPLOY_CHECK_DB": check, "BACKUP_DB": journal}
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 1 and done.stdout.strip() == "pre-deploy dump: FAIL, not a scratch database name"
    assert log.read_text() == ""  # no tool ran, nothing dropped


@pytest.mark.parametrize("free, ok", [(1999, False), (2000, True)])
@needs_bash
def test_the_dump_needs_twice_the_last_dump_free_or_takes_nothing(tmp_path, free, ok):
    """HoE: a free-space check before the dump: under twice the newest dump's size, nothing is dumped."""
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "sleeve_fund-20261015-0012.dump").write_bytes(b"x" * 1000)
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin" / "df").write_text(f"""#!/usr/bin/env bash
echo "Filesystem 1-blocks Used Available Capacity Mounted on"
echo "/dev/x 99999999999 1 {free} 1% /"
""")
    (tmp_path / "bin" / "df").chmod(0o755)
    done, calls = _run(tmp_path)
    if ok:
        assert done.returncode == 0, done.stdout
    else:
        assert done.returncode == 1 and done.stdout.strip() == "pre-deploy dump: FAIL, not enough disk"
        assert not any(c.startswith("pg_dump") for c in calls)
        assert [p.name for p in (tmp_path / "b").iterdir()] == ["sleeve_fund-20261015-0012.dump"]


@pytest.mark.parametrize("kw, words", [({"dump_ok": False}, "the dump didn't complete (exit 1)"),
                                       ({"read_ok": False}, "couldn't read the journal (exit 2)"),
                                       ({"make_ok": False}, "couldn't make the scratch database (exit 3)"),
                                       ({"copy_ok": False}, "couldn't read the copy (exit 4)")])  # HoE; CR pins
@needs_bash
def test_a_tool_that_fails_says_so_and_leaves_nothing(tmp_path, kw, words):
    """QT3 N2: a failed pg_dump (or psql) once exited through set -e with no line in the deploy log."""
    done, _ = _run(tmp_path, **kw)
    assert done.returncode == 1 and done.stdout.strip() == f"pre-deploy dump: FAIL, {words}, details beside the dumps"
    assert "secret-host" not in done.stdout + done.stderr
    assert [p.name for p in (tmp_path / "b").iterdir()] == ["predeploy-last.err"]
    # CR F-PDD-3: the tool's own words are kept for the operator, beside the dumps and private, never in the log
    err = tmp_path / "b" / "predeploy-last.err"
    assert err.stat().st_mode & 0o077 == 0
    assert ("secret-host" in err.read_text()) == (kw != {"copy_ok": False})  # that stand-in fails silently


@needs_bash
def test_the_error_file_starts_afresh_each_run(tmp_path):
    """CR: an older run's words never sit above today's."""
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "predeploy-last.err").write_text("yesterday's words\n")
    done, _ = _run(tmp_path, dump_ok=False)
    assert done.returncode == 1
    assert "yesterday" not in (tmp_path / "b" / "predeploy-last.err").read_text()


@needs_bash
def test_a_whole_copy_passes_and_is_never_named_like_a_nightly_dump(tmp_path):
    done, calls = _run(tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    assert done.stdout.strip() == ("pre-deploy dump: PASS, restored at migration 0028, orders 5, fills 4, events 9, "
                                   "fund_flows 1, book_moves 2, decisions 3")
    (dump,) = (tmp_path / "b").glob("predeploy-*.pgdump")
    assert re.fullmatch(r"predeploy-\d{8}-\d{6}\.pgdump", dump.name)
    assert dump.stat().st_mode & 0o077 == 0  # QT3 N1: docker exec runs with umask 0000; the copy is the owner's only
    assert any(c.startswith("pg_restore") and "--exit-on-error" in c for c in calls)
    # the backup container's environment holds only the password: the journal's host and role are backup.sh's
    tools = [c for c in calls if c.split(" ")[0] in ("pg_dump", "createdb", "dropdb", "pg_restore", "psql")]
    assert len(tools) >= 7 and all(c.split(" ")[1] == "db:sleeve" for c in tools), tools


def test_the_deploy_dumps_before_the_new_code_starts():
    lines = [ln.strip() for ln in (ROOT / "deploy" / "predeploy.sh").read_text().splitlines()]
    assert "docker compose exec -T backup bash -s < deploy/predeploy_dump.sh" in lines
    assert "set -euo pipefail" in lines  # a failed dump stops the deploy
    assert "set -euo pipefail" in (ROOT / "deploy" / "remote_up.sh").read_text()


# --- the host side (HoQA F-PDD-1): skip only on a true first deploy -----------------------------------------------------

HOST = ROOT / "deploy" / "predeploy.sh"


def _host(tmp_path: Path, volume: bool, running: bool, ready: bool = True) -> tuple[subprocess.CompletedProcess, list]:
    """deploy/predeploy.sh as remote_up.sh runs it, with a stand-in docker: the journal's volume there or not, its
    containers running or stopped, the database answering or not."""
    log = tmp_path / "docker.log"
    log.touch()
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "deploy").mkdir(exist_ok=True)
    (tmp_path / "deploy" / "predeploy_dump.sh").write_text("echo 'pre-deploy dump: PASS, stand-in'\n")
    (tmp_path / "bin" / "docker").write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
case "$*" in
  "volume ls"*) {"echo sleeve-fund_pgdata" if volume else "true"};;
  "compose start"*) {"exit 0" if running or volume else "exit 1"};;
  *pg_isready*) {"exit 0" if ready else "exit 2"};;
  "compose exec -T backup bash -s") cat;;
esac
""")
    (tmp_path / "bin" / "docker").chmod(0o755)
    (tmp_path / "bin" / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (tmp_path / "bin" / "sleep").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    done = subprocess.run(["bash", str(HOST)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
                          check=False)
    return done, log.read_text().splitlines()


@needs_bash
def test_a_first_deploy_with_no_journal_volume_skips_the_dump(tmp_path):
    done, calls = _host(tmp_path, volume=False, running=False)
    assert done.returncode == 0 and done.stdout.strip() == "pre-deploy dump: SKIP, no journal yet (first deploy)"
    assert not any(c.startswith(("compose start", "compose exec")) for c in calls)


@needs_bash
@pytest.mark.parametrize("running", [True, False])
def test_a_journal_volume_means_a_dump_even_with_the_database_stopped(tmp_path, running):
    """HoQA F-PDD-1: a stopped db once skipped the dump and the deploy went on with no proved copy. Now the journal's
    containers are started as they are (never recreated: the new config must not start before the dump), then dumped."""
    done, calls = _host(tmp_path, volume=True, running=running)
    assert done.returncode == 0 and "pre-deploy dump: PASS, stand-in" in done.stdout, done.stdout
    start = calls.index("compose start db backup")
    assert start < calls.index("compose exec -T backup bash -s")
    assert not any(c.startswith(("compose up", "compose create", "compose restart")) for c in calls)


@needs_bash
def test_docker_not_answering_is_never_read_as_a_first_deploy(tmp_path):
    _host(tmp_path, volume=True, running=True)
    (tmp_path / "bin" / "docker").write_text("#!/usr/bin/env bash\nexit 1\n")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    done = subprocess.run(["bash", str(HOST)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
                          check=False)
    assert done.returncode != 0 and "SKIP" not in done.stdout


@needs_bash
def test_a_database_that_never_answers_stops_the_deploy(tmp_path):
    done, calls = _host(tmp_path, volume=True, running=True, ready=False)
    assert done.returncode == 1 and "pre-deploy dump: FAIL, the journal's database didn't answer" in done.stdout
    assert "compose exec -T backup bash -s" not in calls


def test_the_deploy_runs_the_host_side_before_the_new_code_starts():
    lines = [ln.strip() for ln in (ROOT / "deploy" / "remote_up.sh").read_text().splitlines()]
    assert lines.index("bash deploy/predeploy.sh") < lines.index("docker compose up -d --build --remove-orphans")
    assert not any("compose ps -q db" in ln for ln in lines)  # never keyed on "db running" again
