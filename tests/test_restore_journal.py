"""Restore-journal (DA, head 3; premerge-3a.md step 4.3): putting a proved dump back as the journal, by hand only, never
without the PM's go. Red first:
1. inside the backup container (deploy/restore_journal_inner.sh) the dump is restored into a stage database and
   counted first; only then is the journal renamed aside (kept, never dropped) and the stage renamed into its place;
2. anything wrong before the swap (a name that isn't a dump, a dump that doesn't restore, the journal still in use)
   leaves the journal exactly as it was, and the stage database goes;
3. on the server (deploy/restore_journal.sh) nothing runs without the PM's go and the typed confirmation; the app is
   stopped before the swap, migrated after it, and started again only on success; on a failure before the swap the
   app is started again on the untouched journal;
4. the workflow is by hand only, behind its own repository switch, and its log prints no tool output."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from test_predeploy_dump import URL, _databases, _journal, _pg_env, needs_bash, needs_pg

ROOT = Path(__file__).resolve().parent.parent
INNER = ROOT / "deploy" / "restore_journal_inner.sh"
HOST = ROOT / "deploy" / "restore_journal.sh"
WORKFLOW = ROOT / ".github" / "workflows" / "restore-journal.yml"
DUMP = "predeploy-20261015-120000.pgdump"
PASS = re.compile(r"^restore journal: PASS, \S+ restored at migration \w+(, [a-z_]+ \d+){6}; "
                  r"the journal it replaced is kept as \w+$")


def _env(db: str, backups: Path, dump: str = DUMP) -> dict:
    env = _pg_env(db, "predeploy_unused", backups)
    return env | {"DUMP": dump, "RESTORE_STAGE_DB": "restore_stage_t"}


def _count(db: str, sql: str) -> int:
    from sqlalchemy import create_engine, text

    eng = create_engine(URL.rsplit("/", 1)[0] + "/" + db)
    with eng.connect() as c:
        n = c.execute(text(sql)).scalar()
    eng.dispose()
    return n


def _drop(*names: str) -> None:
    from sqlalchemy import create_engine, text

    admin = create_engine(URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        for n in names:
            c.execute(text(f'DROP DATABASE IF EXISTS "{n}" WITH (FORCE)'))
    admin.dispose()


@needs_pg
@needs_bash
def test_a_proved_dump_replaces_the_journal_and_the_old_journal_is_kept_beside_it(tmp_path):
    _journal("rj_src")  # one event, one decision: the dump to restore
    backups = tmp_path / "backups"
    backups.mkdir()
    env = _env("rj_src", backups)
    subprocess.run(["pg_dump", "-d", "rj_src", "-Fc", "-f", str(backups / DUMP)], env=env, check=True)
    _journal("rj_live")
    from sqlalchemy import create_engine, text

    eng = create_engine(URL.rsplit("/", 1)[0] + "/rj_live")
    with eng.begin() as c:  # the live journal moved on since the dump: two more events
        c.execute(text("INSERT INTO events (sleeve, ts, level, kind, message) SELECT sleeve, ts, level, kind, 'later' "
                       "FROM events LIMIT 1"))
        c.execute(text("INSERT INTO events (sleeve, ts, level, kind, message) SELECT sleeve, ts, level, kind, 'later' "
                       "FROM events LIMIT 1"))
    eng.dispose()
    before = {d for d in _databases() if d.startswith("rj_live_before_restore_")}
    try:
        done = subprocess.run(["bash", str(INNER)], env=_env("rj_live", backups), capture_output=True, text=True,
                              timeout=120, check=False)
        assert done.returncode == 0, done.stdout + done.stderr
        line = done.stdout.strip().splitlines()[-1]
        assert PASS.match(line) and DUMP in line and "events 1" in line, line
        kept = {d for d in _databases() if d.startswith("rj_live_before_restore_")} - before
        assert len(kept) == 1 and kept.pop() in line
        assert _count("rj_live", "SELECT count(*) FROM events") == 1  # the dump's rows are the journal now
        old = next(d for d in _databases() if d.startswith("rj_live_before_restore_") and d not in before)
        assert _count(old, "SELECT count(*) FROM events") == 3  # and the journal it replaced is whole, beside it
        assert "restore_stage_t" not in _databases()
    finally:
        _drop("rj_src", "rj_live", *[d for d in _databases() if d.startswith("rj_live_before_restore_")])


# --- with stand-in Postgres tools ------------------------------------------------------------------------------------

def _tools(bin_dir: Path, log: Path, restore_ok: bool = True, busy: int = 0, rename_ok: bool = True,
           size: int = 1000) -> None:
    bin_dir.mkdir(exist_ok=True)
    stubs = {
        "createdb": "exit 0",
        "dropdb": "exit 0",
        "pg_restore": "exit 0" if restore_ok else "echo 'pg_restore: error: secret-host row' >&2; exit 1",
        "psql": (f'case "$*" in *pg_stat_activity*) echo {busy};; *pg_database_size*) echo {size};; '
                 f'*"ALTER DATABASE"*) {"exit 0" if rename_ok else "exit 5"};; *) echo "0028 5 4 9 1 2 3";; esac'),
    }
    for name, body in stubs.items():
        (bin_dir / name).write_text(f'#!/usr/bin/env bash\necho "{name} $*" >> {log}\n{body}\n')
        (bin_dir / name).chmod(0o755)


def _run(tmp_path: Path, dump: str = DUMP, present: bool = True, **kw):
    log = tmp_path / "calls.log"
    log.touch()
    _tools(tmp_path / "bin", log, **kw)
    (tmp_path / "b").mkdir(exist_ok=True)
    if present:
        (tmp_path / "b" / DUMP).write_text("dump")
    env = {k: v for k, v in os.environ.items() if k not in ("PGHOST", "PGUSER")}
    env |= {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"), "DUMP": dump}
    done = subprocess.run(["bash", str(INNER)], env=env, capture_output=True, text=True, timeout=30, check=False)
    return done, log.read_text().splitlines()


def _renames(calls):
    return [c for c in calls if "ALTER DATABASE" in c]


@needs_bash
@pytest.mark.parametrize("dump, present", [
    ("../etc/passwd", True), ("sleeve_fund; rm -rf", True), ("other.sql", True), (DUMP, False), ("", True)])
def test_only_a_dump_beside_the_backups_is_ever_restored(tmp_path, dump, present):
    done, calls = _run(tmp_path, dump=dump, present=present)
    assert done.returncode == 1 and "restore journal: FAIL, " in done.stdout and not _renames(calls)
    assert not [c for c in calls if c.startswith("pg_restore")]


@needs_bash
def test_a_dump_that_doesnt_restore_leaves_the_journal_as_it_was(tmp_path):
    done, calls = _run(tmp_path, restore_ok=False)
    assert done.returncode == 1 and "restore journal: FAIL, it didn't restore (exit 1)" in done.stdout
    assert not _renames(calls) and "secret-host" not in done.stdout + done.stderr  # the tool's words stay private
    assert "secret-host" in (tmp_path / "b" / "restore-last.err").read_text()
    assert calls[-1].startswith("dropdb") and "restore_stage" in calls[-1]  # the stage goes


@needs_bash
def test_a_journal_still_in_use_is_never_swapped(tmp_path):
    done, calls = _run(tmp_path, busy=2)
    assert done.returncode == 1 and "restore journal: FAIL, the journal is still in use" in done.stdout
    assert not _renames(calls)


@needs_bash
def test_the_journal_is_renamed_aside_never_dropped(tmp_path):
    done, calls = _run(tmp_path)
    assert done.returncode == 0 and PASS.match(done.stdout.strip().splitlines()[-1]), done.stdout
    first, second = _renames(calls)
    assert 'RENAME TO "sleeve_fund_before_restore_' in first and '"restore_stage" RENAME TO "sleeve_fund"' in second
    assert not [c for c in calls if c.startswith("dropdb") and "sleeve_fund" in c.split()[-1]]


@needs_bash
def test_a_failed_set_aside_leaves_the_journal_and_names_the_exit(tmp_path):
    done, calls = _run(tmp_path, rename_ok=False)
    assert done.returncode == 1 and "couldn't set the journal aside (exit 5), it is as it was" in done.stdout
    assert len(_renames(calls)) == 1


@needs_bash
def test_a_failed_second_rename_puts_the_journal_back_and_names_the_exit(tmp_path):
    log = tmp_path / "calls.log"
    _run(tmp_path)
    psql = tmp_path / "bin" / "psql"  # the set-aside works, the stage's rename into place doesn't
    psql.write_text(f'''#!/usr/bin/env bash
echo "psql $*" >> {log}
case "$*" in *pg_stat_activity*) echo 0;; *pg_database_size*) echo 1000;; *'"restore_stage" RENAME'*) exit 6;;
  *"ALTER DATABASE"*) exit 0;; *) echo "0028 5 4 9 1 2 3";; esac
''')
    log.write_text("")
    env = {k: v for k, v in os.environ.items() if k not in ("PGHOST", "PGUSER")}
    env |= {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"), "DUMP": DUMP}
    done = subprocess.run(["bash", str(INNER)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 1 and "(exit 6), the journal is back as it was" in done.stdout, done.stdout
    back = _renames(log.read_text().splitlines())[-1]
    assert '_before_restore_' in back and back.endswith('RENAME TO "sleeve_fund"')


@needs_bash
def test_a_journal_left_out_of_its_place_exits_2_and_says_so(tmp_path):
    """QT3 F-RJ-1: the stage's rename and the put-back both fail. No database carries the journal's name, which no
    other failure leaves: exit 2, never 1, so the server side keeps the app stopped."""
    log = tmp_path / "calls.log"
    _run(tmp_path)
    (tmp_path / "bin" / "psql").write_text(f'''#!/usr/bin/env bash
echo "psql $*" >> {log}
case "$*" in *pg_stat_activity*) echo 0;; *pg_database_size*) echo 1000;; *'RENAME TO "sleeve_fund_before'*) exit 0;;
  *"ALTER DATABASE"*) exit 6;; *) echo "0028 5 4 9 1 2 3";; esac
''')
    env = {k: v for k, v in os.environ.items() if k not in ("PGHOST", "PGUSER")}
    env |= {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"), "DUMP": DUMP}
    done = subprocess.run(["bash", str(INNER)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 2 and "the journal is still named sleeve_fund_before_restore_" in done.stdout, done.stdout
    assert "as it was" not in done.stdout


@needs_bash
def test_too_little_disk_for_the_stage_restores_nothing(tmp_path):
    """QT3 minor: the stage is a second journal; twice the journal's size free on the host's disk (beside the
    backups, as the pre-deploy dump checks), or nothing is restored."""
    done, calls = _run(tmp_path, size=10**18)
    assert done.returncode == 1 and "restore journal: FAIL, not enough disk" in done.stdout, done.stdout
    assert not [c for c in calls if c.startswith(("createdb", "pg_restore"))] and not _renames(calls)


@needs_bash
@pytest.mark.parametrize("stage", ["sleeve_fund", "postgres", "template1", "restore stage"])
def test_the_stage_is_never_a_database_it_could_harm(tmp_path, stage):
    log = tmp_path / "calls.log"
    _run(tmp_path)  # tools in place
    log.write_text("")
    env = {k: v for k, v in os.environ.items() if k not in ("PGHOST", "PGUSER")}
    env |= {"PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"), "DUMP": DUMP,
            "RESTORE_STAGE_DB": stage}
    done = subprocess.run(["bash", str(INNER)], env=env, capture_output=True, text=True, timeout=30, check=False)
    assert done.returncode == 1 and "not a stage database name" in done.stdout and log.read_text() == ""


# --- the server side, with a stand-in docker -------------------------------------------------------------------------

def _host(tmp_path: Path, go: str = "pm-go", confirm: str | None = None, inner_ok: bool = True,
          migrate_ok: bool = True, inner_exit: int = 1) -> tuple[subprocess.CompletedProcess, list[str]]:
    log = tmp_path / "docker.log"
    log.write_text("")
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "deploy").mkdir(exist_ok=True)
    (tmp_path / "deploy" / "restore_journal_inner.sh").write_text(
        "echo 'restore journal: PASS, stand-in'\n" if inner_ok
        else f"echo 'restore journal: FAIL, stand-in'; exit {inner_exit}\n")
    (tmp_path / "bin" / "docker").write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
case "$*" in
  "compose ps --services --status running") printf 'db\\nbackup\\ncaddy\\nsupervisor\\ndashboard\\nhub-binance\\n';;
  *"backup bash -s") bash -s;;
  *"run --rm --no-deps migrate"*) {"exit 0" if migrate_ok else "exit 1"};;
esac
""")
    (tmp_path / "bin" / "docker").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "RESTORE_JOURNAL_GO": go,
           "DUMP": DUMP, "CONFIRM": f"restore {DUMP}" if confirm is None else confirm}
    done = subprocess.run(["bash", str(HOST)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
                          check=False)
    return done, log.read_text().splitlines()


@needs_bash
@pytest.mark.parametrize("go, confirm", [("", None), ("yes", None), ("pm-go", ""), ("pm-go", "restore other.dump")])
def test_nothing_runs_without_the_pms_go_and_the_typed_confirmation(tmp_path, go, confirm):
    done, calls = _host(tmp_path, go=go, confirm=confirm)
    assert done.returncode == 1 and "restore journal: FAIL" in done.stdout and calls == []


@needs_bash
def test_the_app_stops_before_the_swap_is_migrated_after_and_starts_again(tmp_path):
    done, calls = _host(tmp_path)
    assert done.returncode == 0, done.stdout
    stop = calls.index("compose stop supervisor dashboard hub-binance")
    swap = next(i for i, c in enumerate(calls) if c.endswith("backup bash -s"))
    migrate = next(i for i, c in enumerate(calls) if "migrate" in c)
    start = calls.index("compose start supervisor dashboard hub-binance")
    assert stop < swap < migrate < start
    assert not any(c.startswith(("compose stop db", "compose stop backup", "compose down", "compose rm")) for c in calls)


@needs_bash
def test_a_failure_before_the_swap_starts_the_app_again_on_the_untouched_journal(tmp_path):
    done, calls = _host(tmp_path, inner_ok=False)
    assert done.returncode == 1 and "the journal is as it was" in done.stdout
    assert calls[-1] == "compose start supervisor dashboard hub-binance" and not [c for c in calls if "migrate" in c]


@needs_bash
def test_a_journal_out_of_its_place_keeps_the_app_stopped_and_says_so(tmp_path):
    """QT3 F-RJ-1: the inner script's exit 2 means no database carries the journal's name: the app stays stopped."""
    done, calls = _host(tmp_path, inner_ok=False, inner_exit=2)
    assert done.returncode == 2 and "the app stays stopped" in done.stdout, done.stdout
    assert "as it was" not in done.stdout and not [c for c in calls if c.startswith("compose start")]
    assert not [c for c in calls if "migrate" in c]


def test_a_restore_never_overlaps_a_deploy_or_another_journal_writer():
    """QT3 F-RJ-2: the workflow runs in the deploy's history-writer group, so a push to main waits for it."""
    deploy = (ROOT / ".github" / "workflows" / "deploy.yml").read_text()
    group = re.compile(r"^concurrency: \{ group: ([\w-]+), cancel-in-progress: false \}$", re.M)
    assert group.search(deploy).group(1) == "history-writer" == group.search(WORKFLOW.read_text()).group(1)


@needs_bash
def test_a_failed_migration_after_the_swap_keeps_the_app_stopped(tmp_path):
    done, calls = _host(tmp_path, migrate_ok=False)
    assert done.returncode == 1 and "the app stays stopped" in done.stdout
    assert not [c for c in calls if c.startswith("compose start")]


def test_the_workflow_is_by_hand_only_behind_its_own_switch_and_prints_no_tool_output():
    text = WORKFLOW.read_text()
    assert re.search(r"^on:\n  workflow_dispatch:", text, re.M) and "schedule" not in text and "push:" not in text
    assert "vars.RESTORE_JOURNAL_ENABLED == 'true'" in text and "refs/heads/main" in text
    assert "LogLevel=QUIET" in text and "${{ inputs." not in text.split("run: |", 1)[1]  # inputs only through env
    assert shutil.which("bash") is None or "restore_journal.sh" in text
