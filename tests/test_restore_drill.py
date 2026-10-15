"""DA-7, the restore drill (HoE, 15 Oct): deploy/restore_drill.sh restores the newest nightly dump into a throwaway
database on the server and python -m sleeve_fund.restore_drill checks it. Red first:
1. a dry run on a real dump of a seeded journal passes every step, and a dump that lost rows fails;
2. no log line carries data (the log is public): no Strategy name, message, amount, host, database or path;
3. the script removes what it made, on failure too (stand-in docker, as test_backup_script stands in Postgres);
4. (HoE, 15 Oct) the script refuses to start unless the backups' disk and docker's have twice the newest dump free;
5. (HoQA F-DA7-COUNTS) the money and history tables the backup now counts are matched too, not only four."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "deploy" / "restore_drill.sh"
PG_BIN = next((d for d in (Path("/usr/lib/postgresql/16/bin"),) if (d / "pg_dump").exists()), None)
URL = os.environ.get("TEST_DATABASE_URL", "")
SECRETS = ("zz-hidden-name", "SECRET-MESSAGE-TEXT", "98765.4321")
LINE = re.compile(r"^[a-z][a-z ]*: (PASS|FAIL|NOTE)( \(\d+\.\d+s( in all)?\))?(, [^/@]*)?$")

needs_pg = pytest.mark.skipif(not URL.startswith("postgresql") or (shutil.which("pg_dump") is None and PG_BIN is None),
                              reason="Postgres only: TEST_DATABASE_URL and pg_dump")


def _tool(name: str) -> str:
    return shutil.which(name) or str(PG_BIN / name)


def _libpq(url: str) -> str:
    return re.sub(r"^postgresql\+\w+://", "postgresql://", url)


def _db(name: str) -> str:
    from sqlalchemy import create_engine, text

    admin = create_engine(URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
        c.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()
    return URL.rsplit("/", 1)[0] + "/" + name


def _night(tmp_path: Path, short: str = "", counted: tuple[str, ...] | None = None) -> tuple[Path, str]:
    """A journal at head with a Strategy, an order and its fill, an event and a decision; dumped as backup.sh dumps it,
    its status.json as backup.sh writes it, the same night's research archive; restored into a fresh database."""
    from sqlalchemy import create_engine, text

    from sleeve_fund import schema
    from sleeve_fund.restore_drill import COUNTED
    from sleeve_fund.store import Store

    src = _db("drill_src")
    engine = create_engine(src)
    schema.migrate(engine, log=lambda _: None)
    store = Store(engine=engine)
    store.create_sleeve(name=SECRETS[0], strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, params={"market": "perp", "stop_loss": 0.02}, desired_state="stopped")
    store.event(SECRETS[0], "info", "note", SECRETS[1])
    store.decide("pm", "note", f"{SECRETS[1]} {SECRETS[2]}", SECRETS[0])
    with engine.begin() as c:
        counts = {t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in (counted or COUNTED)}
    engine.dispose()
    backups = tmp_path / "backups"
    backups.mkdir()
    dump = backups / "sleeve_fund-20261015-0012.dump"
    subprocess.run([_tool("pg_dump"), "-d", _libpq(src), "-Fc", "-f", str(dump)], check=True, timeout=60)
    if short:
        counts[short] += 1  # the backup counted a row the dump no longer has
    (backups / "status.json").write_text(json.dumps({"ts": "2026-10-15T00:13:00Z", "ok": True, "file": dump.name,
                                                     "message": "dumped and restored", "restored": counts}))
    research = tmp_path / "research"
    (research / "tearsheets").mkdir(parents=True)
    (research / "idea_ledger.jsonl").write_text('{"idea": "a"}\n{"idea": "b"}\n')
    (research / "tearsheets" / "one.html").write_text("<p>sheet</p>")
    with tarfile.open(backups / "research-20261015-0012.tar.gz", "w:gz") as tf:
        tf.add(research, arcname=".")
    dst = _db("drill_dst")
    subprocess.run([_tool("pg_restore"), "-d", _libpq(dst), "--no-owner", "--no-privileges", "--exit-on-error",
                    str(dump)], check=True, timeout=60)
    return backups, dst


@needs_pg
def test_a_dry_run_on_a_real_dump_passes_every_step(tmp_path, capfd):
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path)
    assert restore_drill.main(["--backups", str(backups), "--db", dst, "--restore-seconds", "1.5"]) == 0
    lines = capfd.readouterr().out.splitlines()
    steps = [ln.split(":")[0] for ln in lines]
    assert steps == ["backup", "restore", "migrate", "counts", "tables", "links", "research", "start check and books",
                     "restore drill"]
    assert all(": PASS" in ln for ln in lines), lines
    assert "idea ledger 2 lines, 1 tear sheets" in next(ln for ln in lines if ln.startswith("research"))


@needs_pg
@pytest.mark.parametrize("short", ["events", "decisions"])  # decisions: a history table the backup counts since F-DA7
def test_a_dump_short_of_the_backups_own_count_fails(tmp_path, capfd, short):
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path, short=short)
    assert restore_drill.main(["--backups", str(backups), "--db", dst]) == 1
    out = capfd.readouterr().out
    assert "counts: FAIL" in out and f"apart from the backup's count: {short}" in out
    assert out.splitlines()[-1].startswith("restore drill: FAIL")


def test_the_backup_counts_every_table_the_drill_matches():
    from sleeve_fund.restore_drill import COUNTED
    from sleeve_fund.store import metadata

    script = (ROOT / "deploy" / "backup.sh").read_text()
    money = {"decisions", "equity", "funding", "insurance", "fund_flows", "book_marks", "book_moves", "sleeve_archive",
             "gate_reservations", "trials"}
    assert money | {"sleeves", "orders", "fills", "events"} <= set(COUNTED) <= set(metadata.tables)
    for t in COUNTED:
        assert f"'{t}', (select count(*) from {t})" in script, t


@needs_pg
def test_a_status_from_before_the_wider_count_is_matched_on_what_it_counted(tmp_path, capfd):
    """A backup taken before backup.sh counted the money tables: its four are matched, the rest are a note."""
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path, counted=("sleeves", "orders", "fills", "events"))
    assert restore_drill.main(["--backups", str(backups), "--db", dst]) == 0
    line = next(ln for ln in capfd.readouterr().out.splitlines() if ln.startswith("counts"))
    assert line.startswith("counts: PASS") and "as the backup counted; not in the backup's count: decisions" in line
    assert "book_marks" in line.split("not in the backup's count")[1]


@needs_pg
def test_no_log_line_carries_data(tmp_path, capfd):
    """The log is public: step names, counts, timings and PASS/FAIL only."""
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path)
    restore_drill.main(["--backups", str(backups), "--db", dst])
    out = capfd.readouterr().out
    host_db = re.match(r"^[^/]+//(?:[^@]*@)?([^:/]+)", URL).group(1)
    for leaked in (*SECRETS, str(tmp_path), str(backups), "drill_dst", "postgresql", host_db + ":"):
        assert leaked not in out, leaked
    assert all(LINE.match(ln) for ln in out.splitlines()), out


def _counts_against(tmp_path: Path, restored) -> tuple[str, str]:
    from sqlalchemy import create_engine

    from sleeve_fund.restore_drill import check_counts
    from sleeve_fund.store import metadata

    engine = create_engine("sqlite://")
    metadata.create_all(engine)
    (tmp_path / "status.json").write_text(json.dumps({"ok": True, "file": "x.dump", "restored": restored}))
    try:
        return check_counts(engine, tmp_path)
    finally:
        engine.dispose()


@pytest.mark.parametrize("restored", [None, {}])
def test_a_backup_that_counted_nothing_fails_the_counts(tmp_path, restored):
    """CR F-RD-1 (a): no count to match is no proof, never a pass."""
    status, words = _counts_against(tmp_path, restored)
    assert status == "FAIL" and "no count" in words


def test_a_counted_table_the_schema_hasnt_fails_without_a_query(tmp_path):
    """CR F-RD-1 (b): a name the backup counted that the restored schema lacks is a FAIL, and is never put in SQL."""
    want = {t: 0 for t in ("sleeves", "orders", "fills", "events")} | {"x; drop table sleeves": 0}
    status, words = _counts_against(tmp_path, want)
    assert status == "FAIL" and "apart from the backup's count: x; drop table sleeves" in words


def test_a_step_that_raises_prints_its_error_type_only(capfd):
    from sleeve_fund.restore_drill import Drill

    drill = Drill()
    drill.step("links", lambda: (_ for _ in ()).throw(RuntimeError(f"host db.internal row {SECRETS[0]}")))
    out = capfd.readouterr().out
    assert drill.failed and out.startswith("links: FAIL (") and "RuntimeError" in out
    assert SECRETS[0] not in out and "db.internal" not in out


# --- the script, with a stand-in docker ---------------------------------------------------------------------------

def _docker(bin_dir: Path, log: Path, fail_on: str = "", free: int = 10**9) -> None:
    bin_dir.mkdir(exist_ok=True)
    disk = bin_dir.parent / "disk"
    (disk / "backups").mkdir(parents=True, exist_ok=True)
    (disk / "docker").mkdir(exist_ok=True)
    (disk / "backups" / "sleeve_fund-20261014-0012.dump").write_bytes(b"x" * 10)
    (disk / "backups" / "sleeve_fund-20261015-0012.dump").write_bytes(b"x" * 1000)  # the newest: 1000 bytes
    (bin_dir / "df").write_text(f"""#!/usr/bin/env bash
echo "df $*" >> {log}
echo "Filesystem 1-blocks Used Available Capacity Mounted on"
echo "/dev/x 99999999999 1 {free} 1% /"
""")
    (bin_dir / "df").chmod(0o755)
    (bin_dir / "docker").write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
case "$*" in
  *"{fail_on or '@never@'}"*) echo "boom: secret-host /var/lib/data row" >&2; exit 1;;
  "volume inspect"*Mountpoint*) echo {disk / "backups"};;
  "info "*DockerRootDir*) echo {disk / "docker"};;
  "compose ps -q"*) echo cid-$4;;
  inspect*Mounts*) echo backups-volume;;
  inspect*Image*) echo some-image;;
  "exec "*" sh -c "*) echo /backups/sleeve_fund-20261015-0012.dump;;
  "run --rm"*) echo "backup: PASS (0.1s)"; echo "restore drill: PASS (0.2s in all)";;
esac
exit 0
""")
    (bin_dir / "docker").chmod(0o755)
    (bin_dir / "sleep").write_text("#!/usr/bin/env bash\nexit 0\n")
    (bin_dir / "sleep").chmod(0o755)


def _run(tmp_path: Path, fail_on: str = "", free: int = 10**9) -> tuple[subprocess.CompletedProcess, list[str]]:
    log = tmp_path / "docker.log"
    _docker(tmp_path / "bin", log, fail_on, free)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "DRILL_ID": "drill-x"}
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60, check=False)
    return done, log.read_text().splitlines()


pytestmark_script = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


@pytestmark_script
def test_the_script_restores_into_its_own_network_and_volume_and_removes_them(tmp_path):
    done, calls = _run(tmp_path)
    assert done.returncode == 0, done.stdout
    assert "network create --internal drill-x" in calls
    run = next(c for c in calls if c.startswith("run -d"))
    assert "--network drill-x" in run and "drill-x:/var/lib/postgresql/data" in run and "backups-volume:/backups:ro" in run
    assert " -p " not in f" {run} " and "5432:" not in run  # nothing published on the host
    assert any(c.startswith("exec drill-x-db pg_restore") and "--exit-on-error" in c for c in calls)
    assert calls[-3:] == ["rm -f drill-x-db", "network rm drill-x", "volume rm drill-x"]
    assert done.stdout.splitlines()[-1] == "cleanup: done"


@pytestmark_script
@pytest.mark.parametrize("fail_on", ["pg_restore", "run --rm", "volume create"])
def test_the_script_removes_what_it_made_when_a_step_fails_and_prints_no_tool_output(tmp_path, fail_on):
    done, calls = _run(tmp_path, fail_on)
    assert done.returncode == 1
    assert calls[-3:] == ["rm -f drill-x-db", "network rm drill-x", "volume rm drill-x"]
    assert "secret-host" not in done.stdout + done.stderr and "/var/lib" not in done.stdout
    assert ": FAIL" in done.stdout or "checks: exit" in done.stdout


@pytestmark_script
@pytest.mark.parametrize("short_on", ["backups", "docker"])
def test_the_script_refuses_to_start_without_twice_the_newest_dump_free(tmp_path, short_on):
    """HoE: free disk on the backups' volume and on docker's root (where the drill's volume lands) must each be at
    least twice the newest dump; else nothing is made and the log says so, without a path or a size."""
    log = tmp_path / "docker.log"
    _docker(tmp_path / "bin", log)
    disk = tmp_path / "disk"
    # df answers 1999 (one byte short of twice the newest dump) for the short path only
    (tmp_path / "bin" / "df").write_text(f"""#!/usr/bin/env bash
echo "df $*" >> {log}
case "$*" in *{disk / short_on}*) free=1999;; *) free=2000;; esac
echo "Filesystem 1-blocks Used Available Capacity Mounted on"
echo "/dev/x 99999999999 1 $free 1% /"
""")
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "DRILL_ID": "drill-x"}
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60, check=False)
    calls = log.read_text().splitlines()
    assert done.returncode == 1
    assert "not enough disk: FAIL" in done.stdout and str(disk) not in done.stdout and "1999" not in done.stdout
    assert not any(c.startswith(("network create", "volume create", "run ")) for c in calls)  # nothing made
    assert any(c.startswith("df") and str(disk / short_on) in c for c in calls)


@pytestmark_script
def test_the_script_starts_with_exactly_twice_the_newest_dump_free(tmp_path):
    done, calls = _run(tmp_path, free=2000)
    assert done.returncode == 0, done.stdout
    assert "disk: PASS" in done.stdout and any(c.startswith("exec drill-x-db pg_restore") for c in calls)


REFRESH = ROOT / "deploy" / "backup_refresh.sh"


def _refresh(tmp_path: Path, script: str, fail: bool = False) -> tuple[subprocess.CompletedProcess, list[str]]:
    """deploy/backup_refresh.sh run as remote_up.sh runs it (from the app folder), with a stand-in docker."""
    (tmp_path / "deploy").mkdir(exist_ok=True)
    (tmp_path / "deploy" / "backup.sh").write_text(script)
    log = tmp_path / "docker.log"
    log.touch()
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin" / "docker").write_text(f'#!/usr/bin/env bash\necho "$*" >> {log}\n{"exit 1" if fail else "exit 0"}\n')
    (tmp_path / "bin" / "docker").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}"}
    done = subprocess.run(["bash", str(REFRESH)], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30,
                          check=False)
    return done, log.read_text().splitlines()


RECREATE = "compose up -d --force-recreate --no-deps backup"


@pytestmark_script
def test_a_changed_backup_script_recreates_the_backup_and_stamps_it(tmp_path):
    """HoE: backup.sh is bind-mounted, so compose up leaves the backup on the old script. A deploy recreates it when
    the script changed (the first deploy too: no stamp yet), and stamps the script it now runs."""
    done, calls = _refresh(tmp_path, "echo one\n")
    assert done.returncode == 0 and calls == [RECREATE]
    stamp = (tmp_path / ".backup_sha").read_text().strip()
    done, calls = _refresh(tmp_path, "echo two\n")
    assert done.returncode == 0 and calls == [RECREATE, RECREATE]
    assert (tmp_path / ".backup_sha").read_text().strip() not in ("", stamp)


@pytestmark_script
def test_an_unchanged_backup_script_leaves_the_backup_running(tmp_path):
    """Each recreate dumps at once and the backup keeps 14 dumps by count: a deploy that didn't change the script
    must not spend a night of history."""
    _refresh(tmp_path, "echo one\n")
    done, calls = _refresh(tmp_path, "echo one\n")
    assert done.returncode == 0 and calls == [RECREATE]  # the first deploy's only


@pytestmark_script
def test_a_failed_recreate_keeps_the_old_stamp_and_fails_the_deploy(tmp_path):
    _refresh(tmp_path, "echo one\n")
    stamp = (tmp_path / ".backup_sha").read_text()
    done, calls = _refresh(tmp_path, "echo two\n", fail=True)
    assert done.returncode != 0 and calls == [RECREATE, RECREATE]
    assert (tmp_path / ".backup_sha").read_text() == stamp  # the next deploy tries again


def test_the_deploy_refreshes_the_backup_after_the_stack_and_keeps_the_stamp():
    lines = [ln.strip() for ln in (ROOT / "deploy" / "remote_up.sh").read_text().splitlines()]
    up = lines.index("docker compose up -d --build --remove-orphans")
    assert lines[up + 1] == "bash deploy/backup_refresh.sh"
    assert "--force-recreate" not in (ROOT / "deploy" / "remote_up.sh").read_text()  # only when the script changed
    # the deploy's rsync --delete would remove the stamp, and every deploy would recreate
    assert "--exclude .backup_sha" in (ROOT / ".github" / "workflows" / "deploy.yml").read_text()


def test_the_workflow_is_by_hand_on_main_quiet_and_copies_nothing_off_the_server():
    import yaml

    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "restore-drill.yml").read_text())
    assert wf[True].keys() == {"workflow_dispatch"}  # YAML 1.1 reads "on" as true: never on a schedule or a push
    job = wf["jobs"]["drill"]
    for gate in ("vars.DEPLOY_ENABLED == 'true'", "github.ref == 'refs/heads/main'"):
        assert gate in job["if"]
    assert job["permissions"] == {"contents": "read"}
    steps = job["steps"]
    assert [s["name"] for s in steps] == ["Configure SSH", "Restore drill"]
    assert "LogLevel=QUIET" in steps[1]["run"] and "sudo bash deploy/restore_drill.sh" in steps[1]["run"]
    for s in steps:
        assert "${{" not in s["run"], s["name"]  # secrets and vars reach a shell line through env only
        assert "uses" not in s  # no upload, no artifact: no dump or row leaves the server
    for word in ("scp", "rsync", "upload-artifact", "pg_dump"):
        assert word not in (ROOT / ".github" / "workflows" / "restore-drill.yml").read_text()
