"""DA-7, the restore drill (HoE, 15 Oct): deploy/restore_drill.sh restores the newest nightly dump into a throwaway
database on the server and python -m sleeve_fund.restore_drill checks it. Red first:
1. a dry run on a real dump of a seeded journal passes every step, and a dump that lost rows fails;
2. no log line carries data (the log is public): no Strategy name, message, amount, host, database or path;
3. the script removes what it made, on failure too (stand-in docker, as test_backup_script stands in Postgres)."""

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


def _night(tmp_path: Path, lose_a_fill: bool = False) -> tuple[Path, str]:
    """A journal at head with a Strategy, an order and its fill, an event and a decision; dumped as backup.sh dumps it,
    its status.json as backup.sh writes it, the same night's research archive; restored into a fresh database."""
    from sqlalchemy import create_engine, text

    from sleeve_fund import schema
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
        counts = {t: c.execute(text(f"SELECT count(*) FROM {t}")).scalar() for t in ("sleeves", "orders", "fills",
                                                                                       "events")}
    engine.dispose()
    backups = tmp_path / "backups"
    backups.mkdir()
    dump = backups / "sleeve_fund-20261015-0012.dump"
    subprocess.run([_tool("pg_dump"), "-d", _libpq(src), "-Fc", "-f", str(dump)], check=True, timeout=60)
    if lose_a_fill:
        counts["events"] += 1  # the backup counted a row the dump no longer has
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
def test_a_dry_run_on_a_real_dump_passes_every_step(tmp_path, capsys):
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path)
    assert restore_drill.main(["--backups", str(backups), "--db", dst, "--restore-seconds", "1.5"]) == 0
    lines = capsys.readouterr().out.splitlines()
    steps = [ln.split(":")[0] for ln in lines]
    assert steps == ["backup", "restore", "migrate", "counts", "tables", "links", "research", "start check and books",
                     "restore drill"]
    assert all(": PASS" in ln for ln in lines), lines
    assert "idea ledger 2 lines, 1 tear sheets" in next(ln for ln in lines if ln.startswith("research"))


@needs_pg
def test_a_dump_short_of_the_backups_own_count_fails(tmp_path, capsys):
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path, lose_a_fill=True)
    assert restore_drill.main(["--backups", str(backups), "--db", dst]) == 1
    out = capsys.readouterr().out
    assert "counts: FAIL" in out and "apart from the backup's count: events" in out
    assert out.splitlines()[-1].startswith("restore drill: FAIL")


@needs_pg
def test_no_log_line_carries_data(tmp_path, capsys):
    """The log is public: step names, counts, timings and PASS/FAIL only."""
    from sleeve_fund import restore_drill

    backups, dst = _night(tmp_path)
    restore_drill.main(["--backups", str(backups), "--db", dst])
    out = capsys.readouterr().out
    host_db = re.match(r"^[^/]+//(?:[^@]*@)?([^:/]+)", URL).group(1)
    for leaked in (*SECRETS, str(tmp_path), str(backups), "drill_dst", "postgresql", host_db + ":"):
        assert leaked not in out, leaked
    assert all(LINE.match(ln) for ln in out.splitlines()), out


def test_a_step_that_raises_prints_its_error_type_only(capsys):
    from sleeve_fund.restore_drill import Drill

    drill = Drill()
    drill.step("links", lambda: (_ for _ in ()).throw(RuntimeError(f"host db.internal row {SECRETS[0]}")))
    out = capsys.readouterr().out
    assert drill.failed and out.startswith("links: FAIL (") and "RuntimeError" in out
    assert SECRETS[0] not in out and "db.internal" not in out


# --- the script, with a stand-in docker ---------------------------------------------------------------------------

def _docker(bin_dir: Path, log: Path, fail_on: str = "") -> None:
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "docker").write_text(f"""#!/usr/bin/env bash
echo "$*" >> {log}
case "$*" in
  *"{fail_on or '@never@'}"*) echo "boom: secret-host /var/lib/data row" >&2; exit 1;;
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


def _run(tmp_path: Path, fail_on: str = "") -> tuple[subprocess.CompletedProcess, list[str]]:
    log = tmp_path / "docker.log"
    _docker(tmp_path / "bin", log, fail_on)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "DRILL_ID": "drill-x"}
    done = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, timeout=60)
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
