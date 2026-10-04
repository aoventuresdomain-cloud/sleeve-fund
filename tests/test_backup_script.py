"""Review round 7, R7-B: deploy/backup.sh wrote error text with tabs into status.json (invalid JSON, so
Ops showed a failed backup as fine) and kept 14 dumps by count, failed ones included, so a failure
retried hourly removed every good dump in about 14 hours. Run the real script with stand-in Postgres
tools."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "backup.sh"
pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def _tools(bin_dir: Path, restore_ok: bool, error: str = "") -> None:
    bin_dir.mkdir(exist_ok=True)
    stubs = {
        "pg_dump": 'while [ $# -gt 0 ]; do [ "$1" = -f ] && { echo dump > "$2"; }; shift; done',
        "createdb": "exit 0",
        "dropdb": "exit 0",
        "pg_restore": "exit 0" if restore_ok else f"printf '%b' {json.dumps(error)} >&2; exit 1",
        "psql": """echo '{"sleeves" : 2, "orders" : 5, "fills" : 4, "events" : 9}'""",
    }
    for name, body in stubs.items():
        p = bin_dir / name
        p.write_text(f"#!/usr/bin/env bash\n{body}\n")
        p.chmod(0o755)


def _round(tmp_path: Path, minute: int, restore_ok: bool, error: str = "") -> dict:
    _tools(tmp_path / "bin", restore_ok, error)
    # Each round's dump gets its own name: the script stamps the minute, so fake the clock with date.
    (tmp_path / "bin" / "date").write_text(
        f'#!/usr/bin/env bash\ncase "$*" in *%M*) echo 20261004-{minute:04d};; *) /bin/date "$@";; esac\n')
    (tmp_path / "bin" / "date").chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}", "BACKUP_DIR": str(tmp_path / "b"),
           "BACKUP_ONCE": "1"}
    subprocess.run(["bash", str(SCRIPT)], env=env, check=True, timeout=30)
    return json.loads((tmp_path / "b" / "status.json").read_text())


def test_a_failing_backup_writes_valid_status_and_never_pushes_out_good_dumps(tmp_path):
    (tmp_path / "b").mkdir()
    ok = _round(tmp_path, 0, True)
    assert ok["ok"] is True and ok["restored"]["orders"] == 5
    for minute in range(1, 4):
        _round(tmp_path, minute, True)
    nasty = 'pg_restore: error:\tbad "header"\\ in C:\\\\path\nsecond line \xe9'
    for minute in range(10, 40):  # 30 failures in a row, more than the 14 kept
        bad = _round(tmp_path, minute, False, nasty)
    assert bad["ok"] is False and "bad \"header\"\\ in C:" in bad["message"] and "\t" not in bad["message"]
    good = sorted(p.name for p in (tmp_path / "b").glob("*.dump"))
    assert good == [f"sleeve_fund-20261004-{m:04d}.dump" for m in range(4)]  # every good one kept
    assert [p.name for p in (tmp_path / "b").glob("*.dump.failed")] == ["sleeve_fund-20261004-0039.dump.failed"]
    for minute in range(40, 56):  # good ones beyond 14 rotate out, oldest first
        _round(tmp_path, minute, True)
    good = sorted(p.name for p in (tmp_path / "b").glob("*.dump"))
    assert len(good) == 14 and good[0] == "sleeve_fund-20261004-0042.dump"
