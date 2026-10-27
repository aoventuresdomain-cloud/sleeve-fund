"""TRIALS-EXPORT (PM yes 25 Oct, again 27 Oct): the one-off read-only workflow that brings the trials register from the
paper server (.github/workflows/trials-export.yml). Done when: the server line is one COPY of the trials table in a
read-only session, run here through a stand-in ssh and docker as the runner and the server's shell would split it; an
empty register fails the run; the CSV is encrypted in sheet-export's layout before upload; and it runs from main with
deploys on only."""

from __future__ import annotations

import pathlib
import re
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "trials-export.yml"
COLUMNS = ("id, definition_hash, idea_hash, code_version, definition_name, family, dataset, stage, source, kind, "
           "status, sharpe, trades, oos_trades, backtest_id, data_start, data_end, created_at, settings")


def _job():
    return yaml.safe_load(WF.read_text())["jobs"]["export"]


def _step(name):
    return next(s for s in _job()["steps"] if s.get("name") == name)


def _read(tmp_path, rows: str):
    """Runs "Read the register" with ssh standing in: the fake ssh hands its remote line to a shell where sudo writes
    each argument it gets, NUL-ended, to argv.txt, and stdout is `rows`, as psql's COPY would write it."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (tmp_path / "rows.csv").write_text(rows)
    (bin_ / "sudo").write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" > {tmp_path}/argv.txt\ncat {tmp_path}/rows.csv\n')
    (bin_ / "ssh").write_text('#!/bin/bash\nfor a; do last="$a"; done\ncd() { :; }\nexport -f cd\n'
                              f'PATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    run = _step("Read the register")["run"]
    res = subprocess.run(["bash", "-e", "-c", run], capture_output=True, text=True, cwd=tmp_path, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path), "USER_AT": "u",
                              "HOST": "h", "HOME": str(tmp_path)})
    argv = (tmp_path / "argv.txt").read_text().split("\0")[:-1] if (tmp_path / "argv.txt").exists() else []
    return res, argv


def test_the_server_line_is_one_copy_of_the_register_in_a_read_only_session(tmp_path):
    res, argv = _read(tmp_path, "id,definition_hash\nabc,def\n")
    assert res.returncode == 0, res.stderr
    assert argv[:6] == ["docker", "compose", "exec", "-T", "-e", "PGOPTIONS=-c default_transaction_read_only=on"]
    assert argv[6:12] == ["db", "psql", "-U", "sleeve", "-d", "sleeve_fund"]
    sql = " ".join(argv[argv.index("-c") + 1].split())
    assert sql == (f"COPY (SELECT {COLUMNS} FROM trials ORDER BY created_at, id) "
                   "TO STDOUT WITH (FORMAT csv, HEADER true)")
    assert not re.search(r"\b(insert|update|delete|drop|alter|truncate|create|grant)\b", sql, re.IGNORECASE)
    assert "1 lines after the header" in res.stdout and "abc" not in res.stdout  # counts in the log, never contents
    assert (tmp_path / "plain" / "sheet.tar.gz").exists()


def test_the_columns_are_the_registers_own(tmp_path):
    """Every column the COPY names is one the trials migrations create."""
    made = set()
    for m in sorted((ROOT / "sleeve_fund" / "migrations" / "versions").glob("*.py")):
        text = m.read_text()
        if "trials" in text:
            made |= set(re.findall(r"sa\.Column\('(\w+)'", text))
    assert set(COLUMNS.split(", ")) <= made


def test_an_empty_register_fails_the_run(tmp_path):
    res, _ = _read(tmp_path, "id,definition_hash\n")
    assert res.returncode != 0 and "the register came back empty" in res.stdout


def test_it_is_encrypted_in_sheet_exports_layout_and_runs_from_main_with_deploys_on():
    job = _job()
    assert "vars.DEPLOY_ENABLED == 'true'" in job["if"] and "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    names = [s.get("name") or s["uses"] for s in job["steps"]]
    assert names == ["actions/checkout@v4", "Configure SSH", "Read the register", "Encrypt",
                     "actions/upload-artifact@v4", "Clean up"]
    enc = _step("Encrypt")["run"]
    assert "deploy/keys/sheet-export.pub.pem" in enc and "rsa_oaep_md:sha256" in enc
    assert 'rm -rf -- "$RUNNER_TEMP/plain"' in enc  # nothing plain reaches the upload
    upload = job["steps"][4]["with"]
    assert upload["path"] == "${{ runner.temp }}/out/" and upload["retention-days"] == 3
    assert "inputs" not in (yaml.safe_load(WF.read_text())[True]["workflow_dispatch"] or {})
