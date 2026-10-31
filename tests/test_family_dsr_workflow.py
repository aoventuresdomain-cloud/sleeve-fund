"""FAMILY-DSR's workflow (.github/workflows/family-dsr.yml): read-only; the server line is the module in a one-off
dashboard container and nothing else; the log shows the outcome's first word, never a Sharpe or an id; the report is
encrypted as trials-export's; it never runs beside a study and only from main."""

from __future__ import annotations

import pathlib
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "family-dsr.yml"
LIST = "configs/studies/next-round-pinned-dates-list.md"


def _wf():
    return yaml.safe_load(WF.read_text())


def _step(name):
    return next(s for s in _wf()["jobs"]["dsr"]["steps"] if s.get("name") == name)


def _run(tmp_path, report: str, code: int = 0):
    """Runs the server step with ssh standing in: sudo writes its arguments, NUL-ended, to argv.txt and prints
    `report` as the module's output."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (tmp_path / "report").write_text(report)
    (bin_ / "sudo").write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" > {tmp_path}/argv.txt\ncat {tmp_path}/report\n'
                               f'exit {code}\n')
    (bin_ / "ssh").write_text('#!/bin/bash\nfor a; do last="$a"; done\ncd() { :; }\nexport -f cd\n'
                              f'PATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    res = subprocess.run(["bash", "-e", "-c", _step("Work out the deflated Sharpe")["run"]], capture_output=True,
                         text=True, cwd=tmp_path, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path), "USER_AT": "u",
                              "HOST": "h", "HOME": str(tmp_path), "RUN_ID": "7", "LIST": LIST, "FAMILY": "B",
                              "SHEET": "h0_x_1"})
    return res, (tmp_path / "argv.txt").read_text().split("\0")[:-1]


def test_it_runs_only_the_module_and_logs_no_figure(tmp_path):
    res, argv = _run(tmp_path, "family B: N 15 (stated 15, register 15)\ndeflated Sharpe probability 0.4321\n"
                               "done: family B, sheet h0_x_1\n")
    assert res.returncode == 0, res.stderr
    assert argv[:7] == ["docker", "compose", "run", "--rm", "-T", "--no-deps", "--name"]
    assert argv[8:] == ["dashboard", "python", "-m", "sleeve_fund.research.family_dsr", "--list", LIST, "--family",
                        "B", "--sheet", "h0_x_1"]
    assert "outcome: done, exit 0" in res.stdout and "0.4321" not in res.stdout and "N 15" not in res.stdout
    assert (tmp_path / "code").read_text().strip() == "0" and (tmp_path / "plain" / "sheet.tar.gz").exists()


def test_a_refusal_still_writes_the_report_and_fails_the_run_later(tmp_path):
    res, _ = _run(tmp_path, "refused: no counted row 0123456789abcdef (a correction not yet applied?)\n", code=1)
    assert "outcome: refused, exit 1" in res.stdout and "0123456789abcdef" not in res.stdout
    assert (tmp_path / "code").read_text().strip() == "1"


def test_it_has_no_write_input_runs_beside_no_study_and_only_from_main():
    wf = _wf()
    assert set(wf[True]["workflow_dispatch"]["inputs"]) == {"list", "family", "sheet"}  # yaml reads "on" as True
    assert wf["concurrency"]["group"] == "study"
    job = wf["jobs"]["dsr"]
    assert "vars.DEPLOY_ENABLED == 'true'" in job["if"] and "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    assert "--apply" not in WF.read_text()
    names = [s.get("name") or s["uses"] for s in job["steps"]]
    assert names.index("Encrypt") < names.index("actions/upload-artifact@v4") < names.index("The script's outcome")
    assert "sheet-export.pub.pem" in _step("Encrypt")["run"]


def test_the_inputs_are_checked_on_the_runner_before_anything_runs(tmp_path):
    def check(list_, family="B", sheet="h0_x_1"):
        return subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                              cwd=ROOT, check=False, env={"PATH": "/usr/bin:/bin", "LIST": list_, "FAMILY": family,
                                                          "SHEET": sheet})
    assert check(LIST).returncode == 0
    assert check("configs/studies/no-such-list.md").returncode == 1  # well formed, but not in this checkout (CR F3)
    assert check("configs/studies/../studies/x.md").returncode == 1
    assert check(LIST, family="B;x").returncode == 1 and check(LIST, sheet="../x").returncode == 1
