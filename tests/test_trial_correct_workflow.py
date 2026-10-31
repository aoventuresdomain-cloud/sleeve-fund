"""TRIAL-CORRECT's workflow (.github/workflows/trial-correct.yml): a dry run unless apply is ticked; the server line is
the correction module in a one-off dashboard container and nothing else; the log shows the outcome's first word, never
a Sharpe or an id; the report is encrypted as trials-export's; it never runs beside a study and only from main."""

from __future__ import annotations

import pathlib
import subprocess

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "trial-correct.yml"
ROW = "0123456789abcdef"


def _wf():
    return yaml.safe_load(WF.read_text())


def _step(name):
    return next(s for s in _wf()["jobs"]["correct"]["steps"] if s.get("name") == name)


def _run(tmp_path, apply: str, report: str, code: int = 0):
    """Runs "Correct the row" with ssh standing in: sudo writes its arguments, NUL-ended, to argv.txt and prints
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
    res = subprocess.run(["bash", "-e", "-c", _step("Correct the row")["run"]], capture_output=True, text=True,
                         cwd=tmp_path, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "RUNNER_TEMP": str(tmp_path), "USER_AT": "u",
                              "HOST": "h", "HOME": str(tmp_path), "RUN_ID": "7", "ROW": ROW, "SHEET": "b0_x_1",
                              "APPLY": apply})
    return res, (tmp_path / "argv.txt").read_text().split("\0")[:-1]


def test_a_dry_run_runs_the_module_without_apply_and_logs_no_figure(tmp_path):
    res, argv = _run(tmp_path, "", f'original: {{"id": "{ROW}", "sharpe": 1.8}}\ndry run: nothing written\n')
    assert res.returncode == 0, res.stderr
    assert argv[:7] == ["docker", "compose", "run", "--rm", "-T", "--no-deps", "--name"]
    assert argv[8:] == ["dashboard", "python", "-m", "sleeve_fund.research.trial_correction", "--row", ROW,
                        "--sheet", "b0_x_1"]
    assert "outcome: dry, exit 0" in res.stdout and ROW not in res.stdout and "1.8" not in res.stdout
    assert (tmp_path / "code").read_text().strip() == "0" and (tmp_path / "plain" / "sheet.tar.gz").exists()


def test_apply_adds_the_flag_and_a_refusal_fails_the_run_later(tmp_path):
    res, argv = _run(tmp_path, "--apply", "refused: 2 counted rows have this basket key, not one\n", code=1)
    assert argv[-1] == "--apply" and "outcome: refused, exit 1" in res.stdout and "counted rows" not in res.stdout
    assert (tmp_path / "code").read_text().strip() == "1"  # the encrypted report still uploads first


def test_it_is_dry_by_default_beside_no_study_and_from_main_with_deploys_on():
    wf = _wf()
    apply = wf[True]["workflow_dispatch"]["inputs"]["apply"]  # yaml reads "on" as True
    assert apply["type"] == "boolean" and apply["default"] is False
    assert wf["concurrency"]["group"] == "study"
    job = wf["jobs"]["correct"]
    assert "vars.DEPLOY_ENABLED == 'true'" in job["if"] and "github.ref == 'refs/heads/main'" in job["if"]
    assert job["permissions"] == {"contents": "read"}
    assert _step("Correct the row")["env"]["APPLY"] == "${{ inputs.apply && '--apply' || '' }}"
    names = [s.get("name") or s["uses"] for s in job["steps"]]
    assert names.index("Encrypt") < names.index("actions/upload-artifact@v4") < names.index("The script's outcome")
    assert "sheet-export.pub.pem" in _step("Encrypt")["run"]
