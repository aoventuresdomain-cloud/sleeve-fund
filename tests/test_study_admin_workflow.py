"""STUDY-ADMIN (HoE 17 Nov): the hand-run workflow for the paper server's one-off study containers
(.github/workflows/study-admin.yml). Done when: check only reads (size, the study containers and their use) and changes
nothing; stop stops the one container named study-<run_id> and nothing else, never removes or prunes, and refuses a
run_id that is not 6 to 20 digits before anything reaches the server; a failed stop fails the run; and it runs by
hand from main with deploys on, one at a time, without a checkout. Run here through a stand-in ssh, sudo and docker as
the runner and the server's shell would split them."""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "study-admin.yml"


def _wf():
    return yaml.safe_load(WF.read_text())


def _step(name):
    return next(s for s in _wf()["jobs"]["admin"]["steps"] if s.get("name") == name)


def _server(tmp_path, step, run_id="", stop_fails=False):
    """Runs a step with ssh standing in: the fake ssh hands its remote line to bash, where sudo runs the command it is
    given and every docker call is written, one line of NUL-ended arguments each, to docker.txt."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "docker.txt"
    (bin_ / "sudo").write_text('#!/bin/bash\nexec "$@"\n')
    (bin_ / "docker").write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" >> {log}\necho >> {log}\n'
                                 f'if [ "$1" = stop ]; then {"exit 1" if stop_fails else 'echo "${@: -1}"'}; fi\n'
                                 'if [ "$1 $2" = "ps -q" ]; then echo abc123; fi\n')
    for tool in ("nproc", "free", "df"):
        (bin_ / tool).write_text(f'#!/bin/bash\necho {tool} "$@"\n')
    (bin_ / "ssh").write_text(f'#!/bin/bash\nfor a; do last="$a"; done\nPATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    res = subprocess.run(["bash", "-e", "-c", _step(step)["run"]], capture_output=True, text=True, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h", "HOME": str(tmp_path),
                              "RUN_ID": run_id})
    calls = [line.split("\0")[:-1] for line in log.read_text().splitlines() if line] if log.exists() else []
    return res, calls


def test_check_only_reads(tmp_path):
    res, calls = _server(tmp_path, "Check the server")
    assert res.returncode == 0, res.stderr
    assert "nproc" in res.stdout and "free -m" in res.stdout and "df -h / /opt" in res.stdout
    assert [c[0] for c in calls] == ["ps", "ps", "stats"]
    assert calls[0] == ["ps", "--filter", "name=study-", "--format", "{{.Names}} {{.Status}} {{.RunningFor}}"]
    assert calls[2] == ["stats", "--no-stream", "abc123"]
    text = _step("Check the server")["run"]
    assert not any(w in text for w in ("stop", " rm", "prune", "kill", "restart"))


def test_stop_stops_only_that_one_container_and_shows_the_list_either_side(tmp_path):
    res, calls = _server(tmp_path, "Stop the study container", run_id="19412345678")
    assert res.returncode == 0, res.stderr
    assert [c[0] for c in calls] == ["ps", "stop", "ps"]
    assert calls[1] == ["stop", "-t", "60", "study-19412345678"]
    assert res.stdout.index("before:") < res.stdout.index("study-19412345678") < res.stdout.index("after:")
    assert not any(w in _step("Stop the study container")["run"] for w in (" rm", "prune", "kill"))


def test_a_failed_stop_fails_the_run(tmp_path):
    res, calls = _server(tmp_path, "Stop the study container", run_id="19412345678", stop_fails=True)
    assert res.returncode != 0 and [c[0] for c in calls] == ["ps", "stop"]


@pytest.mark.parametrize("run_id", ["", "12345", "123456789012345678901", "1941234567x", "19412345678; reboot",
                                    "1941234567 8", "*", "19412345678\n"])
def test_stop_refuses_a_run_id_that_is_not_digits_before_the_server(tmp_path, run_id):
    checked = subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                             check=False, env={"PATH": "/usr/bin:/bin", "ACTION": "stop", "RUN_ID": run_id})
    assert checked.returncode == 1 and "6 to 20 digits" in checked.stdout
    res, calls = _server(tmp_path, "Stop the study container", run_id=run_id)  # and again where it is used
    assert res.returncode != 0 and calls == []


def test_check_needs_no_run_id():
    res = subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                         check=False, env={"PATH": "/usr/bin:/bin", "ACTION": "check", "RUN_ID": ""})
    assert res.returncode == 0


def test_it_runs_by_hand_from_main_with_deploys_on_one_at_a_time_and_checks_nothing_out():
    wf = _wf()
    assert set(wf[True]) == {"workflow_dispatch"}  # yaml reads the key `on` as True
    inputs = wf[True]["workflow_dispatch"]["inputs"]
    assert inputs["action"]["options"] == ["check", "stop", "names"] and inputs["action"]["default"] == "check"
    assert wf["concurrency"] == {"group": "study-admin"}
    job = wf["jobs"]["admin"]
    assert job["if"] == "${{ vars.DEPLOY_ENABLED == 'true' && github.ref == 'refs/heads/main' }}"
    assert job["permissions"] == {"contents": "read"}
    assert not any("uses" in s for s in job["steps"])  # no checkout, no other action
    assert _step("Check the server")["if"] == "inputs.action == 'check'"
    assert _step("Stop the study container")["if"] == "inputs.action == 'stop'"
    assert _step("List tear-sheet names")["if"] == "inputs.action == 'names'"
    assert "needs the PM's word for each use" in WF.read_text()


# names (PM yes 18 Nov 16:54 UTC): the file names of tear sheets by definition prefix and day, nothing else.
SHEETS = ["qrh5_trend_pullback_stop35_basket-240m_20261117-231502.md",
          "qrh7_trend_pullback_short_only_basket-240m_20261118-000917.md",
          "qrh7_trend_pullback_short_only_basket-240m_20261118-000917_trades.csv",
          "qrh1_trend_pullback_ts6_basket-240m_20261116-182615.md",
          "qrn1_daily_breakout_basket-240m_20261118-101638.md",
          "qrh5_trend_pullback_stop35_basket-240m_20261117-231502.md.bak",
          "notes.txt"]


def _names(tmp_path, prefix, day, extra=()):
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    for n in [*SHEETS, *extra]:
        (sheets / n).write_text("contents never printed")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    log = tmp_path / "docker.txt"
    (bin_ / "sudo").write_text('#!/bin/bash\nexec "$@"\n')
    # docker compose run ... dashboard sh -c SCRIPT: log the arguments, then run SCRIPT as the container would.
    (bin_ / "docker").write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" >> {log}\nprintf "\\036" >> {log}\n'
                                 f'TEARSHEET_DIR={sheets} exec sh -c "${{@: -1}}"\n')
    (bin_ / "ssh").write_text(f'#!/bin/bash\nfor a; do last="$a"; done\nlast="${{last//\\/opt\\/sleeve-fund/{tmp_path}}}"\nPATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    res = subprocess.run(["bash", "-e", "-c", _step("List tear-sheet names")["run"]], capture_output=True, text=True,
                         check=False, env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h",
                                           "HOME": str(tmp_path), "PREFIX": prefix, "DAY": day, "RUN_ID": "4242"})
    # one call per record: NUL-ended arguments then a record separator (the script argument has newlines of its own)
    calls = [rec.split("\0")[:-1] for rec in log.read_text().split("\036") if rec] if log.exists() else []
    return res, calls


def test_names_prints_only_matching_sheet_names_over_two_days(tmp_path):
    res, calls = _names(tmp_path, "qrh", "20261117,20261118")
    assert res.returncode == 0, res.stderr
    assert res.stdout.split() == [SHEETS[0], SHEETS[1]]
    assert "contents never printed" not in res.stdout
    assert len(calls) == 1 and calls[0][:7] == ["compose", "run", "--rm", "-T", "--no-deps", "--name",
                                                  "study-admin-names-4242"]
    assert calls[0][7:9] == ["dashboard", "sh"]


def test_names_one_day(tmp_path):
    res, _ = _names(tmp_path, "qrh", "20261117")
    assert res.returncode == 0 and res.stdout.split() == [SHEETS[0]]


def test_names_prints_at_most_40(tmp_path):
    many = [f"qrx{i:02d}_basket-240m_20261117-0000{i:02d}.md" for i in range(45)]
    res, _ = _names(tmp_path, "qrx", "20261117", extra=many)
    assert res.returncode == 0 and res.stdout.split() == sorted(many)[:40]


def test_names_with_nothing_found_still_succeeds(tmp_path):
    res, _ = _names(tmp_path, "qrz", "20261117")
    assert res.returncode == 0 and res.stdout.strip() == ""


@pytest.mark.parametrize("prefix,day", [("", "20261117"), ("q", "20261117"), ("QRH", "20261117"), ("qrh;ls", "20261117"),
                                        ("qrh*", "20261117"), ("qrh", ""), ("qrh", "2026111"), ("qrh", "20261117;x"),
                                        ("qrh", "20261117,20261118,20261119"), ("qrh", "20261117|.*"),
                                        ("qr h", "20261117"), ("qrh", "20261117\n")])
def test_names_refuses_bad_inputs_before_the_server(tmp_path, prefix, day):
    checked = subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                             check=False, env={"PATH": "/usr/bin:/bin", "ACTION": "names", "RUN_ID": "",
                                               "PREFIX": prefix, "DAY": day})
    assert checked.returncode == 1
    res, calls = _names(tmp_path, prefix, day)  # and again where it is used
    assert res.returncode != 0 and calls == []


def test_names_only_lists():
    text = _step("List tear-sheet names")["run"]
    assert "ls -1" in text
    assert not any(w in text for w in (" cat ", " rm", "prune", "kill", "stop", "tar ", "cp ", "mv ", "> "))
