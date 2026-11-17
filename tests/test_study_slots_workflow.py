"""STUDY SLOTS (.github/workflows/study.yml, HoE 17 Nov): study lists run two at a time, one per slot. Each slot is its
own concurrency group, so a dispatch on one slot never drops or waits on the other's; the stacking guard refuses only
while the same slot's container (study-s<slot>-<run id>) is still running, or a container from before slots; and the
run takes optional CPU and memory caps from the repo variables STUDY_CPUS and STUDY_MEMORY, running exactly as before
when both are unset. A trial correction or a family DSR (both still in the old "study" group) refuses while any study
list's container runs, and a study list refuses while either runs. The guard and the run steps are run here as the
runner runs them, through a stand-in ssh."""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "study.yml"
DOC = yaml.safe_load(WF.read_text())
LEGACY = ("cd /opt/sleeve-fund && sudo docker compose run --rm -T --no-deps --name study-s{slot}-42  dashboard python "
          "-u -m sleeve_fund.research.rescore configs/studies/l.md  --venue binance --pair BTC/USDT --dry-run ")


def _step(name: str, wf: dict = DOC, job: str = "study") -> dict:
    return next(s for s in wf["jobs"][job]["steps"] if s.get("name") == name)


def _run(tmp_path, step: str, env: dict, ps: str = "", wf: dict = DOC, job: str = "study"):
    """Runs a step's shell with ssh standing in: it prints the lines of `ps` (the server's docker ps) whose names
    match the remote command's --filter name= prefixes, as docker does, and writes its last argument, the remote
    command, to a file (the first such command, where a step makes two)."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (tmp_path / "ps").write_text(ps)
    ssh = bin_ / "ssh"
    ssh.write_text(f'#!/bin/bash\nfor a; do last="$a"; done\n[ -e {tmp_path}/remote ] || '
                   f'printf "%s" "$last" > {tmp_path}/remote\n'
                   'pre=$(grep -oE "name=[a-z-]+" <<<"$last" | cut -d= -f2 | paste -sd"|")\n'
                   f'[ -z "$pre" ] || grep -E "^($pre)" {tmp_path}/ps || true\n')
    ssh.chmod(0o755)
    res = subprocess.run(["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", _step(step, wf, job)["run"]],
                         capture_output=True, text=True, cwd=ROOT, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h", **env})
    remote = tmp_path / "remote"
    return res, remote.read_text() if remote.exists() else None


def test_each_slot_is_its_own_concurrency_group_and_slot_one_is_the_default():
    slot = DOC[True]["workflow_dispatch"]["inputs"]["slot"]  # yaml reads `on` as True
    assert slot["type"] == "choice" and slot["options"] == ["1", "2"] and slot["default"] == "1"
    assert DOC["concurrency"] == {"group": "study-slot-${{ inputs.slot }}", "cancel-in-progress": False}


@pytest.mark.parametrize("slot, ok", [("1", True), ("2", True), ("3", False), ("", False), ("1;id", False)])
def test_the_inputs_check_takes_slot_one_or_two_only(tmp_path, slot, ok):
    env = {"LIST": "l.md", "ROWS": "", "VENUE": "binance", "PAIR": "BTC/USDT", "SLOT": slot}
    res, _ = _run(tmp_path, "Check the inputs", env)
    assert (res.returncode == 0) is ok, res.stdout


GUARD = "Refuse to stack a second study list on this slot"


@pytest.mark.parametrize("slot, ps, ok", [
    ("1", "", True),
    ("1", "study-s1-100 Up 2 hours\n", False),
    ("2", "study-s1-100 Up 2 hours\n", True),
    ("1", "study-s2-100 Up 2 hours\n", True),
    ("2", "study-s2-100 Up 5 minutes\nstudy-s1-99 Up 2 hours\n", False),
    ("1", "study-100 Up 2 hours\n", False),  # from before slots: holds every slot
    ("2", "study-100 Up 2 hours\n", False),
    ("1", "study-s12-100 Up 1 hour\n", True),  # another slot's name, never a prefix match
])
def test_the_guard_refuses_only_while_the_same_slot_runs(tmp_path, slot, ps, ok):
    res, remote = _run(tmp_path, GUARD, {"SLOT": slot}, ps)
    assert (res.returncode == 0) is ok, res.stdout + res.stderr
    assert "docker ps --filter name=study-" in remote
    if not ok:
        assert f"still running on slot {slot}" in res.stdout
    elif ps:
        assert "running on another slot, left alone" in res.stdout


RUN = {"RUN_ID": "42", "LIST": "l.md", "ROWS": "", "VENUE": "binance", "PAIR": "BTC/USDT", "DRY": "--dry-run",
       "FROM": "image", "SHA": "a" * 40}


@pytest.mark.parametrize("slot", ["1", "2"])
def test_unset_caps_run_exactly_as_before_slots_under_the_slot_name(tmp_path, slot):
    res, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": slot, "CPUS": "", "MEMORY": ""})
    assert res.returncode == 0, res.stderr
    assert remote == LEGACY.format(slot=slot)
    assert "-f " not in remote and "cpus" not in remote


@pytest.mark.parametrize("cpus, memory, caps", [
    ("2", "6g", "cpus: '2', mem_limit: 6g, memswap_limit: 6g"),
    ("1.5", "", "cpus: '1.5'"),
    ("", "4096m", "mem_limit: 4096m, memswap_limit: 4096m"),
])
def test_caps_go_in_a_one_run_override_beside_the_servers_compose_files(tmp_path, cpus, memory, caps):
    res, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": "2", "CPUS": cpus, "MEMORY": memory})
    assert res.returncode == 0, res.stderr
    assert f'"{{services: {{dashboard: {{{caps}}}}}}}"' in remote
    assert "files='-f docker-compose.yml'" in remote and "-f docker-compose.override.yml" in remote
    assert "sudo docker compose $files -f $f run --rm -T --no-deps --name study-s2-42 " in remote
    assert remote.endswith("rc=$?; rm -f $f; exit $rc")


def test_the_override_is_the_caps_on_a_real_shell(tmp_path):
    """The remote line itself, run in a stand-in server folder: the override file holds the caps as YAML, the
    server's own override file is kept when there is one, and the run's exit code comes back after the file goes."""
    _, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": "1", "CPUS": "2", "MEMORY": "6g"})
    server = tmp_path / "srv"
    server.mkdir()
    (server / "docker-compose.override.yml").write_text("{}\n")
    bin_ = tmp_path / "srvbin"
    bin_.mkdir()
    (bin_ / "sudo").write_text(f'#!/bin/bash\necho "$@" > {tmp_path}/argv\n'
                               'while [ "$1" != run ]; do [ "$1" = -f ] && last="$2"; shift; done\n'
                               f'cp "$last" {tmp_path}/caps.yml\nexit 3\n')
    (bin_ / "sudo").chmod(0o755)
    line = remote.replace("cd /opt/sleeve-fund", f"cd {server}")
    res = subprocess.run(["bash", "-c", line], capture_output=True, text=True, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "TMPDIR": str(tmp_path)})
    assert res.returncode == 3
    argv = (tmp_path / "argv").read_text()
    assert argv.startswith("docker compose -f docker-compose.yml -f docker-compose.override.yml -f ")
    caps = yaml.safe_load((tmp_path / "caps.yml").read_text())
    assert caps == {"services": {"dashboard": {"cpus": "2", "mem_limit": "6g", "memswap_limit": "6g"}}}
    assert not [p for p in tmp_path.glob("tmp.*")], "the override file is removed after the run"


@pytest.mark.parametrize("cpus, memory", [("two", ""), ("2;id", ""), ("", "6"), ("", "6g;id"), ("", "6 g")])
def test_a_malformed_cap_fails_before_the_server_is_reached(tmp_path, cpus, memory):
    res, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": "1", "CPUS": cpus, "MEMORY": memory})
    assert res.returncode != 0 and remote is None


@pytest.mark.parametrize("slot, held", [("1", "trial-correct-7 Up 1 minute\n"), ("2", "family-dsr-7 Up 1 minute\n")])
def test_no_study_starts_beside_a_register_job(tmp_path, slot, held):
    res, _ = _run(tmp_path, GUARD, {"SLOT": slot}, held)
    assert res.returncode == 1 and "a register job is running" in res.stdout


REGISTER_JOBS = [("trial-correct.yml", "correct"), ("family-dsr.yml", "dsr")]


@pytest.mark.parametrize("name, job", REGISTER_JOBS)
@pytest.mark.parametrize("ps, ok", [
    ("", True),
    ("study-s1-100 Up 2 hours\n", False),
    ("study-s2-100 Up 2 hours\n", False),
    ("study-100 Up 2 hours\n", False),
    ("studying-1 Up\n", True),
])
def test_a_register_job_refuses_while_any_study_list_runs(tmp_path, name, job, ps, ok):
    wf = yaml.safe_load((WF.parent / name).read_text())
    assert wf["concurrency"]["group"] == "study"
    steps = [s.get("name") for s in wf["jobs"][job]["steps"]]
    guard = "Refuse to run beside a study list"
    assert steps.index("Configure SSH") < steps.index(guard) == steps.index("Configure SSH") + 1
    res, _ = _run(tmp_path, guard, {}, ps, wf, job)
    assert (res.returncode == 0) is ok, res.stdout + res.stderr
