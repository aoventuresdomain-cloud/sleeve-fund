"""STUDY SLOTS (.github/workflows/study.yml, HoE 17 Nov): study lists run two at a time, one per slot. Each slot is its
own concurrency group, so a dispatch on one slot never drops or waits on the other's; the stacking guard refuses only
while the same slot's container (study-s<slot>-<run id>) is still running, or a container from before slots; and the
run takes optional CPU and memory caps from the repo variables STUDY_CPUS and STUDY_MEMORY, running exactly as before
when both are unset. A trial correction or a family DSR (both still in the old "study" group) refuses while any study
list's container runs, and a study list refuses while either runs. CR #358 follow-ups: the same exclusion is held
atomically by a server-side flock (studies shared, register jobs exclusive; F1); a list already running for real on
the other slot is not started again on rows it covers (F2); and the research-config prune spares a commit's copy a
running study still mounts (F3). The guard and the run steps are run here as the runner runs them, through a
stand-in ssh."""

from __future__ import annotations

import os
import pathlib
import signal
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "study.yml"
DOC = yaml.safe_load(WF.read_text())
LOCK = "sudo flock -s -n -E 75 /run/lock/sleeve-register.lock"
LEGACY = (f"cd /opt/sleeve-fund && {LOCK} docker compose run --rm -T --no-deps --name study-s{{slot}}-42 "
          "-l 'sleeve.study=l.md:all:dry'  dashboard python "
          "-u -m sleeve_fund.research.rescore configs/studies/l.md  --venue binance --pair BTC/USDT --dry-run ")


def _step(name: str, wf: dict = DOC, job: str = "study") -> dict:
    return next(s for s in wf["jobs"][job]["steps"] if s.get("name") == name)


def _run(tmp_path, step: str, env: dict, ps: str = "", wf: dict = DOC, job: str = "study", labels: str = "",
         rc: int = 0):
    """Runs a step's shell with ssh standing in: it prints the lines of `ps` (the server's docker ps) whose names
    match the remote command's --filter name= prefixes, as docker does, or `labels` (the running studies'
    sleeve.study labels and names) for a --filter label= query, writes its last argument, the remote command, to a
    file (the first such command, where a step makes several; all of them go to `remotes`, one per NUL), and exits
    `rc`. rsync is a no-op."""
    bin_ = tmp_path / "bin"
    bin_.mkdir(exist_ok=True)
    (tmp_path / "ps").write_text(ps)
    (tmp_path / "labels").write_text(labels)
    ssh = bin_ / "ssh"
    ssh.write_text(f'#!/bin/bash\nfor a; do last="$a"; done\n[ -e {tmp_path}/remote ] || '
                   f'printf "%s" "$last" > {tmp_path}/remote\n'
                   f'printf "%s\\0" "$last" >> {tmp_path}/remotes\n'
                   f'if grep -q "label=sleeve.study" <<<"$last"; then cat {tmp_path}/labels; exit 0; fi\n'
                   'pre=$(grep -oE "name=[a-z-]+" <<<"$last" | cut -d= -f2 | paste -sd"|")\n'
                   f'[ -z "$pre" ] || grep -E "^($pre)" {tmp_path}/ps || true\nexit {rc}\n')
    ssh.chmod(0o755)
    (bin_ / "rsync").write_text("#!/bin/bash\nexit 0\n")
    (bin_ / "rsync").chmod(0o755)
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
    res, remote = _run(tmp_path, GUARD, {"SLOT": slot, "LIST": "l.md", "ROWS": "", "DRY": "true"}, ps)
    assert (res.returncode == 0) is ok, res.stdout + res.stderr
    assert "docker ps --filter name=study-" in remote
    if not ok:
        assert f"still running on slot {slot}" in res.stdout
    elif ps:
        assert "running on another slot, left alone" in res.stdout


RUN = {"RUN_ID": "42", "LIST": "l.md", "ROWS": "", "VENUE": "binance", "PAIR": "BTC/USDT", "DRY": "--dry-run",
       "FROM": "image", "SHA": "a" * 40, "ROWSET": "", "MODE": "dry"}


@pytest.mark.parametrize("slot", ["1", "2"])
def test_unset_caps_run_as_before_slots_under_the_slot_name_the_label_and_the_lock(tmp_path, slot):
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
    assert f"{LOCK} docker compose $files -f $f run --rm -T --no-deps --name study-s2-42 " in remote
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
    assert argv.startswith("flock -s -n -E 75 /run/lock/sleeve-register.lock docker compose -f docker-compose.yml "
                           "-f docker-compose.override.yml -f ")
    caps = yaml.safe_load((tmp_path / "caps.yml").read_text())
    assert caps == {"services": {"dashboard": {"cpus": "2", "mem_limit": "6g", "memswap_limit": "6g"}}}
    assert not [p for p in tmp_path.glob("tmp.*")], "the override file is removed after the run"


@pytest.mark.parametrize("cpus, memory", [("two", ""), ("2;id", ""), ("", "6"), ("", "6g;id"), ("", "6 g")])
def test_a_malformed_cap_fails_before_the_server_is_reached(tmp_path, cpus, memory):
    res, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": "1", "CPUS": cpus, "MEMORY": memory})
    assert res.returncode != 0 and remote is None


@pytest.mark.parametrize("slot, held", [("1", "trial-correct-7 Up 1 minute\n"), ("2", "family-dsr-7 Up 1 minute\n")])
def test_no_study_starts_beside_a_register_job(tmp_path, slot, held):
    res, _ = _run(tmp_path, GUARD, {"SLOT": slot, "LIST": "l.md", "ROWS": "", "DRY": "false"}, held)
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


# F1: the register lock.

@pytest.mark.parametrize("name, job, step", [("trial-correct.yml", "correct", "Correct the row"),
                                             ("family-dsr.yml", "dsr", "Work out the deflated Sharpe")])
def test_a_register_job_takes_the_lock_exclusive_and_says_when_it_is_held(name, job, step):
    run = _step(step, yaml.safe_load((WF.parent / name).read_text()), job)["run"]
    assert "cd /opt/sleeve-fund && sudo flock -x -n -E 75 /run/lock/sleeve-register.lock docker compose run " in run
    assert '[ "$code" != 75 ] || echo "::error::the register lock is held' in run


def test_shared_holders_run_together_and_an_exclusive_one_waits_for_none(tmp_path):
    """flock itself, as the server runs it: two studies share the lock, a register job is refused beside one (75,
    before its command runs), and a study is refused beside a register job."""
    lock = tmp_path / "l.lock"

    def hold(mode):
        # its own process group: the command flock runs holds the lock too, so the whole group is ended
        return subprocess.Popen(["flock", mode, str(lock), "sleep", "5"], start_new_session=True)

    def end(h):
        os.killpg(h.pid, signal.SIGKILL)
        h.wait()

    def try_(mode):
        return subprocess.run(["flock", mode, "-n", "-E", "75", str(lock), "touch", str(tmp_path / "ran")],
                              check=False).returncode

    import time
    h = hold("-s")
    time.sleep(0.3)
    try:
        assert try_("-s") == 0 and (tmp_path / "ran").exists()
        (tmp_path / "ran").unlink()
        assert try_("-x") == 75 and not (tmp_path / "ran").exists()
    finally:
        end(h)
    h = hold("-x")
    time.sleep(0.3)
    try:
        assert try_("-s") == 75 and not (tmp_path / "ran").exists()
    finally:
        end(h)
    assert try_("-x") == 0


def test_a_study_held_off_by_the_lock_says_so_and_fails(tmp_path):
    res, _ = _run(tmp_path, "Run the list", {**RUN, "SLOT": "1", "CPUS": "", "MEMORY": ""}, rc=75)
    assert res.returncode == 75 and "the register lock is held" in res.stdout


# F2: one list is not run twice over the same rows.

@pytest.mark.parametrize("rows, dry, labels, ok", [
    ("", "false", "", True),
    ("", "false", "l.md:all:run study-s1-9\n", False),
    ("1-4", "false", "l.md:5-8:run study-s1-9\n", True),
    ("1-4", "false", "l.md:4,9:run study-s1-9\n", False),
    ("1,3", "false", "l.md:2-2,4-9:run study-s1-9\n", True),
    ("7", "false", "l.md:all:run study-s1-9\n", False),
    ("", "false", "l.md:3:run study-s1-9\n", False),
    ("", "false", "l.md:all:dry study-s1-9\n", True),  # a dry run writes nothing, so holds nothing
    ("", "true", "l.md:all:run study-s1-9\n", True),  # nor is one refused
    ("", "false", "other.md:all:run study-s1-9\n", True),
    ("", "false", "l.md.bak:all:run study-s1-9\nother.md:all:run study-s1-8\n", True),
    ("2", "false", "other.md:all:run study-s1-8\nl.md:1-3:run study-s1-9\n", False),
])
def test_a_list_running_for_real_on_the_other_slot_is_not_run_again_on_its_rows(tmp_path, rows, dry, labels, ok):
    res, _ = _run(tmp_path, GUARD, {"SLOT": "2", "LIST": "l.md", "ROWS": rows, "DRY": dry}, labels=labels)
    assert (res.returncode == 0) is ok, res.stdout + res.stderr
    if not ok:
        assert "l.md is already running on rows" in res.stdout and "study-s1-9" in res.stdout


@pytest.mark.parametrize("rowset, mode, label", [("", "run", "l.md:all:run"), ("1-4,7", "dry", "l.md:1-4,7:dry")])
def test_each_run_carries_its_list_rows_and_mode_on_a_label(tmp_path, rowset, mode, label):
    res, remote = _run(tmp_path, "Run the list", {**RUN, "SLOT": "1", "CPUS": "", "MEMORY": "", "ROWSET": rowset,
                                                  "MODE": mode})
    assert res.returncode == 0 and f"--name study-s1-42 -l 'sleeve.study={label}' " in remote
    assert DOC["jobs"]["study"]["steps"][[s.get("name") for s in DOC["jobs"]["study"]["steps"]].index(
        "Run the list")]["env"]["MODE"] == "${{ inputs.dry_run && 'dry' || 'run' }}"


# F3: the prune spares what a running study mounts.

def test_the_prune_keeps_the_newest_five_and_any_copy_a_running_study_mounts(tmp_path):
    shas = [f"{i:x}" * 40 for i in range(1, 9)]  # oldest first
    res, _ = _run(tmp_path, "Ship the research configs", {"SHA": shas[-1]})
    assert res.returncode == 0, res.stderr
    prune = (tmp_path / "remotes").read_text().split("\0")[1]
    srv = tmp_path / "research"
    srv.mkdir()
    for i, sha in enumerate(shas):
        (srv / sha).mkdir()
        os.utime(srv / sha, (1_000_000 + i, 1_000_000 + i))
    (srv / "notes").mkdir()
    bin_ = tmp_path / "srvbin"
    bin_.mkdir()
    (bin_ / "sudo").write_text('#!/bin/bash\n[ "$1" = touch ] && exit 0\nexec "$@"\n')
    # docker: one running study (id c1) mounting the second-oldest copy
    (bin_ / "docker").write_text(f'#!/bin/bash\nif [ "$1" = ps ]; then echo c1; exit 0; fi\n'
                                 f'[ "$1 ${{@: -1}}" = "inspect c1" ] && echo "/opt/sleeve-research/{shas[1]}/definitions '
                                 f'/opt/sleeve-research/{shas[1]}/studies /srv/data "\n')
    for b in ("sudo", "docker"):
        (bin_ / b).chmod(0o755)
    line = prune.replace("cd /opt/sleeve-research", f"cd {srv}")
    out = subprocess.run(["bash", "-c", line], capture_output=True, text=True, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert sorted(p.name for p in srv.iterdir()) == sorted(shas[3:] + [shas[1], "notes"])
    # docker unreachable: nothing pruned
    (bin_ / "docker").write_text("#!/bin/bash\nexit 1\n")
    out = subprocess.run(["bash", "-c", line], capture_output=True, text=True, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin"})
    assert out.returncode != 0 and sorted(p.name for p in srv.iterdir()) == sorted(shas[3:] + [shas[1], "notes"])
