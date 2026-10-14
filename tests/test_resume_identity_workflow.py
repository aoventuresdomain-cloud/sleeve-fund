"""RESUME-WF (HoE, 14 Oct): the dispatch workflow that lifts the book check's identity hold on the PM's written word
(.github/workflows/resume-identity.yml, PE2's `supervisor resume-identity`). Done when (4): a bad reason or actor is
rejected before ssh, by the workflow's own input check, run here as the runner runs it. (1)-(3), the dry run, the lift
and the refusal while the gap is open, are the subcommand's: PE2's cells, and the runbook's step 6."""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml

WF = pathlib.Path(__file__).resolve().parents[1] / ".github" / "workflows" / "resume-identity.yml"


def _wf():
    return yaml.safe_load(WF.read_text())


def _step(name):
    return next(s for s in _wf()["jobs"]["resume"]["steps"] if s["name"] == name)


def _check(actor, reason):
    return subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                          env={"ACTOR": actor, "REASON": reason, "PATH": "/usr/bin:/bin"}).returncode


@pytest.mark.parametrize("reason", ["PM: book adds up again (checked 06:50 UK)", "Resume, as agreed!", "ok?"])
def test_a_plain_reason_passes_the_check(reason):
    assert _check("hoe", reason) == 0


@pytest.mark.parametrize("reason", ["it's fine", 'say "go"', "$(reboot)", "`id`", "a\\b", "x; rm -rf /'", "ab",
                                    "x" * 201, "line\nbreak", "a|b", "a&b", "a>b"])
def test_a_reason_with_anything_that_could_reach_the_remote_line_is_rejected_before_ssh(reason):
    assert _check("hoe", reason) == 1


@pytest.mark.parametrize("actor", ["", "o'brien", "$USER", "a" * 41])
def test_a_bad_actor_is_rejected_before_ssh(actor):
    assert _check(actor, "PM said resume") == 1


def test_the_check_runs_first_and_the_dry_run_is_the_default():
    wf = _wf()
    steps = [s["name"] for s in wf["jobs"]["resume"]["steps"]]
    assert steps == ["Check the inputs", "Configure SSH", "Lift the identity hold"]
    inputs = wf[True]["workflow_dispatch"]["inputs"]  # YAML 1.1 reads "on" as true
    assert inputs["dry_run"]["default"] is True and inputs["reason"]["required"] and inputs["actor"]["required"]
    line = _step("Lift the identity hold")["run"]
    assert "supervisor resume-identity --actor '$ACTOR' --reason '$REASON' $DRY" in line
    assert "github.ref == 'refs/heads/main'" in wf["jobs"]["resume"]["if"]
