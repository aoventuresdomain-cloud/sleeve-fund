"""DEPLOY-NEWEST: deploy/pick_commit.sh, the commit deploy.yml ships. Run against a throwaway git history and a stand-in
`gh` that lists the main push runs whose tests passed: only a green commit ships, never one older than (or not
descended from) the server's, and a run whose commit main has moved past ships main's newest green commit."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "deploy" / "pick_commit.sh"

pytestmark = pytest.mark.skipif(not shutil.which("git") or not shutil.which("bash"), reason="needs git and bash")


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """origin with main = c1..c5 (first parent), a clone of it, and a `gh` that prints the shas in green.txt."""
    origin = tmp_path / "origin"
    origin.mkdir()
    _git(origin, "init", "-q", "-b", "main")
    _git(origin, "config", "user.email", "t@t")
    _git(origin, "config", "user.name", "t")
    shas = []
    for i in range(1, 6):
        (origin / "f").write_text(str(i))
        _git(origin, "add", "f")
        _git(origin, "commit", "-q", "-m", f"c{i}")
        shas.append(_git(origin, "rev-parse", "HEAD"))
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(origin), str(clone))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    green = tmp_path / "green.txt"
    (bin_dir / "gh").write_text(f'#!/bin/sh\ncat "{green}"\n')
    (bin_dir / "gh").chmod(0o755)
    return {"origin": origin, "clone": clone, "shas": shas, "green": green, "path": f"{bin_dir}:{os.environ['PATH']}"}


def pick(repo, ship: str, deployed: str = "", green=()) -> tuple[str, str, int]:
    repo["green"].write_text("".join(f"{s}\n" for s in green))
    r = subprocess.run(["bash", str(SCRIPT), ship, deployed], cwd=repo["clone"], capture_output=True, text=True, check=False,
                       env={**os.environ, "PATH": repo["path"]})
    return r.stdout.strip(), r.stderr, r.returncode


def test_main_still_at_the_tested_commit_ships_it(repo):
    c = repo["shas"]
    assert pick(repo, c[4], deployed=c[2])[0] == c[4]


def test_main_past_the_tested_commit_ships_mains_newest_green_commit_never_an_in_flight_one(repo):
    c = repo["shas"]
    out, err, code = pick(repo, c[1], deployed=c[0], green=[c[1], c[3]])  # c5 in flight, c4 green
    assert (out, code) == (c[3], 0) and "shipping" in err


def test_an_older_green_commit_ships_and_a_newer_red_or_running_one_never_does(repo):
    c = repo["shas"]
    # c5 red, c4 still running, c3 green: gh lists only the runs whose tests passed, so c3 ships.
    out, err, code = pick(repo, c[0], deployed=c[0], green=[c[0], c[2]])
    assert (out, code) == (c[2], 0) and "shipping" in err


def test_no_green_commit_ships_nothing(repo):
    c = repo["shas"]
    out, err, code = pick(repo, c[1], deployed=c[0], green=[])
    assert (out, code) == ("", 0) and "Nothing shipped" in err


def test_the_commit_the_server_runs_is_not_shipped_again(repo):
    c = repo["shas"]
    out, err, code = pick(repo, c[2], deployed=c[3], green=[c[3]])  # a newer run already shipped c4
    assert (out, code) == ("", 0) and "already runs" in err


def test_never_a_commit_older_than_the_servers(repo):
    c = repo["shas"]
    out, err, code = pick(repo, c[4], deployed=c[4])
    assert (out, code) == ("", 0)
    out, err, code = pick(repo, c[1], deployed=c[3], green=[c[1]])  # only the older commit is green
    assert (out, code) == ("", 0) and "no rollback" in err


def test_never_a_commit_not_descended_from_the_servers(repo):
    c = repo["shas"]
    _git(repo["origin"], "checkout", "-q", "-b", "side", c[0])
    (repo["origin"] / "g").write_text("x")
    _git(repo["origin"], "add", "g")
    _git(repo["origin"], "commit", "-q", "-m", "side")
    side = _git(repo["origin"], "rev-parse", "HEAD")
    _git(repo["clone"], "fetch", "-q", "origin", "side")
    out, err, code = pick(repo, c[4], deployed=side)
    assert (out, code) == ("", 0) and "does not descend" in err


def test_no_record_on_the_server_ships_with_a_notice(repo):
    c = repo["shas"]
    out, err, code = pick(repo, c[4])
    assert (out, code) == (c[4], 0) and "no record" in err


@pytest.mark.parametrize("ship, deployed", [("main", ""), ("x" * 40, ""), ("0" * 40, "not-a-sha")])
def test_anything_but_a_full_sha_is_refused(repo, ship, deployed):
    out, err, code = pick(repo, ship, deployed)
    assert code == 1 and out == "" and "::error::" in err
