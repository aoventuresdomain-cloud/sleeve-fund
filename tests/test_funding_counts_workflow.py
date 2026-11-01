"""FUNDING-COUNTS: the one-off read-only workflow that prints the next round's funding table from the paper server's
stored rates (.github/workflows/funding-counts.yml). Done when: the server line is one run of the dashboard's image,
with no other service started, fed this commit's scripts/funding_counts.py on stdin, for BTC, ETH and SOL only, run
here through a stand-in ssh and docker as the runner and the server's shell would split it; a malformed day is refused
before anything reaches the server; and it runs from main with deploys on only."""

from __future__ import annotations

import pathlib
import subprocess

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "funding-counts.yml"


def _job():
    return yaml.safe_load(WF.read_text())["jobs"]["counts"]


def _step(name):
    return next(s for s in _job()["steps"] if s.get("name") == name)


def _count(tmp_path, start="2024-01-01", end="2026-10-30", gate=None, check8=None):
    """Runs "Count the settlements" with ssh standing in: the fake ssh hands its remote line to a shell where sudo
    writes each argument it gets, NUL-ended, to argv.txt, and what it reads on stdin to stdin.txt."""
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sudo").write_text(f'#!/bin/bash\nprintf "%s\\0" "$@" > {tmp_path}/argv.txt\ncat > {tmp_path}/stdin.txt\n'
                               'echo "| BTC | 3 | 3 | 0 | 0 |"\n')
    (bin_ / "ssh").write_text('#!/bin/bash\nfor a; do last="$a"; done\ncd() { :; }\nexport -f cd\n'
                              f'PATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    res = subprocess.run(["bash", "-e", "-c", _step("Count the settlements")["run"]], capture_output=True, text=True,
                         cwd=ROOT, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h", "HOME": str(tmp_path),
                              "RUN_ID": "7", "START": start, "END": end, **({} if gate is None else {"GATE": gate}),
                              **({} if check8 is None else {"CHECK8": check8})})
    argv = (tmp_path / "argv.txt").read_text().split("\0")[:-1] if (tmp_path / "argv.txt").exists() else []
    return res, argv


def test_the_server_line_runs_this_commits_script_in_one_dashboard_container_for_btc_eth_and_sol(tmp_path):
    res, argv = _count(tmp_path)
    assert res.returncode == 0, res.stderr
    assert argv[:8] == ["docker", "compose", "run", "--rm", "-T", "--no-deps", "--name", "funding-counts-7"]
    assert argv[8:] == ["dashboard", "python", "-", "--venue", "BINANCE", "--pairs", "BTC/USDT,ETH/USDT,SOL/USDT",
                        "--start", "2024-01-01", "--end", "2026-10-30"]
    assert (tmp_path / "stdin.txt").read_text() == (ROOT / "scripts" / "funding_counts.py").read_text()
    assert "| BTC | 3 | 3 | 0 | 0 |" in res.stdout


def test_the_dashboard_reads_the_history_read_only():
    dash = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["dashboard"]
    assert "history:/data/history:ro" in dash["volumes"] and dash["environment"]["HISTORY_DIR"] == "/data/history"


@pytest.mark.parametrize("start,end", [("2024-1-1", "2026-10-30"), ("2024-01-01", "2026-10-30; reboot"),
                                       ("2024-01-01 --root /", "2026-10-30"), ("", "2026-10-30")])
def test_a_malformed_day_is_refused_before_the_server(start, end):
    res = subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                         check=False, env={"PATH": "/usr/bin:/bin", "START": start, "END": end})
    assert res.returncode == 1 and "must be days" in res.stdout


def test_it_runs_by_hand_from_main_with_deploys_on_and_reads_only():
    wf = yaml.safe_load(WF.read_text())
    assert set(wf[True]) == {"workflow_dispatch"}  # yaml reads the key `on` as True
    job = _job()
    assert job["if"] == "${{ vars.DEPLOY_ENABLED == 'true' && github.ref == 'refs/heads/main' }}"
    assert job["permissions"] == {"contents": "read"}
    assert "actions/upload-artifact@v4" not in WF.read_text()  # nothing leaves the run but the log's counts


def test_the_gate_tick_adds_only_the_gate_flag_and_anything_else_is_refused(tmp_path):
    """A0's short-gate counts (Advisor, 1 Nov): the tick box adds --gate to the same one container; the step passes
    nothing else on, so the environment can't carry other arguments to the server."""
    assert _step("Count the settlements")["env"]["GATE"] == "${{ inputs.gate && '--gate' || '' }}"
    assert yaml.safe_load(WF.read_text())[True]["workflow_dispatch"]["inputs"]["gate"]["type"] == "boolean"
    assert yaml.safe_load(WF.read_text())[True]["workflow_dispatch"]["inputs"]["gate"]["default"] is False
    head = ["dashboard", "python", "-", "--venue", "BINANCE", "--pairs", "BTC/USDT,ETH/USDT,SOL/USDT", "--start",
            "2024-01-01", "--end", "2026-10-30"]
    for n, (gate, tail) in enumerate((("--gate", ["--gate"]), ("", []))):
        (tmp_path / str(n)).mkdir()
        res, argv = _count(tmp_path / str(n), gate=gate)
        assert res.returncode == 0, res.stderr
        assert argv[8:] == head + tail
    for n, bad in enumerate(("--root /", "--gate --root /", "x")):
        (tmp_path / f"bad{n}").mkdir()
        res, argv = _count(tmp_path / f"bad{n}", gate=bad)
        assert res.returncode == 1 and "gate is a tick box" in res.stdout and argv == []


STEM = "qra0_funding_fade_binance-basket-btc-eth-sol-usdt-store-240m_20261101-165919"
BAD_STEMS = ("x; reboot", "../etc/passwd", f"{STEM} --root /", "h0_x_1", f"{STEM}\nreboot", "$(id)")


def test_check_8_takes_one_sheet_stem_to_the_same_container_and_refuses_anything_else(tmp_path):
    """Check 8 (Advisor, 1 Nov): a basket sheet's stem adds --check8 STEM to the same one container; anything that
    isn't one is refused at the inputs and again before ssh."""
    assert _step("Count the settlements")["env"]["CHECK8"] == "${{ inputs.check8 }}"
    assert _step("Check the inputs")["env"]["CHECK8"] == "${{ inputs.check8 }}"
    assert yaml.safe_load(WF.read_text())[True]["workflow_dispatch"]["inputs"]["check8"]["default"] == ""
    head = ["dashboard", "python", "-", "--venue", "BINANCE", "--pairs", "BTC/USDT,ETH/USDT,SOL/USDT", "--start",
            "2024-01-01", "--end", "2026-10-30"]
    (tmp_path / "a").mkdir()
    res, argv = _count(tmp_path / "a", gate="--gate", check8=STEM)
    assert res.returncode == 0, res.stderr
    assert argv[8:] == head + ["--gate", "--check8", STEM]
    (tmp_path / "b").mkdir()
    res, argv = _count(tmp_path / "b", check8="")
    assert res.returncode == 0 and argv[8:] == head
    for n, bad in enumerate(BAD_STEMS):
        (tmp_path / f"bad{n}").mkdir()
        res, argv = _count(tmp_path / f"bad{n}", check8=bad)
        assert res.returncode == 1 and "check8 must be a basket sheet's stem" in res.stdout and argv == [], bad
        res = subprocess.run(["bash", "-e", "-c", _step("Check the inputs")["run"]], capture_output=True, text=True,
                             check=False, env={"PATH": "/usr/bin:/bin", "START": "2024-01-01", "END": "2026-10-30",
                                               "CHECK8": bad})
        assert res.returncode == 1 and "check8 must be a basket sheet's stem" in res.stdout, bad
