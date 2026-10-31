"""SHEET-EXPORT (.github/workflows/sheet-export.yml): the encrypted read-only route for a finished tear sheet. Done when
(M1 item 9, the accuracy checklist proof): every file the sheet writer puts beside a sheet (run.BESIDE, and the
results, returns and test-window returns) comes out, so scripts/recompute_sheet.py can rebuild a basket's sheet from
the export alone; and nothing else does. Run here through a stand-in ssh and docker: the one-off container's shell
runs against a temporary tear-sheet folder."""

from __future__ import annotations

import pathlib
import subprocess
import tarfile

import yaml

from sleeve_fund.research.run import BESIDE

ROOT = pathlib.Path(__file__).resolve().parents[1]
WF = ROOT / ".github" / "workflows" / "sheet-export.yml"
OWN = (".md", ".results.json", ".returns.csv", ".oos-returns.csv")


def _read(tmp_path, sheet, files):
    """Runs "Read the sheet" with ssh and sudo standing in: sudo drops "docker compose run ... dashboard" and runs
    the container's sh -c line itself, in a tear-sheet folder holding `files`."""
    tears = tmp_path / "tears"
    tears.mkdir()
    for name in files:
        (tears / name).write_text(name, encoding="utf-8")
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sudo").write_text('#!/bin/bash\nwhile [ "$1" != dashboard ]; do\n'
                               '  case "$1" in -e) shift; export "$1";; esac; shift\ndone\nshift\n'
                               f'export TEARSHEET_DIR={tears}\nexec "$@"\n')
    (bin_ / "ssh").write_text('#!/bin/bash\nfor a; do last="$a"; done\ncd() { :; }\nexport -f cd\n'
                              f'PATH={bin_}:/usr/bin:/bin bash -c "$last"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    step = next(s for s in yaml.safe_load(WF.read_text())["jobs"]["export"]["steps"] if s.get("name") == "Read the sheet")
    run_temp = tmp_path / "runner"
    run_temp.mkdir()
    res = subprocess.run(["bash", "-e", "-c", step["run"]], capture_output=True, text=True, cwd=ROOT, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h", "HOME": str(tmp_path),
                              "RUN_ID": "7", "SHEET": sheet, "RUNNER_TEMP": str(run_temp)})
    with tarfile.open(run_temp / "plain" / "sheet.tar.gz") as tar:
        return res, sorted(tar.getnames())


def test_a_basket_sheet_comes_out_with_every_file_the_recompute_reads(tmp_path):
    stem = "trend-basket"
    wanted = sorted([stem + s for s in OWN] + [f"{stem}-{kind}" for kind in BESIDE])
    res, names = _read(tmp_path, stem, wanted + ["other-sheet.md", f"{stem}-notes.txt"])
    assert res.returncode == 0, res.stderr
    assert names == wanted
    for kind in ("basket-trades.csv", "pooled.csv", "oos-trades.csv"):
        assert f"{stem}-{kind} " in res.stdout


def test_a_sheet_with_only_some_files_brings_those(tmp_path):
    res, names = _read(tmp_path, "single", ["single.md", "single-trades.csv"])
    assert res.returncode == 0, res.stderr
    assert names == ["single-trades.csv", "single.md"]
