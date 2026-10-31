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


def _read(tmp_path, sheet, files, links=(), sudo=None):
    """Runs "Read the sheet" with ssh and sudo standing in: sudo drops "docker compose run ... dashboard" and runs
    the container's sh -c line itself, in a tear-sheet folder holding `files` (and `links`, each a symlink to a file
    there). The server's `cd /opt/sleeve-fund` is taken out of the remote line itself rather than stubbed as a shell
    function, so the container's own `cd "$TEARSHEET_DIR"` runs whichever shell sh is (CR F2 on #339). `sudo`
    replaces the stand-in, to feed the runner's checks an archive the container would never make."""
    tears = tmp_path / "tears"
    tears.mkdir()
    for name in files:
        (tears / name).write_text(name, encoding="utf-8")
    for name in links:
        (tears / name).symlink_to(tears / files[0])
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "sudo").write_text(sudo or ('#!/bin/bash\nwhile [ "$1" != dashboard ]; do\n'
                                        '  case "$1" in -e) shift; export "$1";; esac; shift\ndone\nshift\n'
                                        f'export TEARSHEET_DIR={tears}\nexec "$@"\n'))
    (bin_ / "ssh").write_text('#!/bin/bash\nfor a; do last="$a"; done\n'
                              'case "$last" in "cd /opt/sleeve-fund && "*) ;; *) exit 9;; esac\n'
                              f'PATH={bin_}:/usr/bin:/bin bash -c "${{last#cd /opt/sleeve-fund && }}"\n')
    for f in bin_.iterdir():
        f.chmod(0o755)
    step = next(s for s in yaml.safe_load(WF.read_text())["jobs"]["export"]["steps"] if s.get("name") == "Read the sheet")
    run_temp = tmp_path / "runner"
    run_temp.mkdir()
    res = subprocess.run(["bash", "-e", "-c", step["run"]], capture_output=True, text=True, cwd=ROOT, check=False,
                         env={"PATH": f"{bin_}:/usr/bin:/bin", "USER_AT": "u", "HOST": "h", "HOME": str(tmp_path),
                              "RUN_ID": "7", "SHEET": sheet, "RUNNER_TEMP": str(run_temp)})
    if res.returncode:
        return res, None
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


def test_a_linked_file_beside_the_sheet_stays_behind(tmp_path):
    res, names = _read(tmp_path, "b", ["b.md", "b-trades.csv"], links=["b-pooled.csv", "b-basket-trades.csv"])
    assert res.returncode == 0, res.stderr
    assert names == ["b-trades.csv", "b.md"]


def test_the_runner_refuses_an_archive_with_any_other_name(tmp_path):
    """The runner's own check, past the container's: an archive holding a name off the list fails the run."""
    odd = tmp_path / "odd"
    odd.mkdir()
    for name in ("b.md", "b-secrets.env"):
        (odd / name).write_text(name, encoding="utf-8")
    res, _ = _read(tmp_path, "b", ["b.md"], sudo=f'#!/bin/bash\ncat >/dev/null\ntar -czf - -C {odd} b.md b-secrets.env\n')
    assert res.returncode != 0 and "unexpected name in the archive" in res.stdout + res.stderr
    assert "b-secrets.env" not in res.stdout
