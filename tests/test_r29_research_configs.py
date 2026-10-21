"""R-29 (HoE, PM 21 Oct 2026): research configs reach a study without a deploy. study.yml copies only
configs/definitions/*.toml and configs/studies/*.md from main at the dispatch commit to the server, outside the
deployed tree, and mounts them read-only into the one-off study container only; deploy.yml skips the restart, and
keeps the server's deployed-commit record, when main moved in those paths only (deploy/research_config_only.sh)."""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RULE = ROOT / "deploy" / "research_config_only.sh"
STUDY = ROOT / ".github" / "workflows" / "study.yml"
DEPLOY = ROOT / ".github" / "workflows" / "deploy.yml"


def _git(repo: Path, *args: str) -> str:
    env = {**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@t"}
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True,
                          env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    for path, body in (("configs/definitions/qr01_a.toml", "version = 1\n"), ("configs/studies/list-a.md", "# a\n"),
                       ("sleeve_fund/x.py", "x = 1\n"),
                       ("configs/definition_pins.toml", '[qr01_a]\ndefinition_hash = "aa"\nidea_hash = "bb"\n')):
        (r / path).parent.mkdir(parents=True, exist_ok=True)
        (r / path).write_text(body)
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "base")
    return r


def _change(repo: Path, edit) -> tuple[str, str]:
    base = _git(repo, "rev-parse", "HEAD")
    edit(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "change", "--allow-empty")
    return base, _git(repo, "rev-parse", "HEAD")


def _config_only(repo: Path, base: str, head: str) -> bool:
    return subprocess.run(["bash", str(RULE), base, head], cwd=repo, capture_output=True).returncode == 0


def _write(path: str, body: str = "new\n"):
    def edit(r: Path):
        (r / path).parent.mkdir(parents=True, exist_ok=True)
        (r / path).write_text(body)
    return edit


def _append(path: str, body: str):
    def edit(r: Path):
        with open(r / path, "a") as f:
            f.write(body)
    return edit


PIN = "configs/definition_pins.toml"


def _both(*edits):
    def edit(r: Path):
        for e in edits:
            e(r)
    return edit


CASES = {
    "a new definition": (_write("configs/definitions/qr20_volume_surge.toml"), True),
    # HoE R-29 v3 (CR M1, HoQA): an edited definition is a new trials key: a code change, a deploy and a TEC.
    "an edited definition": (_write("configs/definitions/qr01_a.toml", "version = 2\n"), False),
    "an edited study list": (_write("configs/studies/list-a.md", "# a, rows\n"), True),
    "a study list only": (_write("configs/studies/qr-rescore-list.md", "# rows\n"), True),
    "definition and study list": (_both(_write("configs/definitions/qr21_x.toml"),
                                        _write("configs/studies/list-b.md")), True),
    "mixed with code": (_both(_write("configs/definitions/qr21_x.toml"), _write("sleeve_fund/x.py", "x = 2\n")), False),
    "a deleted definition": (lambda r: (r / "configs/definitions/qr01_a.toml").unlink(), False),
    "a rename into the allowlist": (lambda r: (r / "sleeve_fund/x.py").rename(r / "configs/definitions/x.toml"), False),
    "a rename inside the allowlist": (lambda r: (r / "configs/definitions/qr01_a.toml").rename(
        r / "configs/definitions/qr01_b.toml"), False),
    "a variant definition": (_write("configs/definitions/variants/qr01_trend_ema_long_v2_funding.toml"), True),
    "a variant and a study list": (_both(_write("configs/definitions/variants/qr01b_v1.toml"),
                                         _write("configs/studies/qr-rescore-list.md")), True),
    "a new definition and its pin": (_both(_write("configs/definitions/qr24_x_long.toml"),
                                           _append(PIN, '\n[qr24_x_long]\ndefinition_hash = "cc"\n')), True),
    "a pin added alone": (_append(PIN, '\n[qr25_y]\ndefinition_hash = "dd"\n'), True),
    "a pin changed": (_write(PIN, '[qr01_a]\ndefinition_hash = "a2"\nidea_hash = "bb"\n'), False),
    "a pin removed": (_write(PIN, '[qr01_a]\ndefinition_hash = "aa"\n'), False),
    "an edited definition with its changed pin": (_both(_write("configs/definitions/qr01_a.toml", "version = 2\n"),
                                                        _write(PIN, '[qr01_a]\ndefinition_hash = "a2"\n'
                                                                    'idea_hash = "b2"\n')), False),
    "a nested directory": (_write("configs/definitions/sub/qr22.toml"), False),
    "a directory under the variants": (_write("configs/definitions/variants/sub/qr22.toml"), False),
    "an upper-case variant": (_write("configs/definitions/variants/QR22.toml"), False),
    "an upper-case name": (_write("configs/definitions/QR22.toml"), False),
    "another suffix": (_write("configs/definitions/qr22.py"), False),
    "a study list's other suffix": (_write("configs/studies/list.toml"), False),
    "a dot-dot name": (_write("configs/definitions/..qr22.toml"), False),
    "another configs file": (_write("configs/risk.toml"), False),
    "an executable definition": (lambda r: (_write("configs/definitions/qr23.toml")(r),
                                            os.chmod(r / "configs/definitions/qr23.toml", 0o755)), False),
    "a symlinked definition": (lambda r: os.symlink("../../sleeve_fund/x.py", r / "configs/definitions/qr24.toml"),
                               False),
    "no change at all": (lambda r: None, False),
}


@pytest.mark.parametrize("name", list(CASES))
def test_the_path_rule(repo, name):
    edit, want = CASES[name]
    base, head = _change(repo, edit)
    assert _config_only(repo, base, head) is want, name


def test_an_edited_variant_is_not_config_only(repo):
    _change(repo, _write("configs/definitions/variants/qr01b_v1.toml"))
    base, head = _change(repo, _write("configs/definitions/variants/qr01b_v1.toml", "v2\n"))
    assert not _config_only(repo, base, head)


def test_the_path_rule_refuses_what_is_not_a_commit(repo):
    head = _git(repo, "rev-parse", "HEAD")
    for base in ("", "HEAD~1", "../" + head[:37], "0" * 40):
        assert not _config_only(repo, base, head), base


def _steps(text: str) -> str:
    return text.split("\n    steps:", 1)[1]


def test_only_the_study_container_mounts_the_research_configs_read_only():
    study = STUDY.read_text()
    mounts = re.findall(r"-v (/opt/sleeve-research/\S+):(/app/configs/\w+):(\w+)", study)
    assert sorted(mounts) == [("/opt/sleeve-research/$SHA/definitions", "/app/configs/definitions", "ro"),
                              ("/opt/sleeve-research/$SHA/studies", "/app/configs/studies", "ro")]
    run = study.split("- name: Run the list", 1)[1]
    assert "sudo docker compose run --rm -T --no-deps --name study-$RUN_ID $MOUNTS dashboard" in run
    for wf in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
        if wf != STUDY:
            assert "/opt/sleeve-research" not in wf.read_text(), f"{wf.name} must not see the research configs"
    for f in ("docker-compose.yml", "compose.yml", "docker-compose.yaml"):
        if (ROOT / f).exists():
            assert "sleeve-research" not in (ROOT / f).read_text()


def test_the_study_job_stages_main_at_its_own_commit_by_the_allowlist():
    study = STUDY.read_text()
    steps = _steps(study)
    assert "uses: actions/checkout@v4" in steps and "ref: ${{ github.sha }}" in steps
    assert "configs_from:" in study and "options: [main, image]" in study and "default: main" in study
    stage = study.split("- name: Stage the research configs", 1)[1].split("- name:", 1)[0]
    assert "git ls-tree -r" in stage and "100644 blob" in stage
    assert r"^configs/definitions/[a-z0-9_]+\.toml$" in stage and r"^configs/studies/[a-z0-9_-]+\.md$" in stage
    assert r"^configs/definitions/variants/[a-z0-9_]+\.toml$" in stage
    assert "symlink" in stage
    ship = study.split("- name: Ship the research configs", 1)[1].split("- name:", 1)[0]
    assert "--no-links" in ship and "/opt/sleeve-research/$SHA/" in ship and "tail -n +6" in ship
    assert "research configs from main at $SHA" in study


def test_the_deploy_skips_a_research_config_only_change_and_keeps_its_record():
    deploy = DEPLOY.read_text()
    pick = deploy.split("- name: Pick the commit to ship", 1)[1].split("- name:", 1)[0]
    assert 'bash deploy/research_config_only.sh "$deployed" "$pick"' in pick
    assert "research-config-only change, no restart" in pick
    skip = pick.split("research_config_only.sh", 1)[1]
    assert 'echo "current=false"' in skip.split("else", 1)[0], "nothing ships, so .deployed_sha stays"


def test_the_research_page_shows_a_study_whose_definition_is_not_in_the_image(tmp_path, monkeypatch):
    """A study of a definition merged after the deploy writes its sheet as any other; the Research page lists and
    renders it from the sheet alone, never looking the definition up in the image's configs."""
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.research import rescore
    from sleeve_fund.store import Store

    stem = "qr99_not_in_image_long_binance-btcusdt-store_20261021-130000"
    assert not (rescore.DEFINITIONS / "qr99_not_in_image_long.toml").exists()
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    (tmp_path / f"{stem}.md").write_text("# Tear sheet: qr99_not_in_image_long\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    c = TestClient(app_mod.create_app(Store(f"sqlite:///{tmp_path}/t.db")))
    auth = ("pm", "test-pw")
    assert stem in c.get("/research", auth=auth).text
    page = c.get(f"/research/{stem}", auth=auth)
    assert page.status_code == 200 and "qr99_not_in_image_long" in page.text


def _stage_script() -> str:
    """The shell of study.yml's "Stage the research configs" step, as the runner runs it."""
    stage = STUDY.read_text().split("- name: Stage the research configs", 1)[1].split("- name:", 1)[0]
    lines = stage.split("run: |", 1)[1].splitlines()[1:]
    indent = min(len(line) - len(line.lstrip()) for line in lines if line.strip())
    return "\n".join(line[indent:] for line in lines) + "\n"


def test_the_study_job_stages_the_r06_r07_variants_beside_the_definitions(repo):
    """CR M1 on R-06/R-07 (HoE): the mount replaces the image's configs/definitions, so the variants a study list
    names (configs/definitions/variants/*.toml) are staged under it too, or a configs_from=main study can't find them.
    Anything deeper, or not named as a definition, fails the run (the next cell)."""
    _write("configs/definitions/variants/qr01b_v1_funding.toml")(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "variants")
    out = subprocess.run(["bash", "-c", _stage_script()], cwd=repo, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    staged = sorted(str(p.relative_to(repo / "staged")) for p in (repo / "staged").rglob("*") if p.is_file())
    assert staged == ["definitions/qr01_a.toml", "definitions/variants/qr01b_v1_funding.toml", "studies/list-a.md"]


@pytest.mark.parametrize("stray", ["configs/definitions/variants/sub/x.toml", "configs/definitions/variants/QR2.toml",
                                   "configs/definitions/notes.txt", "configs/studies/list.toml"])
def test_the_study_job_refuses_anything_else_in_the_research_configs(repo, stray):
    """Advisor R-29 (4): a file under configs/definitions or configs/studies not named as a research config fails
    the run, named, rather than being left out."""
    _write(stray)(repo)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "stray")
    out = subprocess.run(["bash", "-c", _stage_script()], cwd=repo, capture_output=True, text=True)
    assert out.returncode != 0 and f"not a research config by name: {stray}" in out.stdout


def test_the_study_job_reads_the_dispatched_commit_and_only_one_on_main():
    """Advisor R-29 (1): the configs are taken from the commit the run was dispatched on, by SHA, never by branch name,
    so main moving after dispatch changes nothing, and that commit must be on main."""
    study = STUDY.read_text()
    steps = _steps(study)
    checkout = steps.split("uses: actions/checkout@v4", 1)[1].split("- name:", 1)[0]
    assert "ref: ${{ github.sha }}" in checkout and "fetch-depth: 0" in checkout
    assert "persist-credentials: false" in checkout, "CR F1"
    on_main = study.split("- name: Check the commit is on main", 1)[1].split("- name:", 1)[0]
    assert 'git merge-base --is-ancestor "$SHA" origin/main' in on_main and "SHA: ${{ github.sha }}" in on_main
    assert "inputs.configs_from == 'main'" in on_main
    assert study.index("- name: Check the commit is on main") < study.index("- name: Stage the research configs")
    for step in ("Ship the research configs", "Run the list"):
        body = study.split(f"- name: {step}", 1)[1].split("- name:", 1)[0]
        assert "SHA: ${{ github.sha }}" in body and "github.ref_name" not in body and "refs/heads" not in body


def test_a_study_from_main_says_its_tries_are_counted_by_hand(tmp_path, monkeypatch):
    """HoE (R-29 v2): every study run with configs_from=main prints "Tries counted by hand until R-25." on its sheet,
    by a rescore flag study.yml passes on main only; image runs and the dashboard's studies are unchanged."""
    from test_binance import INFO
    from test_research import _stored_minutes

    from sleeve_fund import funding
    from sleeve_fund.history import HistoryStore
    from sleeve_fund.research import rescore
    from sleeve_fund.research.run import StudyRequest, run_store_study
    from sleeve_fund.venues import binance_contract, venue

    assert rescore.TRIES_BY_HAND == "Tries counted by hand until R-25."
    step = STUDY.read_text().split("- name: Run the list", 1)[1]
    main_branch, image_branch = step.split('if [ "$FROM" = main ]; then', 1)[1].split("else", 1)
    assert 'TRIES="--tries-by-hand"' in main_branch and "TRIES" not in image_branch.split("fi", 1)[0]
    assert "$DRY $TRIES" in step
    lst = ROOT / "configs" / "studies" / "qr-rescore-list.md"
    seen = []
    spy = lambda req, store=None, **kw: seen.append(kw)  # noqa: E731
    assert rescore.main([str(lst), "--rows", "1", "--tries-by-hand"], run=spy, journal=lambda: None) == 0
    assert rescore.main([str(lst), "--rows", "1"], run=spy, journal=lambda: None) == 0
    assert seen == [{"notes": (rescore.TRIES_BY_HAND,)}, {}]
    # The study itself stubbed (its sheet's caveats list r.notes: tearsheet.render): the run adds the notes it is given.
    import types

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.research import study, tearsheet

    monkeypatch.setattr(study, "run_study", lambda *a, **k: types.SimpleNamespace(
        notes=[], fee_basis=None, breakeven=None))
    monkeypatch.setattr(tearsheet, "render", lambda r, *a, **k: "\n".join(f"- {n}" for n in r.notes))
    monkeypatch.setattr(run_mod, "write_results", lambda r, out: None)
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    hist = HistoryStore(tmp_path / "hist")
    hist.append("BINANCE", "BTC/USDT", _stored_minutes(3), cursor="x")
    req = StudyRequest(strategy="buy_and_hold", pair="BTC/USDT", venue="binance", minutes=240,
                       train_days=60, test_days=30, holdout_days=30)
    marked = run_store_study(req, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "a", history=hist,
                             notes=(rescore.TRIES_BY_HAND,)).read_text()
    plain = run_store_study(req, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "b", history=hist).read_text()
    assert f"- {rescore.TRIES_BY_HAND}" in marked and rescore.TRIES_BY_HAND not in plain


def test_the_definitions_merge_rule_is_written_down():
    """HoE (R-29 v2): the manual merge gate for a definitions PR, until R-25 counts tries by family."""
    doc = (ROOT / "docs" / "research-definitions.md").read_text()
    for must in ("pre-registration", "family", "Advisor", "tries so far", "R-25", "Strategy tab",
                 "configs_from=image"):
        assert must in doc, must


def test_the_definition_pins_live_in_one_config_file_both_hash_checks_read():
    """HoE (R-29 v2): a rule PR touches only its definition, configs/definition_pins.toml and a study list, so its
    pins are data under configs, not tests/ code; both pin checks read that one file, exactly as strictly."""
    import tomllib

    import test_b16_definition_titles as b16
    import test_research_gaps as gaps

    pins = tomllib.loads((ROOT / "configs" / "definition_pins.toml").read_text())
    assert sorted(pins) == sorted(p.stem for p in (ROOT / "configs" / "definitions").glob("qr*.toml"))
    assert all(set(v) == {"definition_hash", "idea_hash"} for v in pins.values())
    assert b16.BEFORE == {k: (v["definition_hash"], v["idea_hash"]) for k, v in pins.items()}
    # HoE R-29 v3 (CR M1): today's 21 pins stay frozen in test code, so changing one is a test edit and a TEC.
    assert len(b16.FROZEN) == 21 and b16.FROZEN.items() <= b16.BEFORE.items()
    assert gaps.QR_HASHES == {k: v["definition_hash"] for k, v in pins.items()}


def test_a_study_from_main_needs_its_commits_tests_passed():
    """HoE R-29 v3 (HoQA F2): configs_from=main reads only a commit whose push run of tests.yml on main passed, by the
    same gh run list gate as deploy.yml."""
    study = STUDY.read_text()
    assert "permissions: { contents: read, actions: read }" in study
    gate = study.split("- name: Require green tests for this commit", 1)[1].split("- name:", 1)[0]
    assert "inputs.configs_from == 'main'" in gate and "GH_TOKEN: ${{ github.token }}" in gate
    assert ('gh run list --workflow tests.yml --commit "$GITHUB_SHA" --branch main --event push --status success'
            in gate) and "exit 1" in gate
    assert study.index("- name: Require green tests for this commit") < study.index("- name: Stage the research configs")
