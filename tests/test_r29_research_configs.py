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
                       ("sleeve_fund/x.py", "x = 1\n")):
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


def _both(*edits):
    def edit(r: Path):
        for e in edits:
            e(r)
    return edit


CASES = {
    "a new definition": (_write("configs/definitions/qr20_volume_surge.toml"), True),
    "an edited definition": (_write("configs/definitions/qr01_a.toml", "version = 2\n"), True),
    "a study list only": (_write("configs/studies/qr-rescore-list.md", "# rows\n"), True),
    "definition and study list": (_both(_write("configs/definitions/qr21_x.toml"),
                                        _write("configs/studies/list-b.md")), True),
    "mixed with code": (_both(_write("configs/definitions/qr21_x.toml"), _write("sleeve_fund/x.py", "x = 2\n")), False),
    "a deleted definition": (lambda r: (r / "configs/definitions/qr01_a.toml").unlink(), False),
    "a rename into the allowlist": (lambda r: (r / "sleeve_fund/x.py").rename(r / "configs/definitions/x.toml"), False),
    "a rename inside the allowlist": (lambda r: (r / "configs/definitions/qr01_a.toml").rename(
        r / "configs/definitions/qr01_b.toml"), False),
    "a nested directory": (_write("configs/definitions/sub/qr22.toml"), False),
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
