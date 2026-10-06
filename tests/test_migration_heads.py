"""Migration history stays one straight line: exactly one alembic head and no revision id used twice. Two
branches that each add the next-numbered migration both pass on their own and only collide once merged, so
this fails the merged branch in CI before the deploy's upgrade does (DA-12)."""

import ast
import shutil
from collections import Counter
from pathlib import Path

import pytest
from alembic.script import ScriptDirectory

import sleeve_fund

MIGRATIONS = Path(sleeve_fund.__file__).parent / "migrations"


def _revisions(versions: Path) -> list[tuple[str, str]]:
    """(file, revision id) for every migration file, read from the source so a duplicate id is still seen
    (alembic keeps only one of two files sharing an id)."""
    out = []
    for f in sorted(versions.glob("*.py")):
        for node in ast.parse(f.read_text()).body:
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "revision" for t in node.targets):
                out.append((f.name, ast.literal_eval(node.value)))
    return out


def problems(migrations: Path) -> list[str]:
    found = []
    revisions = _revisions(migrations / "versions")
    for rev, n in sorted(Counter(rev for _, rev in revisions).items()):
        if n > 1:
            files = [f for f, r in revisions if r == rev]
            found.append(f"revision id {rev!r} is used by {n} files: {', '.join(files)}")
    heads = ScriptDirectory(str(migrations)).get_heads()
    if len(heads) != 1:
        found.append(f"expected one head, found {len(heads)}: {', '.join(sorted(heads))}; renumber the newer "
                     "migration so it revises the other")
    return found


def test_one_head_and_unique_revision_ids():
    assert problems(MIGRATIONS) == []


def _copy(tmp_path):
    dest = tmp_path / "migrations"
    shutil.copytree(MIGRATIONS, dest, ignore=shutil.ignore_patterns("__pycache__"))
    return dest


def _add(dest, name, revision, down):
    (dest / "versions" / name).write_text(
        f"revision = {revision!r}\ndown_revision = {down!r}\nbranch_labels = None\ndepends_on = None\n\n"
        "def upgrade():\n    pass\n\n\ndef downgrade():\n    pass\n")


def _head():
    return ScriptDirectory(str(MIGRATIONS)).get_current_head()


def test_two_heads_fail(tmp_path):
    dest = _copy(tmp_path)
    _add(dest, "9001_a.py", "9001", _head())
    _add(dest, "9002_b.py", "9002", _head())
    assert any("expected one head, found 2" in p for p in problems(dest))


@pytest.mark.filterwarnings("ignore:Revision 9001 is present more than once")
def test_duplicate_revision_id_fails(tmp_path):
    dest = _copy(tmp_path)
    _add(dest, "9001_a.py", "9001", _head())
    _add(dest, "9001_b.py", "9001", _head())
    assert any("revision id '9001' is used by 2 files" in p for p in problems(dest))


def test_a_straight_line_passes(tmp_path):
    dest = _copy(tmp_path)
    _add(dest, "9001_a.py", "9001", _head())
    _add(dest, "9002_b.py", "9002", "9001")
    assert problems(dest) == []
