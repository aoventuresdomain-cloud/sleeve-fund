"""M1 items 4-6 (PM 30 Oct, next strategy round): a study list names its round, its rows are the registered tries
("try k of N" on each sheet, under the list's SHA-256), each family has a try budget the runner refuses past, and the
round's count prints in one line, all read off the trials register."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.study import idea_of

ROOT = Path(__file__).resolve().parents[1]
BASE = (ROOT / "configs" / "studies" / "qr-rescore-list.md").read_text()
BLOCK = "- **Round:** next-1\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-10\n- **Family B:** rows 11-21\n"


def _list(tmp_path: Path, block: str = BLOCK) -> Path:
    at = BASE.index("## Pinned dates")
    path = tmp_path / "round-list.md"
    path.write_text(BASE[:at] + block + "\n" + BASE[at:])
    return path


class _Register:
    """The trials register's counts, by idea: what rounds.used reads."""

    def __init__(self, by_idea: dict):
        self.by_idea = by_idea

    def counts(self, idea):
        return {"evaluations": self.by_idea.get(idea, 0)}


def test_a_list_names_its_round_and_puts_every_row_in_one_family(tmp_path):
    text = _list(tmp_path).read_text()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert (r.name, r.budget) == ("next-1", 3)
    assert r.families == {"A": tuple(range(1, 11)), "B": tuple(range(11, 22))}
    assert r.sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert r.place(1) == (1, 21) and r.place(21) == (21, 21) and r.family_of(11) == "B"
    assert rounds.round_of(BASE, list(range(1, 22))) is None  # a list naming no round runs as before


@pytest.mark.parametrize("block, words", [
    ("- **Round:** x\n- **Family A:** rows 1-21\n", "sets its budget"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n", "puts its rows in families"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-20\n", "not 21"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-21\n- **Family B:** rows 3\n",
     "in two families: 3"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-22\n", "doesn't have: 22"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 9-1\n", "runs backwards"),
    ("- **Budget per family:** 3 tries\n- **Family A:** rows 1-21\n", "names itself once"),
])
def test_a_half_written_round_is_refused_in_words(tmp_path, block, words):
    text = _list(tmp_path, block).read_text()
    with pytest.raises(ValueError, match=words):
        rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])


def test_a_familys_tries_are_its_rows_ideas_counted_evaluations():
    reg = _Register({"i1": 2, "i2": 5, "other": 9})
    assert rounds.used(reg, {"i1", "i2", None}) == 7


def _ideas_of(text: str) -> dict:
    out = {}
    for n, name, minutes in rescore.rows_of(text):
        out.setdefault(n, idea_of(rescore.request(name, minutes, "binance", "BTC/USDT").params, None))
    return out


def test_each_sheet_says_which_registered_try_it_is_and_a_spent_family_is_refused(tmp_path, capfd, monkeypatch):
    path = _list(tmp_path)
    text = path.read_text()
    ideas = _ideas_of(text)
    a_ideas = {ideas[n] for n in range(1, 11)}
    counted = dict.fromkeys(a_ideas, 0)
    first_a = ideas[1]
    counted[first_a] = 2  # family A has two tries already; family B none
    seen = []

    def run(req, store=None, notes=()):
        seen.append((req.label, notes))
        idea = idea_of(req.params, None)
        counted[idea] = counted.get(idea, 0) + 1  # the study records its try, as run_store_study does
        return tmp_path / f"{req.label}.md"

    monkeypatch.setattr("sleeve_fund.research.trials.TrialsRegister", lambda store: _Register(counted))
    code = rescore.main([str(path), "--rows", "1-3,11"], run=run, journal=lambda: None)
    out = capfd.readouterr().out
    assert code == 1  # a refused row is a failed row
    assert [label for label, _ in seen] == ["qr01_trend_ema_long", rescore.rows_of(text)[10][1]]
    note = seen[0][1][0]
    sha = hashlib.sha256(text.encode()).hexdigest()
    assert note == (f"Round next-1 (study list SHA-256 {sha[:12]}): registered try 1 of 21, family A; the family had "
                    "used 2 of its 3 tries before this one (trials register).")
    assert "row  2 " in out and "REFUSED, family A has used 3 of its 3 tries" in out
    assert "registered try 11 of 21, family B; the family had used 0 of its 3 tries" in seen[1][1][0]
    assert f"round next-1, study list SHA-256 {sha}: 21 registered tries, a budget of 3 tries per family" in out
    assert out.rstrip().splitlines()[-1] == (f"Round next-1 (study list SHA-256 {sha[:12]}), 21 registered tries: "
                                            "family A 3 of 3 tries, family B 1 of 3 tries")


def test_a_dry_run_shows_each_rows_place_and_runs_nothing(tmp_path, capfd):
    path = _list(tmp_path)
    called = []
    assert rescore.main([str(path), "--rows", "12", "--dry-run"], run=lambda *a, **k: called.append(1),
                        journal=lambda: None) == 0
    out = capfd.readouterr().out
    assert not called and "registered try 12 of 21 in family B" in out


def test_a_list_without_a_round_runs_as_before_with_the_hand_count_note(tmp_path):
    seen = []

    def run(req, store=None, notes=()):
        seen.append(notes)
        return tmp_path / "x.md"

    rescore.main([str(ROOT / "configs" / "studies" / "qr-rescore-list.md"), "--rows", "1", "--tries-by-hand"], run=run,
                 journal=lambda: None)
    assert seen == [(rescore.TRIES_BY_HAND,)]


def test_a_basket_sheet_carries_the_lists_notes(monkeypatch, tmp_path):
    from test_study_asset_list import _History, _req

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.store import Store

    out = run_mod.run_store_study(_req(), store=Store(f"sqlite:///{tmp_path}/t.db"), out_dir=tmp_path / "sheets",
                                  history=_History(), notes=("Round next-1: registered try 1 of 2.",))
    assert out.read_text().rstrip().endswith("## Notes\n\n- Round next-1: registered try 1 of 2.")


def test_a_lists_pinned_dates_set_its_walk_forward_and_holdout_lengths():
    from sleeve_fund.research.pinned import parse

    seal = parse("- **Data start:** 2024-01-01\n- **Research end:** 2026-04-30\n"
                 "- **Holdout:** 2026-05-01 to 2026-10-30\n| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |\n")
    assert rescore.fold_days(seal) == (366, 485, 183)  # the next round's (PM 30 Oct, "seal May to Oct 2026")
    req = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT", seal)
    assert (req.train_days, req.test_days, req.holdout_days) == (366, 485, 183)
    assert rescore.fold_days(parse(BASE)) == (rescore.TRAIN_DAYS, rescore.TEST_DAYS, rescore.HOLDOUT_DAYS)
