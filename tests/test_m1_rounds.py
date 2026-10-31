"""M1 items 4-6 (PM 30 Oct, next strategy round): a study list names its round, its rows are the registered tries
("try k of N" on each sheet, under the list's SHA-256), each family has a try budget the runner refuses past, and the
round's count prints in one line, all read off the trials register."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
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
    """The trials register's counted rows, by idea: n distinct variants for an idea counted n."""

    def __init__(self, by_idea: dict):
        self.by_idea = by_idea

    def _counted(self, idea=None):
        ideas = self.by_idea if idea is None else {idea: self.by_idea.get(idea, 0)}
        return [{"definition_hash": f"{i}-{k}", "code_version": "c", "dataset": "d", "definition_name": "x",
                 "settings": "{}"} for i, n in ideas.items() for k in range(n)]


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
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-21\n- **Prior tries, family C:** 2\n",
     "which the round doesn't have"),
    ("- **Round:** x\n- **Budget per family:** 3 tries\n- **Family A:** rows 1-21\n- **Prior tries, family A:** many\n",
     "a prior tries line reads"),
])
def test_a_half_written_round_is_refused_in_words(tmp_path, block, words):
    text = _list(tmp_path, block).read_text()
    with pytest.raises(ValueError, match=words):
        rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])


def test_a_familys_tries_are_its_rows_ideas_counted_variants():
    reg = _Register({"i1": 2, "i2": 5, "other": 9})
    assert rounds.used(reg, {"i1", "i2", None}) == 7


START, END = datetime(2024, 1, 1, tzinfo=timezone.utc), datetime(2026, 4, 30, tzinfo=timezone.utc)


def _real_register(tmp_path):
    from sleeve_fund.research.trials import TrialsRegister
    from sleeve_fund.store import Store

    return TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))


def test_one_studys_several_register_rows_are_one_try(tmp_path):
    reg = _real_register(tmp_path)
    params = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT").params
    idea = idea_of(params, None)
    for _ in range(3):  # one study: its sensitivity row and a wf_train row per fold, as study.log writes them
        reg.record(definition_hash="d1", idea_hash=idea, name="qr01_trend_ema_long", family="trend", settings=params,
                   dataset="BINANCE:BTC/USDT:240", stage="in_sample", source="study", sharpe=0.1, data_start=START,
                   data_end=END)
    assert reg.counts(idea)["evaluations"] == 3 and rounds.used(reg, {idea}) == 1  # CR M1 on #337
    reg.record(definition_hash="d1", idea_hash=idea, name="qr01_trend_ema_long", family="trend", settings=params,
               dataset="BINANCE:ETH/USDT:240", stage="in_sample", source="study", sharpe=0.1, data_start=START,
               data_end=END)
    assert rounds.used(reg, {idea}) == 2  # the same definition on another asset is another try


def test_a_basket_try_counts_in_the_family_of_the_definition_it_ran(tmp_path):
    from sleeve_fund.research import basket as bk

    reg = _real_register(tmp_path)
    a = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT").params
    b = rescore.request("qr01c_trend_donchian_long", 240, "binance", "BTC/USDT").params
    basket = bk.Basket("trend basket", "rules", a, (), "2024-01-01")
    bk.record_try(reg, basket, "BINANCE:BTC/USDT,ETH/USDT,SOL/USDT:240", START, END)
    assert reg.counts(idea_of(a, None))["evaluations"] == 0  # filed under the "rules" model's idea, as baskets are
    assert rounds.used(reg, {idea_of(a, None)}) == 1  # but counted as its definition's family's try (CR M2 on #337)
    assert rounds.used(reg, {idea_of(b, None)}) == 0  # and in no other family's


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
    assert note == (f"Round next-1 (study list SHA-256 {sha[:12]}): registered try 1 of 21, family A try 3 (2 in the "
                    "trials register before this one); the family had used 2 of its 3 tries before this one (trials "
                    "register).")
    assert "row  2 " in out and "REFUSED, family A has used 3 of its 3 tries" in out
    lines = out.splitlines()  # a refused row has its memory line too (CR F1 on #339)
    refused = next(i for i, line in enumerate(lines) if "REFUSED, family A" in line)
    assert re.fullmatch(r"    peak memory so far: \d+ MB( \(largest child process \d+ MB\))?", lines[refused + 1])
    assert "registered try 11 of 21, family B try 1 (0 in the trials register before this one); the family had used 0 of its 3 tries" in seen[1][1][0]
    assert f"round next-1, study list SHA-256 {sha}: 21 registered tries, a budget of 3 tries per family" in out
    assert out.rstrip().splitlines()[-1] == (f"Round next-1 (study list SHA-256 {sha[:12]}), 21 registered tries: "
                                            "family A 3 of 3 tries, family B 1 of 3 tries")


def test_a_dry_run_shows_each_rows_place_and_runs_nothing(tmp_path, capfd):
    path = _list(tmp_path)
    called = []
    assert rescore.main([str(path), "--rows", "12", "--dry-run"], run=lambda *a, **k: called.append(1),
                        journal=lambda: None) == 0
    out = capfd.readouterr().out
    assert not called and "registered try 12 of 21 in the round, family B" in out


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


def test_a_basket_sheets_tries_line_gives_its_rounds_count(monkeypatch, tmp_path):
    """The basket counter files every definition-built basket under one model ("try 2 of 20 for rules's family" on
    B0's sheet), so a sheet with its round's try line gives that line as its Tries line instead (QD 31 Oct)."""
    from test_study_asset_list import _History, _req

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.store import Store

    line = ("Round next-1 (study list SHA-256 0123456789ab): registered try 12 of 21, family B try 14 (13 stated before "
            "the round, 0 in the trials register before this one); the family had used 0 of its 3 tries before this one "
            "(trials register).")
    assert rounds.is_try_line(line) and not rounds.is_try_line("Round next-1: registered try 1 of 2.")
    out = run_mod.run_store_study(_req(), store=Store(f"sqlite:///{tmp_path}/t.db"), out_dir=tmp_path / "sheets",
                                  history=_History(), notes=(line,))
    text = out.read_text()
    assert f"**Tries:** {line}" in text and " of 20 for " not in text


def test_a_sealed_basket_sheet_says_already_viewed_only_where_an_earlier_study_holds_the_lock(tmp_path):
    """The holdout row reads the seal itself (CR M1 on #341): a lock an earlier study settled under the basket's idea
    shows on its asset; a register with no lock shows a sealed holdout with nothing viewed."""
    from test_study_asset_list import _History, _req

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.research.holdout import HoldoutLocks
    from sleeve_fund.research.trials import legacy_idea_hash
    from sleeve_fund.store import Store

    def holdout_row(store, out):
        text = run_mod.run_store_study(_req(), store=store, out_dir=tmp_path / out, history=_History()).read_text()
        return next(x for x in text.splitlines() if x.startswith("| Locked holdout"))

    held = Store(f"sqlite:///{tmp_path}/held.db")
    locks, idea = HoldoutLocks(held), legacy_idea_hash("rsi_cross")  # the basket's idea (basket.idea_hash)
    assert locks.open(idea, "ETH", START, END)
    locks.settle(idea, "ETH", None)
    assert holdout_row(held, "held").endswith(
        "sealed: opened once, as its own final step; already viewed on ETH by an earlier study, so paper is the "
        "forward test |")
    assert holdout_row(Store(f"sqlite:///{tmp_path}/clean.db"), "clean").endswith(
        "sealed: opened once, as its own final step |")


def test_a_lists_pinned_dates_set_its_walk_forward_and_holdout_lengths():
    from sleeve_fund.research.pinned import parse

    seal = parse("- **Data start:** 2024-01-01\n- **Research end:** 2026-04-30\n"
                 "- **Holdout:** 2026-05-01 to 2026-10-30\n| 1 | 2024-01-01 | 2025-01-01 | 2026-04-30 |\n")
    assert rescore.fold_days(seal) == (366, 485, 183)  # the next round's (PM 30 Oct, "seal May to Oct 2026")
    req = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT", seal)
    assert (req.train_days, req.test_days, req.holdout_days) == (366, 485, 183)
    assert rescore.fold_days(parse(BASE)) == (rescore.TRAIN_DAYS, rescore.TEST_DAYS, rescore.HOLDOUT_DAYS)


def test_a_familys_stated_prior_tries_number_its_tries_but_not_its_budget(tmp_path):
    text = _list(tmp_path, BLOCK + "- **Prior tries, family B:** 13\n").read_text()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert r.prior == {"B": 13}
    line = rounds.try_line(r, 11, 0)
    assert "family B try 14 (13 stated before the round, 0 in the trials register before this one)" in line
    assert "used 0 of its 3 tries" in line  # the budget is the round's own tries
    assert rounds.summary(r, {"A": 1, "B": 0}).endswith(
        "family A 1 of 3 tries, family B 0 of 3 tries (and 13 stated before the round)")
