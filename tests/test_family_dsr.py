"""FAMILY-DSR (Advisor, 31 Oct 2026, on H0): a try's deflated Sharpe across its whole family. N is the larger of the
stated and the register's counts, flagged when they differ; the spread is one Sharpe per included row, floored at 0.5
a year below 3 rows; every row newer than a pin must itself be pinned, included or excluded with its reason; the try's
own row joins when its sheet shows 1-minute fills; left-out variants are listed. Figures are synthetic."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.research import family_dsr as fd
from sleeve_fund.research import trials
from sleeve_fund.research.metrics import deflated_sharpe_probability
from sleeve_fund.store import Store

ROOT = Path(__file__).resolve().parents[1]
ROUND = (ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md").read_text()
IDEAS = ("a" * 16, "b" * 16, "c" * 16)


def _returns(seed=3, n=485):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0008, 0.02, n), index=pd.date_range("2025-01-01", periods=n, freq="D"))


def _register(tmp_path):
    """Four variants under three member ideas (synthetic Sharpes); one variant re-run, so it has a newer row."""
    reg = trials.TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))
    ids = {}
    for key, (defn, idea, sharpe, dataset) in {
        "x1": ("1" * 16, IDEAS[0], 0.31, "d1"), "y1": ("2" * 16, IDEAS[1], -0.22, "d1"),
        "z1": ("3" * 16, IDEAS[2], 0.44, "d1"), "z2": ("3" * 16, IDEAS[2], 0.47, "d2"),
    }.items():
        ids[key] = reg.record(definition_hash=defn, idea_hash=idea, name=key, family="rules", settings={},
                              dataset=dataset, stage="in_sample", source="study", sharpe=sharpe)
    ids["z2b"] = reg.record(definition_hash="3" * 16, idea_hash=IDEAS[2], name="z2", family="rules", settings={},
                            dataset="d2", stage="in_sample", source="study", sharpe=0.12)  # z2 again: newer row
    return reg, ids


def _text(valid: str, members=IDEAS, excluded: str = "") -> str:
    return (ROUND + f"\n- **Family members, B:** {', '.join(members)}\n"
            + (f"- **Valid-fill rows, B:** {valid}\n" if valid else "")
            + (f"- **Excluded rows, B:** {excluded}\n" if excluded else ""))


Z = "3" * 16


def test_n_is_the_larger_count_and_the_spread_is_the_pinned_rows(tmp_path):
    reg, ids = _register(tmp_path)
    text = _text(f"{'1' * 16}/{ids['x1']} (stand-ins 12%), {'2' * 16}/{ids['y1']}, {'3' * 16}/{ids['z2b']}")
    r = fd.compute(reg, text, "B", _returns(), row_ideas={IDEAS[0]})
    assert (r.n_stated, r.n_register, r.n) == (13 + 1, 4, 14)  # 13 stated prior + 1 in the register; 4 variants
    assert [s for _, _, s, _ in r.spread_rows] == [0.31, -0.22, 0.12] and r.spread_rows[0][3] == 12
    assert not r.floored
    assert r.dsr == pytest.approx(deflated_sharpe_probability(_returns(), 14, [0.31, -0.22, 0.12]))
    lines = fd.report(r)
    assert lines[0] == f"list SHA-256 {hashlib.sha256(text.encode()).hexdigest()}"  # the list it read (CR F3)
    assert lines[1] == "family B: N 14 (stated 14, register 4) FLAG: the counts differ"
    assert "funding stand-ins 12%" in lines[2] and lines[-1].startswith("deflated Sharpe probability ")
    assert r.left_out == ((f"row {ids['z1']} ({Z})", "not valid-fill"),)  # z's d1 variant: ran, not pinned
    assert f"  left out: row {ids['z1']} ({Z}): not valid-fill" in lines
    # 485 days: the no-skill error, sqrt(365/484) = 0.87 a year, is above the rows' spread (0.33)
    assert r.binding[0] == "the no-skill error 1/sqrt(T-1)" and r.binding[1] == pytest.approx((365 / 484) ** 0.5)
    assert lines[-2] == "spread bound: the no-skill error 1/sqrt(T-1), 0.87 a year"


def test_the_register_count_wins_when_it_is_larger(tmp_path):
    reg, ids = _register(tmp_path)
    text = _text(f"{'1' * 16}/{ids['x1']}, {'2' * 16}/{ids['y1']}, {'3' * 16}/{ids['z1']}").replace(
        "- **Prior tries, family B:** 13", "- **Prior tries, family B:** 2")
    r = fd.compute(reg, text, "B", _returns(), row_ideas={IDEAS[0]})
    assert (r.n_stated, r.n_register, r.n) == (3, 4, 4)
    # the probability itself is judged at the larger N, not only the printed one (CR M1)
    sharpes = [s for _, _, s, _ in r.spread_rows]
    assert r.dsr == pytest.approx(deflated_sharpe_probability(_returns(), 4, sharpes))
    assert r.dsr != pytest.approx(deflated_sharpe_probability(_returns(), 3, sharpes))


def test_below_three_rows_the_spread_is_floored_at_half_a_year(tmp_path):
    reg, ids = _register(tmp_path)
    long = _returns(n=3000)  # long enough that the no-skill error 1/sqrt(T) (0.35 a year) is under the floor
    r = fd.compute(reg, _text(f"{'1' * 16}/{ids['x1']}, {'3' * 16}/{ids['z1']}"), "B", long, {IDEAS[0]})
    assert r.floored
    floored = deflated_sharpe_probability(long, r.n, [0.31, 0.44], spread_floor=0.5)
    assert r.dsr == pytest.approx(floored) and floored < deflated_sharpe_probability(long, r.n, [0.31, 0.44])
    assert "floored at 0.5 a year (fewer than 3)" in fd.report(r)[-3]
    assert r.binding == ("the 0.5 floor", 0.5)  # 3000 days: 0.35 a year of error, a rows' spread of 0.09
    # On the round's 485 test days 1/sqrt(T) is 0.87 a year already, so the 0.5 floor changes nothing there
    short = _returns()
    assert deflated_sharpe_probability(short, 14, [0.31, 0.44], spread_floor=0.5) == pytest.approx(
        deflated_sharpe_probability(short, 14, [0.31, 0.44]))


def test_a_wide_spread_of_rows_binds_and_is_said(tmp_path):
    reg = trials.TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))
    pins = []
    for k, s in enumerate((-2.1, 0.2, 2.4)):
        rid = reg.record(definition_hash=f"{k + 1}" * 16, idea_hash=IDEAS[k], name=f"w{k}", family="rules",
                         settings={}, dataset="d", stage="in_sample", source="study", sharpe=s)
        pins.append(f"{f'{k + 1}' * 16}/{rid}")
    r = fd.compute(reg, _text(", ".join(pins)), "B", _returns(), {IDEAS[0]})
    assert r.binding[0] == "the valid-fill rows' spread" and r.binding[1] == pytest.approx(np.std([-2.1, 0.2, 2.4],
                                                                                                    ddof=1))


@pytest.mark.parametrize("pin, words", [
    ("{z}/{z2}", "is newer than pinned row"),  # z2 has a newer row, unpinned: it may be its correction
    ("{z}/" + "9" * 16, "no counted row"),  # a correction not yet applied
    ("{x}/{y1}", "is definition"),
    ("{z}/{z2b}, {z}/{z2b}", "pins a trials id twice"),
])
def test_a_stale_missing_or_wrong_pin_is_refused(tmp_path, pin, words):
    reg, ids = _register(tmp_path)
    valid = pin.format(z="3" * 16, x="1" * 16, z2=ids["z2"], z2b=ids["z2b"], y1=ids["y1"])
    with pytest.raises(ValueError, match=words):
        fd.compute(reg, _text(valid), "B", _returns(), {IDEAS[0]})


@pytest.mark.parametrize("extra, words", [
    ("- **Family members, B:** short\n", "distinct 16-character idea hashes"),
    (f"- **Family members, B:** {'a' * 16}\n- **Family members, B:** {'b' * 16}\n", "listed twice"),
    (f"- **Valid-fill rows, A:** {'1' * 16}/{'2' * 16}\n", "whose members aren't"),
    (f"- **Family members, B:** {'a' * 16}\n- **Valid-fill rows, B:** {'1' * 16} {'2' * 16}\n", "<definition hash>/"),
    ("- **Family members B:** x\n", "a family line reads"),
])
def test_malformed_family_lines_are_refused(extra, words):
    with pytest.raises(ValueError, match=words):
        fd.families_of(ROUND + "\n" + extra)


def test_the_round_list_as_committed_pins_no_family_yet():
    assert fd.families_of(ROUND) == {}  # pinned only once the rows are chosen from the sheets (Advisor 31 Oct)


def test_a_newer_coarse_re_run_pinned_as_excluded_leaves_the_older_row_in(tmp_path):
    reg, ids = _register(tmp_path)
    text = _text(f"{'1' * 16}/{ids['x1']}, {Z}/{ids['z2']}", excluded=f"{Z}/{ids['z2b']} (coarse re-run)")
    r = fd.compute(reg, text, "B", _returns(), {IDEAS[0]})
    assert [s for _, _, s, _ in r.spread_rows] == [0.31, 0.47]


def test_a_correction_is_included_and_its_original_excluded(tmp_path):
    reg, ids = _register(tmp_path)
    ok = _text(f"{Z}/{ids['z2b']}", excluded=f"{Z}/{ids['z2']} (corrected by row {ids['z2b']})")
    assert [s for _, _, s, _ in fd.compute(reg, ok, "B", _returns(), {IDEAS[0]}).spread_rows] == [0.12]
    # the original kept in while a newer row is marked as its correction: refused (Advisor and CR, 31 Oct)
    backwards = _text(f"{Z}/{ids['z2']}", excluded=f"{Z}/{ids['z2']} (corrected by row {ids['z2b']})")
    with pytest.raises(ValueError, match=f"row {ids['z2']} is pinned as included, but row {ids['z2b']} is marked"):
        fd.compute(reg, backwards, "B", _returns(), {IDEAS[0]})
    for valid, by, words in ((f"{'1' * 16}/{ids['x1']}", ids["z2b"], "which isn't included"),  # its correction left out
                             (f"{Z}/{ids['z1']}", ids["z1"], "not a newer row")):  # another variant's, and older
        with pytest.raises(ValueError, match=words):
            fd.compute(reg, _text(valid, excluded=f"{Z}/{ids['z2']} (corrected by row {by})"), "B", _returns(),
                       {IDEAS[0]})
    # its own variant's, but older than the row it would correct
    with pytest.raises(ValueError, match="not a newer row"):
        fd.compute(reg, _text(f"{Z}/{ids['z2']}", excluded=f"{Z}/{ids['z2b']} (corrected by row {ids['z2']})"), "B",
                   _returns(), {IDEAS[0]})


def test_left_out_variants_and_ideas_are_listed_with_why(tmp_path):
    reg, ids = _register(tmp_path)
    never = "d" * 16
    text = _text(f"{'1' * 16}/{ids['x1']}, {'2' * 16}/{ids['y1']}, {Z}/{ids['z1']}", members=(*IDEAS, never),
                 excluded=f"{Z}/{ids['z2b']} (Sharpe unverified), {Z}/{ids['z2']} (coarse re-run)")
    r = fd.compute(reg, text, "B", _returns(), {IDEAS[0]})
    assert r.left_out == ((f"idea {never}", "never ran"), (f"row {ids['z2b']} ({Z})", "Sharpe unverified"))


def test_a_row_of_no_member_is_refused(tmp_path):
    reg, _ = _register(tmp_path)
    foreign = reg.record(definition_hash="5" * 16, idea_hash="e" * 16, name="f", family="rules", settings={},
                         dataset="d", stage="in_sample", source="study", sharpe=0.2)
    with pytest.raises(ValueError, match="a try of none of family B's members"):
        fd.compute(reg, _text(f"{'5' * 16}/{foreign}"), "B", _returns(), {IDEAS[0]})


def test_the_try_joins_unpinned_and_refuses_when_excluded(tmp_path):
    reg, ids = _register(tmp_path)
    r = fd.compute(reg, _text(""), "B", _returns(), {IDEAS[0]}, try_id=ids["x1"])
    assert [i for i, _, _, _ in r.spread_rows] == [ids["x1"]] and r.floored
    with pytest.raises(ValueError, match="the try's own row .* is pinned as excluded"):
        fd.compute(reg, _text("", excluded=f"{'1' * 16}/{ids['x1']} (coarse re-run)"), "B", _returns(), {IDEAS[0]},
                   try_id=ids["x1"])


def test_three_rows_on_a_long_window_are_not_floored(tmp_path):
    reg, ids = _register(tmp_path)
    long = _returns(n=3000)  # here the 0.5 floor would bind (0.35 a year of error, a rows' spread of 0.27)
    r = fd.compute(reg, _text(f"{'1' * 16}/{ids['x1']}, {'2' * 16}/{ids['y1']}, {Z}/{ids['z2b']}"), "B", long,
                   {IDEAS[0]})
    sharpes = [0.31, -0.22, 0.12]
    assert not r.floored and r.dsr == pytest.approx(deflated_sharpe_probability(long, r.n, sharpes))
    assert r.dsr != pytest.approx(deflated_sharpe_probability(long, r.n, sharpes, spread_floor=0.5))


def test_a_non_finite_sharpe_or_too_few_days_is_refused(tmp_path, monkeypatch):
    reg, ids = _register(tmp_path)
    text = _text(f"{'1' * 16}/{ids['x1']}")
    with pytest.raises(ValueError, match="20 return days, under the 30"):  # the probability would be nan (CR F6)
        fd.compute(reg, text, "B", _returns(n=20), {IDEAS[0]})
    rows = reg._counted()
    monkeypatch.setattr(reg, "_counted", lambda idea=None: [
        {**t, "sharpe": math.nan} if t["id"] == ids["x1"] else t for t in rows if idea is None or t["idea_hash"] == idea])
    with pytest.raises(ValueError, match="has no finite Sharpe"):  # the register writes None, but never trust it (F5)
        fd.compute(reg, text, "B", _returns(), {IDEAS[0]})


def test_the_cli_reads_the_sheets_pooled_returns_and_refuses_in_words(tmp_path, capfd, monkeypatch):
    reg, ids = _register(tmp_path)
    data = "binance-basket-btc-eth-sol-usdt-store-240m"
    own = reg.record(definition_hash="4" * 16, idea_hash=IDEAS[0], name="basket:qrx", family="rules", settings={},
                     dataset=data, stage="in_sample", source="study", sharpe=0.05)
    stem = f"qrx_{data}_20261031-104331"
    lst = tmp_path / "configs" / "studies" / "r.md"
    lst.parent.mkdir(parents=True)
    lst.write_text(_text(f"{'1' * 16}/{ids['x1']}, {'2' * 16}/{ids['y1']}, {Z}/{ids['z2b']}"))
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    ret = _returns()
    pd.DataFrame({"date": ret.index.strftime("%Y-%m-%d"), "pooled_return": ret.values}).to_csv(
        sheets / f"{stem}-pooled.csv", index=False)
    trades = sheets / f"{stem}-basket-trades.csv"
    trades.write_text("asset,entry_time,exit_time\nA,2025-01-07T16:00:00,2025-01-08T11:39:00\n")  # a stop's minute
    monkeypatch.chdir(tmp_path)
    argv = ["--list", "configs/studies/r.md", "--family", "B", "--sheet", stem]
    assert fd.main(argv, store=reg.store, sheets=sheets) == 0
    out = capfd.readouterr().out.strip().splitlines()
    assert out[0].startswith("list SHA-256 ") and out[1].startswith("family B: N ")
    assert f"  spread row {own} ({'4' * 16}): Sharpe 0.0500" in out and out[-1] == f"done: family B, sheet {stem}"
    for exit_ in ("2025-01-08T12:00:00", "2025-01-08T12:15:00"):  # on the 4h grid; inside the candle, on 15-minute bars
        trades.write_text(f"asset,entry_time,exit_time\nA,2025-01-07T16:00:00,{exit_}\n")
        assert fd.main(argv, store=reg.store, sheets=sheets) == 1
        assert "1-minute execution isn't shown" in capfd.readouterr().out
    trades.write_text("asset,entry_time,exit_time\nA,2025-01-07T16:00:00,2025-01-08T11:39:00\n")
    assert fd.main(argv[:5] + ["h0_x_1"], store=reg.store, sheets=sheets) == 1  # not a basket sheet's stem
    assert capfd.readouterr().out.startswith("refused: h0_x_1 is not a basket sheet's stem")
    assert fd.main(argv[:3] + ["A"] + argv[4:], store=reg.store, sheets=sheets) == 1
    assert capfd.readouterr().out.strip() == "refused: the list pins no members for family A"
    assert fd.main(argv[:5] + ["../x"], store=reg.store, sheets=sheets) == 2
    assert fd.main(["--list", "/etc/passwd", "--family", "B", "--sheet", stem], store=reg.store) == 2
    capfd.readouterr()
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert fd.main(argv) == 2 and "no DATABASE_URL" in capfd.readouterr().out
