"""Idea batch S1-S4 (Researcher 18 Nov; Advisor cleared for the 2024 signal count as family B tries 27-30): the four
definitions exactly as the Researcher froze them (raw sha256), S3's and S4's pre-registered fallbacks as one-figure
changes, each sized to 40% a year of 30-day volatility on the study's own settings, and the count-only list that
records no try. Figures in fixtures are synthetic."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.research.variants import BATCH_VOL, PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import (
    check_definition,
    definition_hash,
    load_definition_file,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_signal_count import PAIRS, _bars, _History

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qrs-batch-count-list.md"
# The Researcher's final raw sha256 values (idea-batch-18nov.md, FINAL for the count, Advisor ~10:56 UTC)
FROZEN = {# qrs1: its reason reworded without H0's figures (CR M1 on #368, public repo); the reason is outside its hash
          "qrs1_short_pullback_downtrend": "24b4759143eae46c3b2ccaa7823e1a76f6575233fc5b173bf0e87856950d606c",
          "qrs2_london_asia_range_fade": "d0ba96f92145be9629686382ef7b68c7c51a5812299ce1f757af3ac63c12f30b",
          "qrs3_shock_fade_4h": "30dbdfcadc5cc9497fe1371e2c87801ce7de777467e4e3bbf001d55a48bfd6a6",
          "qrs4_vwap_range_reversion_4h": "6ada6c15052213c391b6fbe25d54a3b2a35611d83d46621ff3333d2a85d1b5b1"}
FALLBACKS = {"qrs3_shock_fade_4h_fb175": ("qrs3_shock_fade_4h", "2", "1.75"),
             "qrs4_vwap_range_reversion_4h_fb125": ("qrs4_vwap_range_reversion_4h", "1.5", "1.25")}
ROWS = [*FROZEN, *FALLBACKS]


@pytest.mark.parametrize("name", FROZEN)
def test_each_definition_is_the_researchers_file_byte_for_byte(name):
    assert hashlib.sha256(definition_path(name).read_bytes()).hexdigest() == FROZEN[name]


def test_the_reworded_reason_keeps_the_cleared_variant_and_carries_no_figure():
    defn = load_definition_file(definition_path("qrs1_short_pullback_downtrend"))
    assert definition_hash(defn) == "1d119165a8dfdd64"  # the hash of the Researcher's file as cleared
    assert not re.search(r"\d+\.\d+ bps|\d,\d{3}|over \d+ trades", defn["reason"])


@pytest.mark.parametrize("name", FALLBACKS)
def test_each_fallback_changes_its_threshold_and_nothing_else(name):
    parent, was, now = FALLBACKS[name]
    a, b = load_definition_file(definition_path(parent)), load_definition_file(definition_path(name))
    assert {k: v for k, v in a.items() if k not in ("name", "title")} != {k: v for k, v in b.items()
                                                                          if k not in ("name", "title")}

    def blank(defn):  # the threshold, written as a figure in each entry rule, blanked on both
        text = repr({k: v for k, v in defn.items() if k in ("long", "short")})
        return re.sub(rf"(?<![\d.])-?{re.escape(was if defn is a else now)}(?![\d])", "T", text)

    assert blank(a) == blank(b)
    assert {k: v for k, v in a.items() if k in ("blocks", "exits")} == {k: v for k, v in b.items()
                                                                       if k in ("blocks", "exits")}


@pytest.mark.parametrize("name", ROWS)
def test_each_row_checks_on_4h_and_is_sized_to_the_batch_volatility_target(name):
    check_definition(load_definition_file(definition_path(name)), bar_minutes=240)
    variant = PREREGISTERED[name.upper()]
    assert variant == {"definition": name, "params": {"vol_target_size": BATCH_VOL}}
    assert BATCH_VOL == {"annual": 0.40, "lookback_days": 30}
    assert "sizing" not in load_definition_file(definition_path(name))  # nothing about sizing in the definitions


def test_the_list_is_the_six_rows_on_the_rounds_pinned_dates_and_records_no_try():
    text = LIST.read_text()
    assert [(n, name, m) for n, name, m in rescore.rows_of(text)] == [(i + 1, r, 240) for i, r in enumerate(ROWS)]
    pinned = parse(text)
    assert pinned == parse((ROOT / "configs" / "studies" / "next-round-pinned-dates-list.md").read_text())
    assert rescore.confirming_of(text) == {"fund_guards": False, "size_scale": 1.0}  # a run by mistake is no try
    assert rounds.round_of(text, ROWS) is None
    req = rescore.request("qrs4_vwap_range_reversion_4h", 240, "binance", PAIRS, pinned)
    assert req.params["vol_target_size"] == BATCH_VOL and list(req.pairs) == PAIRS.split(",")


def test_the_count_runs_every_row_on_the_train_window_and_writes_nothing(capfd):
    history = _History({p: _bars(seed=i) for i, p in enumerate(PAIRS.split(","))})

    def no_run(*a, **k):
        raise AssertionError("a signal count runs no study and opens no journal")

    code = rescore.main([str(LIST), "--venue", "binance", "--pair", PAIRS, "--count-signals"], run=no_run,
                        journal=no_run, history=history)
    out = capfd.readouterr().out
    assert code == 0, out
    for i, name in enumerate(ROWS, 1):
        assert f"row {i:>2} {name}: train window 2024-01-01 to 2024-12-31 (366 days)" in out
        assert f"row {i:>2} {name}: pre-check at least 60 expected trades: " in out
        assert f"row {i:>2} {name}: sizing to 40% a year over 30 days" in out
        assert "FAILED" not in out
    assert "dry run: 6 checked, 0 run, train-window signals counted (no try)" in out
    assert {m for _, m in history.reads} == {240}
    # S4's day VWAP holds its first decision 1,441 candles (sized for 1-minute bars): on 4h, to late August
    s4 = next(line for line in out.splitlines() if line.startswith("row  4 ") and "decided from" in line)
    assert "decided from 2024-08-2" in s4
    # CR F2: the count reads the engine's decisions, so a change to their keys must not read as zero entries
    s1 = next(line for line in out.splitlines() if line.startswith("row  1 ") and "in all" in line)
    assert int(re.search(r"; ([\d,]+) in all", s1).group(1).replace(",", "")) > 0
    s4_exits = next(line for line in out.splitlines() if line.startswith("row  4 ") and "exit-rule exits" in line)
    assert int(re.search(r"(\d+) of (\d+)", s4_exits).group(2)) > 0
