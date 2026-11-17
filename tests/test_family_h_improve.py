"""Family H improve items (PM reopened H 16 Nov; Advisor cleared items 1-5, 7, 8 and P1). Done when:

1. each definition is H0 with exactly its one change (plus its name, title and reason), pinned by content hash;
2. the list runs the eight rows on H0's sealed dates as family B, each a counted try, and dry-runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "configs" / "definitions" / "variants"
LIST = ROOT / "configs" / "studies" / "family-h-improve-list.md"
H0 = load_definition_file(VARIANTS / "qrh0_trend_pullback.toml")
LABELS = ("name", "title", "reason")

# definition -> (its content hash, the one change from H0 as {path: (H0's value, its value)})
ITEMS = {
    "qrh1_trend_pullback_ts6": ("0c28a5cdc295c700", {
        "exits.long.time_stop.bars": (10, 6), "exits.short.time_stop.bars": (10, 6)}),
    "qrh2_trend_pullback_runners": ("d87d06472b5f5d32", {
        "blocks.sma5.kind": ("sma", None), "blocks.sma5.period": (5, None),
        "blocks.sma10.kind": (None, "sma"), "blocks.sma10.period": (None, 10),
        "long.exit.any[0].right": (70, 90), "long.exit.any[1].right": ("sma5", "sma10"),
        "short.exit.any[0].right": (30, 10), "short.exit.any[1].right": ("sma5", "sma10")}),
    "qrh3_trend_pullback_stop15": ("aeca2eccb7c81120", {
        "exits.long.stop.level.sub[1].mul[0]": (2.5, 1.5), "exits.short.stop.level.add[1].mul[0]": (2.5, 1.5)}),
    "qrh4_trend_pullback_sma100": ("8890da699283cf47", {"blocks.trend.period": (200, 100)}),
    "qrh5_trend_pullback_stop35": ("580f9fcbdea883b1", {
        "exits.long.stop.level.sub[1].mul[0]": (2.5, 3.5), "exits.short.stop.level.add[1].mul[0]": (2.5, 3.5)}),
    "qrh7_trend_pullback_short_only": ("7b2f8f27dfd2867a", None),  # the long side and its exits removed, below
    "qrh8_trend_pullback_rsi595": ("92229f8814446bcb", {
        "long.entry.all[1].right": (10, 5), "short.entry.all[1].right": (90, 95)}),
    "qrhp1_trend_pullback_break_filter": ("fd4915d7f0c3e0de", {
        "long.entry.all[2].left": (None, "rsi"), "long.entry.all[2].op": (None, ">="),
        "long.entry.all[2].right": (None, 2), "short.entry.all[2].left": (None, "rsi"),
        "short.entry.all[2].op": (None, "<="), "short.entry.all[2].right": (None, 98)}),
}


def _flat(d, p=""):
    if isinstance(d, dict):
        return {k2: v2 for k, v in d.items() for k2, v2 in _flat(v, f"{p}.{k}" if p else k).items()}
    if isinstance(d, list):
        return {k2: v2 for i, v in enumerate(d) for k2, v2 in _flat(v, f"{p}[{i}]").items()}
    return {p: d}


def _changes(d):
    a = _flat({k: v for k, v in H0.items() if k not in LABELS})
    b = _flat({k: v for k, v in d.items() if k not in LABELS})
    return {k: (a.get(k), b.get(k)) for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


def test_h0_is_the_cleared_round_0_definition():
    assert definition_hash(H0)[:16] == "2161c99b4cf29d08"


@pytest.mark.parametrize("name", ITEMS)
def test_each_item_is_h0_with_its_one_change(name):
    d = load_definition_file(VARIANTS / f"{name}.toml")
    want_hash, want = ITEMS[name]
    assert definition_hash(d)[:16] == want_hash
    if want is not None:
        assert _changes(d) == want
    else:  # short only: H0's short side and exits untouched, no long side at all
        assert "long" not in d and "long" not in d["exits"]
        assert d["short"] == H0["short"] and d["exits"]["short"] == H0["exits"]["short"] and d["blocks"] == H0["blocks"]


def test_the_list_runs_the_eight_rows_on_h0s_sealed_dates_as_family_b():
    text = LIST.read_text(encoding="utf-8")
    p = parse(text)
    assert rescore.rows_of(text) == [(n, name, 240) for n, name in enumerate(ITEMS, 1)]
    for _, name, minutes in rescore.rows_of(text):
        req = rescore.request(name, minutes, "binance", "BTC/USDT,ETH/USDT,SOL/USDT", p)
        assert tuple(req.pairs) == ("BTC/USDT", "ETH/USDT", "SOL/USDT") and req.params["market"] == "perp"
        assert req.params["allow_short"] is True and req.risk_profile is None
        assert (req.use_holdout, req.train_days, req.test_days, req.holdout_days) == (False, 366, 485, 183)
        req.validate()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert (r.name, r.budget, r.families, r.prior) == ("h-improve", 12, {"B": tuple(range(1, 9))}, {"B": 14})
    # H0 is in the register under row 1's idea, so H1 prints as family B try 16
    assert "family B try 16 (14 stated before the round, 1 in the trials register" in rounds.try_line(r, 1, 1)


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--pair", "BTC/USDT,ETH/USDT,SOL/USDT", "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    assert "dry run: 8 checked, 0 run" in capfd.readouterr().out
