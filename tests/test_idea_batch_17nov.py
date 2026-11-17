"""Idea batch N1 to N4 (Advisor cleared 17 Nov with pre-run implementation fixes). Done when each definition is the
rechecked file (pinned by content hash), the daily reads are the completed day (the fixes), and the list runs the four
rows on the sealed dates with each family's try number, and dry-runs."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleeve_fund.research import rescore, rounds
from sleeve_fund.research.pinned import parse
from sleeve_fund.strategies.definitions import definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ROOT / "configs" / "definitions" / "variants"
LIST = ROOT / "configs" / "studies" / "idea-batch-17nov-list.md"
HASHES = {"qrn1_capitulation_reversal": "0039b701dfa7e320", "qrn2_short_trend_sleeve": "e2df6c21cf61008f",
          "qrn3_weekend_fade": "df0f1d4e12d172ff", "qrn4_wide_range_continuation": "36bf0e6d71352009"}


@pytest.mark.parametrize("name", HASHES)
def test_each_definition_is_the_rechecked_file(name):
    d = load_definition_file(VARIANTS / f"{name}.toml")
    assert definition_hash(d)[:16] == HASHES[name]
    daily = {k: b for k, b in d["blocks"].items() if b.get("timeframe") == "1d"}
    # every daily ATR settles inside the 366 train days (20x its period, strategies/indicators/averages.py)
    assert all(b["period"] * 20 + 1 <= 366 for b in daily.values() if b["kind"] == "atr")
    # no prior_period day read: it is a day stale at the 00:00 decision (HoE dry run 17 Nov)
    assert not any(b["kind"] == "prior_period" for b in d["blocks"].values())


@pytest.mark.parametrize("name", ["qrn2_short_trend_sleeve", "qrn4_wide_range_continuation"])
def test_breakouts_compare_against_yesterdays_channel(name):
    """A 1d Donchian includes the day just closed, so its breakout reads a lag-1 copy (HoE dry run 17 Nov)."""
    d = load_definition_file(VARIANTS / f"{name}.toml")
    lags = {k for k, b in d["blocks"].items() if b["kind"] == "lag" and b.get("timeframe") == "1d"
            and str(b.get("input", "")).startswith("dc.")}
    assert lags
    text = repr({k: v for k, v in d.items() if k in ("long", "short")})
    assert "'dc.lower'" not in text and "'dc.upper'" not in text


def test_the_list_runs_the_four_rows_on_the_sealed_dates_with_their_try_numbers():
    text = LIST.read_text(encoding="utf-8")
    p = parse(text)
    assert rescore.rows_of(text) == [(n, name, 240) for n, name in enumerate(HASHES, 1)]
    for _, name, minutes in rescore.rows_of(text):
        req = rescore.request(name, minutes, "binance", "BTC/USDT,ETH/USDT,SOL/USDT", p)
        assert tuple(req.pairs) == ("BTC/USDT", "ETH/USDT", "SOL/USDT") and req.params["market"] == "perp"
        assert (req.use_holdout, req.train_days, req.test_days, req.holdout_days) == (False, 366, 485, 183)
        assert req.risk_profile is None
        req.validate()
    r = rounds.round_of(text, [n for n, _, _ in rescore.rows_of(text)])
    assert r.families == {"B": (1,), "trend": (2, 4), "E": (3,)} and r.prior == {"B": 24, "trend": 43, "E": 1}
    assert "family B try 25 " in rounds.try_line(r, 1, 0)
    assert "family trend try 44 " in rounds.try_line(r, 2, 0) and "family trend try 45 " in rounds.try_line(r, 4, 1)
    assert "family E try 2 " in rounds.try_line(r, 3, 0)


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--pair", "BTC/USDT,ETH/USDT,SOL/USDT", "--dry-run"], run=lambda *a, **k: None,
                        journal=lambda: None) == 0
    assert "dry run: 4 checked, 0 run" in capfd.readouterr().out
