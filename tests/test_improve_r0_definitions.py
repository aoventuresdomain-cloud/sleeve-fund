"""Improve-loop round 0 (research/quant-researcher/improve-loop-plan-24oct.md; PM 24 Oct: the top 3, long and short in
one Strategy): qr23 and qr21 as long and short, each long leg its parent's exactly and the short leg its mirror, and
the BTC study list on round 1's dates with the holdout locked."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleeve_fund.research import rescore
from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import check_definition, definition_hash, load_definition_file

ROOT = Path(__file__).resolve().parents[1]
LIST = ROOT / "configs" / "studies" / "qr-improve-r0-list.md"


def _mirror_stop(stop: dict) -> dict:
    (op, args), = stop["level"].items()
    return {"level": {{"sub": "add", "add": "sub"}[op]: args}}


@pytest.mark.parametrize("key", ["QR23_LS", "QR21_LS"])
def test_each_long_leg_is_its_parent_and_the_short_leg_its_mirror(key):
    v = PREREGISTERED[key]
    mine = load_definition_file(definition_path(v["definition"]))
    parent = load_definition_file(definition_path(v["parent"]))
    assert mine["blocks"] == parent["blocks"] and mine["long"] == parent["long"]
    # the mirror: short where the long leaves, cover where it enters (qr23: below the lower band, above the upper one)
    assert mine["short"] == {"entry": parent["long"]["exit"], "exit": parent["long"]["entry"]}
    if key == "QR23_LS":
        assert "exits" not in mine and "exits" not in parent  # no stop, as the parent
    else:
        assert mine["exits"] == {"long": parent["exits"], "short": {"stop": _mirror_stop(parent["exits"]["stop"])}}
    check_definition(mine, bar_minutes=1440)
    assert definition_hash(mine) != definition_hash(parent)


def test_the_list_is_two_btc_rows_on_the_round1_dates_on_the_perpetual_with_shorts():
    from sleeve_fund.research.pinned import parse

    text = LIST.read_text()
    round1 = (ROOT / "configs" / "studies" / "qr-round1-list.md").read_text()
    assert rescore.rows_of(text) == [(1, "qr23_ls", 1440), (2, "qr21_ls", 1440)]
    assert parse(text) == parse(round1) and parse(text).holdout_days == rescore.HOLDOUT_DAYS
    for name, key in (("qr23_ls", "QR23_LS"), ("qr21_ls", "QR21_LS")):
        req = rescore.request(name, 1440, "binance", "BTC/USDT", parse(text))
        defn = load_definition_file(definition_path(PREREGISTERED[key]["definition"]))
        assert req.params["definition"] == defn and req.params["market"] == "perp" and req.params["allow_short"] is True
        assert (req.use_holdout, req.holdout_days, req.train_days, req.test_days) == (False, 365, 730, 365)
        req.validate()


def test_the_list_dry_runs(capfd):
    assert rescore.main([str(LIST), "--dry-run"], run=lambda *a, **k: None, journal=lambda: None) == 0
    assert "dry run: 2 checked, 0 run" in capfd.readouterr().out
