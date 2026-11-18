"""The H combination (Advisor 18 Nov 17:38; the Researcher's file, byte for byte): H0's short side only (H7), H1's
6-candle time stop and P1's short-side break filter, combined by rule and nothing else."""

from __future__ import annotations

import hashlib
from pathlib import Path

from sleeve_fund.research.variants import PREREGISTERED, definition_path
from sleeve_fund.strategies.definitions import check_definition, load_definition_file

VARIANTS = Path(__file__).resolve().parents[1] / "configs" / "definitions" / "variants"
NAME = "qrhc_trend_pullback_combination"
RAW = "87d851f87d810cbe8a99b05b50034d779b4f2a3a5f36db3b3e077c0f9d824339"


def _load(name):
    return load_definition_file(VARIANTS / f"{name}.toml")


def test_the_combination_is_the_researchers_file_and_checks_on_4h():
    assert hashlib.sha256(definition_path(NAME).read_bytes()).hexdigest() == RAW
    check_definition(_load(NAME), bar_minutes=240)
    assert PREREGISTERED[NAME.upper()] == {"definition": NAME, "params": {}}


def test_the_combination_is_h7_with_h1s_time_stop_and_p1s_short_filter_and_nothing_else():
    c, h0 = _load(NAME), _load("qrh0_trend_pullback")
    h1, h7, p1 = _load("qrh1_trend_pullback_ts6"), _load("qrh7_trend_pullback_short_only"), _load(
        "qrhp1_trend_pullback_break_filter")
    assert "long" not in c and set(c["exits"]) == {"short"}  # H7: short only
    assert c["blocks"] == h0["blocks"] == h7["blocks"]
    assert c["short"]["entry"] == p1["short"]["entry"]  # P1: H0's short entry plus RSI(2) at most 98
    assert c["short"]["exit"] == h0["short"]["exit"] == h7["short"]["exit"]
    assert c["exits"]["short"] == h1["exits"]["short"]  # H1: H0's stop, the time stop at 6 candles
    assert c["exits"]["short"]["time_stop"] == {"bars": 6} and h0["exits"]["short"]["time_stop"] == {"bars": 10}
    assert set(c) - {"name", "title", "reason"} == {"version", "blocks", "short", "exits"}
