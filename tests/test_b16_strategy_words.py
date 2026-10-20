"""B-16: a built Strategy's rules in plain English (dashboard/strategy_words), written from the checked definition so
the words can't drift from what the engine trades. Every rule file renders; every rule and operand form the checker
accepts has words; a form it doesn't know is refused rather than skipped."""

import copy
from pathlib import Path

import pytest

from sleeve_fund.dashboard import strategy_words as words
from sleeve_fund.strategies.definitions import (
    BLOCKS,
    check_definition,
    load_definition_file,
)

DEFS = Path(__file__).resolve().parents[1] / "configs" / "definitions"
QR = sorted(p for p in DEFS.glob("qr*.toml"))


def test_sixteen_qr_files():
    assert len(QR) == 16


@pytest.mark.parametrize("path", sorted(DEFS.glob("*.toml")), ids=lambda p: p.stem)
def test_every_file_in_words(path):
    """Each file: one entry and one exit sentence per side it trades, and a row per block."""
    defn = load_definition_file(path)
    checked = check_definition(defn)
    sides = words.rules(checked)
    assert [s["side"] for s in sides] == [s for s in ("long", "short") if s in defn]
    for s in sides:
        enter, leave = words.SIDES[s["side"]]
        assert s["entry"].startswith(f"{enter} when ") and len(s["entry"]) > len(enter) + 10
        assert s["exit"].startswith(f"{leave} ")
        assert s["stopless"] == (not s["levels"])
    assert [b["name"] for b in words.indicators(checked)] == list(checked.order)


def test_dip_long_reads_as_the_mockup():
    """The approved mock-up's example, as generated (decision candle from the file's header: 4h)."""
    checked = check_definition(load_definition_file(DEFS / "qr02_dip_long.toml"))
    (s,) = words.rules(checked, 240)
    assert s["entry"] == "Buy when the close is above the 200-day average and RSI(2) is below 10"
    assert s["exit"] == "Sell when RSI(2) is above 70 or the close is above the 5-candle average or 10 candles have passed"
    assert s["levels"] == [("Stop", "the close minus 2.5 × ATR(14), set as the trade opens")]
    rows = {b["name"]: b for b in words.indicators(checked, "4h")}
    assert (rows["trend"]["period"], rows["trend"]["candle"]) == ("200", "1d")
    assert rows["rsi"]["candle"] == "4h"


def test_short_side_and_daily_candle():
    checked = check_definition(load_definition_file(DEFS / "qr01c_trend_donchian_short.toml"))
    (s,) = words.rules(checked, 1440)
    assert s["entry"].startswith("Sell short when the close is below the lowest low of the last 55 days as of the candle before")
    assert s["exit"] == "Buy back only on the exits below"
    assert s["levels"][0][0] == "Trailing stop" and "highest high of the last 20 days" in s["levels"][0][1]


def test_words_follow_the_file():
    """Generated, not hand-written: change a number in the definition and the sentence changes with it."""
    defn = load_definition_file(DEFS / "qr02_dip_long.toml")
    before = words.rules(check_definition(defn))[0]["entry"]
    edited = copy.deepcopy(defn)
    edited["long"]["entry"]["all"][1]["right"] = 15
    after = words.rules(check_definition(edited))[0]["entry"]
    assert "below 10" in before and "below 15" in after and before != after


EVERY_FORM = {
    "version": 1, "name": "every-form", "reason": "Exercises every rule and operand form the checker accepts.",
    "blocks": {"rsi": {"kind": "rsi", "period": 14}, "atr": {"kind": "atr", "period": 14},
               "dc": {"kind": "donchian", "period": 20}, "avg": {"kind": "sma", "period": 3, "input": "rsi"},
               "rsi1h": {"kind": "rsi", "period": 14, "timeframe": "1h"}},
    "long": {
        "entry": {"setup": {"left": "rsi1h", "op": "<", "right": 30},
                  "trigger": {"all": [
                      {"left": "rsi", "op": "crosses_above", "right": "avg"},
                      {"only_when": {"left": {"time": "weekday"}, "op": "<", "right": 5},
                       "then": {"left": {"time": "hour", "tz": "Europe/London"}, "op": ">=", "right": 8}},
                      {"event": {"left": "close", "op": ">", "right": {"prev": "dc.upper"}},
                       "confirm": {"left": {"div": ["close", "atr"]}, "op": ">", "right": 1}, "within": 2},
                      {"any": [{"left": {"time": "day_of_month"}, "op": "<=", "right": 28},
                               {"left": {"time": "days_to_month_end"}, "op": ">", "right": 0}]},
                  ]},
                  "cancel": {"left": "rsi", "op": ">", "right": 70},
                  "expire_after": {"bars": 4, "timeframe": "1h"}},
        "exit": {"left": "close", "op": "<", "right": {"sub": [{"max_since_entry": "high"}, {"mul": [3, "atr"]}]}},
        "reverse": True,
    },
    "short": {
        "entry": {"first_touch": {"reach": {"sub": ["dc.lower", "atr"]}, "before": {"add": ["dc.lower", "atr"]}}},
        "exit": {"left": {"min_since_entry": "low"}, "op": "<", "right": "dc.lower"},
    },
    "exits": {"time_stop": {"bars": 10, "count": "bars"},
              "trail": {"level": "dc.mid", "ratchet": False},
              "exit_at_level": {"level": "dc.mid", "rest": "previous_close"}},
}


def test_every_form_the_checker_accepts():
    checked = check_definition(EVERY_FORM)
    long_, short = words.rules(checked)
    assert "within 4 candles of 1h" in long_["entry"] and "unless RSI(14) is above 70 first" in long_["entry"]
    assert "only while the weekday is before Saturday" in long_["entry"]
    assert "it is 08:00 London time or later" in long_["entry"]
    assert "confirmed when" in long_["entry"] and "one of the next 2" in long_["entry"]
    assert "the highest high since the entry candle" in long_["exit"]
    assert "reverse into the short" in long_["exit"] and "10 candles have closed" in long_["exit"]
    assert short["entry"].startswith("Sell short when inside the candle, the price reaches")
    assert [k for k, _ in short["levels"]] == ["Trailing stop", "Exit at"]
    assert "moves both ways" in short["levels"][0][1]


@pytest.mark.parametrize("kind", sorted(BLOCKS))
def test_every_block_kind_has_words(kind):
    """Each indicator block, with its default settings and each of its outputs, reads as words, never a raw id."""
    from sleeve_fund.strategies.definitions import BUILDER_BLOCKS

    settings = {"period": 20} if kind in ("sma", "atr_sma") else {}
    if kind == "session_levels":
        settings = {"session": "london"}
    defn = {"version": 1, "name": "k", "reason": "r", "blocks": {"b": {"kind": kind, **settings}}, "long": {}}
    for output in BUILDER_BLOCKS[kind].OUTPUTS:
        ref = "b" if len(BUILDER_BLOCKS[kind].OUTPUTS) == 1 else f"b.{output}"
        defn["long"] = {"entry": {"left": ref, "op": ">", "right": 0}}
        try:
            checked = check_definition(defn, bar_minutes=60)
        except ValueError:
            pytest.skip(f"{kind} needs settings the default doesn't give")
        text = words.rules(checked)[0]["entry"]
        assert " b " not in f" {text} " and "b." not in text


def test_unknown_form_is_refused():
    w = words.Words(check_definition(load_definition_file(DEFS / "qr02_dip_long.toml")))
    with pytest.raises(ValueError):
        w.rule({"not": {"left": "close", "op": ">", "right": 1}})
    with pytest.raises(ValueError):
        w.value({"pow": [1, 2]})
