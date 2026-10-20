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
        assert s["entry"].startswith(f"{enter} when") and len(s["entry"]) > len(enter) + 10
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
    assert "within the next 4 one-hour candles" in long_["entry"] and "unless RSI(14) is above 70 first" in long_["entry"]
    assert "only while the weekday is before Saturday" in long_["entry"]
    assert "the candle closes at or after 08:00 London time" in long_["entry"]
    assert "confirmed when" in long_["entry"] and "one of the next 2" in long_["entry"]
    assert "the highest high since the entry candle" in long_["exit"]
    assert "reverse into the short" in long_["exit"] and "10 candles have closed" in long_["exit"]
    assert short["entry"].startswith("Sell short when inside this candle, the price reaches")
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


def _sentences(stem: str, decision: int | None = None) -> list[dict]:
    path = next(DEFS.glob(f"{stem}*.toml"))
    return words.rules(check_definition(load_definition_file(path)), decision)


def test_same_kind_on_two_candles_says_which():
    """QD wording review 1: qr11 has RSI on 1h and on its 15m decision candle; each reference names its candle."""
    (s,) = _sentences("qr11_exhaustion_long", 15)
    assert "RSI(14) on 1h is below 15" in s["entry"] and "RSI(14) on 15m crosses above 30" in s["entry"]


def test_clock_hours_say_the_candle_closes():
    """QD 2: the rule tests the hour at the candle's close, not the clock."""
    (s,) = _sentences("qr12")
    assert "the candle closes at or after 21:00 or before 06:00 London time" in s["entry"]
    assert s["exit"] == "Sell when the candle closes between 08:00 and 21:00 London time"


def test_squeeze_and_exhaustion_wording():
    """QD 3, 4, 7, 8: the next N candles in words, volume as 'at least', no brackets round one setup, slippage."""
    (sq,) = _sentences("qr03_squeeze_long", 60)
    assert "then within the next 10 one-hour candles" in sq["entry"] and "volume is at least 1.5 × its 20-candle average" in sq["entry"]
    assert sq["entry"].startswith("Buy when the upper Bollinger band") and "(a breakout, so the fill allows for extra slippage)" in sq["entry"]
    assert "volume as a multiple" not in sq["entry"]


def test_candle_before_is_said_once():
    """QD 5 and 6: qr13 says 'at the candle before' once; qr10 reads 'the previous candle's high', no 'daily' on its own candle."""
    (lng,) = _sentences("qr13_level_continuation_long", 60)
    assert lng["entry"].count("as of the candle before") == 0 and lng["entry"].startswith("Buy when, at the candle before, the close was above")
    assert "inside this candle, the price reaches" in lng["entry"] and "levels as set at the candle before" in lng["entry"]
    (cap,) = _sentences("qr10", 1440)
    assert "the previous candle's high" in cap["exit"] and "daily" not in cap["entry"]


def test_indicator_settings_in_words():
    """QD 9: '1.5 × ATR(20)', '2 standard deviations', 'highs and lows'."""
    ch = check_definition(load_definition_file(next(DEFS.glob("qr03_squeeze_long*.toml"))))
    by = {b["name"]: b["settings"] for b in words.indicators(ch)}
    assert by["bb"] == "2 standard deviations" and by["kc"] == "1.5 × ATR(20)" and by["dc"] == "highs and lows"
