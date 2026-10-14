"""F227-2: a portfolio run's book counts a held position's open risk from the stop the engine holds now, not from
the stop it opened with. Pins only: one for a stop that tightens (a ratcheted trail), one for a stop that follows
its level both ways (exit_at_level)."""

from decimal import Decimal as D
from itertools import pairwise

import pytest
from test_portfolio_run import LOOSE, _spec

from sleeve_fund.portfolio.holding import Position, holding_for, underlying
from sleeve_fund.research.portfolio_run import run_portfolio
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import Rules

ALWAYS_LONG = {"entry": {"left": "close", "op": ">", "right": 0}}
CHANNEL = {"dc": {"kind": "donchian", "period": 5}, "ma": {"kind": "sma", "period": 10}}


def _defn(exits: dict) -> dict:
    return {"version": 1, "reason": "A test case, written before it is run.", "blocks": CHANNEL, "long": ALWAYS_LONG,
            "exits": exits}


def _book_hours(monkeypatch, exits: dict) -> list[dict]:
    """Every hour the book sees the rules leg holding: the risk it counted, the level the engine holds then, and the
    stop the position opened with."""
    seen, opened = [], {}
    real = Rules.portfolio_book

    def spy(self, name, close, atr_pct):
        equity, held = real(self, name, close, atr_pct)
        if held is not None and self._levels:
            side = self._entry_side or 1
            level = self._tightest(self._levels, side)
            entry_stop = opened.setdefault((self._entry_px, self._entry_side), level)
            seen.append({"close": close, "atr": atr_pct, "qty": self._mark(close)[2], "side": side, "level": level,
                         "entry_stop": entry_stop, "risk": held.risk, "base": self._base_code(),
                         "through": (close - level) * side <= 0, "entry": (self._entry_px, self._entry_side)})
        return equity, held

    monkeypatch.setattr(Rules, "portfolio_book", spy)
    run_portfolio([_spec("r", 3, to_params(_defn(exits)), model="rules")], LOOSE)
    return seen


def _risk(h: dict, stop: float) -> D:
    pos = Position("r", underlying(h["base"]), D(repr(float(h["qty"]))), D(repr(float(stop))))
    return holding_for(pos, D(repr(float(h["close"]))), h["atr"]).risk


def _check_reads_the_current_level(seen: list[dict]) -> list[dict]:
    held = [h for h in seen if not h["through"]]  # through its level, the exit is pending: counted stopless
    assert held
    for h in held:
        assert float(h["risk"]) == pytest.approx(float(_risk(h, h["level"])), rel=1e-9)
    moved = [h for h in held if h["level"] != h["entry_stop"] and (h["close"] - h["entry_stop"]) * h["side"] > 0]
    assert moved, "no held hour with the stop moved off its entry level: the pin proves nothing"
    assert any(abs(float(h["risk"]) - float(_risk(h, h["entry_stop"]))) > 1e-6 for h in moved)
    return held


def test_f227_2_the_book_counts_a_tightened_stop_where_it_rests_now(monkeypatch):
    seen = _book_hours(monkeypatch, {"trail": {"level": "dc.lower"}})  # ratchets by default: only ever tightens
    held = _check_reads_the_current_level(seen)
    assert any(h["level"] > h["entry_stop"] for h in held)
    assert all(h["level"] >= h["entry_stop"] for h in held)


def test_f227_2_the_book_follows_a_trailing_stop_both_ways(monkeypatch):
    seen = _book_hours(monkeypatch, {"exit_at_level": {"level": "ma"}})  # an average falls without a new low
    held = _check_reads_the_current_level(seen)
    looser = [b for a, b in pairwise(held) if a["entry"] == b["entry"] and b["level"] < a["level"]]
    tighter = [b for a, b in pairwise(held) if a["entry"] == b["entry"] and b["level"] > a["level"]]
    assert looser and tighter
