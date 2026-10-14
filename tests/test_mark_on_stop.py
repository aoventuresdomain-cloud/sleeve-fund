"""MARK-ON-STOP (B3 start check, 14 Oct 16:58 UTC): the runtime stamps marks and fills to the second, and the tick that
sends a flatten marks before its fill is booked, so the flatten tick's mark and the flatten fill share a second. Paper
stopped a few seconds later wrote no mark, so the latest mark of the stopped Strategy still held the position the
journal had closed: the start check compared it and failed, the book page and the Open-risk tile showed it.
Paper now marks the book once on stop, and the start check reads a fill in its mark's own second as late. Replayed
on the hub-fed paper node (tests/qa_ir_hub_path_parity_lib), the paper runtime not in backtest mode."""

from datetime import timedelta

import numpy as np
import pytest

import tests.qa_ir_hub_path_parity_lib as hp
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.start_check import FAIL, NOTE, PASS, check_books
from tests.qa_ir_hub_path_parity_lib import (
    _probe,  # noqa: F401 - autouse: registers the probe strategy
)

PRICES = 30_000 + 50 * np.sin(np.arange(40 * 60) / 200)


def _flatten_on_first_holding_tick(monkeypatch):
    """The round trip's flatten: the PM's command lands just before the first tick that sees the position held, so
    that tick marks the position and then sends the flatten (as scripts/round_trip.py and the B3 smoke did)."""
    seen = {}
    tick = SleeveRuntime.tick

    def flatten_then_tick(self, **k):
        if seen.get("at") == self.now():
            # The replay ticks twice a second where a trade and the timer land together (each sees the tick due);
            # live, the timer's tick moves the due time on, so the flatten tick is that second's only mark (B3).
            return None
        if k["qty"] and "at" not in seen and self.now().second:  # off the minute: no bar decides in that second
            seen["at"] = self.now()
            self.store.command(self.name, "flatten", "round trip: close", actor="test")
        return tick(self, **k)

    monkeypatch.setattr(SleeveRuntime, "tick", flatten_then_tick)
    return seen


def _stopped_seconds_after_the_flatten(monkeypatch, after: int = 4):
    """The run stopped `after` seconds after the flatten tick, well inside the 30 s tick: no tick marks again."""
    seen = _flatten_on_first_holding_tick(monkeypatch)
    hp.hub_paper(PRICES, {"period": 7})
    assert "at" in seen, "the probe never held a position: the fixture needs a longer window"
    first = (seen["at"] - hp.pd.Timestamp(hp.START, tz="UTC").to_pydatetime()) // timedelta(seconds=1)
    seen.clear()
    _, fills, _, store = hp.hub_paper(PRICES[: first + after], {"period": 7})
    return seen["at"], fills, store


def test_the_flatten_fill_lands_in_the_flatten_ticks_own_second(monkeypatch):
    """The root cause, pinned: the tie the start check misread."""
    at, fills, store = _stopped_seconds_after_the_flatten(monkeypatch)
    flat = [f for f in fills if f["side"] == "SELL"]
    assert flat and flat[-1]["ts"] == at
    marked = [m for m in store.equity_series("hubqa", limit=100_000) if m["ts"] == at]
    assert marked and float(marked[0]["qty"]) > 0  # the flatten tick's mark: still holding


def test_a_stopped_paper_strategys_latest_mark_is_its_journal(monkeypatch):
    _, _, store = _stopped_seconds_after_the_flatten(monkeypatch)
    book = store.journal_book("hubqa", 10_000)
    last = store.last_equity("hubqa")
    assert book["qty"] == 0 and float(last["qty"]) == 0
    assert float(last["cash"]) == pytest.approx(float(book["cash"]), abs=0.01)
    assert float(last["equity"]) == pytest.approx(float(book["cash"]), abs=0.01)


def test_the_start_check_passes_the_stopped_round_trip(monkeypatch):
    _, _, store = _stopped_seconds_after_the_flatten(monkeypatch)
    sec = check_books(store)
    assert [r[0] for r in sec.lines] == [PASS], sec.lines


def test_a_fill_in_its_marks_own_second_is_late_not_compared(monkeypatch):
    """Without the stop's mark (a process killed, not stopped): the flatten tick's mark is the latest, in the fill's
    own second. It can't show the fill, so it is not compared: compare again after the next mark."""
    monkeypatch.setattr(SleeveRuntime, "final_mark", lambda self, **k: None, raising=False)
    _, _, store = _stopped_seconds_after_the_flatten(monkeypatch)
    sec = check_books(store)
    assert [r[0] for r in sec.lines] == [NOTE, FAIL], sec.lines
    assert "landed after its latest mark" in sec.lines[0][1] and "nothing was compared" in sec.lines[1][1]


def test_a_backtest_takes_no_mark_on_stop():
    """Golden guard: a backtest's last mark is its last bar's tick; the stop's mark is paper's only."""
    rt = SleeveRuntime.for_backtest(strategy="ping_pong", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=10_000, risk_profile="aggressive")
    rt.final_mark(equity=9_000.0, cash=9_000.0, qty=0.0, price=30_000.0)
    assert rt.store.last_equity("backtest") is None
