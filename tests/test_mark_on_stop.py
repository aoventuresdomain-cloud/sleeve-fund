"""MARK-ON-STOP (B3 start check, 14 Oct 16:58 UTC): the runtime stamps marks and fills to the second, and the tick that
sends a flatten marks before its fill is booked, so the flatten tick's mark and the flatten fill share a second. Paper
stopped a few seconds later wrote no mark, so the latest mark of the stopped Strategy still held the position the
journal had closed: the start check compared it and failed, the book page and the Open-risk tile showed it.
Paper now marks the book once on stop, and the start check reads a fill in its mark's own second as late. Replayed
on the hub-fed paper node (tests/qa_ir_hub_path_parity_lib), the paper runtime not in backtest mode.
HoQA light-delta cells: (i) flatten then stop inside one tick, (ii) close now then stop, (iii) a stop with no fill
since the last mark, (iv) a backtest stop writes no row, (v) the start check names the tie and the qty mismatch,
(vi) the stop's mark can't pause or halt on its own."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import tests.qa_ir_hub_path_parity_lib as hp
from sleeve_fund import lifecycle
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.start_check import FAIL, PASS, check_books
from sleeve_fund.store import Store
from tests.qa_ir_hub_path_parity_lib import (
    _probe,  # noqa: F401 - autouse: registers the probe strategy
)

PRICES = 30_000 + 50 * np.sin(np.arange(40 * 60) / 200)
NAME = "hubqa"


def _on_first_holding_tick(monkeypatch, act, skip: int = 0):
    """`act(runtime)` just before the first tick that sees the position held, off the minute (no bar decides in that
    second), as the PM's flatten or close now landed before the B3 tick that marked the position and sent the exit."""
    seen = {}
    tick = SleeveRuntime.tick

    def act_then_tick(self, **k):
        if seen.get("at") == self.now():
            # The replay ticks twice a second where a trade and the timer land together (each sees the tick due);
            # live, the timer's tick moves the due time on, so that tick is its second's only mark (B3).
            return None
        if k["qty"] and "at" not in seen and self.now().second:
            seen["held"] = seen.get("held", 0) + 1
            if seen["held"] > skip:
                seen["at"] = self.now()
                act(self)
        return tick(self, **k)

    monkeypatch.setattr(SleeveRuntime, "tick", act_then_tick)
    return seen


def flatten(rt):
    rt.store.command(rt.name, "flatten", "round trip: close", actor="test")


def close_now(rt):
    rt.store.set_lifecycle(rt.name, "close_now", lifecycle.PM, "round trip: close now", now=rt.now())


def nothing(rt):
    pass


def _stopped_after(monkeypatch, act, after: int = 4, skip: int = 0):
    """The run stopped `after` seconds after that tick, well inside the 30 s tick: no tick marks again."""
    seen = _on_first_holding_tick(monkeypatch, act, skip)
    hp.hub_paper(PRICES, {"period": 7})
    assert "at" in seen, "the probe never held a position: the fixture needs a longer window"
    first = (seen["at"] - datetime.fromtimestamp(hp.START / 1e9, tz=UTC)) // timedelta(seconds=1)
    seen.clear()
    _, fills, _, store = hp.hub_paper(PRICES[: first + after], {"period": 7})
    return seen["at"], fills, store


def _latest_is_the_journal(store):
    book, last = store.journal_book(NAME, 10_000), store.last_equity(NAME)
    assert float(last["qty"]) == pytest.approx(float(book["qty"]), abs=1e-12)
    assert float(last["cash"]) == pytest.approx(float(book["cash"]), abs=0.01)
    assert float(last["equity"]) == pytest.approx(float(book["cash"]) + float(book["qty"]) * float(last["price"]),
                                                  abs=0.01)


def test_root_cause_the_flatten_fill_lands_in_the_flatten_ticks_own_second(monkeypatch):
    at, fills, store = _stopped_after(monkeypatch, flatten)
    assert fills[-1]["side"] == "SELL" and fills[-1]["ts"] == at
    marked = [m for m in store.equity_series(NAME, limit=100_000) if m["ts"] == at]
    assert marked and float(marked[0]["qty"]) > 0  # the flatten tick's own mark: still holding


@pytest.mark.parametrize("act", [flatten, close_now], ids=["i_flatten_then_stop", "ii_close_now_then_stop"])
def test_i_ii_a_stopped_paper_strategys_latest_mark_is_its_journal(monkeypatch, act):
    _, fills, store = _stopped_after(monkeypatch, act)
    assert fills[-1]["side"] == "SELL" and store.journal_book(NAME, 10_000)["qty"] == 0
    _latest_is_the_journal(store)
    assert [r[0] for r in check_books(store).lines] == [PASS]


def test_iii_a_stop_with_no_fill_since_the_last_mark_moves_only_by_price(monkeypatch):
    _, fills, store = _stopped_after(monkeypatch, nothing, skip=3)  # held a while: no bar decides near the stop
    marks = store.equity_series(NAME, limit=100_000)
    before, last = marks[-2], marks[-1]
    assert fills[-1]["ts"] < before["ts"] < last["ts"]
    assert float(last["qty"]) == float(before["qty"]) > 0 and float(last["cash"]) == float(before["cash"])
    moved = float(last["qty"]) * (float(last["price"]) - float(before["price"]))
    assert float(last["equity"]) - float(before["equity"]) == pytest.approx(moved, abs=1e-6)
    _latest_is_the_journal(store)


def test_iv_a_backtest_takes_no_mark_on_stop():
    """Golden guard: a backtest's last mark is its last bar's tick; the stop's mark is paper's only."""
    rt = SleeveRuntime.for_backtest(strategy="ping_pong", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                    starting_balance=10_000, risk_profile="aggressive")
    rt.final_mark(equity=9_000.0, cash=9_000.0, qty=0.0, price=30_000.0)
    assert rt.store.last_equity("backtest") is None


def test_v_without_the_stops_mark_the_start_check_names_the_tie(monkeypatch):
    """A process killed, not stopped: the flatten tick's mark is the latest, in the fill's own second, and doesn't hold
    it. Not compared, and not on its own "nothing compared" (HoQA, #289): a FAIL naming the Strategy and the tie
    (F289-2, HoE OK 18:48 UK 14 Oct: no start check or re-base is signed off it; compare again after the next mark)."""
    monkeypatch.setattr(SleeveRuntime, "final_mark", lambda self, **k: None, raising=False)
    _, _, store = _stopped_after(monkeypatch, flatten)
    lines = check_books(store).lines
    assert [r[0] for r in lines] == [FAIL], lines
    assert lines[0][1].startswith("hubqa: its latest mark predates a fill in the same second") \
        and lines[0][1].endswith("compare again after the next mark"), lines


def _tie(tmp_path, held: float):
    """A BUY of 0.1 at 16:56:09 and a mark in the same second holding `held`."""
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    t = datetime(2026, 10, 14, 16, 56, 9, tzinfo=UTC)
    store.record_fill("s1", side="BUY", qty=0.1, price=100.0, fee=0.0, order_id="o1", trade_id="t1", ts=t)
    cash = 10_000.0 - 10.0 * (held > 0)
    store.record_equity("s1", equity=cash + held * 100.0, cash=cash, qty=held, price=100.0, benchmark=10_000.0, ts=t)
    return store


def test_v_a_tie_the_mark_holds_is_compared_and_passes(tmp_path):
    """The usual case (a tick books a fill, then marks, in one second): HALT-REF-1's re-base reads check_books on it."""
    lines = check_books(_tie(tmp_path, 0.1)).lines
    assert [r[0] for r in lines] == [PASS], lines


def test_v_a_tie_the_mark_doesnt_hold_is_a_fail_naming_the_strategy(tmp_path):
    lines = check_books(_tie(tmp_path, 0.0)).lines
    want = "s1: its latest mark predates a fill in the same second (16:56:09 UTC): compare again after the next mark"
    assert lines == [(FAIL, want)], lines


def test_v_a_mark_short_of_a_fill_before_it_names_the_position(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    t = datetime(2026, 10, 14, 16, 56, 9, tzinfo=UTC)
    store.record_fill("s1", side="BUY", qty=0.1, price=100.0, fee=0.0, order_id="o1", trade_id="t1", ts=t)
    store.record_equity("s1", equity=10_000.0, cash=10_000.0, qty=0.0, price=100.0, benchmark=10_000.0,
                        ts=t + timedelta(seconds=1))
    lines = check_books(store).lines
    assert lines[0][0] == FAIL and "doesn't hold the journal's position: it predates or misses a fill" in lines[0][1]


def test_vi_the_stops_mark_never_pauses_or_halts_on_its_own(tmp_path):
    """A day loss just under the 3% pause (conservative, as the round trip runs), then a stop whose mark is past it:
    the stop only marks. The next start's first tick judges the book as it is, as it would any mark."""
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, risk_profile="conservative")
    t = [datetime(2026, 10, 14, 12, 0, tzinfo=UTC)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    store.set_status("s1", "running", "test")
    rt.tick(equity=10_000.0, cash=10_000.0, qty=0.0, price=100.0)
    t[0] += timedelta(minutes=5)
    rt.tick(equity=9_701.0, cash=9_701.0, qty=0.0, price=100.0)  # 2.99% down on the day: no pause
    events = len(store.events("s1"))
    t[0] += timedelta(seconds=4)
    rt.final_mark(equity=9_650.0, cash=9_650.0, qty=0.0, price=100.0)  # 3.5% down: only marked
    rt.on_stop()
    s = store.sleeve("s1")
    assert s.status == "stopped" and s.paused_until is None and len(store.events("s1")) == events
    assert float(store.last_equity("s1")["equity"]) == 9_650.0


# ---- F289-3 (CR #289): the stop's mark, each guard pinned on its own --------------------------------------------------
def _stopping(mark=lambda: (10_050.0, 4_050.0, 0.1, 60_000.0), restore=None, backtest=False, funded=0.0):
    from types import SimpleNamespace

    got, warned = [], []
    s = SimpleNamespace(_restore=restore, _mark=mark, log=SimpleNamespace(warning=warned.append),
                        runtime=SimpleNamespace(backtest=backtest, funded=funded, final_mark=lambda **k: got.append(k)))
    return s, got, warned


def test_f289_3_the_stop_marks_the_book_as_it_stands():
    from sleeve_fund.strategies.base import LongFlatStrategy

    s, got, _ = _stopping()
    LongFlatStrategy._final_mark(s)
    assert got == [{"equity": 10_050.0, "cash": 4_050.0, "qty": 0.1, "price": 60_000.0}]


def test_f289_3_r8_a_mark_that_fails_is_logged_and_never_stops_the_stop():
    from sleeve_fund.strategies.base import LongFlatStrategy

    def boom():
        raise RuntimeError("no price yet")

    s, got, warned = _stopping(mark=boom)
    LongFlatStrategy._final_mark(s)  # no exception out of it
    assert got == [] and warned == ["no final mark on stop: no price yet"]


@pytest.mark.parametrize("why", ["restore", "backtest"])
def test_f289_3_r6_a_carried_over_book_or_a_backtest_takes_no_stop_mark(why):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s, got, _ = _stopping(restore=object() if why == "restore" else None, backtest=why == "backtest")
    LongFlatStrategy._final_mark(s)
    assert got == []


@pytest.mark.parametrize("equity", [0.0, -5.0])
def test_f289_3_r7_a_wiped_out_account_takes_no_stop_mark(equity):
    from sleeve_fund.strategies.base import LongFlatStrategy

    s, got, _ = _stopping(mark=lambda: (equity, equity, 0.0, 60_000.0))
    LongFlatStrategy._final_mark(s)
    assert got == []


@pytest.mark.parametrize("price", [0.0, -1.0])
def test_f289_3_r4_a_stop_mark_with_no_price_is_not_written(tmp_path, price):
    store = Store(f"sqlite:///{tmp_path}/m.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    rt = SleeveRuntime(store, "s1", now=lambda: datetime(2026, 10, 15, 1, 0, tzinfo=UTC))
    rt.on_start(0.0005)
    rt.final_mark(equity=10_000.0, cash=10_000.0, qty=0.0, price=price)
    assert store.last_equity("s1") is None


def test_f289_3_r2_the_stop_marks_before_the_runtime_stops():
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    order = []
    s = SimpleNamespace(_drop_kept=lambda: None, _maker={}, cancel_all_orders=lambda i: None, recorder=None,
                        _cfg=SimpleNamespace(instrument_id=None), clock=SimpleNamespace(timer_names=lambda: []),
                        _final_mark=lambda: order.append("mark"),
                        runtime=SimpleNamespace(on_stop=lambda: order.append("runtime stopped")))
    LongFlatStrategy.on_stop(s)
    assert order == ["mark", "runtime stopped"]


# ---- F289-4 (CR #289): the tie's cash half, and the benchmark's base set once -----------------------------------------
def test_f289_4_r13_a_tie_whose_position_matches_but_whose_cash_does_not_is_a_fail(tmp_path):
    store = _tie(tmp_path, 0.1)
    t = datetime(2026, 10, 14, 16, 56, 9, tzinfo=UTC)
    store.record_fill("s1", side="BUY", qty=0.0, price=100.0, fee=5.0, order_id="o2", trade_id="t2", ts=t)  # a fee only
    lines = check_books(store).lines
    assert [r[0] for r in lines] == [FAIL] and "predates a fill in the same second" in lines[0][1], lines


def test_f289_4_r10_the_benchmark_keeps_the_first_marks_price_through_the_stops_mark(tmp_path):
    store = Store(f"sqlite:///{tmp_path}/b.db")
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    t = [datetime(2026, 10, 15, 1, 0, tzinfo=UTC)]
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0)
    rt.tick(equity=10_000.0, cash=10_000.0, qty=0.0, price=100.0)
    t[0] += timedelta(minutes=1)
    rt.final_mark(equity=10_000.0, cash=10_000.0, qty=0.0, price=150.0)
    assert rt.bench_base_price == 100.0
    assert store.last_equity("s1")["benchmark"] == pytest.approx(rt.starting_balance * (
        1 - rt.profile.max_position_pct + rt.profile.max_position_pct * 1.5))
