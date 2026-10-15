"""GAP-DOWN (Advisor 15 Oct ~02:00 UK, HoE Drop B): a stop the venue would have filled while the strategy was down is
booked as a resting venue stop would have filled (NA-1), never better than its level less the floor (the taker's
slippage). A minute that opened past the level: its open less the floor, as before (D13-INTRAMINUTE-JUMP). Else, from
the trigger minute's own trades: the worse of the level less the floor and the first trade at or through the level,
less the floor. With no trade data for that minute: the worse of the level less the floor and the minute's adverse
extreme, labelled "outage fill: pessimistic". Each booking journals its source (open, trade or extreme). A target is
never improved. With the engine up, a stop the venue filled past its level by more than the floor and 7 bp is
recorded in the jump bucket, with that excess. Backtests are unchanged."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401 - the paper harness's offline venue (autouse)
from test_timing import NS, _booked, _late_strategy, _minutes

FLOOR = 0.0005  # the taker's slippage: max(half spread, 0.05%)
MIN = 60_000_000_000


def _down(side, minutes, trades=None, raises=False):
    """A position on `side` from 100 with a 2% stop, restarted after the stored `minutes`; trades: the trigger
    minute's trades the venue serves, as (ns, price), or None when no trade loader is attached."""
    s, bar, events, sold, opened = _late_strategy()
    _booked(s, sold)
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac = 100.0, side, 0.02, 0.05
    s._last_alive = s._last_seen = datetime.fromtimestamp(NS / 1e9, tz=timezone.utc)
    s.runtime.store.fills = lambda name, limit: [{"ts": datetime.fromtimestamp((NS - 600e9) / 1e9, tz=timezone.utc)}]
    s._mark = lambda: (1e4, 1e4, 0.0, 101.0)
    s.history_loader = lambda instrument, bar_type, n: _minutes(*minutes)
    s.instrument = None
    asked = []
    if trades is not None or raises:
        def load(start_ns, end_ns):
            asked.append((start_ns, end_ns))
            if raises:
                raise OSError("venue unreachable")
            return [t for t in trades if start_ns <= t[0] < end_ns]
        s.attach_trade_loader(load)
    assert s._check_outage_exits() is True
    (intent, reason, book), = sold
    return intent, reason, book, asked, events


def _at(k, sec):
    """A time `sec` seconds into stored minute k (1-based; minute k closes at NS + k minutes)."""
    return NS + (k - 1) * MIN + sec * 1_000_000_000


# (1) A gap between minutes: the minute opened past the stop, so it fills at that open less the floor, as before; no
# trade is fetched for it.
@pytest.mark.parametrize("side, minutes, open_", [(1, [(100.5, 99.5), (97.0, 96.0)], 96.5),
                                                  (-1, [(100.5, 99.5), (104.0, 103.0)], 103.5)])
def test_gap_down_1_a_minute_opening_past_the_stop_fills_at_its_open_less_the_floor(side, minutes, open_):
    intent, reason, book, asked, _ = _down(side, minutes, trades=[(_at(2, 5), open_)])
    assert intent == "stop_loss" and book["book_px"] == pytest.approx(open_ * (1 - side * FLOOR))
    assert book["fill_source"] == "open" and asked == []
    assert "opened past it" in reason


# (2) An intra-minute jump, trades served: the worse of the level less the floor and the first trade at or through the
# level, less the floor. Only the trigger minute is fetched.
@pytest.mark.parametrize("side, minutes, trades, first", [
    (1, [(100.5, 99.5), (100.2, 97.0)], [(_at(2, 10), 99.0), (_at(2, 20), 97.6), (_at(2, 30), 97.0)], 97.6),
    (1, [(100.5, 99.5), (100.2, 97.9)], [(_at(2, 10), 98.0), (_at(2, 20), 97.9)], 98.0),  # a touch: the level
    (-1, [(100.5, 99.5), (103.0, 99.8)], [(_at(2, 10), 101.0), (_at(2, 20), 102.5), (_at(2, 30), 103.0)], 102.5),
])
def test_gap_down_2_an_intra_minute_jump_books_the_first_trade_through_the_stop_when_worse(side, minutes, trades,
                                                                                            first):
    intent, reason, book, asked, _ = _down(side, minutes, trades=trades)
    level = 100.0 * (1 - side * 0.02)
    worse = min if side > 0 else max
    assert intent == "stop_loss"
    assert book["book_px"] == pytest.approx(worse(level, first) * (1 - side * FLOOR))
    assert book["fill_source"] == "trade" and book["outage_level"] == pytest.approx(level)
    assert asked == [(NS + MIN, NS + 2 * MIN)]  # the trigger minute, by open time to close time
    assert "pessimistic" not in reason


# (3) No trade data for the trigger minute (none served, the request failed, or no loader for this venue): the worse of
# the level less the floor and the minute's adverse extreme, labelled.
@pytest.mark.parametrize("how", ["no_loader", "empty", "raises"])
@pytest.mark.parametrize("side, minutes, extreme", [(1, [(100.5, 99.5), (100.2, 97.0)], 97.0),
                                                    (1, [(100.5, 99.5), (100.2, 97.97)], None),
                                                    (-1, [(100.5, 99.5), (103.0, 99.8)], 103.0)])
def test_gap_down_3_no_trade_data_books_the_minutes_adverse_extreme_when_worse_and_says_so(how, side, minutes,
                                                                                           extreme):
    trades = {"no_loader": None, "empty": [], "raises": None}[how]
    intent, reason, book, _, events = _down(side, minutes, trades=trades, raises=how == "raises")
    level = 100.0 * (1 - side * 0.02)
    floored = level * (1 - side * FLOOR)
    want = floored if extreme is None else (min if side > 0 else max)(floored, extreme)
    assert intent == "stop_loss" and book["book_px"] == pytest.approx(want)
    assert book["fill_source"] == "extreme"
    assert "outage fill: pessimistic" in reason


def test_gap_down_a_target_is_never_improved_by_the_trades():
    # The target touched in minute 2, a trade far through it: still its level less the taker's slippage (L12).
    intent, _, book, asked, _ = _down(1, [(100.5, 99.5), (107.0, 101.0)], trades=[(_at(2, 10), 107.0)])
    assert intent == "take_profit" and book["book_px"] == pytest.approx(105.0 * (1 - FLOOR)) and asked == []


# (4) The engine up: the venue's stop filled on the tick that gapped through it, booked there as before, and recorded in
# the jump bucket when worse than its level by more than the floor and 7 bp.
@pytest.mark.parametrize("side, level, px, bucket", [(1, 58_800.0, 58_200.0, True), (1, 58_800.0, 58_760.0, False),
                                                     (-1, 61_200.0, 61_800.0, True), (-1, 61_200.0, 61_240.0, False)])
def test_gap_down_4_with_the_engine_up_a_stop_through_its_level_is_booked_there_and_in_the_jump_bucket(
        side, level, px, bucket):
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    events = []
    s = SimpleNamespace(_backtest=False, _entry_side=side, _half_spread=lambda: 0.0001,
                        decisions={"o1": {"intent": "stop_loss", "signal": {"stop_px": level}}},
                        runtime=SimpleNamespace(name="s1", now=lambda: None,
                                                store=SimpleNamespace(event=lambda *a, **k: events.append(a))))
    LongFlatStrategy._jump_bucket(s, "o1", px)
    excess = (side * (level - px) / level - FLOOR) * 1e4
    if not bucket:
        assert events == []
        return
    (_, level_word, kind, msg), = events
    assert (level_word, kind) == ("info", "stop_jump")
    assert f"{excess:.1f} bp" in msg and f"{level:,.6g}" in msg and f"{px:,.6g}" in msg


def test_gap_down_4_a_paper_stop_gapped_through_with_the_engine_up_lands_in_the_jump_bucket(tmp_path, monkeypatch):
    """End to end on the paper engine: a long's 2% stop rests at the venue; the price jumps from about 60,000 to
    58,000 between two trades, so the venue fills the stop on the 58,000 trade, booked there, and the jump bucket says
    so once."""
    import test_exposure_gate_xfails as h

    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    t0 = h.STD_T0
    plan = h.Plan(t0=t0, minutes=18, windows=[(h.M(t0, 2), h.M(t0, 17), 1)],
                  price=lambda s: 60_000 + s * 0.01 if s < 720 else 58_000.0)
    h.run(tmp_path, store, plan, monkeypatch)
    stops = [o for o in h._orders(store) if o["intent"] == "stop_loss" and o["status"] == "filled"]
    fill = next(f for f in h._fills(store) if f["order_id"] == stops[0]["order_id"])
    assert float(fill["price"]) == pytest.approx(58_000.0, abs=1.0)  # booked where the venue filled it
    (jump,) = [e for e in h._events(store) if e["kind"] == "stop_jump"]
    assert "58,000" in jump["message"] and " bp worse than the level less the floor" in jump["message"]


def test_gap_down_binance_trades_reads_the_minute_in_order_and_pages_on_by_id():
    from sleeve_fund.venues import binance_trades

    asked = []
    pages = [[{"a": i, "p": str(100 + i), "T": 1_000 + i} for i in range(1000)],
             [{"a": 1000, "p": "1", "T": 1_999}, {"a": 1001, "p": "2", "T": 61_000}]]

    def get_json(url):
        asked.append(url)
        return pages[len(asked) - 1]

    got = binance_trades("BTC/USDT", 0, 60_000 + 1_000, get_json=get_json)
    assert len(got) == 1001 and got[0] == (1_000 * 1_000_000, 100.0) and got[-1] == (1_999 * 1_000_000, 1.0)
    assert "startTime=0" in asked[0] and "endTime=60999" in asked[0] and "fromId=1000" in asked[1]
    with pytest.raises(ValueError, match="Binance: bad"):
        binance_trades("BTC/USDT", 0, 1, get_json=lambda url: {"code": -1, "msg": "bad"})


def test_gap_down_a_slow_venue_never_holds_the_replay_past_the_deadline(monkeypatch):
    """F-GD-DEADLINE (HoE 02:59 UK, MUST): the replay runs on the first trade after a restart, ahead of every stop and
    tick behind it, so it waits for the trigger minute's trades TRADES_DEADLINE at most, every page together; past
    that the minute's extreme books the stop, said, and a late answer is never read."""
    import threading
    import time

    from sleeve_fund.strategies import base

    monkeypatch.setattr(base, "TRADES_DEADLINE", 0.3, raising=False)
    release = threading.Event()
    s, bar, events, sold, opened = _late_strategy()
    _booked(s, sold)
    s._entry_px, s._entry_side, s._stop_frac, s._tp_frac = 100.0, 1, 0.02, 0.05
    s._last_alive = s._last_seen = datetime.fromtimestamp(NS / 1e9, tz=timezone.utc)
    s.runtime.store.fills = lambda name, limit: [{"ts": datetime.fromtimestamp((NS - 600e9) / 1e9, tz=timezone.utc)}]
    s._mark = lambda: (1e4, 1e4, 0.0, 101.0)
    s.history_loader = lambda instrument, bar_type, n: _minutes((100.5, 99.5), (100.2, 97.0))

    def slow(start_ns, end_ns):  # a venue that answers long after the deadline, with a trade far better than the low
        release.wait(3)
        return [(_at(2, 10), 97.9)]

    s.instrument = None
    s.attach_trade_loader(slow)
    began = time.monotonic()
    try:
        assert s._check_outage_exits() is True
        took = time.monotonic() - began
    finally:
        release.set()
    (intent, reason, book), = sold
    assert took < 0.3 + 0.5, took
    assert intent == "stop_loss" and book["fill_source"] == "extreme" and book["book_px"] == pytest.approx(97.0)
    assert "outage fill: pessimistic" in reason
    assert any(e[2] == "outage_trades_unavailable" and "no answer within 0.3 s" in e[3] for e in events), events
