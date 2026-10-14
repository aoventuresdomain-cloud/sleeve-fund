"""F-WED1-3 end to end, on the real engine (recorded sessions replayed through the paper runtime, as
tests/test_gate_845_u35.py): a short opened in session 1 whose orders the journal holds on another venue than its pin
starts for its exits only. Nothing opens, its 2% stop closes it, and that stop is journaled on the position's venue;
flat, it is refused until a reset."""
import re
from datetime import datetime, timedelta, timezone

import pytest
import test_replay
from sqlalchemy import text
from test_degraded_155_qa import _record_session, _session_meta
from test_restart_x_guards_on import NAME, OPENS_SHORT, PARAMS, whole_equity  # noqa: F401

from sleeve_fund import router
from sleeve_fund.paper import node
from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from sleeve_fund.supervisor import VENUE_EXITS

LIFT = pytest.mark.no_open_risk_limit(reason="the 2x short must open in session 1 to be carried")
ENTRY = 60_900.0  # the price after session 1's 1.5% rise (ping_pong shorts at about 60,628)
STOPPED = {**PARAMS, "stop_loss": 0.02}


@LIFT
def test_a_short_held_on_another_venue_runs_for_its_exits_only_its_stop_closes_it_there_then_it_is_refused(
        tmp_path, whole_equity):  # noqa: F811
    store = Store(f"sqlite:///{tmp_path}/t.db")
    start0 = test_replay.START
    try:
        s1 = tmp_path / "s1.jsonl.gz"
        _record_session(s1, _session_meta(10_000, STOPPED), OPENS_SHORT)
        replay(s1, store=store)
        book = store.journal_book(NAME, 10_000)
        assert book["qty"] < 0, f"setup: session 1 opened no short: {book}"
        with store.engine.begin() as c:  # its orders went to another venue than the default it is now built on
            c.execute(text("UPDATE orders SET venue = 'BINANCE' WHERE sleeve = :s"), {"s": NAME})
        row = node.venue_start(store, store.sleeve(NAME))  # the paper process's start
        assert row.venue == "BINANCE" and store.sleeve(NAME).status_reason == VENUE_EXITS
        store.create_sleeve = lambda **kw: store.sleeve(kw["name"])
        test_replay.START = start0 + 2 * 3600 * 10**9
        s2 = tmp_path / "s2.jsonl.gz"  # session 2: a 3% rise through the stop, then another ping_pong would short
        _record_session(s2, _session_meta(book["cash"] + book["qty"] * book["entry_px"], STOPPED),
                        [(5, 0.0), (5, 0.03), (5, 0.0), (10, 0.015), (5, 0.0)], px=ENTRY)
        replay(s2, store=store)
    finally:
        test_replay.START = start0
    since = datetime.fromtimestamp(test_replay.START / 1e9, timezone.utc) + timedelta(hours=2)
    after = [o for o in store.orders(NAME, limit=10_000) if o["ts"] >= since]
    filled = [o for o in after if o["filled_qty"] > 0]
    assert filled and all(o["intent"] == "stop_loss" and o["venue"] == "BINANCE" for o in filled), [
        (o["intent"], o["side"], o["venue"], o["status"]) for o in after]
    assert not [o for o in after if o["intent"] in ("entry", "rebalance")]
    kinds = [e["kind"] for e in store.events(NAME, limit=10_000) if e["ts"] >= since]
    assert "entry_blocked" in kinds and "incident" in kinds and kinds.count("venue_mismatch_exits_only") == 1
    assert abs(store.journal_book(NAME, 10_000)["qty"]) < 1e-12
    assert store.venue_start(store.sleeve(NAME)) == (router.REFUSE, "BINANCE")
    assert node.venue_start(store, store.sleeve(NAME)) is None
    (ev,) = [e for e in store.events(NAME, limit=10_000) if e["kind"] == "venue_mismatch"]
    assert not re.search("BINANCE|KRAKEN", ev["message"], re.I)
