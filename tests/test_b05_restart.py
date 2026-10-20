"""B-05 M5: restart and ownership (drill-prep MUST 5, 11, 19, 20, 21; Done-when 4). At a start the venue's position
and open orders are set against the journal; what the fund's own orders left behind is cancelled by its id. The
framework's cancel takes the order's client id, never the order: an order passed in raises, and the leftover stays
resting at the venue with entries held (found on 274ecd88 at the restart reconcile and the reconcile-halt loop)."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace

import pytest
from exec_stub import SYMBOL, VenueStub
from nautilus_trader.model import ClientOrderId

from sleeve_fund.paper import exec_client, exec_reconcile, exec_start
from sleeve_fund.strategies.base import EXIT_LEGS, LongFlatStrategy


class _Spy:
    """Strategy.cancel_order's contract on NautilusTrader 2.0.0rc5: a ClientOrderId, else TypeError."""

    def __init__(self):
        self.sent = []

    def __call__(self, coid):
        if not isinstance(coid, ClientOrderId):
            raise TypeError(f"cancel_order takes a ClientOrderId, got {type(coid).__name__}")
        self.sent.append(str(coid))


def _order(oid: str):
    return SimpleNamespace(client_order_id=ClientOrderId(oid))


def test_the_restart_reconcile_cancels_a_leftover_by_its_id(monkeypatch):
    from sleeve_fund.paper import orphan_guard

    left = _order("sf-" + "a" * 32)
    spy = _Spy()
    fake = SimpleNamespace(
        _guarding=lambda: True, cancel_order=spy, strategy_id="S-1",
        runtime=SimpleNamespace(store=None, name="a", in_process_venue=None, sole_instance=None),
        cache=SimpleNamespace(orders_open=lambda strategy_id=None: [left],
                              order=lambda coid: left if str(coid) == str(left.client_order_id) else None),
        clock=SimpleNamespace(timestamp_ns=lambda: 1_760_000_000_000_000_000))

    def asks(store, name, now, venue_open, send=None, **kw):
        for oid in venue_open:
            send(oid)
        return venue_open

    monkeypatch.setattr(orphan_guard, "restart_reconcile", asks)
    LongFlatStrategy._reconcile_resting(fake)
    assert spy.sent == [str(left.client_order_id)]


def test_a_reconcile_halt_cancels_all_but_the_resting_exits_by_their_ids():
    entry, stop, target = _order("sf-" + "b" * 32), _order("sf-" + "c" * 32), _order("sf-" + "d" * 32)
    spy = _Spy()
    fake = SimpleNamespace(
        cancel_order=spy, strategy_id="S-1",
        decisions={str(stop.client_order_id): {"intent": EXIT_LEGS[0]},
                   str(target.client_order_id): {"intent": EXIT_LEGS[1]},
                   str(entry.client_order_id): {"intent": "entry"}},
        cache=SimpleNamespace(orders_open=lambda strategy_id=None: [entry, stop, target]))
    LongFlatStrategy._cancel_all_but_exits(fake)
    assert spy.sent == [str(entry.client_order_id)]


# the start, on recorded responses ------------------------------------------------------------------------------------
NOW = datetime(2026, 10, 20, 20, 0, tzinfo=UTC)
INSTR = "BTC/USDT"
OURS, LEFT = "sf-" + "1" * 32, "sf-" + "2" * 32  # a journaled open order, and one of ours no journal owns


@pytest.fixture(params=["sqlite", "postgres"])
def store(request, tmp_path):
    from sleeve_fund.store import Store, make_engine, metadata

    if request.param == "sqlite":
        s = Store(f"sqlite:///{tmp_path}/j.db")
    else:
        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("CI runs these against Postgres too (TEST_DATABASE_URL)")
        engine = make_engine(url)
        metadata.drop_all(engine)
        s = Store(engine=engine)
    for name in ("s1", "s2"):
        s.create_sleeve(name=name, strategy="ping_pong", instrument=INSTR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, desired_state="running")
    return s


def _long(store, name, qty, oid):
    store.record_order(name, order_id=oid, side="BUY", qty=qty, intent="entry", reason="cell", ts=NOW - timedelta(hours=1))
    store.book_fill(name, side="BUY", qty=qty, price=60_000.0, fee=0.0, order_id=oid, trade_id=f"t-{oid}",
                    ts=NOW - timedelta(hours=1))


class _Account(VenueStub):
    """The stub with the account's position, open orders only for openOnly, and the transaction log."""

    def __init__(self, size="0", side="", settlements=()):
        super().__init__()
        self.size, self.side, self.settlements = size, side, list(settlements)

    def route(self, method, path, body):
        if path.startswith("/v5/position/list"):
            return {"retCode": 0, "result": {"category": "linear", "nextPageCursor": "", "list": [
                {"symbol": SYMBOL, "positionIdx": 0, "side": self.side, "size": self.size, "leverage": "2"}]}}
        if path.startswith("/v5/account/transaction-log"):
            return {"retCode": 0, "result": {"list": self.settlements, "nextPageCursor": ""}}
        if path.startswith("/v5/order/realtime") and "orderLinkId" not in path:
            with self._lock:
                rows = [self.lookup_row(o) for o in self.orders.values() if o["orderStatus"] != "Cancelled"]
            return {"retCode": 0, "result": {"category": "linear", "list": rows, "nextPageCursor": ""}}
        return super().route(method, path, body)

    def rest(self, link, qty="0.01", key=None):
        self.orders[key or link] = {"orderLinkId": link, "orderId": str(next(self._ids)), "qty": qty, "side": "Sell",
                                    "orderStatus": "New", "orderType": "Limit"}


def _clients(venue):
    from sleeve_fund.paper.venue_reads import VenueReads
    from sleeve_fund.paper.venue_stops import VenueStops

    limiter = exec_client.Limiter()
    return (VenueReads(venue.http_url, api_key="k", api_secret="s", limiter=limiter, timeout_s=2.0),
            VenueStops(venue.http_url, symbol=SYMBOL, api_key="k", api_secret="s", limiter=limiter, timeout_s=2.0))


def _start(store, venue, fund_ids=frozenset()):
    reads, stops = _clients(venue)
    return exec_start.start_reconcile(store, reads, stops, account="demo-1", symbol=SYMBOL, instrument=INSTR,
                                      tags=("s1", "s2"), fund_ids=set(fund_ids), now=NOW)


def test_a_start_cancels_our_orphan_holds_on_a_foreign_order_and_ties_the_rest(store):
    _long(store, "s1", 0.03, "sf-" + "a" * 32)
    _long(store, "s2", 0.02, "sf-" + "b" * 32)
    store.record_order("s1", order_id=OURS, side="SELL", qty=0.01, intent="exit", reason="cell", ts=NOW)
    venue = _Account(size="0.05", side="Buy").start()
    try:
        venue.rest(OURS)
        venue.rest(LEFT)
        venue.rest("", key="manual-1")  # placed by hand on the account: no orderLinkId at all
        done = _start(store, venue)
        again = _start(store, venue)  # a second start: the same foreign order is not alerted again
        venue.orders.pop("manual-1")
        cleared = _start(store, venue)
    finally:
        venue.stop()
    assert [oid for oid, _ in done["cancelled"]] == [LEFT] and done["cancelled"][0][1][0] == "accepted"
    assert [b["orderLinkId"] for b in venue.bodies("order/cancel")] == [LEFT]  # ours only: the foreign one never
    assert done["result"].clean and len(done["foreign"]) == 1 and again["cancelled"] == []
    crit = [e for e in store.sleeve_events_since(None, (exec_start.FOREIGN_HOLD,))]
    assert len(crit) == 1 and crit[0]["level"] == "error" and crit[0]["message"].startswith("CRITICAL")
    assert cleared["result"].clean and exec_start.foreign_hold(store, INSTR) is None
    assert exec_reconcile.hold(store, "s1") is None and exec_reconcile.hold(store, "s2") is None


def test_a_foreign_order_refuses_every_strategys_entries_on_its_instrument(store, monkeypatch):
    from sleeve_fund.paper import runtime

    monkeypatch.setenv("EXEC_MODE", "demo")
    venue = _Account().start()
    try:
        venue.rest("", key="manual-1")
        _start(store, venue)
    finally:
        venue.stop()
    for name in ("s1", "s2"):
        assert any("in no journal of the fund" in h for h in exec_client.holds(store, name))
    why = runtime.exec_target_block(store=store, name="s2")
    assert why is not None and why.code == "exec_target"


def test_a_position_the_journal_does_not_hold_holds_entries_and_splits_nothing(store):
    _long(store, "s1", 0.03, "sf-" + "a" * 32)
    venue = _Account(size="0.05", side="Buy").start()
    try:
        done = _start(store, venue)
    finally:
        venue.stop()
    assert not done["result"].clean and done["result"].per_tag == {}
    assert "0.05" in exec_reconcile.hold(store, "s1") and exec_reconcile.hold(store, "s2") is not None


def test_no_answer_at_the_start_is_a_hold_never_a_pass(store):
    from sleeve_fund.paper.venue_reads import VenueReads
    from sleeve_fund.paper.venue_stops import VenueStops

    lim = exec_client.Limiter()
    reads = VenueReads("http://127.0.0.1:9", api_key="k", api_secret="s", limiter=lim, timeout_s=0.5)
    stops = VenueStops("http://127.0.0.1:9", symbol=SYMBOL, api_key="k", api_secret="s", limiter=lim, timeout_s=0.5)
    done = exec_start.start_reconcile(store, reads, stops, account="demo-1", symbol=SYMBOL, instrument=INSTR,
                                      tags=("s1",), fund_ids=set(), now=NOW)
    assert not done["result"].clean and "didn't answer" in exec_reconcile.hold(store, "s1")


# a late fill's resting stop (MUST 11) and its read-back (MUST 19) ----------------------------------------------------
def test_a_late_fill_gets_a_resting_reduce_only_stop_at_once_and_it_is_read_back(store):
    venue = _Account().start()
    try:
        _, stops = _clients(venue)
        done = exec_start.protect_late_fill(store, "s1", stops, side="SELL", qty="0.01", level="58000", now=NOW,
                                            sleep=lambda s: None)
    finally:
        venue.stop()
    [sent] = venue.bodies("order/create")
    assert sent["orderLinkId"] == done["id"] and sent["reduceOnly"] is True and sent["triggerBy"] == "LastPrice"
    assert (sent["side"], sent["qty"], sent["triggerPrice"]) == ("Sell", "0.01", "58000")
    assert done["read"]["orderLinkId"] == done["id"]
    assert not [e for e in store.events("s1") if "CRITICAL" in e["message"]]


class _Lookups:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), 0

    def lookup(self, oid):
        self.asked += 1
        a = self.answers.pop(0)
        if a is ConnectionError:
            raise ConnectionError("no answer")
        return a


def test_a_failed_read_back_is_retried_once_after_the_back_off_then_critical(store):
    slept = []
    once = _Lookups([ConnectionError, {"orderLinkId": "x"}])
    assert exec_start.read_back(store, "s1", once, "x", now=NOW, backoff_s=1.5, sleep=slept.append) == {
        "orderLinkId": "x"}
    assert once.asked == 2 and slept == [1.5] and not store.events("s1")
    twice = _Lookups([ConnectionError, ConnectionError])
    assert exec_start.read_back(store, "s1", twice, "x", now=NOW, backoff_s=1.5, sleep=slept.append) is False
    [e] = store.events("s1")
    assert twice.asked == 2 and e["level"] == "error" and "CRITICAL" in e["message"] and "stop unverified on the venue" in e["message"]


def test_a_refused_late_fill_stop_is_critical_cannot_protect(store):
    class Refuses:
        def place(self, oid, **kw):
            return ("rejected", 110017, "reduce-only rule not satisfied")

    done = exec_start.protect_late_fill(store, "s1", Refuses(), side="SELL", qty="0.01", level="58000", now=NOW)
    [e] = store.events("s1")
    assert done["read"] is None and e["level"] == "error" and e["kind"] == "cannot_protect" and "CRITICAL" in e["message"]


# funding from settlement records (MUST 5) ----------------------------------------------------------------------------
def _settled(sid="S-1", size="0.05", side="Buy", change="-1.23456789", at=NOW):
    return {"id": sid, "symbol": SYMBOL, "side": side, "size": size, "type": "SETTLEMENT",
            "transactionTime": str(int(at.timestamp() * 1000) + 123), "tradePrice": "60000", "feeRate": "0.0001",
            "change": change, "funding": str(-Decimal(change))}


def _book(store, rows, held):
    return exec_start.book_settlements(store, rows, lambda at: held, account="demo-1", instrument=INSTR,
                                       tags=("s1", "s2"), now=NOW)


def test_funding_is_booked_from_the_venues_settlement_once_and_adds_up_to_the_cent(store):
    held = {"s1": Decimal("0.03"), "s2": Decimal("0.02")}
    done = _book(store, [_settled(), _settled()], held)  # the same settlement twice in one read
    assert set(done["booked"]) == {"S-1"} and not done["gaps"]
    parts = {t: a for t, (_, a) in done["booked"]["S-1"].items()}
    assert sum(parts.values()) == Decimal("-1.23456789")  # the venue's figure exactly, not only to the cent
    assert store.funding_total("s1") + store.funding_total("s2") == Decimal("-1.23456789")
    again = _book(store, [_settled()], held)  # a restart reads it again: nothing moves twice
    assert {k for k, _ in again["booked"]["S-1"].values()} == {"same"}
    assert store.funding_total("s1") + store.funding_total("s2") == Decimal("-1.23456789")
    [row] = store.funding("s1")
    assert row["ts"].replace(tzinfo=UTC) == NOW  # the settlement's minute, as the modelled funding books it


def test_a_settlement_the_journal_cant_split_is_a_gap_and_books_nothing(store):
    done = _book(store, [_settled(size="0.06")], {"s1": Decimal("0.03"), "s2": Decimal("0.02")})
    assert done["booked"] == {} and done["gaps"] and store.funding_total("s1") == 0
    assert exec_reconcile.hold(store, "s1") is not None and exec_reconcile.hold(store, "s2") is not None


def test_a_settlement_whose_signs_dont_agree_is_never_booked(store):
    row = dict(_settled(), funding="-1.23456789")  # the venue's funding field must be the cash change's negative
    done = _book(store, [row], {"s1": Decimal("0.05")})
    assert done["booked"] == {} and "negative" in done["gaps"][0] and store.funding_total("s1") == 0


# the OwnershipGate inside the Strategy (MUST 21) ---------------------------------------------------------------------
def _strategy(store, *, on=True, now=NOW):
    clock = {"now": now}
    spy = _Spy()
    me = SimpleNamespace(_backtest=False, _cfg=SimpleNamespace(exec_on=on), decisions={}, cancel_order=spy,
                         runtime=SimpleNamespace(store=store, name="s1", now=lambda: clock["now"]))
    me._recheck_claims = lambda: LongFlatStrategy._recheck_claims(me)
    return me, spy, clock


def _event(oid):
    return SimpleNamespace(client_order_id=ClientOrderId(oid))


def test_off_the_strategy_books_every_event_as_today(store):
    me, _, _ = _strategy(store, on=False)
    assert LongFlatStrategy._claimed(me, _event("O-20261020-1")) is False


def test_on_only_this_strategys_journal_is_booked(store):
    store.record_order("s1", order_id=OURS, side="BUY", qty=0.01, intent="entry", reason="cell", ts=NOW)
    sib = "sf-" + "5" * 32
    store.record_order("s2", order_id=sib, side="BUY", qty=0.01, intent="entry", reason="cell", ts=NOW)
    me, _, _ = _strategy(store)
    me.decisions["sf-" + "6" * 32] = {"intent": "entry"}
    assert LongFlatStrategy._claimed(me, _event("sf-" + "6" * 32)) is False  # sent by this process
    assert LongFlatStrategy._claimed(me, _event(OURS)) is False  # its own journal's, from before a restart
    assert LongFlatStrategy._claimed(me, _event(sib)) is True  # a sibling's: never booked here
    assert exec_start.foreign_hold(store, INSTR) is None and not store.events("s1")


def test_on_a_foreign_order_is_never_booked_and_holds_the_instrument(store):
    me, spy, _ = _strategy(store)
    assert LongFlatStrategy._claimed(me, _event("web-order-77")) is True
    assert exec_start.foreign_hold(store, INSTR) is not None and spy.sent == []  # never cancelled


def test_on_one_of_ours_in_no_journal_is_held_then_cancelled_by_its_id_past_the_grace(store):
    me, spy, clock = _strategy(store)
    assert LongFlatStrategy._claimed(me, _event(LEFT)) is True
    LongFlatStrategy._recheck_claims(me)
    assert spy.sent == []  # inside the grace: its row may not be visible yet
    clock["now"] = NOW + timedelta(seconds=3)
    LongFlatStrategy._recheck_claims(me)
    assert spy.sent == [LEFT] and [e["kind"] for e in store.events("s1")] == ["orphan_order"]


def test_on_one_whose_row_turns_up_in_the_grace_holds_entries_until_a_start_check_ties(store):
    me, spy, clock = _strategy(store)
    assert LongFlatStrategy._claimed(me, _event(LEFT)) is True
    store.record_order("s1", order_id=LEFT, side="BUY", qty=0.01, intent="entry", reason="cell", ts=NOW)
    clock["now"] = NOW + timedelta(seconds=1)
    LongFlatStrategy._recheck_claims(me)
    assert spy.sent == [] and LEFT in exec_reconcile.hold(store, "s1")
