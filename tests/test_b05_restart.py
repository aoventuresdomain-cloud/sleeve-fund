"""B-05 M5: restart and ownership (drill-prep MUST 5, 11, 19, 20, 21; Done-when 4). At a start the venue's position
and open orders are set against the journal; what the fund's own orders left behind is cancelled by its id. The
framework's cancel takes the order's client id, never the order: an order passed in raises, and the leftover stays
resting at the venue with entries held (found on 274ecd88 at the restart reconcile and the reconcile-halt loop)."""

from __future__ import annotations

from types import SimpleNamespace

from nautilus_trader.model import ClientOrderId

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
