"""A1 (BGL-1, HoE 15 Oct): reduce-only stops resting at the venue, built OFF. With resting stops on, paper rests the
stop a backtest rests, for the whole position at the Strategy's level, cancels it before any other close and keeps
watching the level in the process as a backstop. Off (the default) paper is as it was, and the supervisor refuses
it on until the test-network drills pass and the PM switches it on."""

from types import SimpleNamespace

import pytest
from test_long_short import PERP, _meta, _record

from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.wording import no_venues

# flat, a rise (long), a pause, then a 3% fall through the long's 1% stop
LEGS = [(5, 0.0), (20, 0.015), (5, 0.0), (10, -0.03), (10, 0.0)]


def _run(tmp_path, resting: bool, legs=LEGS, taker_fee=None):
    params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, **PERP, **({"resting_stops": True} if resting else {})}
    meta = _meta(10_000, params)
    if taker_fee is not None:  # the venue terms every fill takes its fee from
        meta["sleeve"]["taker_fee"] = taker_fee
    _record(tmp_path / "a.jsonl.gz", meta, legs)
    store = Store.in_memory()
    orders, fills = replay(tmp_path / "a.jsonl.gz", with_fills=True, store=store)
    return store, orders, fills


def _stops(orders):
    return [o for o in orders if o["intent"] == "stop_loss" and o["order_type"] == "STOP"]


def test_a1a_each_position_has_one_resting_stop_for_all_of_it_at_its_level(tmp_path):
    store, orders, fills = _run(tmp_path, resting=True)
    entries = [o for o in orders if o["intent"] == "entry"]
    stops = _stops(orders)
    assert len(entries) >= 3 and len(stops) == len(entries), [(o["intent"], o["order_type"]) for o in orders]
    for entry, stop in zip(entries, stops):
        assert stop["qty"] == pytest.approx(entry["filled_qty"]) and stop["side"] != entry["side"]
        side = 1 if entry["side"] == "BUY" else -1
        px = next(f["price"] for f in fills if f["order_id"] == entry["order_id"])
        assert stop["signal"]["trigger"] == pytest.approx(px * (1 - side * 0.01), rel=1e-6)
    # the venue order is the journal's row: no watched one beside it
    assert not [o for o in store.orders("ping-pong-test") if o["order_type"] == "STOP (watched)"]
    assert not [o for o in orders if o["status"] in ("rejected", "denied")]


def test_a1b_every_other_close_cancels_the_resting_stop_first_and_none_rests_once_flat(tmp_path):
    store, orders, fills = _run(tmp_path, resting=True)
    stops = _stops(orders)
    assert [s["status"] for s in stops[:-1]] == ["canceled"] * (len(stops) - 1)  # closed by the signal's exits
    assert stops[-1]["status"] == "filled"  # the fall took the last one at the venue
    last = store.journal_book("ping-pong-test", 10_000)
    assert last["qty"] == pytest.approx(0, abs=1e-9)  # never over-closed: reduce-only, and cancelled before exits
    assert not [o for o in store.orders("ping-pong-test") if o["status"] in ("submitted", "accepted")]


@pytest.mark.parametrize("taker", ["0.0005", "0.0008"])
def test_a1c_the_resting_stop_fills_on_the_fall_as_a_stop_loss(tmp_path, taker):
    """Its fee is the configured taker fee, as every other fill's (Advisor): change the terms and it follows."""
    store, orders, fills = _run(tmp_path, resting=True, taker_fee=taker)
    stop = _stops(orders)[-1]
    mine = [f for f in fills if f["order_id"] == stop["order_id"]]
    assert sum(f["qty"] for f in mine) == pytest.approx(stop["qty"])
    for f in mine:  # at its own fill price, past the trigger, and charged the configured taker fee on it
        assert f["side"] == "SELL" and f["price"] <= stop["signal"]["trigger"] + 1e-9
        assert f["fee"] == pytest.approx(f["qty"] * f["price"] * float(taker), abs=0.01)  # USD to the cent
    trip_after = [o for o in orders if o["ts"] > stop["ts"] and o["intent"] == "entry"]
    assert trip_after == []  # no re-entry on the side the stop closed, as paper's watched stop


def test_a1d_with_no_venue_stop_the_watched_level_still_closes_it(tmp_path, monkeypatch):
    """The backstop: a stop the venue never placed (refused, or not yet there) leaves the process watching the level."""
    monkeypatch.setattr(LongFlatStrategy, "_rest_exits", lambda self: None)
    store, orders, fills = _run(tmp_path, resting=True)
    assert not _stops(orders)
    closes = [o for o in orders if o["intent"] == "stop_loss"]
    assert closes and closes[-1]["order_type"] == "MARKET" and closes[-1]["filled_qty"] > 0


def test_a1e_off_by_default_paper_watches_its_stop_as_before(tmp_path):
    store, orders, fills = _run(tmp_path, resting=False)
    assert not _stops(orders)
    assert [o for o in store.orders("ping-pong-test") if o["order_type"] == "STOP (watched)"]
    assert [o for o in orders if o["intent"] == "stop_loss" and o["order_type"] == "MARKET"]


@pytest.mark.parametrize("on", [True, 1, "yes"])
def test_a1f_the_supervisor_refuses_resting_stops_until_they_are_switched_on(on):
    from sleeve_fund.supervisor import RESTING_STOPS_OFF, check_resting_entries

    with pytest.raises(ValueError) as e:
        check_resting_entries(SimpleNamespace(params={"stop_loss": 0.01, "resting_stops": on}))
    assert str(e.value) == RESTING_STOPS_OFF == no_venues(RESTING_STOPS_OFF)
    check_resting_entries(SimpleNamespace(params={"stop_loss": 0.01, "resting_stops": False}))
    check_resting_entries(SimpleNamespace(params={"stop_loss": 0.01}))


# ---- Advisor additions (15 Oct): the backstop's grace, benign rejects, the trigger's price ----------------------

def _waits(status="ACCEPTED", grace=1000, margin=False, liq=None, trigger=59_400.0):
    from nautilus_trader.model import OrderStatus

    t = [10**18]
    order = SimpleNamespace(status=getattr(OrderStatus, status), trigger_price=SimpleNamespace(as_double=lambda: trigger))
    strat = SimpleNamespace(_resting_stops=lambda: True, _resting_exits=lambda: {"stop_loss": order},
                            _margin=margin, _mark=lambda: (0, 1_000.0, 0.1, 0), _liq_now=lambda cash, qty: liq,
                            clock=SimpleNamespace(timestamp_ns=lambda: t[0]), _backstop_since=None,
                            _cfg=SimpleNamespace(backstop_grace_ms=grace))
    return strat, t


def test_a1g_the_backstop_waits_its_grace_for_a_resting_stop_then_closes(tmp_path):
    strat, t = _waits()
    assert LongFlatStrategy._backstop_waits(strat, 59_300.0, 1)
    t[0] += 999_000_000
    assert LongFlatStrategy._backstop_waits(strat, 59_300.0, 1)
    t[0] += 1_000_000
    assert not LongFlatStrategy._backstop_waits(strat, 59_300.0, 1)  # 1 s on: the watched path closes the rest


@pytest.mark.parametrize("status", ["REJECTED", "CANCELED", "SUBMITTED", "INITIALIZED"])
def test_a1h_no_grace_for_a_stop_the_venue_has_not_accepted_or_refused(status):
    strat, _ = _waits(status=status)
    assert not LongFlatStrategy._backstop_waits(strat, 59_300.0, 1)


def test_a1i_no_grace_once_past_half_the_way_to_liquidation():
    strat, _ = _waits(margin=True, liq=50_000.0)  # half of 59,400 to 50,000 is 4,700
    assert LongFlatStrategy._backstop_waits(strat, 54_800.0, 1)
    strat, _ = _waits(margin=True, liq=50_000.0)
    assert not LongFlatStrategy._backstop_waits(strat, 54_600.0, 1)
    short, _ = _waits(margin=True, liq=70_000.0, trigger=60_600.0)  # a short's stop above, its liquidation further up
    assert not LongFlatStrategy._backstop_waits(short, 65_400.0, -1)


@pytest.mark.parametrize("bad", [-1, 3_001, 10_000, 2.5])
def test_a1j_a_grace_outside_0_to_3s_is_refused(bad, instrument):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.base import LongFlatConfig

    with pytest.raises(ValueError, match="backstop_grace_ms"):
        LongFlatConfig(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-INTERNAL"),
                       assumed_taker_fee=0.001, backstop_grace_ms=bad)


def test_a1k_a_reduce_only_close_refused_once_flat_is_benign_and_raises_nothing():
    from sleeve_fund.paper.runtime import BENIGN_REJECT, SleeveRuntime

    store = Store.in_memory()
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT-PERP",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=1_000)
    store.record_order("s1", order_id="x1", side="SELL", qty=0.1, intent="stop_loss", reason="backstop")
    rt = SleeveRuntime(store, "s1")
    rt.on_benign_reject("x1", "ReduceOnly Order is rejected")
    row = store.orders("s1")[0]
    assert row["status"] == "rejected" and row["signal"][BENIGN_REJECT] is True
    assert not store.events("s1", min_level="warning")
    assert [e["kind"] for e in store.events("s1")] == ["order_rejected_benign"]


@pytest.mark.parametrize("pos, intent, reduce, book, benign", [
    (0, "stop_loss", True, 0.0, True), (1, "stop_loss", True, 0.0, False), (0, "entry", True, 0.0, False),
    (0, "exit", False, 0.0, False),
    (0, "stop_loss", True, 0.1, False),  # the journal still holds where the venue is flat: drift, which counts
])
def test_a1l_only_a_reduce_only_close_with_nothing_left_is_benign(pos, intent, reduce, book, benign):
    rt = SimpleNamespace(name="s1", starting_balance=1_000,
                         store=SimpleNamespace(journal_book=lambda name, bal: {"qty": book}))
    strat = SimpleNamespace(_backtest=False, runtime=rt, _pos_side=lambda: pos,
                            cache=SimpleNamespace(order=lambda c: SimpleNamespace(is_reduce_only=reduce)),
                            decisions={"x1": {"intent": intent}})
    assert LongFlatStrategy._benign_reject(strat, SimpleNamespace(client_order_id="x1")) is benign


def test_a1m_paper_rests_its_stop_on_the_last_trade_price_as_it_watches_it(tmp_path, monkeypatch):
    from nautilus_trader.model import OrderType, TriggerType

    seen, real = [], LongFlatStrategy.submit_order

    def spy(self, order, *a, **k):
        if order.order_type == OrderType.STOP_MARKET:
            seen.append((order.trigger_type, order.is_reduce_only))
        return real(self, order, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "submit_order", spy)
    _run(tmp_path, resting=True)
    assert seen and set(seen) == {(TriggerType.LAST_PRICE, True)}


def test_a1n_a_real_reduce_only_refusal_on_a_flat_row_goes_through_on_order_rejected_as_benign(tmp_path, monkeypatch):
    """CR (F-RX-B1): driven end to end. Once the resting stop has closed the position, a second reduce-only close (the
    backstop crossing it) reaches the venue, which refuses it: on_order_rejected journals it benign, with no warning."""
    from nautilus_trader.model import OrderSide, OrderType, Quantity

    from sleeve_fund.paper.runtime import BENIGN_REJECT

    real, sent = LongFlatStrategy.on_order_filled, []

    def then_a_late_close(self, event):
        real(self, event)
        order = self.cache.order(event.client_order_id)
        if sent or order.order_type != OrderType.STOP_MARKET or self._pos_side() != 0:
            return
        late = self.order_factory.market(instrument_id=self._cfg.instrument_id, order_side=OrderSide.SELL,
                                         quantity=Quantity.from_str("0.01"), reduce_only=True)
        coid = str(late.client_order_id)
        self.decisions[coid] = {"intent": "stop_loss", "reason": "backstop", "signal": {}}
        self.runtime.on_order(order_id=coid, side="SELL", qty=0.01, intent="stop_loss", reason="backstop", signal={},
                              order_type="MARKET")
        sent.append(coid)
        self.submit_order(late)

    monkeypatch.setattr(LongFlatStrategy, "on_order_filled", then_a_late_close)
    store, orders, fills = _run(tmp_path, resting=True)
    [late] = [o for o in store.orders("ping-pong-test") if o["order_id"] in sent]
    assert late["status"] == "rejected" and late["signal"].get(BENIGN_REJECT) is True, late
    assert [r["benign"] for r in store.closing_orders("ping-pong-test") if r["order_id"] in sent] == [True]
    kinds = [(e["level"], e["kind"]) for e in store.events("ping-pong-test", limit=1000)]
    assert ("info", "order_rejected_benign") in kinds
    assert not [k for k in kinds if k[1] == "order_rejected"]  # no warning, so nothing to forward or count
