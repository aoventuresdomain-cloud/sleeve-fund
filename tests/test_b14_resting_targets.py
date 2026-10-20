"""B-14 (Head 6): reduce-only take-profits resting at the venue, built OFF. With resting targets on, paper rests the
target as a reduce-only limit at its level for the whole position, filled as a maker order when the price trades
through it; the stop (watched, or resting under A1) still guards the position, and whichever exit fills first cancels
the other. Off (the default) paper is as it was, and the supervisor refuses it on until the PM switches it on.

Expected values are worked from the recorded tape and the settings here, never read back from the strategy."""

from decimal import Decimal
from types import SimpleNamespace

import pytest
from test_long_short import PERP, _meta, _record

from sleeve_fund.research.replay import replay
from sleeve_fund.store import Store
from sleeve_fund.wording import no_venues

NAME = "ping-pong-test"
MAKER, TAKER = Decimal("0.0002"), Decimal("0.0005")  # test_long_short._meta's fees
# flat, a rise (ping_pong buys 1% up), a further 3% rise through the 1% target, flat, a 3% fall, flat
UP = [(5, 0.0), (20, 0.015), (10, 0.03), (5, 0.0), (10, -0.03), (10, 0.0)]
# flat, a rise (long), a pause, then a 3% fall through the long's 1% stop before any target
DOWN = [(5, 0.0), (20, 0.008), (5, 0.0), (10, -0.03), (10, 0.0)]


def _run(tmp_path, *, targets=None, stops=False, legs=UP, tp=0.01, size=1.0):
    params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, "take_profit": tp, **PERP}
    if targets is not None:
        params["resting_targets"] = targets
    if stops:
        params["resting_stops"] = True
    _record(tmp_path / "a.jsonl.gz", _meta(10_000, params), legs, size=size)
    store = Store.in_memory()
    replay(tmp_path / "a.jsonl.gz", with_fills=True, store=store)
    orders = store.orders(NAME)[::-1]  # oldest first
    fills = store.fills(NAME, limit=10_000)[::-1]
    return store, orders, fills


def _targets(orders):
    return [o for o in orders if o["intent"] == "take_profit" and o["order_type"] == "LIMIT"]


def _strip(rows):
    """Rows without the ids and stamps a second run renumbers."""
    return [{k: v for k, v in r.items() if k not in ("id", "order_id", "trade_id", "updated_at", "ts")} for r in rows]


# ---- off: today's paper, unchanged ------------------------------------------------------------------------------

@pytest.mark.parametrize("value", [None, False])
def test_b14a_off_by_default_paper_closes_at_market_on_touch_as_before(tmp_path, value):
    store, orders, fills = _run(tmp_path, targets=value)
    assert not _targets(orders)
    tps = [o for o in orders if o["intent"] == "take_profit"]
    assert tps and {o["order_type"] for o in tps} == {"MARKET"}  # market-on-touch, the taker's fee
    base_store, base_orders, base_fills = _run(tmp_path / "base", targets=None) if value is False else (None, None, None)
    if value is False:  # an explicit off is byte-for-byte the default
        assert _strip(orders) == _strip(base_orders) and _strip(fills) == _strip(base_fills)


# ---- on: one resting reduce-only limit at the target ---------------------------------------------------------------

def test_b14b_each_position_has_one_resting_target_for_all_of_it_at_its_level(tmp_path):
    store, orders, fills = _run(tmp_path, targets=True)
    entries = [o for o in orders if o["intent"] == "entry" and o["filled_qty"] > 0]
    targets = _targets(orders)
    assert entries and len(targets) == len(entries), [(o["intent"], o["order_type"], o["status"]) for o in orders]
    for entry, target in zip(entries, targets):
        side = 1 if entry["side"] == "BUY" else -1
        assert target["side"] != entry["side"] and target["qty"] == pytest.approx(entry["filled_qty"])
        px = Decimal(str(entry["avg_px"]))
        level = (px * (1 + side * Decimal("0.01"))).quantize(Decimal("0.1"))  # the tape's 1-dp price precision
        assert Decimal(str(target["signal"]["target_px"])) == level


def test_b14c_the_target_fills_at_its_level_with_the_maker_fee(tmp_path):
    store, orders, fills = _run(tmp_path, targets=True)
    filled = [t for t in _targets(orders) if t["status"] == "filled"]
    assert filled, [(o["intent"], o["order_type"], o["status"]) for o in orders]
    for t in filled:
        mine = [f for f in fills if f["order_id"] == t["order_id"]]
        assert sum(Decimal(str(f["qty"])) for f in mine) == Decimal(str(t["qty"]))
        level = Decimal(str(t["signal"]["target_px"]))
        for f in mine:  # a limit: at its level exactly, never past it, charged the maker's fee to the cent
            assert Decimal(str(f["price"])) == level
            assert abs(Decimal(str(f["fee"])) - Decimal(str(f["qty"])) * level * MAKER) <= Decimal("0.01")
    # no market take-profit went out beside it: one close per target
    assert not [o for o in orders if o["intent"] == "take_profit" and o["order_type"] == "MARKET"]


def test_b14d_the_stop_closing_first_cancels_the_target_and_nothing_rests_once_flat(tmp_path):
    store, orders, fills = _run(tmp_path, targets=True, legs=DOWN, tp=0.05)
    targets = _targets(orders)
    assert targets and {t["status"] for t in targets} == {"canceled"}, [(t["status"], t["message"]) for t in targets]
    assert [o for o in orders if o["intent"] == "stop_loss" and o["filled_qty"] > 0]
    assert store.journal_book(NAME, 10_000)["qty"] == pytest.approx(0, abs=1e-9)  # never over-closed
    assert not [o for o in store.orders(NAME) if o["status"] in ("submitted", "accepted")]


def test_b14e_with_resting_stops_too_whichever_fills_first_cancels_the_other(tmp_path):
    """UP: the long's target fills (its stop cancelled); the short that follows is stopped out (its target cancelled)."""
    store, orders, fills = _run(tmp_path, targets=True, stops=True)
    entries = [o for o in orders if o["intent"] == "entry" and o["filled_qty"] > 0]
    stops = [o for o in orders if o["intent"] == "stop_loss" and o["order_type"] == "STOP"]
    targets = _targets(orders)
    assert len(entries) == len(stops) == len(targets) == 2, [(o["intent"], o["order_type"]) for o in orders]
    assert [(s["status"], t["status"]) for s, t in zip(stops, targets)] == [("canceled", "filled"),
                                                                             ("filled", "canceled")]
    for o in orders:  # never both: each position closed once, for what it held
        assert o["status"] not in ("rejected", "denied"), o
    assert store.journal_book(NAME, 10_000)["qty"] == pytest.approx(0, abs=1e-9)
    assert not [o for o in store.orders(NAME) if o["status"] in ("submitted", "accepted")]


def test_b14f_the_resting_target_is_a_reduce_only_post_only_gtc_limit(tmp_path, monkeypatch):
    from nautilus_trader.model import OrderType, TimeInForce

    from sleeve_fund.strategies.base import LongFlatStrategy

    seen, real = [], LongFlatStrategy.submit_order

    def spy(self, order, *a, **k):
        if order.order_type == OrderType.LIMIT and self.decisions.get(str(order.client_order_id), {}).get(
                "intent") == "take_profit":
            seen.append((order.is_reduce_only, order.time_in_force, order.is_post_only))
        return real(self, order, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "submit_order", spy)
    _run(tmp_path, targets=True)
    assert seen and set(seen) == {(True, TimeInForce.GTC, True)}  # post-only: it only ever rests


def test_b14g_with_no_venue_target_the_watched_level_still_closes_it(tmp_path, monkeypatch):
    """A target the venue never placed (refused, or lost) leaves the process closing at market on touch, as today."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(LongFlatStrategy, "_place_exits", lambda self, plan: None)
    store, orders, fills = _run(tmp_path, targets=True)
    assert not _targets(orders)
    tps = [o for o in orders if o["intent"] == "take_profit"]
    assert tps and tps[0]["order_type"] == "MARKET" and tps[0]["filled_qty"] > 0


def test_b14h_a_backtest_never_rests_a_target_whatever_the_setting():
    from sleeve_fund.strategies.base import LongFlatStrategy

    on = SimpleNamespace(_backtest=True, _cfg=SimpleNamespace(resting_targets=True))
    assert LongFlatStrategy._resting_targets(on) is False
    paper = SimpleNamespace(_backtest=False, _cfg=SimpleNamespace(resting_targets=True))
    assert LongFlatStrategy._resting_targets(paper) is True
    assert LongFlatStrategy._resting_targets(SimpleNamespace(_backtest=False, _cfg=SimpleNamespace())) is False


# ---- the switch: refused until the PM turns it on ------------------------------------------------------------------

@pytest.mark.parametrize("on", [True, 1, "yes"])
def test_b14i_the_supervisor_refuses_resting_targets_until_they_are_switched_on(on):
    from sleeve_fund.supervisor import RESTING_TARGETS_OFF, check_resting_entries

    with pytest.raises(ValueError) as e:
        check_resting_entries(SimpleNamespace(params={"take_profit": 0.01, "resting_targets": on}))
    assert str(e.value) == RESTING_TARGETS_OFF == no_venues(RESTING_TARGETS_OFF)
    check_resting_entries(SimpleNamespace(params={"take_profit": 0.01, "resting_targets": False}))
    check_resting_entries(SimpleNamespace(params={"take_profit": 0.01}))


def test_b14j_the_setting_is_a_strategy_parameter_off_by_default(instrument):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.base import LongFlatConfig

    kw = dict(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-MINUTE-LAST-INTERNAL"),
              assumed_taker_fee=0.001)
    assert LongFlatConfig(**kw).resting_targets is False
    assert LongFlatConfig(**kw, resting_targets=True).resting_targets is True


@pytest.mark.parametrize("rests, closes", [(True, False), (False, True)])
def test_b14k_a_touch_of_the_target_leaves_the_close_to_the_resting_limit(rests, closes):
    """A trade at the level only touches it: the limit fills only on a trade through it (instruments.fill_model), so
    the watched target must not close at market beside it while the venue holds it; without one, it closes as today."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []
    strat = SimpleNamespace(
        _safety_pending=None, _entry_px=60_000.0, _stop_frac=0.01, _tp_frac=0.01, _busy=lambda: False,
        _backtest=False, _restore=None, _entry_side=1, _backstop_since=None, _target_since=None,
        _target_waits=lambda price, side: rests,
        _backstop_waits=lambda price, side: False, runtime=None, _margin=False,
        log=SimpleNamespace(info=lambda *a: None), _lock_exit_leg=lambda *a: None,
        _exit_at_market=lambda *a: sent.append(a))
    assert LongFlatStrategy._check_exits(strat, 60_600.0) is closes
    assert bool(sent) is closes and (not sent or sent[0][0] == "take_profit")


# ---- the Advisor's ruling (HEAD6-B13-B14, 20 Oct): points 5 to 9 ---------------------------------------------------

HELD = {"rise": 0.4, "dip": 0.005, "stop_loss": 0.01, "take_profit": 0.02, **PERP}  # a long held: never rises 40%


def _held(tmp_path, prices, **more):
    """ping_pong's long held on a given price path (one trade a second, 0.05 each), resting targets on."""
    from test_prop_inv import HOUR0, _record_prices

    _record_prices(tmp_path / "a.jsonl.gz", _meta(10_000, {**HELD, "resting_targets": True, **more}), prices, HOUR0)
    store = Store.in_memory()
    replay(tmp_path / "a.jsonl.gz", with_fills=True, store=store)
    return store, store.orders(NAME)[::-1], store.fills(NAME, limit=10_000)[::-1]


def _opening():
    from test_r_i5_1_exit_lock import _ticks

    return _ticks([(5, 0.0, 0.05), (10, 0.004, 0.05)])[0]  # flat, then the buy


@pytest.mark.parametrize("past, through", [(0.0, False), (0.1, True)])
def test_b14l_a_touch_of_the_target_fills_nothing_a_trade_through_fills_it_at_its_level(tmp_path, past, through):
    """HEAD6 6 and 9a: paper's limit fills on a trade through its level, never a touch. The price climbs to the
    target's level and trades there for a minute (or a tick, 0.1, above it), then falls back 50."""
    prices = _opening()
    _, orders, _ = _held(tmp_path / "pre", prices)
    level = _targets(orders)[0]["signal"]["target_px"]  # where the target rests, read off the run up to it
    last = prices[-1]
    tape = prices + [round(last + (level - last) * i / 120, 1) for i in range(1, 121)] + [level + past] * 60
    store, orders, fills = _held(tmp_path / "run", tape + [level - 50] * 60)
    targets = _targets(orders)
    assert len(targets) == 1 and targets[0]["signal"]["target_px"] == level
    closes = [f for f in fills if f["order_id"] != next(o for o in orders if o["intent"] == "entry")["order_id"]]
    if not through:
        assert targets[0]["status"] == "accepted" and targets[0]["filled_qty"] == 0  # still resting, untouched
        assert closes == [] and store.journal_book(NAME, 10_000)["qty"] == pytest.approx(0.11)
        assert not [o for o in orders if o["intent"] == "take_profit" and o["order_type"] == "MARKET"]
    else:  # all of it at its level, as a maker, in the trades' sizes (0.05 a trade)
        assert targets[0]["status"] == "filled" and targets[0]["filled_qty"] == pytest.approx(0.11)
        assert [f["order_id"] for f in closes] == [targets[0]["order_id"]] * 3
        assert [round(f["qty"], 8) for f in closes] == [0.05, 0.05, 0.01]
        for f in closes:
            assert Decimal(str(f["price"])) == Decimal(str(level))
            assert abs(Decimal(str(f["fee"])) - Decimal(str(f["qty"])) * Decimal(str(level)) * MAKER) <= Decimal("0.01")
        assert store.journal_book(NAME, 10_000)["qty"] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize("stops", [False, True])
def test_b14m_a_part_filled_target_leaves_the_stop_and_target_sized_to_what_is_left(tmp_path, stops):
    """HEAD6 5 and 9b: trades of 0.03 take the long's 0.11 target in parts. After each part the stop covers what is
    left, and the target keeps covering it too: all 0.11 fill at its level, nothing at market. With resting stops on
    (A1), the short after it is stopped in parts: its target follows it down to what is left."""
    store, orders, fills = _run(tmp_path, targets=True, stops=stops, size=0.03)
    long_target = _targets(orders)[0]
    parts = [Decimal(str(f["qty"])) for f in fills if f["order_id"] == long_target["order_id"]]
    assert len(parts) > 1 and sum(parts) == Decimal("0.11") and long_target["status"] == "filled"
    assert not [o for o in orders if o["intent"] == "take_profit" and o["order_type"] == "MARKET"]
    left = [float(Decimal("0.11") - sum(parts[:i + 1])) for i in range(len(parts) - 1)]  # after each part but the last
    if not stops:  # the watched stop's journal rows, resized as each part filled
        first = next(f["ts"] for f in fills if f["order_id"] == long_target["order_id"])
        sizes = [o["qty"] for o in orders if o["order_type"] == "STOP (watched)" and o["side"] == "SELL"
                 and o["ts"] >= first]
        assert [round(q, 8) for q in sizes] == left
    else:
        stop = next(o for o in orders if o["intent"] == "stop_loss" and o["order_type"] == "STOP")
        assert stop["status"] == "canceled" and round(stop["qty"], 8) == left[-1]
        assert stop["message"].endswith("as part of the position closed")
        short_target = _targets(orders)[1]
        stop2 = next(o for o in orders if o["intent"] == "stop_loss" and o["side"] == "BUY")
        parts2 = [Decimal(str(f["qty"])) for f in fills if f["order_id"] == stop2["order_id"]]
        assert len(parts2) > 1 and short_target["status"] == "canceled"
        assert Decimal(str(short_target["qty"])) == sum(parts2) - parts2[0]  # what the stop's first part left
    assert store.journal_book(NAME, 10_000)["qty"] == pytest.approx(0, abs=1e-9)


def test_b14n_after_a_restart_each_position_has_exactly_one_resting_target():
    """HEAD6 9c: killed 30 s before a candle close and restarted 30 s after it, with the long held: the target
    rested before is ended on the restart, and one is placed again, for all of the position at the same level."""
    from datetime import timedelta

    from paper_scenarios import NAME as SCENARIO
    from test_prop_inv import HOUR0, _at, _start_ns
    from test_r_i5_1_exit_lock import _ticks
    from test_r_i5_2_missed_candle import _ab

    a, b, _, (restart, _) = _ab("ping_pong", {**HELD, "take_profit": 0.05, "resting_targets": True},
                                _ticks([(5, 0.0, 0.05), (30, 0.004, 0.05)]),
                                lambda a: _at(_start_ns(HOUR0)) + timedelta(minutes=20))
    for store in (a, b):
        live = [o for o in _targets(store.orders(SCENARIO)) if o["status"] in ("submitted", "accepted")]
        assert len(live) == 1 and live[0]["qty"] == pytest.approx(store.journal_book(SCENARIO, 10_000)["qty"])
    before, after = _targets(b.orders(SCENARIO)[::-1])
    assert before["status"] == "canceled" and "restarted" in before["message"]
    assert before["signal"]["target_px"] == after["signal"]["target_px"]
    assert _targets(a.orders(SCENARIO))[0]["signal"]["target_px"] == after["signal"]["target_px"]


def test_b14o_the_compliance_tests_read_the_same_book_with_targets_on(tmp_path, monkeypatch):
    """HEAD6 9d: T1 to T4 never see a target. The paper gate passes it through unchecked (a reduce-only close),
    and the book the tests measure (every position at its stop) is the same with a target resting as without."""
    from sleeve_fund import supervisor
    from sleeve_fund.paper.portfolio_gate import fund_holdings, paper_gate

    monkeypatch.setattr(supervisor, "outside_fund_book", lambda *a, **k: False)  # the replay's USD test Strategy, as if in

    legs = [(5, 0.0), (10, 0.0)]  # the long opened, its target resting at the end
    on, on_orders, _ = _run(tmp_path / "on", targets=True, legs=legs, tp=0.05)
    off, off_orders, _ = _run(tmp_path / "off", legs=legs, tp=0.05)
    assert [t["status"] for t in _targets(on_orders)] == ["accepted"] and not _targets(off_orders)
    entry = [_strip([o]) for o in on_orders if o["intent"] == "entry"]
    assert entry and entry == [_strip([o]) for o in off_orders if o["intent"] == "entry"]
    held = fund_holdings(on, atr_pct=lambda s: 0.02)
    assert held and held == fund_holdings(off, atr_pct=lambda s: 0.02)
    gate = paper_gate(on)
    assert gate(NAME, {"what": "take_profit", "reduce_only": True, "qty": 0.11}, 0) is None


@pytest.mark.parametrize("stops", [False, True])
def test_b14p_the_target_never_holds_more_than_the_position(tmp_path, monkeypatch, stops):
    """HEAD6 9f: every size the target is ever given (sent, or resized: what it has left to fill) is at most the
    position held when it is given, through entries filled in slices and exits filled in parts."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    sizes, send, modify = [], LongFlatStrategy.submit_order, LongFlatStrategy.modify_order

    def spy_send(self, order, *a, **k):
        if self._target_order(order):
            sizes.append((order.quantity.as_double(), float(self._position_qty())))
        return send(self, order, *a, **k)

    def spy_modify(self, coid, quantity=None, *a, **k):
        order = self.cache.order(coid)
        if self._target_order(order) and quantity is not None:
            sizes.append((quantity.as_double() - order.filled_qty.as_double(), float(self._position_qty())))
        return modify(self, coid, quantity, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "submit_order", spy_send)
    monkeypatch.setattr(LongFlatStrategy, "modify_order", spy_modify)
    _run(tmp_path, targets=True, stops=stops, size=0.03)
    assert len(sizes) >= 2 and all(qty <= pos + 1e-12 for qty, pos in sizes), sizes


@pytest.mark.parametrize("filled, held, total", [("0", "0.11", "0.11"),  # an add: 0.05 grows to the 0.11 held
                                                 ("0.03", "0.08", "0.11"),  # its own part fill: nothing to change
                                                 ("0.03", "0.13", "0.16")])  # an add after a part fill
def test_b14p2_an_add_grows_the_target_to_the_position_and_no_further(filled, held, total):
    _p2(filled, held, total, on=True)


@pytest.mark.parametrize("intent", ["stop_loss", "take_profit"])
def test_b14p3_with_resting_targets_off_a_part_filled_exit_is_resized_as_before(intent):
    """HoQA: off, _resize_exits sizes a part-filled exit (a backtest's or A1's stop) to the position alone, as on
    274ecd88: 0.03 filled of 0.11 with 0.08 held goes to 0.08, never to 0.11."""
    _p2("0.03", "0.08", "0.08", on=False, intent=intent)


def _p2(filled, held, total, on, intent="take_profit"):
    from decimal import ROUND_DOWN as _

    from nautilus_trader.model import OrderStatus, Price, Quantity

    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []
    order = SimpleNamespace(client_order_id="T", status=OrderStatus.ACCEPTED, price=Price(60_600.0, 1),
                            trigger_price=Price(59_400.0, 1),
                            quantity=Quantity.from_str("0.11" if filled != "0" else "0.05"),
                            filled_qty=Quantity.from_str(filled))
    strat = SimpleNamespace(_resting_targets=lambda: on,
        _position_qty=lambda: Decimal(held), _lot=lambda: Decimal("0.00000001"), _min_qty=lambda: Decimal("0.00000001"),
        instrument=SimpleNamespace(size_precision=8, price_precision=1), _entry_side=1, _entry_px=60_000.0,
        modify_order=lambda coid, quantity=None, price=None, trigger_price=None: sent.append(
            (quantity, price or trigger_price)),
        _open_target=lambda *a, **k: None, _resizing={}, _resize_due=False)
    LongFlatStrategy._resize_exits(strat, {intent: order}, {intent: 0.01})
    level = 60_600.0 if intent == "take_profit" else 59_400.0
    if Decimal(total) == order.quantity.as_decimal():
        assert sent == []
    else:
        assert [(q.as_decimal(), p.as_double()) for q, p in sent] == [(Decimal(total), level)]
        if on:
            assert Decimal(total) - Decimal(filled) == Decimal(held)  # what it has left to fill is the position


def _crossing(monkeypatch):
    """Every resting target goes out 5% on the wrong side of its entry, so the venue refuses it (a post-only limit
    that would take liquidity): what a refusal looks like, end to end."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    real = LongFlatStrategy._place_exits

    def crossing(self, plan):
        if "take_profit" in plan:
            was = self._entry_px
            self._entry_px = was * (1 - (self._entry_side or 1) * 0.05)
            try:
                real(self, {"take_profit": plan["take_profit"]})
            finally:
                self._entry_px = was
            plan = {k: v for k, v in plan.items() if k != "take_profit"}
        if plan:
            real(self, plan)

    monkeypatch.setattr(LongFlatStrategy, "_place_exits", crossing)
    monkeypatch.setattr(LongFlatStrategy, "_through_target", lambda self, level, side: False)


def test_b14q_a_refused_target_is_benign_journaled_and_the_watched_target_closes(tmp_path, monkeypatch):
    """HEAD6 7: a refused resting target is not a safety exit: benign for F-RX-2 (never one of its refused closes in
    a row, no block), journaled with the venue's words and an info event, sent once per position (no retry on every
    trade), and the watched target closes the position at market in its place."""
    from sleeve_fund.ops_health import REJECTS_OPEN

    _crossing(monkeypatch)
    store, orders, fills = _run(tmp_path, targets=True)
    targets = _targets(orders)
    entries = [o for o in orders if o["intent"] == "entry" and o["filled_qty"] > 0]
    assert len(targets) == len(entries) == 2  # one each, never sent again on the next trade
    for t in targets:
        assert t["status"] == "rejected" and "POST_ONLY" in t["message"] and t["signal"]["benign_reject"] is True
    closing = {r["order_id"]: r for r in store.closing_orders(NAME)}
    assert all(closing[t["order_id"]]["benign"] for t in targets)
    events = store.events(NAME, limit=1_000)
    assert not [e for e in events if e["kind"] == REJECTS_OPEN]
    assert not [e for e in events if e["kind"] == "order_rejected"]  # no warning for it
    refused = [e for e in events if e["kind"] == "resting_target_refused"]
    assert len(refused) == 2 and {e["level"] for e in refused} == {"info"}
    market = [o for o in orders if o["intent"] == "take_profit" and o["order_type"] == "MARKET"]
    assert market and market[0]["filled_qty"] == pytest.approx(entries[0]["filled_qty"])  # the watched one, at market


def test_b14r_three_targets_refused_in_a_row_raise_an_info_alert_on_their_own_count():
    from sleeve_fund.paper.runtime import BENIGN_REJECT, RESTING_TARGETS_REFUSED, SleeveRuntime

    calls = []
    store = SimpleNamespace(update_order=lambda oid, **k: calls.append(("order", oid, k["status"])),
                            merge_order_signal=lambda oid, sig: calls.append(("signal", oid, sig)),
                            event=lambda name, level, kind, msg, ts=None: calls.append(("event", level, kind, msg)))
    rt = SimpleNamespace(store=store, name=NAME, now=lambda: None)
    for n in range(1, 8):
        SleeveRuntime.on_target_refused(rt, f"T{n}", "would take liquidity", n)
    assert [c[1] for c in calls if c[0] == "order"] == [f"T{n}" for n in range(1, 8)]
    assert all(c[2] == {BENIGN_REJECT: True} for c in calls if c[0] == "signal")
    events = [c for c in calls if c[0] == "event"]
    assert {e[1] for e in events} == {"info"}
    alerts = [e for e in events if e[2] == RESTING_TARGETS_REFUSED]
    assert [e[3].split()[0] for e in alerts] == ["3", "6"]  # at 3 in a row, and each further 3
    assert no_venues(alerts[0][3]) == alerts[0][3]


@pytest.mark.parametrize("refusals, target, left", [(2, True, 0), (2, False, 2)])
def test_b14s_a_target_the_venue_takes_ends_the_refusals_in_a_row(refusals, target, left):
    from sleeve_fund.strategies.base import LongFlatStrategy

    order = object()
    strat = SimpleNamespace(
        _target_refusals=refusals, _target_order=lambda o: target and o is order,
        cache=SimpleNamespace(order=lambda coid: order), _order_status=lambda e, s: None, runtime=None,
        _cancel_on_accept=set(), _resize_if_due=lambda: None,
        _claimed=lambda e: False, _backtest=False)  # TEC-B14-STUBS: what the D3 tree's handler reads first
    LongFlatStrategy.on_order_accepted(strat, SimpleNamespace(client_order_id="T", ts_init=0))
    assert strat._target_refusals == left


@pytest.mark.parametrize("status, grace_passed, waits", [
    ("ACCEPTED", True, True), ("PARTIALLY_FILLED", True, True), ("PENDING_UPDATE", True, True),
    ("SUBMITTED", False, True), ("SUBMITTED", True, False), ("REJECTED", False, False), ("CANCELED", False, False)])
def test_b14t_through_the_target_a_live_one_waits_an_unanswered_one_gets_the_grace(status, grace_passed, waits):
    """HEAD6 6: through the level, a target the venue holds is left to fill (filled or live: nothing to do); one sent
    and unanswered waits backstop_grace_ms, then is given up (never waited on again) and the watched one closes at
    market; refused, cancelled or none: the watched one closes at once."""
    from nautilus_trader.model import OrderStatus

    from sleeve_fund.strategies.base import LongFlatStrategy

    events = []
    order = SimpleNamespace(status=getattr(OrderStatus, status), price=SimpleNamespace(as_double=lambda: 61_200.0),
                            client_order_id="T")
    now = [10_000_000_000]
    strat = SimpleNamespace(
        _resting_targets=lambda: True, _resting_exits=lambda: {"take_profit": order}, _target_since=None,
        clock=SimpleNamespace(timestamp_ns=lambda: now[0]), _cfg=SimpleNamespace(backstop_grace_ms=1_000),
        _targets_lost=set(), _cancel_on_accept=set(),
        runtime=SimpleNamespace(name=NAME, now=lambda: None, store=SimpleNamespace(
            event=lambda *a, **k: events.append(a))))
    assert LongFlatStrategy._target_waits(strat, 61_200.0, 1) is (status != "REJECTED" and status != "CANCELED")
    first = LongFlatStrategy._target_waits(strat, 61_200.1, 1)  # through it: the grace starts here
    now[0] += 1_000_000_000 if grace_passed else 999_000_000
    assert LongFlatStrategy._target_waits(strat, 61_200.1, 1) is waits
    lost = status == "SUBMITTED" and grace_passed
    assert first is (status not in ("REJECTED", "CANCELED"))
    assert strat._targets_lost == ({"T"} if lost else set()) and strat._cancel_on_accept == strat._targets_lost
    assert [e[2] for e in events] == (["resting_target_unanswered"] if lost else [])


@pytest.mark.parametrize("what, status, busy", [("target", "SUBMITTED", False), ("target", "PENDING_UPDATE", False),
                                                ("target", "PENDING_CANCEL", True), ("entry", "SUBMITTED", True)])
def test_b14u_an_unanswered_target_never_holds_up_the_stop_and_a_lost_one_is_not_waited_on(what, status, busy):
    """HEAD6 6 and 8: targets are independent of stops. A target on its way or being resized doesn't make the
    strategy busy, so the watched stop still fires (a cancel in flight still counts: a close is under way); and one
    given up after the grace is left out of the orders a close waits for (_targets_lost)."""
    from nautilus_trader.model import OrderStatus, OrderType

    from sleeve_fund.strategies.base import LongFlatStrategy

    order = SimpleNamespace(client_order_id="T", status=getattr(OrderStatus, status), is_closed=False,
                            order_type=OrderType.LIMIT if what == "target" else OrderType.MARKET)
    strat = SimpleNamespace(
        _risk_stop_id=None, strategy_id="S", _cfg=SimpleNamespace(instrument_id="I"), _unsent=lambda: [],
        cache=SimpleNamespace(orders_inflight=lambda **k: [order], orders_open=lambda **k: []),
        _resting_targets=lambda: True, decisions={"T": {"intent": "take_profit" if what == "target" else "entry"}},
        _targets_lost=set())
    strat._target_order = lambda o: LongFlatStrategy._target_order(strat, o)
    assert LongFlatStrategy._busy(strat) is busy
    assert LongFlatStrategy._working(strat) == [order]
    strat._targets_lost.add("T")
    assert LongFlatStrategy._working(strat) == []


def test_b14v_a_touch_never_starts_an_unanswered_targets_grace():
    """HEAD6 6: the grace counts from a trade through the level. Trades at it, however long, leave an unanswered
    target waiting (a limit there wouldn't have filled either); the first trade through starts it."""
    from nautilus_trader.model import OrderStatus

    from sleeve_fund.strategies.base import LongFlatStrategy

    order = SimpleNamespace(status=OrderStatus.SUBMITTED, price=SimpleNamespace(as_double=lambda: 61_200.0),
                            client_order_id="T")
    now = [10_000_000_000]
    strat = SimpleNamespace(
        _resting_targets=lambda: True, _resting_exits=lambda: {"take_profit": order}, _target_since=None,
        clock=SimpleNamespace(timestamp_ns=lambda: now[0]), _cfg=SimpleNamespace(backstop_grace_ms=1_000),
        _targets_lost=set(), _cancel_on_accept=set(), runtime=None)
    for _ in range(3):
        assert LongFlatStrategy._target_waits(strat, 61_200.0, 1) is True
        now[0] += 2_000_000_000
    assert strat._target_since is None and not strat._targets_lost
    assert LongFlatStrategy._target_waits(strat, 61_200.1, 1) is True and strat._target_since == now[0]


def test_b14w_a_reconcile_halt_leaves_the_resting_target_and_stop_and_cancels_the_rest():
    """HEAD6 5: halted, the target stays resting at the venue, as A1's stop does (only openers are cancelled)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    cancelled = []
    orders = [SimpleNamespace(client_order_id=c) for c in ("E", "T", "S", "X")]
    strat = SimpleNamespace(
        strategy_id="S1", cache=SimpleNamespace(orders_open=lambda **k: orders),
        decisions={"E": {"intent": "entry"}, "T": {"intent": "take_profit"}, "S": {"intent": "stop_loss"},
                   "X": {"intent": "exit"}},
        cancel_order=cancelled.append)
    LongFlatStrategy._cancel_all_but_exits(strat)
    assert cancelled == ["E", "X"]


@pytest.mark.parametrize("status, cancels, on_accept", [("ACCEPTED", ["T"], set()), ("PARTIALLY_FILLED", ["T"], set()),
                                                        ("SUBMITTED", [], {"T"}), ("PENDING_CANCEL", [], set())])
def test_b14x_every_close_cancels_the_resting_target_first_and_sells_only_once_it_has_gone(status, cancels, on_accept):
    """HEAD6 5: a close (a signal's exit, a flatten, a halt's close) cancels the resting target before anything is
    sold; one not yet at the venue is cancelled as soon as it gets there. Nothing is sent at market meanwhile."""
    from nautilus_trader.model import OrderStatus, OrderType

    from sleeve_fund.strategies.base import LongFlatStrategy

    cancelled, sent = [], []
    target = SimpleNamespace(client_order_id="T", status=getattr(OrderStatus, status), order_type=OrderType.LIMIT)
    strat = SimpleNamespace(_pending_exit=None, pair=None, _drop_kept=lambda earn: None, _working=lambda: [target],
                            _cancel_on_accept=set(), cancel_order=cancelled.append,
                            _submit=lambda *a, **k: sent.append(a))
    LongFlatStrategy._sell_all(strat, "pm_flatten", "Flattened", {})
    assert cancelled == cancels and strat._cancel_on_accept == on_accept and sent == []
    assert strat._pending_exit == ("pm_flatten", "Flattened", {})  # sold once every working order has closed


@pytest.mark.parametrize("status, exits_only", [("halted", False), ("paused", False), ("flattening", False)])
def test_b14y_halted_or_paused_nothing_new_rests_and_nothing_resting_is_cancelled(status, exits_only):
    """HEAD6 5: halt and pause place no new target (and resize none) but cancel none either: the one resting stays."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    calls = []
    strat = SimpleNamespace(
        _cfg=SimpleNamespace(), _backtest=False, _resting_stops=lambda: False, _resting_targets=lambda: True,
        _tp_frac=0.01, _target_refused_at=None, _entry_px=60_000.0, _exits_only=exits_only,
        runtime=SimpleNamespace(status=status), _resting_exits=lambda: calls.append("read") or {},
        _place_exits=lambda plan: calls.append(("place", plan)), cancel_order=lambda c: calls.append(("cancel", c)))
    LongFlatStrategy._rest_exits(strat)
    assert calls == []
    strat.runtime.status = "running"
    LongFlatStrategy._rest_exits(strat)
    assert calls == ["read", ("place", {"take_profit": 0.01})]


def test_b14z_stopping_the_strategy_cancels_its_resting_target_with_every_other_order():
    """HEAD6 5 (HoE: every closing path, Stop-strategy included): a stopped Strategy leaves no target resting at the
    venue with nothing watching it; a restart rests one again for what is held (b14n)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    calls = []
    strat = SimpleNamespace(_gate_done=None, _drop_kept=lambda: calls.append("drop"), _maker={},
                            _cfg=SimpleNamespace(instrument_id="I"), runtime=None, recorder=None,
                            cancel_all_orders=lambda iid: calls.append(("cancel all", iid)))
    LongFlatStrategy.on_stop(strat)
    assert ("cancel all", "I") in calls


# ---- QD's independent pass (for QD3): OFF before and after, and the hunks no cell above held -------------------------

def test_b14aa_off_orders_and_fills_are_byte_identical_to_before_b14_on_both_engines():
    """With resting targets off, a target path and a stop path give the orders and fills the tree before B-14 gave, on
    the paper replay and the backtest engine (tests/data/golden/b14_off_662e8a0b.json, made by b14_off_golden on the D3
    base 662e8a0b, which has no B-14; TEC-B14AA-REBASE: it differs from b14_off_274ecd88.json, kept, only in B-15's
    liquidation prices, b14am)."""
    import json
    from pathlib import Path

    from b14_off_golden import off_runs

    want = json.loads((Path(__file__).parent / "data" / "golden" / "b14_off_662e8a0b.json").read_text())
    got = json.loads(off_runs())
    assert set(got) == {"paper_target", "paper_stop", "backtest_target", "backtest_stop"}
    assert {o["order_type"] for o in got["paper_target"]["orders"] if o["intent"] == "take_profit"} == {"MARKET"}
    assert any(o["intent"] == "stop_loss" and o["order_type"] == "MARKET" for o in got["paper_stop"]["orders"])
    assert "take_profit" in got["backtest_target"]["intents"] and "stop_loss" in got["backtest_stop"]["intents"]
    for k in want:
        assert got[k] == want[k], k


@pytest.mark.parametrize("bid, ask, last, through", [(60_599.9, 60_600.0, None, False),  # only the ask at the level
                                                     (60_600.0, 60_600.1, None, True),  # the bid at it: post-only takes
                                                     (None, None, 60_600.0, True),  # unquoted: the last trade at it
                                                     (None, None, 60_599.9, False)])
def test_b14ab_a_long_targets_level_is_judged_against_the_bid_at_or_through_it(bid, ask, last, through):
    """A long's target sells: a post-only sell at the level takes liquidity once the best bid is at or above it (the
    ask alone there leaves it resting). A short's buys against the ask."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    strat = SimpleNamespace(_bid=bid, _ask=ask, _last_close=last)
    assert LongFlatStrategy._through_target(strat, 60_600.0, 1) is through
    short = SimpleNamespace(_bid=59_399.9 if bid else None, _ask=59_400.0 if ask else None,
                            _last_close=last and 59_400.0)
    assert LongFlatStrategy._through_target(short, 59_400.0, -1) is (bool(ask) or bool(last))


@pytest.mark.parametrize("intent, refused", [("take_profit", True), ("entry", False), ("exit", False)])
def test_b14ac_only_a_refused_resting_target_counts_as_one_never_a_maker_entry(intent, refused):
    """A post-only maker entry refused is a refused entry as before, never one of the targets' refusals in a row."""
    from nautilus_trader.model import OrderType

    from sleeve_fund.strategies.base import LongFlatStrategy

    order = SimpleNamespace(order_type=OrderType.LIMIT)
    strat = SimpleNamespace(_resting_targets=lambda: True, runtime=object(),
                            cache=SimpleNamespace(order=lambda c: order),
                            decisions={"T": {"intent": intent}})
    assert LongFlatStrategy._target_refused(strat, SimpleNamespace(client_order_id="T")) is refused


def _watching(price, since):
    from sleeve_fund.strategies.base import LongFlatStrategy

    sent = []
    strat = SimpleNamespace(
        _safety_pending=None, _stop_frac=None, _tp_frac=0.01, _entry_px=60_000.0, _busy=lambda: False, _backtest=False,
        _restore=None, _entry_side=1, _backstop_since=None, _target_since=since, _target_waits=lambda p, s: False,
        log=SimpleNamespace(info=lambda m: None), runtime=None, _margin=False, _lock_exit_leg=lambda s: None,
        _exit_at_market=lambda *a: sent.append(a[0]))
    return LongFlatStrategy._check_exits(strat, price), strat, sent


@pytest.mark.parametrize("price, closes", [(60_500.0, False), (60_700.0, True)])
def test_b14ad_an_unanswered_targets_grace_ends_with_the_trade_through_and_with_the_close(price, closes):
    """Back below the level, the grace counts again from the next trade through; and the watched target's close ends
    it, so the next position's target gets its own grace, never one already used up."""
    sent, strat, closed = _watching(price, since=5)
    assert sent is closes and closed == (["take_profit"] if closes else [])
    assert strat._target_since is None


def test_b14ae_a_maker_entry_on_its_way_still_holds_up_the_watched_exits():
    """Only the resting target is left out of _busy: a post-only entry (a LIMIT too) sent and unanswered is busy."""
    from nautilus_trader.model import OrderStatus, OrderType

    from sleeve_fund.strategies.base import LongFlatStrategy

    order = SimpleNamespace(client_order_id="E", status=OrderStatus.SUBMITTED, is_closed=False,
                            order_type=OrderType.LIMIT)
    strat = SimpleNamespace(
        _risk_stop_id=None, strategy_id="S", _cfg=SimpleNamespace(instrument_id="I"), _unsent=lambda: [],
        cache=SimpleNamespace(orders_inflight=lambda **k: [order], orders_open=lambda **k: []),
        _resting_targets=lambda: True, decisions={"E": {"intent": "entry"}}, _targets_lost=set())
    strat._target_order = lambda o: LongFlatStrategy._target_order(strat, o)
    assert LongFlatStrategy._busy(strat) is True


def test_b14af_each_refused_target_adds_one_to_the_refusals_in_a_row_the_runtime_alerts_on():
    from sleeve_fund.strategies.base import LongFlatStrategy

    told = []
    strat = SimpleNamespace(
        _benign_reject=lambda e: False, _target_refused=lambda e: True, _target_refusals=0, _target_refused_at=None,
        _entry_px=60_000.0, _restore_id="T", runtime=SimpleNamespace(on_target_refused=lambda c, m, n: told.append(n)),
        _claimed=lambda e: False, _backtest=False, _cfg=SimpleNamespace())  # TEC-B14-STUBS: paper, order client off
    for _ in range(3):
        LongFlatStrategy.on_order_rejected(strat, SimpleNamespace(client_order_id="T", reason="would take liquidity"))
        strat._restore_id = "T"
    assert told == [1, 2, 3] and strat._target_refusals == 3 and strat._target_refused_at == 60_000.0


def test_b14ag_a_target_still_to_place_beside_a_resting_stop_is_placed_and_the_stop_resized():
    """With A1's stop resting and no target yet (one through its level when the position opened), the target is
    placed for the position as it is now; the stop is left to its own resize."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    calls = []
    stop = object()
    strat = SimpleNamespace(
        _cfg=SimpleNamespace(), _backtest=False, _resting_stops=lambda: True, _resting_targets=lambda: True,
        _stop_frac=0.01, _tp_frac=0.02, _target_refused_at=None, _entry_px=60_000.0, _exits_only=False,
        runtime=SimpleNamespace(status="running"), _resting_exits=lambda: {"stop_loss": stop},
        _place_exits=lambda plan: calls.append(("place", plan)), clock=SimpleNamespace(timestamp_ns=lambda: 7),
        _resized_ns=None, _resize_exits=lambda resting, plan: calls.append(("resize", resting, plan)))
    LongFlatStrategy._rest_exits(strat)
    assert calls == [("place", {"take_profit": 0.02}),
                     ("resize", {"stop_loss": stop}, {"stop_loss": 0.01, "take_profit": 0.02})]


@pytest.mark.parametrize("through", [True, False])
def test_b14ah_a_price_already_through_the_level_places_no_resting_target(through):
    from nautilus_trader.model import OrderSide

    from sleeve_fund.strategies.base import LongFlatStrategy

    judged, sent = [], []
    strat = SimpleNamespace(
        _cfg=SimpleNamespace(instrument_id="I"), _position_qty=lambda free=False: Decimal("0.1"),
        _lot=lambda: Decimal("0.00000001"), _min_qty=lambda: Decimal("0.00000001"),
        instrument=SimpleNamespace(size_precision=8, price_precision=1), _entry_side=1, _entry_px=60_000.0,
        clock=SimpleNamespace(timestamp_ns=lambda: 7), _through_target=lambda lv, s: judged.append((lv, s)) or through,
        order_factory=SimpleNamespace(limit=lambda **k: SimpleNamespace(client_order_id="T", **k)), decisions={},
        runtime=None, _open_target=lambda c: None, _sent=[], submit_order=sent.append, _backtest=False)
    LongFlatStrategy._place_exits(strat, {"take_profit": 0.01})
    assert judged == [(pytest.approx(60_600.0), 1)]
    assert [(o.order_side, o.price.as_double()) for o in sent] == ([] if through else [(OrderSide.SELL, 60_600.0)])


@pytest.mark.parametrize("intent, journaled", [("take_profit", True), ("stop_loss", True), ("entry", False)])
def test_b14ai_the_venue_resizing_a_resting_target_journals_its_new_quantity(intent, journaled):
    from sleeve_fund.strategies.base import LongFlatStrategy

    rows = []
    store = SimpleNamespace(update_order=lambda c, **k: rows.append((c, k)))
    strat = SimpleNamespace(_resizing={}, decisions={"T": {"intent": intent}}, _resize_if_due=lambda: None,
                            runtime=SimpleNamespace(store=store))
    LongFlatStrategy.on_order_updated(strat, SimpleNamespace(client_order_id="T",
                                                             quantity=SimpleNamespace(as_double=lambda: 0.08)))
    assert rows == ([("T", {"qty": 0.08, "message": None})] if journaled else [])


@pytest.mark.parametrize("targets, stops, rests", [(True, False, True), (False, True, True), (False, False, False)])
def test_b14aj_a_resize_left_due_is_done_with_resting_targets_alone(targets, stops, rests):
    from sleeve_fund.strategies.base import LongFlatStrategy

    calls = []
    strat = SimpleNamespace(_resize_due=True, _backtest=False, _resting_stops=lambda: stops,
                            _resting_targets=lambda: targets, _pending_exit=None, _rest_exits=lambda: calls.append(1))
    LongFlatStrategy._resize_if_due(strat)
    assert (calls == [1]) is rests and strat._resize_due is not rests


def test_b14ak_a_refused_target_is_marked_rejected_and_benign_on_its_row():
    from sleeve_fund.paper.runtime import BENIGN_REJECT, SleeveRuntime

    rows, signals = [], []
    store = SimpleNamespace(update_order=lambda oid, **k: rows.append((oid, k["status"])),
                            merge_order_signal=lambda oid, sig: signals.append((oid, sig)),
                            event=lambda *a, **k: None)
    SleeveRuntime.on_target_refused(SimpleNamespace(store=store, name=NAME, now=lambda: None), "T", "post-only", 1)
    assert rows == [("T", "rejected")] and signals == [("T", {BENIGN_REJECT: True})]


@pytest.mark.parametrize("setting", [{}, {"resting_targets": False}])
def test_b14al_off_survives_a_restart(setting):
    """D4 gate (Advisor HEAD6-B14, HoE): with resting targets off, a restart holding a long rests no target, before or
    after, and the restarted Strategy keeps the setting off; its target is watched and closes at market, as before."""
    from datetime import timedelta

    from paper_scenarios import NAME as SCENARIO
    from test_prop_inv import HOUR0, _at, _start_ns
    from test_r_i5_1_exit_lock import _ticks
    from test_r_i5_2_missed_candle import _ab

    from sleeve_fund.supervisor import check_resting_entries

    a, b, _, _ = _ab("ping_pong", {**HELD, "take_profit": 0.05, **setting},
                     _ticks([(5, 0.0, 0.05), (30, 0.004, 0.05)]),
                     lambda a: _at(_start_ns(HOUR0)) + timedelta(minutes=20))
    for store in (a, b):
        assert store.journal_book(SCENARIO, 10_000)["qty"] > 0  # held through the restart
        assert not _targets(store.orders(SCENARIO))  # no resting target, ever
        assert not [o for o in store.orders(SCENARIO) if o["order_type"] == "LIMIT" and o["intent"] == "take_profit"]
        s = store.sleeve(SCENARIO)
        assert not s.params.get("resting_targets")
        check_resting_entries(s)  # the stored settings start as they are: nothing to refuse


def test_b14am_the_rebased_off_golden_moves_only_b15s_liquidation_prices_to_the_oracle():
    """TEC-B14AA-REBASE (HoE/HoQA): against the 274ecd88 golden, the D3 one changes liquidation_px and
    position_liquidation_px and nothing else (orders, fills, fees and quantities as they were), and each new figure is
    the B-15 oracle worked here: isolated margin less the entry fee held, the nearer of the maintenance on value and
    on entry value (VS-1-TABLE), on the instrument's 0.1 tick (the order's nearest, the position's the safer side)."""
    import json
    from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN
    from pathlib import Path

    golden = Path(__file__).parent / "data" / "golden"
    old = json.loads((golden / "b14_off_274ecd88.json").read_text())
    new = json.loads((golden / "b14_off_662e8a0b.json").read_text())
    LIQ = ("liquidation_px", "position_liquidation_px")

    def without(rows):
        return [{**r, "signal": {k: v for k, v in (r.get("signal") or {}).items() if k not in LIQ}}
                if isinstance(r, dict) and "signal" in r else r for r in rows]

    assert set(old) == set(new)
    for k in old:
        assert set(old[k]) == set(new[k]), k
        for part in old[k]:
            a, b = old[k][part], new[k][part]
            assert (without(a) == without(b)) if isinstance(a, list) else a == b, (k, part)

    def oracle(q, e, fee, lev=Decimal(2), mm=Decimal("0.005")):  # balanced 2x; test_long_short's 0.5% maintenance
        q, e, fee = Decimal(str(q)), Decimal(str(e)), Decimal(str(fee))
        cash = abs(q) * e / lev - fee - q * e
        on_value, on_entry = cash / (mm * abs(q) - q), (abs(q) * e * mm - cash) / q
        p = max(on_value, on_entry) if q > 0 else min(on_value, on_entry)
        tick = Decimal("0.1")
        return float(p.quantize(tick, ROUND_HALF_EVEN)), float(p.quantize(tick, ROUND_CEILING if q > 0 else ROUND_FLOOR))

    for k in ("paper_target", "paper_stop"):
        filled = [o for o in new[k]["orders"] if Decimal(str(o["filled_qty"]))]  # the golden keeps them as text
        entries = [o for o in filled if o["intent"] == "entry"]
        for entry in entries:
            q = Decimal(str(entry["filled_qty"])) * (1 if entry["side"] == "BUY" else -1)
            nearest, safer = oracle(q, entry["avg_px"], entry["fee"])
            assert (entry["signal"]["liquidation_px"], entry["signal"]["position_liquidation_px"]) == (nearest, safer)
            exit_ = filled[filled.index(entry) + 1]
            assert exit_["signal"]["liquidation_px"] == safer
    assert [o["signal"]["liquidation_px"] for o in new["paper_target"]["orders"] if "liquidation_px" in o["signal"]] == [
        30330.0, 30330.1, 90460.2, 90460.1]
