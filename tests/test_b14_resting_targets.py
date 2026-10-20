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


def _run(tmp_path, *, targets=None, stops=False, legs=UP, tp=0.01):
    params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.01, "take_profit": tp, **PERP}
    if targets is not None:
        params["resting_targets"] = targets
    if stops:
        params["resting_stops"] = True
    _record(tmp_path / "a.jsonl.gz", _meta(10_000, params), legs)
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
        _backtest=False, _restore=None, _entry_side=1, _backstop_since=None, _target_rests=lambda: rests,
        _backstop_waits=lambda price, side: False, runtime=None, _margin=False,
        log=SimpleNamespace(info=lambda *a: None), _lock_exit_leg=lambda *a: None,
        _exit_at_market=lambda *a: sent.append(a))
    assert LongFlatStrategy._check_exits(strat, 60_600.0) is closes
    assert bool(sent) is closes and (not sent or sent[0][0] == "take_profit")
