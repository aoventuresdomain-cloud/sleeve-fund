"""RESUME-LEG restore (DA, head 3; plan-tonight-2130.md): a multi-asset Strategy restarted over open positions puts
every leg back from its own journal: its position (side, quantity, entry price), the stop and target its own entry
journaled, and its own marks. A leg never takes another leg's stop or book, and a flat leg comes back flat. Each leg is
a row of its own ("trend-x~ETH"), so this pins that every restore read is keyed by the leg, not the parent. CI also runs
these against Postgres via TEST_DATABASE_URL."""

from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper.runtime import SleeveRuntime

from test_multi_asset_legs import _fund
from test_sleeve_runtime import store as _store_fixture

store = _store_fixture  # the same journal, on Postgres when TEST_DATABASE_URL is set

T0 = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=2)
COST = 0.0026
ASSETS = ["BTC/USD", "ETH/USD", "SOL/USD"]
# Each open leg's own entry: price, quantity, side and the exit plan journaled with it (distinct per leg on purpose)
OPEN = {
    "trend-x": dict(px=60_000.0, qty=0.5, side="BUY", stop_frac=0.02, tp_frac=0.05),
    "trend-x~ETH": dict(px=2_500.0, qty=4.0, side="BUY", stop_frac=0.035, tp_frac=0.08),
}


def _legs(store):
    _fund(store)
    rows = store.create_legs(name="trend-x", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL", assets=ASSETS,
                             params={}, starting_balance=0, universe_date="2026-10-15", venue="kraken")
    return [r.name for r in rows]


def _enter(store, name, px, qty, side, stop_frac, tp_frac, minutes):
    rt = SleeveRuntime(store, name, now=lambda: T0 + timedelta(minutes=minutes))
    oid = f"E-{name}"
    rt.on_order(order_id=oid, side=side, qty=qty, intent="entry", reason="Signal to be long",
                signal={"stop_frac": stop_frac, "tp_frac": tp_frac, "stop_basis": f"{stop_frac:.1%} below entry",
                        "stop_cfg": {"stop_loss": stop_frac}})
    rt.on_fill(side=side, qty=qty, price=px, fee=round(px * qty * COST, 2), order_id=oid, trade_id=f"T-{name}")
    return rt


def _restart(store, instrument, name, minutes=60):
    """The leg's strategy over its journal after a restart, as on_start does: the runtime's book, then the plan."""
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=COST, stop_loss=0.10, take_profit=0.20))
    strat.runtime = SleeveRuntime(store, name, now=lambda: T0 + timedelta(minutes=minutes))
    strat.runtime.on_start(COST, now=lambda: T0 + timedelta(minutes=minutes))
    book = strat.runtime.book
    if book["entry_px"] and book["qty"]:
        strat._entry_px, strat._entry_qty = book["entry_px"], abs(book["qty"])
        strat._entry_side = 1 if book["qty"] > 0 else -1
        strat._restore_plan()
    return strat


def test_every_leg_comes_back_with_its_own_position_and_a_flat_leg_comes_back_flat(store, instrument):
    names = _legs(store)
    assert names == ["trend-x", "trend-x~ETH", "trend-x~SOL"]
    for n, (name, e) in enumerate(OPEN.items()):
        _enter(store, name, minutes=n, **e)
    for name in names:
        book = _restart(store, instrument, name).runtime.book
        if name in OPEN:
            assert (book["qty"], book["entry_px"]) == pytest.approx((OPEN[name]["qty"], OPEN[name]["px"])), name
        else:
            assert book["qty"] == 0 and not book["entry_px"], name


def test_every_open_leg_works_to_the_stop_and_target_its_own_entry_journaled(store, instrument):
    names = _legs(store)
    for n, (name, e) in enumerate(OPEN.items()):
        _enter(store, name, minutes=n, **e)
    for name in OPEN:
        strat = _restart(store, instrument, name)
        assert (strat._stop_frac, strat._tp_frac) == pytest.approx((OPEN[name]["stop_frac"], OPEN[name]["tp_frac"])), name
        assert strat._plan_entry["order_id"] == f"E-{name}", name
    flat = _restart(store, instrument, names[2])
    assert flat.runtime.book["qty"] == 0 and not flat._entry_px  # nothing to restore, no plan taken from a sibling


def test_a_stop_edit_on_one_leg_moves_only_that_legs_stop_after_the_restart(store, instrument):
    _legs(store)
    for n, (name, e) in enumerate(OPEN.items()):
        _enter(store, name, minutes=n, **e)
    store.event("trend-x~ETH", "info", "exits_change", "Settings changed by PM: stop 3.5% -> 1.5%. Restarting to apply them.",
                ts=T0 + timedelta(minutes=30))
    btc = _restart(store, instrument, "trend-x")
    assert btc._stop_frac == pytest.approx(OPEN["trend-x"]["stop_frac"])  # the other leg's edit never reaches it
    assert not [e for e in store.events("trend-x", limit=200) if e["kind"] in ("exits_applied", "stop_reset")]


def test_every_leg_keeps_its_own_marks_over_a_restart(store, instrument):
    names = _legs(store)
    for n, (name, e) in enumerate(OPEN.items()):
        rt = _enter(store, name, minutes=n, **e)
        rt.final_mark(equity=e["px"] * e["qty"], cash=0.0, qty=e["qty"], price=e["px"] * 1.01)
    before = {n: store.equity_series(n) for n in names}
    for name in names:
        _restart(store, instrument, name)
    for name in OPEN:
        last = store.last_equity(name)
        assert float(last["price"]) == pytest.approx(OPEN[name]["px"] * 1.01), name  # its own mark, not another's
        assert float(last["qty"]) == pytest.approx(OPEN[name]["qty"]), name
        assert store.equity_series(name)[: len(before[name])] == before[name], name  # nothing rewritten
    assert store.last_equity("trend-x~SOL") is None or float(store.last_equity("trend-x~SOL")["qty"]) == 0


def test_a_restore_read_keyed_on_the_parent_turns_these_cells_red(store, instrument, monkeypatch):
    """The deliberate reader: were the journal reads keyed on the parent's row instead of the leg's, the ETH leg would
    come back with BTC's stop. These cells see it, so they are not green by construction."""
    from sleeve_fund.store import Store

    _legs(store)
    for n, (name, e) in enumerate(OPEN.items()):
        _enter(store, name, minutes=n, **e)
    for method in ("orders", "exit_plan"):
        real = getattr(Store, method)
        monkeypatch.setattr(Store, method,
                            lambda self, sleeve, *a, _real=real, **kw: _real(self, sleeve.split("~")[0], *a, **kw))
    eth = _restart(store, instrument, "trend-x~ETH")
    assert eth._stop_frac == pytest.approx(OPEN["trend-x"]["stop_frac"]) != OPEN["trend-x~ETH"]["stop_frac"]


# --- perp legs (Advisor RESUME-CELLS, cell 6) --------------------------------------------------------------------------
# Two perp legs of one Strategy (one settings block, so one leverage cap: create_legs), each entered at its own price
# and size, each with the liquidation price its own fill journaled (deliberately not the formula's, so a value worked
# out afresh, or a sibling's, shows) and a stop looser than half the way to it.
PERP_OPEN = {
    "perp-x": dict(px=60_000.0, qty=0.5, side="BUY", stop_frac=0.30, tp_frac=0.40, liq=41_000.0),
    "perp-x~ETH": dict(px=2_500.0, qty=4.0, side="SELL", stop_frac=0.25, tp_frac=0.30, liq=3_200.0),
}


def _perp_legs(store):
    _fund(store)
    return [r.name for r in store.create_legs(
        name="perp-x", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL", assets=ASSETS,
        params={"market": "perp", "allow_short": True}, starting_balance=0, universe_date="2026-10-15",
        venue="kraken", risk_profile="aggressive")]


def _perp_enter(store, name, px, qty, side, stop_frac, tp_frac, liq, minutes):
    rt = SleeveRuntime(store, name, now=lambda: T0 + timedelta(minutes=minutes))
    oid = f"E-{name}"
    rt.on_order(order_id=oid, side=side, qty=qty, intent="entry", reason="Signal",
                signal={"stop_frac": stop_frac, "tp_frac": tp_frac, "stop_basis": f"{stop_frac:.0%} from entry",
                        "stop_cfg": {"stop_loss": stop_frac}})
    rt.on_fill(side=side, qty=qty, price=px, fee=round(px * qty * COST, 2), order_id=oid, trade_id=f"T-{name}")
    store.merge_order_signal(oid, {"position_liquidation_px": liq})  # as _set_liq journals it after the fill


def _perp_restart(store, instrument, name):
    from nautilus_trader.model import BarType

    from sleeve_fund.strategies import REGISTRY

    cls, config_cls = REGISTRY["buy_and_hold"]
    strat = cls(config_cls(instrument_id=instrument.id, bar_type=BarType.from_str(f"{instrument.id}-1-DAY-LAST-EXTERNAL"),
                           assumed_taker_fee=COST, stop_loss=0.10, take_profit=0.20, market="perp", allow_short=True))
    strat.runtime = SleeveRuntime(store, name, now=lambda: T0 + timedelta(minutes=60))
    strat.runtime.on_start(COST, now=lambda: T0 + timedelta(minutes=60))
    book = strat.runtime.book
    strat._entry_px, strat._entry_qty = book["entry_px"], abs(book["qty"])
    strat._entry_side = 1 if book["qty"] > 0 else -1
    strat.instrument = instrument  # as on_start sets it: the restore counts the position's fills in its lots
    strat._restore_plan()
    strat._mark = lambda price=None: (0.0, book["cash"], book["qty"], book["entry_px"])  # the leg's own book
    return strat


def _half_way_cap(e, share):
    side = 1 if e["side"] == "BUY" else -1
    return side * (1 - (e["px"] - side * share * abs(e["px"] - e["liq"])) / e["px"])


def test_every_perp_leg_comes_back_on_its_own_margin_liquidation_price_and_half_way_stop(store, instrument):
    """Advisor RESUME-CELLS cell 6: each perp leg comes back with its own position and side, the leverage cap and
    isolated margin of its Strategy, the liquidation price ITS OWN fill journaled, and its stop re-checked against half
    the way to that price: held there (STOP-LOOSEN-1), never measured from a sibling's."""
    names = _perp_legs(store)
    for n, (name, e) in enumerate(PERP_OPEN.items()):
        _perp_enter(store, name, minutes=n, **e)
    for name, e in PERP_OPEN.items():
        strat = _perp_restart(store, instrument, name)
        side = 1 if e["side"] == "BUY" else -1
        assert strat._margin and strat._entry_side == side and strat._entry_px == pytest.approx(e["px"]), name
        assert strat._max_leverage() == strat.runtime.profile.max_leverage, name  # isolated, at its Strategy's cap
        assert strat._liq_px == pytest.approx(e["liq"]), name  # its own journaled price, not the formula's
        assert strat._stop_frac == pytest.approx(e["stop_frac"]), name  # restored as journaled ...
        held, against = strat._capped_stop(strat._stop_frac)  # ... then held inside half the way to its own price
        assert against == pytest.approx(e["liq"]), name
        cap = _half_way_cap(e, strat.runtime.profile.stop_to_liquidation)
        assert held == pytest.approx(cap) and cap < e["stop_frac"], name
    assert _perp_restart(store, instrument, names[2]).runtime.book["qty"] == 0  # the flat leg stays flat


def test_crossing_the_legs_liquidation_prices_turns_cell_6_red(store, instrument, monkeypatch):
    """The deliberate mutation: the journal's orders read for the other perp leg. The ETH leg then no longer comes
    back with its own liquidation price (the BTC leg's orders hold no short to take one from), and cell 6's own
    check sees it."""
    from sleeve_fund.store import Store

    _perp_legs(store)
    for n, (name, e) in enumerate(PERP_OPEN.items()):
        _perp_enter(store, name, minutes=n, **e)
    swap = {"perp-x": "perp-x~ETH", "perp-x~ETH": "perp-x"}
    real = Store.orders
    monkeypatch.setattr(Store, "orders", lambda self, sleeve, *a, **kw: real(self, swap.get(sleeve, sleeve), *a, **kw))
    eth = _perp_restart(store, instrument, "perp-x~ETH")
    assert eth._liq_px != pytest.approx(PERP_OPEN["perp-x~ETH"]["liq"])
