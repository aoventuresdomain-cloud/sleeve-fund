"""Stale data (Independent Quant Advisor, 7 Oct, HoE 02:09): a fixed 5 minutes for every candle interval. Pinned here:
(1) it is measured on the venue feed of the traded instrument (its trades and quotes): candles still arriving from the
hub, or another instrument's prints, do not keep it fresh, and it is its own cause, never funding; (2) it clears on the
first fresh trade or quote; (3) it cancels resting entries. (4), exits and stops never held, is
test_choke_flip_after_target.test_stale_data_never_holds_a_stop_or_a_target.

Also here: a fired stop's watched row stays linked to the market stop-loss sent for it when _sell_all prunes the sent
list on the way (Code Reviewer on #182)."""
from __future__ import annotations

import pytest
import test_hub_146_qa as qa
from test_exposure_gate_xfails import (M, STD_T0, Plan, _fills, _live_openers, _offline, _orders,  # noqa: F401
                                      run, store)  # _offline: the exposure-gate strategy, offline (pin 3)
from test_hub_146_qa import START, S, _probe, _probe_classes, adverse, flat_prices, paper, shape  # noqa: F401

from sleeve_fund.strategies.base import STALE_PRICE_WARN_MINUTES

HOLE = (20 * 60, 28 * 60)  # seconds: no trade or quote for 8 minutes (past the 5-minute warning, short of a restart)


def _recording(monkeypatch, record):
    """The probe, recording through `record(strategy, what, before)` around each watchdog look and each trade or
    quote; `before` is the hold before the handler ran."""
    from sleeve_fund.strategies import REGISTRY

    probe, config = _probe_classes()

    class Recording(probe):
        def _feed_dead(self) -> bool:
            out = super()._feed_dead()
            record(self, "watchdog", None)
            return out

        def on_trade(self, tick) -> None:
            before = dict(self.runtime.holds)
            super().on_trade(tick)
            record(self, "trade", before)

        def on_quote(self, quote) -> None:
            before = dict(self.runtime.holds)
            super().on_quote(quote)
            record(self, "quote", before)

    monkeypatch.setitem(REGISTRY, "probe", (Recording, config))


def _codes(why) -> tuple:
    return tuple(getattr(why, "codes", ()) or ())


def _other_instruments_trade_in_the_hole(monkeypatch):
    """Another instrument at the same venue keeps printing every second through the hole, in the same engine."""
    from nautilus_trader.backtest import BacktestEngine
    from nautilus_trader.model import AggressorSide, Price, Quantity, QuoteTick, TradeId, TradeTick

    from sleeve_fund.venues import venue

    other = venue("KRAKEN").instrument("ETH", "USD", price_precision=1)
    real = BacktestEngine.add_data

    def add_data(self, data, *a, **kw):
        self.add_instrument(other)
        extra = []
        for s in range(*HOLE):
            t = START + s * S
            extra.append(TradeTick(other.id, Price(2_500.0, 1), Quantity(1.0, 8), AggressorSide.BUY, TradeId(f"e{s}"),
                                   t, t + 1000))
            extra.append(QuoteTick(other.id, Price(2_499.9, 1), Price(2_500.1, 1), Quantity(1, 8), Quantity(1, 8),
                                   t + 2000, t + 3000))
        return real(self, list(data) + extra, *a, **kw)

    monkeypatch.setattr(BacktestEngine, "add_data", add_data)


# (1) ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("other", [False, True], ids=["alone", "another-instrument-printing"])
def test_stale_data_is_measured_on_the_traded_instruments_own_feed(monkeypatch, other):
    """The hub's candles keep arriving through the hole (only the trades and quotes are missing), and with `other`
    another instrument prints every second through it: neither keeps this strategy's data fresh. From 5 minutes after
    its last trade or quote, entries are held for stale data alone (funding is a separate cause, O17b), with the age
    given."""
    if other:
        _other_instruments_trade_in_the_hole(monkeypatch)
    looks = []

    def record(st, what, before):
        if what == "watchdog":
            looks.append((st.clock.timestamp_ns(), st.runtime.entry_blocked()))

    _recording(monkeypatch, record)
    got = paper(flat_prices(40), side=1, perp=True, profile="conservative", leave=35,
                gone=frozenset(range(*HOLE)))
    last_seen = START + (HOLE[0] - 1) * S
    warn = last_seen + STALE_PRICE_WARN_MINUTES * 60 * S
    in_hole = [(t, b) for t, b in looks if START + HOLE[0] * S <= t < START + HOLE[1] * S]
    assert in_hole, "set-up: no watchdog look inside the hole"
    early = [(t, b) for t, b in in_hole if t < warn]
    late = [(t, b) for t, b in in_hole if t >= warn]
    assert early and late, f"set-up: looks either side of the 5 minutes: {[(t - START) / S for t, _ in in_hole]}"
    assert all("stale_data" not in _codes(why) for _, (_, why) in early), f"stale before 5 minutes: {early}"
    for t, (blocked, why) in late:
        assert blocked and _codes(why) == ("stale_data",), f"at +{(t - START) / S:.0f} s: {why!r}"
        assert f"{(t - last_seen) / S:.0f} s old" in str(why), f"no age in {str(why)!r}"
    took = [b for b in got.bars if START + HOLE[0] * S < b[0] <= START + HOLE[1] * S]
    assert took, "set-up: the hub's candles did not arrive through the hole"


# (2) ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("first", ["trade", "quote"])
def test_stale_data_clears_on_the_first_fresh_trade_or_quote(monkeypatch, first):
    """When data resumes, the first trade or quote of the traded instrument clears the hold, in its own handler: held
    as it arrives, not held once it is taken. `quote`: the first second back carries a quote and no trade."""
    if first == "quote":
        from nautilus_trader.model import TradeTick

        real = qa.ticks
        back = START + HOLE[1] * S
        monkeypatch.setattr(qa, "ticks", lambda inst, prices, gone=frozenset(): [
            d for d in real(inst, prices, gone) if not (isinstance(d, TradeTick) and d.ts_event == back)])
    seen = []

    def record(st, what, before):
        if what != "watchdog" and before is not None and st.clock.timestamp_ns() >= START + HOLE[1] * S and not seen:
            seen.append((what, before, dict(st.runtime.holds), st.runtime.entry_blocked()))

    _recording(monkeypatch, record)
    paper(flat_prices(40), side=1, perp=True, profile="conservative", leave=35, gone=frozenset(range(*HOLE)))
    assert seen, "set-up: no trade or quote after the hole"
    what, before, after, (blocked, why) = seen[0]
    assert what == first, f"set-up: the first fresh data was a {what}"
    assert "stale_data" in before, f"set-up: not held as data resumed: {before}"
    assert "stale_data" not in after, f"the first fresh {first} did not clear the hold: {after}"
    assert "stale_data" not in _codes(why), f"still blocked for stale data after the first fresh {first}: {why!r}"


# (3) ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("hole", [False, True], ids=["control", "stale"])
def test_stale_data_cancels_a_resting_entry_and_it_never_fills(tmp_path, store, monkeypatch, hole):
    """A stop entry rests from t0+5, 1 % above the price; the price steps 2 % up at t0+35. With no trade or quote from
    t0+20 to t0+28 the hold starts at t0+25 and the resting entry is cancelled then, so the step after data resumes
    fills nothing. The control, with no hole, fills it."""
    plan = Plan(t0=STD_T0, tag="sr")
    base = lambda s: 60_000 + s * 0.01  # noqa: E731
    p0 = base(5 * 60)
    plan.price = lambda s: base(s) * (1.02 if s >= 35 * 60 else 1.0)
    plan.rest.append((M(plan.t0, 5), 1, round(p0 * 1.01, 1), 0.05))
    if hole:
        plan.holes.append(HOLE)
    run(tmp_path, store, plan, monkeypatch)
    entries = [o for o in _orders(store) if o["intent"] == "entry"]
    assert len(entries) == 1 and entries[0]["order_type"] == "STOP_MARKET", f"set-up: the resting entry: {entries}"
    filled = {f["order_id"] for f in _fills(store)}
    if not hole:
        assert entries[0]["order_id"] in filled, f"control: the resting entry did not fill: {entries}"
        return
    assert entries[0]["status"] in ("canceled", "cancelled"), f"the resting entry was not cancelled: {entries}"
    assert entries[0]["order_id"] not in filled, f"the resting entry filled after stale data: {entries}"
    assert not _live_openers(store, M(plan.t0, 25).to_pydatetime()), "a resting entry outlived the stale-data hold"


# The watched row's link ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_fired_stops_watched_row_links_its_market_stop_loss_when_the_sent_list_is_pruned(monkeypatch, side):
    """The live path's 2 % dip through a 1 % stop, with an order the venue already has (the filled entry) still in the
    sent list as the stop fires: _sell_all prunes it while sending the market stop-loss, so the list's length does not
    grow. The watched row still ends "triggered" naming that stop-loss's order id."""
    from nautilus_trader.model import ClientOrderId

    from sleeve_fund.strategies import REGISTRY

    probe, config = _probe_classes()

    class Pruned(probe):
        def _exit_at_market(self, intent, *a, **kw):
            entry = next((c for c, d in self.decisions.items() if d.get("intent") == "entry"), None)
            if intent == "stop_loss" and entry is not None:
                self._sent.append(ClientOrderId(entry))
            return super()._exit_at_market(intent, *a, **kw)

    monkeypatch.setitem(REGISTRY, "probe", (Pruned, config))
    got = paper(shape(flat_prices(30), 10.5, 30, adverse(side, 0.02)), side=side, perp=True, profile="conservative",
                leave=25, stop=0.01)
    (watched,) = [o for o in got.orders if o.get("order_type") == qa.WATCHED]
    markets = [o for o in got.orders if o["intent"] == "stop_loss" and o.get("order_type") != qa.WATCHED]
    assert len(markets) == 1 and markets[0]["status"] == "filled", f"set-up: the market stop-loss: {markets}"
    assert watched["status"] == "triggered", watched
    assert (watched.get("signal") or {}).get("triggered_order") == markets[0]["order_id"], (watched, markets[0])
