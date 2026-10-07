"""Stale data (Independent Quant Advisor, 7 Oct, HoE 02:09): a fixed 5 minutes for every candle interval. Pinned here:
(1) it is measured on the venue feed of the traded instrument (its trades and quotes): candles still arriving from the
hub, or another instrument's prints, do not keep it fresh, and it is its own cause, never funding; (2) it clears on the
first fresh trade or quote; (3) it cancels resting entries. (4), exits and stops never held, is
test_choke_flip_after_target.test_stale_data_never_holds_a_stop_or_a_target.

Also here: a fired stop's watched row stays linked to the market stop-loss sent for it when _sell_all prunes the sent
list on the way (Code Reviewer on #182)."""
from __future__ import annotations

import pandas as pd
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


# P1-SG21: the staleness clock across a restart ------------------------------------------------------------------


@pytest.mark.parametrize("enter", [3, 8], ids=["wanted-3min-in", "wanted-8min-in"])
def test_a_process_started_on_a_dead_feed_holds_entries_until_its_first_trade_or_quote(monkeypatch, enter):
    """A process starts with no trade or quote reaching it for 30 minutes (a restart on a dead feed; the hub's candles
    still arrive). The model wants long from `enter` minutes in: nothing opens until trades resume, the hold says it
    counts from the process start, and the entry goes once data is back."""
    holds = []

    def record(st, what, before):
        if what == "watchdog":
            holds.append(st.runtime.holds.get("stale_data"))

    _recording(monkeypatch, record)
    resume = 30 * 60
    got = paper(flat_prices(40), enter=enter, leave=38, side=1, profile="conservative",
                gone=frozenset(range(0, resume)))
    entries = [o for o in got.orders if o["intent"] == "entry"]
    assert entries, "set-up: no entry once data resumed"
    first = int(pd.Timestamp(entries[0]["ts"]).value)
    assert first >= START + resume * S, f"an entry went {(first - START) / S:.0f} s in, before any trade or quote"
    assert holds and all(h and "since this process started" in h for h in holds[:3]), holds[:3]


# P1-SG15: nothing opens from the instant the Stop is accepted -----------------------------------------------------


def _stop_at(monkeypatch, store, at):  # noqa: F811
    """The PM's Stop written to the journal on the first trade at or after `at`: between two 30 s runtime ticks."""
    from test_exposure_gate_xfails import Egx, _pm_stop

    real, accepted = Egx.on_trade, []

    def on_trade(self, tick):
        if not accepted and tick.ts_event >= at.value:
            accepted.append(pd.Timestamp(tick.ts_event, tz="UTC").to_pydatetime())
            _pm_stop(store)
        return real(self, tick)

    monkeypatch.setattr(Egx, "on_trade", on_trade)
    return accepted


def test_a_stop_accepted_between_ticks_cancels_a_resting_entry_before_the_next_trade(tmp_path, store, monkeypatch):  # noqa: F811
    """A stop entry rests from t0+5, 1 % above; the Stop is accepted at t0+25:10 (ticks fall on :00 and :30) and the
    price steps 2 % up at t0+25:11. The resting entry is cancelled on the Stop's own trade and never fills."""
    plan = Plan(t0=STD_T0, tag="sb")
    k = 25 * 60 + 10
    p0 = 60_000 + 5 * 60 * 0.01
    plan.price = lambda s: (60_000 + s * 0.01) * (1.02 if s >= k + 1 else 1.0)
    plan.rest.append((M(plan.t0, 5), 1, round(p0 * 1.01, 1), 0.05))
    accepted = _stop_at(monkeypatch, store, plan.t0 + pd.Timedelta(seconds=k))
    run(tmp_path, store, plan, monkeypatch)
    assert accepted, "set-up: the Stop never landed"
    (entry,) = [o for o in _orders(store) if o["intent"] == "entry"]
    assert entry["status"] in ("canceled", "cancelled"), entry
    assert entry["order_id"] not in {f["order_id"] for f in _fills(store)}, "the resting entry filled after the Stop"


@pytest.mark.parametrize("case", ["crossing-first-print-within-1s", "crossing-first-print-at-1.5s",
                                  "first-print-not-crossing", "no-market-data-after-the-stop"])
def test_a_stop_between_prints_lets_a_resting_entry_fill_only_on_a_crossing_print_within_a_second(
        tmp_path, store, monkeypatch, case):  # noqa: F811
    """The one-print residual (HoE, logged before G2; Advisor 7 Oct 05:01 bounds). A Stop accepted between two prints:
    its cancel goes out within GATE_WATCH_SECONDS whatever the feed does, so a resting stop entry fills only when the
    first print after the Stop crosses it within that time, as a real venue can fill a resting order before a cancel
    lands. Then the raced-fill rule holds: kept with its stop, one incident, one raced_fill row with the ms after the
    Stop. A first print 1.5 s after the Stop finds the entry cancelled; a first print that doesn't cross cancels it
    cleanly; with no market data at all after the Stop, the gate read alone cancels it (Advisor 05:47). Every cancel is
    on record with the ms from the Stop's acceptance (the instant its decision is stamped and committed)."""
    from test_exposure_gate_xfails import NAME, Egx, _events, _pm_stop

    from sleeve_fund import store as store_mod
    from sleeve_fund.strategies import base

    k = 25 * 60 + 11  # the price steps 2% here and crosses the entry (60 s later when the first print doesn't cross)
    lead = 1.5 if case == "crossing-first-print-at-1.5s" else 0.25
    accepted = pd.Timestamp(STD_T0) + pd.Timedelta(seconds=k - lead)
    real_start, real_bar = Egx.on_start, Egx.on_bar

    def stop(e):
        with monkeypatch.context() as m:  # the decision is stamped as the dashboard stamps a Stop: its own instant
            m.setattr(store_mod, "utcnow", lambda: accepted.to_pydatetime())
            _pm_stop(store)

    def on_start(self):
        real_start(self)
        self.clock.set_time_alert("qa-stop", accepted.to_pydatetime(), callback=stop)

    def on_bar(self, bar):
        resting = bool(self._rest)
        real_bar(self, bar)
        if resting and not self._rest and self._stop_frac is None:
            # A venue-resting entry plans its exits when it is placed, as every entry does at its decision (the
            # QA harness places the order bare); paper's own post-only entries are kept in the process instead.
            self._stop_frac, self._tp_frac, self._stop_basis = self._plan_exits(bar.close.as_double(), 1)

    monkeypatch.setattr(Egx, "on_start", on_start)
    monkeypatch.setattr(Egx, "on_bar", on_bar)
    if case == "crossing-first-print-within-1s":
        # Here the gate reads land on whole seconds, as the prints do, and a read beats a print at the same instant;
        # a live feed's prints fall anywhere between reads. Reads 0.75 s apart (still inside 1 s) put the next one
        # after the crossing print, at t0+25:11.25.
        monkeypatch.setattr(base, "GATE_WATCH_SECONDS", 0.75)
    plan = Plan(t0=STD_T0, tag="sr1")
    p0 = 60_000 + 5 * 60 * 0.01
    cross = k + (60 if case in ("first-print-not-crossing", "no-market-data-after-the-stop") else 0)
    plan.price = lambda s: (60_000 + s * 0.01) * (1.02 if s >= cross else 1.0)
    if case == "crossing-first-print-at-1.5s":
        plan.holes.append((k - 1, k))  # no print between the Stop and the crossing one
    if case == "no-market-data-after-the-stop":
        plan.holes.append((k - 1, k + 120))  # nothing for two minutes; the price is past the trigger when it returns
    # the strategy wants the long from the candle after the Stop on (no market entry of its own before it)
    plan.windows.append((M(plan.t0, 25) + pd.Timedelta(seconds=1), M(plan.t0, plan.minutes), 1))
    plan.rest.append((M(plan.t0, 5), 1, round(p0 * 1.01, 1), 0.05))
    run(tmp_path, store, plan, monkeypatch)
    assert store.sleeve(NAME).desired_state == "stopped", "set-up: the Stop never landed"
    (entry,) = [o for o in _orders(store) if o["intent"] == "entry"]
    fills = [f for f in _fills(store) if f["order_id"] == entry["order_id"]]
    incidents = [e["message"][:120] for e in _events(store, ("incident",))]
    raced = [e["message"] for e in _events(store, ("raced_fill",))]
    if case != "crossing-first-print-within-1s":
        assert not fills and entry["status"] == "canceled", f"the resting entry filled after the Stop: {entry}"
        assert not incidents and not raced, (incidents, raced)
        (cancel,) = _events(store, ("resting_entry_cancelled",))
        ms = int(cancel["message"].split(" cancelled ")[1].split(" ms ")[0])
        assert ms <= 500 and "(stopped" in cancel["message"], cancel["message"]
        if case == "no-market-data-after-the-stop":  # the read at t0+25:11, with no print since t0+25:09
            assert ms == 250 and cancel["ts"] == pd.Timestamp(STD_T0) + pd.Timedelta(seconds=k), cancel
        return
    assert fills, f"set-up: the crossing print did not fill the resting entry: {entry}"
    stops = [o for o in _orders(store) if o["intent"] == "stop_loss" and o["ts"] >= fills[0]["ts"]]
    assert stops, f"the raced entry has no stop: {[(o['intent'], o['order_type'], o['status']) for o in _orders(store)]}"
    assert len([m for m in incidents if "filled while nothing may open" in m]) == 1, incidents
    assert not [o for o in _orders(store) if o["intent"] == "exit"], "the raced entry was closed, not kept"
    assert len(raced) == 1 and " 250 ms after nothing could open any more (stopped" in raced[0], raced
