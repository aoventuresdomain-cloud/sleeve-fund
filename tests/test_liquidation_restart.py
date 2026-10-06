"""QA P1-L22 (#146, Advisor 22:36): a perp position's liquidation price is worked out once, from its own average
entry and posted margin, journaled on the order that opened or added to it and read back after a restart; never
from the restore's price at the simulated venue. And P1-L23: a replayed stop says it gapped only when it did."""

import pytest

import test_hub_146_qa as qa
from sleeve_fund import markets
from sleeve_fund.store import Store
from test_hub_146_qa import _probe  # noqa: F401 - the autouse fixture that registers the probe strategy


def _liq_events(run):
    return [e for e in run.events if e["kind"] == "liquidation"]


@pytest.mark.parametrize("profile, side, spike, on_return", [
    ("aggressive", -1, 0.40, 0.10),  # QA's repro: a 3x short, the minute to 00:08 opens 40 % against, 10 % on return
    ("balanced", 1, 0.60, 0.30),  # a 2x long 30 % down on return
], ids=["3x-short", "2x-long"])
def test_after_a_restart_a_replayed_minute_opening_past_liquidation_liquidates_however_far_the_price_is_now(
        profile, side, spike, on_return):
    p = qa.shape(qa.shape(qa.flat_prices(30), 7.0, 8.0, qa.adverse(side, spike)), 8.0, 30, qa.adverse(side, on_return))
    run = qa.restart(p, 6, 12, side=side, perp=True, profile=profile, stop=0.10, qty=0.25)
    seq = run.sequence()
    assert qa.first_exit(seq)[0] == "liquidation", seq
    assert not [r for r in seq if r[0] == "stop_loss"], seq
    assert len(_liq_events(run)) == 1


def test_the_journaled_liquidation_price_is_the_one_used_after_a_restart(monkeypatch):
    """The entry's order carries a liquidation price further off than the formula's: the restarted strategy uses
    the journaled one, so the minute opening 40 % against the 3x short is a stop past it, not a liquidation."""
    record = Store.record_order

    def with_liq(self, sleeve, **kw):
        if kw.get("order_id") == "O-held":
            kw["signal"] = {**kw["signal"], "position_liquidation_px": qa.BASE * 1.5}
        return record(self, sleeve, **kw)

    monkeypatch.setattr(Store, "record_order", with_liq)
    p = qa.shape(qa.flat_prices(30), 7.0, 8.0, qa.adverse(-1, 0.40))
    run = qa.restart(p, 6, 12, side=-1, perp=True, profile="aggressive", stop=0.10, qty=0.25)
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()


def test_a_later_entry_that_never_filled_does_not_hide_the_held_entrys_journaled_price(monkeypatch):
    """An expired post-only entry, journaled after the held one and never filled: the restart still reads the
    held entry's journaled price (CR #146), so the same minute is a stop past it, not a liquidation."""
    from datetime import timedelta

    record = Store.record_order

    def with_liq(self, sleeve, **kw):
        if kw.get("order_id") == "O-held":
            kw["signal"] = {**kw["signal"], "position_liquidation_px": qa.BASE * 1.5}
            record(self, sleeve, **kw)
            return record(self, sleeve, **{**kw, "order_id": "O-expired", "ts": kw["ts"] + timedelta(seconds=30),
                                           "signal": {k: v for k, v in kw["signal"].items()
                                                      if k != "position_liquidation_px"}})
        return record(self, sleeve, **kw)

    monkeypatch.setattr(Store, "record_order", with_liq)
    p = qa.shape(qa.flat_prices(30), 7.0, 8.0, qa.adverse(-1, 0.40))
    run = qa.restart(p, 6, 12, side=-1, perp=True, profile="aggressive", stop=0.10, qty=0.25)
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()


def test_an_entry_journals_the_liquidation_price_from_its_own_fill():
    run = qa.paper(qa.flat_prices(12), side=-1, perp=True, profile="aggressive", leave=10)
    entry = next(o for o in run.orders if o["intent"] == "entry")
    fill = next(f for f in run.fills if f["order_id"] == entry["order_id"])
    qty, px = -fill["qty"], fill["price"]
    cash = 10_000 - fill["fee"] - qty * px  # the journal's spot-style cash: a short holds its sale's proceeds
    expected = markets.isolated_liquidation(cash, qty, px, 3.0, 0.005)
    assert entry["signal"]["position_liquidation_px"] == pytest.approx(expected, rel=1e-6)
    assert "liquidation_px" in entry["signal"]  # the decision's own estimate from the close stays


@pytest.mark.parametrize("gap", [False, True])
def test_a_replayed_stop_says_it_opened_past_its_level_only_when_it_did(gap):
    p = qa.flat_prices(30)
    p = qa.shape(p, 7.0, 9.0, qa.adverse(1, 0.03)) if gap else qa.shape(p, 7.5, 7 + 50 / 60, qa.adverse(1, 0.02))
    run = qa.restart(p, 6, 12, side=1, perp=False, stop=0.01)
    why = next(e["message"] for e in run.events if e["kind"] == "outage_exit")
    assert ("(it opened past it)" in why) is gap, why


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_an_add_sets_the_price_again_from_the_average_entry_and_the_whole_positions_margin(side):
    """An entry, then an add on the same side at another price (a perp can't rebalance yet, so this drives the fill
    book directly): the add's order journals the whole position's price at its average entry with the cash then
    held, not the first fill's."""
    from decimal import Decimal
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    merged = {}

    class Book:
        _track_entry, _set_liq = LongFlatStrategy._track_entry, LongFlatStrategy._set_liq
        _liq_from_entry, _backtest = LongFlatStrategy._liq_from_entry, False
        _entry_px, _entry_qty, _entry_side, _liq_px = None, 0.0, 0, None
        runtime = SimpleNamespace(profile=SimpleNamespace(max_leverage=3.0),
                                  store=SimpleNamespace(merge_order_signal=lambda oid, v: merged.update({oid: v})))
        _cfg = SimpleNamespace(perp=SimpleNamespace(maintenance_margin=0.005))
        post, cash = Decimal(0), 10_000.0

        def _signed_qty(self):
            return self.post

        def _lot(self):
            return Decimal("0.001")

        def _mark(self):
            return 0.0, self.cash, float(self.post), 0.0

        def fill(self, oid, qty, px):
            self.post += side * Decimal(str(qty))
            self.cash -= side * qty * px + qty * px * 0.0005
            assert self._track_entry(side, Decimal(str(qty)), qty, px)
            self._set_liq(oid)

    book = Book()
    book.fill("A", 0.05, 60_000.0)
    book.fill("B", 0.10, 60_000.0 * qa.favourable(side, 0.05))
    avg = (0.05 * 60_000.0 + 0.10 * 60_000.0 * qa.favourable(side, 0.05)) / 0.15
    expected = markets.isolated_liquidation(book.cash, side * 0.15, avg, 3.0, 0.005)
    first, add = merged["A"]["position_liquidation_px"], merged["B"]["position_liquidation_px"]
    assert add == pytest.approx(expected, rel=1e-6) and book._liq_px == pytest.approx(expected, rel=1e-6)
    assert first != pytest.approx(add, rel=1e-4)  # the add moved it
