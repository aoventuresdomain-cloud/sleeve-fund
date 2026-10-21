"""FIX-SPOTPERP-CHECK (HoE 21 Oct, 01:12 UK): a perp simulated on a spot venue (profile.perpetual False, e.g. KRAKEN
BTC/USD) trades on markets.LOW_FEE_PERP, with no venue tiers, tick or lot kept, so the price tick and lot size checks
skip it, as markets.terms does. A perp on a perpetual venue with no kept tick, or a lot mismatch, still FAILs.
Advisor (01:15 UK): only those two checks skip it, never silently: one NOTE line per Strategy in each section, which
stays green; the stop and leverage rules still refuse it. "Start allowed" here is the section not failing."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from sleeve_fund import start_check
from sleeve_fund.store import Store

pytestmark = [pytest.mark.no_maintenance_tables]
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}
LOT = (D("0.001"), D("0.001"))


def _store(venue, pair):
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=pair, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, venue=venue, params=PERP)
    return store


def _fails(sec):
    return [w for k, w in sec.lines if k == "FAIL"]


def _no_read(venue, pair):
    pytest.fail(f"no contract read for a perp simulated on a spot venue ({venue} {pair})")


SKIPPED = ("NOTE", "p1: SKIPPED (simulated perp, no perpetual venue)")


def test_a_perp_simulated_on_a_spot_venue_has_no_price_tick_fail():
    sec = start_check.check_price_ticks(_store("KRAKEN", "BTC/USD"), live_tick=_no_read)
    assert not sec.failed and not _fails(sec), sec.lines


def test_a_perp_simulated_on_a_spot_venue_has_no_lot_size_fail():
    sec = start_check.check_lot_sizes(_store("KRAKEN", "BTC/USD"), hub_lot=_no_read, paper_lot=_no_read)
    assert not sec.failed and not _fails(sec), sec.lines


@pytest.mark.parametrize("check", ["tick", "lot"])
def test_advisor_each_skipped_strategy_gets_one_visible_line_and_the_section_stays_green(check):
    store = _store("KRAKEN", "BTC/USD")
    sec = (start_check.check_price_ticks(store, live_tick=_no_read) if check == "tick"
           else start_check.check_lot_sizes(store, hub_lot=_no_read, paper_lot=_no_read))
    assert sec.lines.count(SKIPPED) == 1 and not sec.failed, sec.lines


def test_advisor_the_stop_rule_still_refuses_a_perp_simulated_on_a_spot_venue(prices, full_margin):
    """Only the tick and lot checks skip it: a 25% stop past half way to liquidation at 3x is still refused."""
    from test_backtest import _path

    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    spot = venue("KRAKEN")
    assert not spot.perpetual
    feed = _path(prices, [100.0, 100.5, 100.8, 101.5, 101.5, 101.0, 100.0])
    res = run_backtest("ping_pong", feed, spot.instrument("BTC", "USD"), {**PERP, "stop_loss": 0.25}, half_spread=0,
                       risk_profile="aggressive")
    assert res.fills.empty and "entry_refused_liquidation" in {e["kind"] for e in res.journal.events_}


def test_a_perp_on_a_perpetual_venue_with_no_kept_tick_still_fails():
    sec = start_check.check_price_ticks(_store("BINANCE", "SUI/USDT"), live_tick=lambda v, p: D("0.01"))
    assert sec.failed and any("SUI/USDT" in w and "no price tick kept" in w for w in _fails(sec)), sec.lines


def test_a_perp_on_a_perpetual_venue_with_a_lot_mismatch_still_fails():
    sec = start_check.check_lot_sizes(_store("BINANCE", "BTC/USDT"), hub_lot=lambda v, p: LOT,
                                      paper_lot=lambda v, p: (D("0.01"), D("0.01")))
    assert sec.failed and "0.01" in _fails(sec)[0], sec.lines
