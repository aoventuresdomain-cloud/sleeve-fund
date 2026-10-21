"""FIX-SPOTPERP-CHECK (HoE 21 Oct, 01:12 UK): a perp simulated on a spot venue (profile.perpetual False, e.g. KRAKEN
BTC/USD) trades on markets.LOW_FEE_PERP, with no venue tiers, tick or lot kept, so the price tick and lot size checks
skip it, as markets.terms does. A perp on a perpetual venue with no kept tick, or a lot mismatch, still FAILs."""
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


def test_a_perp_simulated_on_a_spot_venue_has_no_price_tick_fail():
    sec = start_check.check_price_ticks(_store("KRAKEN", "BTC/USD"), live_tick=_no_read)
    assert not _fails(sec), sec.lines


def test_a_perp_simulated_on_a_spot_venue_has_no_lot_size_fail():
    sec = start_check.check_lot_sizes(_store("KRAKEN", "BTC/USD"), hub_lot=_no_read, paper_lot=_no_read)
    assert not _fails(sec), sec.lines


def test_a_perp_on_a_perpetual_venue_with_no_kept_tick_still_fails():
    sec = start_check.check_price_ticks(_store("BINANCE", "ETH/USDT"), live_tick=lambda v, p: D("0.01"))
    assert sec.failed and any("ETH/USDT" in w and "no price tick kept" in w for w in _fails(sec)), sec.lines


def test_a_perp_on_a_perpetual_venue_with_a_lot_mismatch_still_fails():
    sec = start_check.check_lot_sizes(_store("BINANCE", "BTC/USDT"), hub_lot=lambda v, p: LOT,
                                      paper_lot=lambda v, p: (D("0.01"), D("0.01")))
    assert sec.failed and "0.01" in _fails(sec)[0], sec.lines
