"""VS-1-TABLE (Advisor via HoE, 15 Oct): which liquidation formula the venue uses can't be proved before an execution
client exists, so both are worked and the NEARER price is used, for the liquidation guard, the half-distance stop rule
and every reader of margin.isolated_liquidation.

A: the equity falls to the maintenance on the position's value at the price (solved over the tiers):
   long entry x (1 - 1/L) / (1 - MMR), short entry x (1 + 1/L) / (1 + MMR) in one tier.
B: on the entry's value: liq = entry -/+ (IM - MM) / qty, MM = entry notional x MMR - deduction, IM = notional / L.
Nearer: a long's higher price, a short's lower.

Hand-worked (3x at 60,000, qty 2.5, 150,000 notional, tier 1: 0.33%, no deduction; IM 50,000, MM 495):
  long  A = 40,000 / 0.9967 = 40,132.44   B = 60,000 - (50,000 - 495) / 2.5 = 40,198.00  -> 40,198.00
  short A = 80,000 / 1.0033 = 79,736.87   B = 60,000 + 49,505 / 2.5 = 79,802.00      -> 79,736.87
At exactly 300,000 notional (qty 5): the entry is in tier 2 (0.5%, 510: MM 990, the same as tier 1's by continuity).
  long  A = 40,132.44 (tier 1 at 200,660)   B = 60,000 - (100,000 - 990) / 5 = 40,198.00  -> 40,198.00
  short A = 400,510 / 5.025 = 79,703.48 (tier 2 at 398,517)   B = 60,000 + 99,010 / 5 = 79,802.00  -> 79,703.48
"""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from sleeve_fund import margin, markets
from sleeve_fund.margin import entry_liquidation, shifted_liquidation, tier_at
from sleeve_fund.venues import venue

pytestmark = pytest.mark.no_maintenance_tables  # the venue's own BTC/USDT table


def _btc():
    return markets.maintenance_for(venue("BINANCE"), "BTC/USDT")


@pytest.mark.parametrize("qty, cash, side, want", [
    (2.5, 100_000, 1, 40_198.00), (2.5, 100_000, -1, 79_736.87),  # tier 1: B nearer for the long, A for the short
    (5, 200_000, 1, 40_198.00), (5, 200_000, -1, 79_703.48),  # exactly 300,000: the entry is in tier 2
])
def test_the_nearer_of_the_two_formulas_is_the_liquidation_price(qty, cash, side, want):
    liq, distance = entry_liquidation(cash, qty, 60_000, side, 0.0, _btc(), 3)
    assert round(liq, 2) == want
    assert distance == pytest.approx(abs(want / 60_000 - 1), abs=1e-6)


def test_the_entry_at_exactly_300000_takes_tier_2_and_the_table_meets_there():
    t = tier_at(_btc(), 300_000)
    assert (t.rate, t.amount) == (0.005, 510)
    assert 300_000 * t.rate - t.amount == pytest.approx(300_000 * 0.0033)  # 990 either side of the ceiling


def test_a_fully_paid_1x_long_still_has_no_liquidation():
    """A finds none (the long is paid for); B's ~0.33%-of-entry price isn't taken as one (PE1, to the HoE)."""
    assert entry_liquidation(60_000, 1, 60_000, 1, 0.0, _btc(), 1)[0] is None


@pytest.mark.parametrize("side, fixed, want", [(1, 40_198.00, 40_598.00), (-1, 79_736.87, 79_338.19)])
def test_funding_paid_moves_the_nearer_price_solved_afresh_from_the_positions_cash(side, fixed, want):
    """1,000 of funding paid out of the 150,000 position's margin: the cash it implies is recovered from the fixed
    price under both formulas (the larger is the position's), then both are re-solved and the nearer kept.
    long  A = 101,000 / 2.49175 = 40,533.76   B = (495 + 101,000) / 2.5 = 40,598.00  -> 40,598.00
    short A = 199,000 / 2.50825 = 79,338.19   B = (199,000 - 495) / 2.5 = 79,402.00  -> 79,338.19"""
    liq = shifted_liquidation(fixed, side * 2.5, -1_000, _btc(), entry=60_000)
    assert round(liq, 2) == want


def test_btc_s_table_carries_its_venue_date_and_the_venues_top_ceiling():
    perp = venue("BINANCE")
    assert "paper venue" in perp.maintenance_basis and "15 Oct 2026" in perp.maintenance_basis
    assert perp.maintenance_read_on == date(2026, 10, 15)
    assert markets.terms({"market": "perp"}, "BINANCE", "BTC/USDT").max_notional == 1_200_000_000


def test_the_start_check_notes_a_table_older_than_30_days(tmp_path):
    from datetime import UTC, datetime

    from sleeve_fund import start_check
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/j.db")
    store.create_sleeve(name="p1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", params={"market": "perp"})
    fresh = start_check.check_maintenance(store, now=datetime(2026, 11, 14, tzinfo=UTC))  # 30 days: still fresh
    assert not any("days ago" in w for _, w in fresh.lines), fresh.lines
    old = start_check.check_maintenance(store, now=datetime(2026, 11, 15, tzinfo=UTC))
    assert not old.failed
    assert ("NOTE", "BTC/USDT on the paper venue: its maintenance table was read on 15 Oct 2026, 31 days ago; "
                    "re-paste and re-check it (DA-11)") in [(s, w.replace(perp_label(), "the paper venue"))
                                                            for s, w in old.lines], old.lines


def perp_label():
    return venue("BINANCE").label


def test_an_entry_at_or_past_the_venues_top_tier_is_refused(prices, instrument, full_margin, monkeypatch):
    """Fail closed: past the venue's last published ceiling there is no tier, so no liquidation price; the entry is
    refused and says why. (Simulated perp, with its ceiling set low so a 100-price bar's entry reaches it.)"""
    from sleeve_fund.research.runner import run_backtest
    from test_backtest import _path

    feed = _path(prices, [100.0, 100.5, 100.8, 101.5, 101.5, 101.0, 100.0])
    perp = {"market": "perp", "allow_short": True}
    monkeypatch.setattr(markets, "LOW_FEE_PERP", replace(markets.LOW_FEE_PERP, max_notional=1.0))
    res = run_backtest("ping_pong", feed, instrument, perp, half_spread=0, risk_profile="aggressive")
    assert res.fills.empty
    assert "entry_refused_top_tier" in {e["kind"] for e in res.journal.events_}
    monkeypatch.setattr(markets, "LOW_FEE_PERP", replace(markets.LOW_FEE_PERP, max_notional=float("inf")))
    assert not run_backtest("ping_pong", feed, instrument, perp, half_spread=0, risk_profile="aggressive").fills.empty


def test_margin_still_reads_one_rate_as_a_flat_table():
    assert margin.isolated_liquidation(-100_000, 2.5, 60_000, 3, 0.0033) == pytest.approx(40_198.0)
