"""CR MUST (B-15 review), with B-36b: the dashboard's liquidation price is on the contract tick, as the engine's is. The
engine rounds toward the entry to its instrument's tick (margin.to_tick); the dashboard reads the tick kept for the
instrument (markets.price_tick), which the start check holds equal to the venue's. A position whose unrounded price is off
the 0.1 grid shows the same price on the pages as the engine guards, and the most it can lose is worked from that price.
Expected prices are rounded here by hand, toward the entry, from the unrounded one."""
from __future__ import annotations

from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from types import SimpleNamespace

import pytest

from sleeve_fund import markets, risk
from sleeve_fund.dashboard import riskops, trading
from sleeve_fund.strategies.base import LongFlatStrategy

pytestmark = pytest.mark.no_maintenance_tables  # the venue's real BTC/USDT tiers
PAIR, VENUE = "BTC/USDT", "BINANCE"
ENTRY, EQUITY, TAKER = 60_003.37, 50_000.0, 0.00055
PROFILE = risk.PROFILES["balanced"]
SLEEVE = SimpleNamespace(params={"market": "perp"}, venue=VENUE, instrument=PAIR)


def _position(qty):
    fees = abs(qty) * ENTRY * TAKER
    cash = EQUITY - fees - qty * ENTRY  # the spot-style balance less qty x entry, after the entry fee
    return {"sleeve": SLEEVE, "qty": qty, "cash": cash, "price": ENTRY, "entry_px": ENTRY, "equity": EQUITY - fees,
            "profile": PROFILE, "unrealised": 0.0, "entry_fees": fees}


def _by_hand(qty):
    x = _position(qty)
    lev = risk.leverage_cap(PROFILE, SLEEVE.params)
    mm = markets.terms(SLEEVE.params, VENUE, PAIR).maintenance_margin
    raw = markets.isolated_liquidation(x["cash"], qty, ENTRY, lev, mm, fees=x["entry_fees"])
    tenths = Decimal(repr(raw)) * 10
    assert tenths != tenths.to_integral_value()  # off the grid: the cell can tell rounded from not
    return x, lev, mm, raw, float((tenths.to_integral_value(ROUND_CEILING if qty > 0 else ROUND_FLOOR)) / 10)


@pytest.mark.parametrize("qty", [0.5, -0.5])
def test_the_dashboard_and_the_engine_give_the_same_liquidation_price_on_the_kept_tick(monkeypatch, qty):
    x, lev, mm, raw, want = _by_hand(qty)
    assert markets.price_tick(VENUE, PAIR) == Decimal("0.1")
    _, shown = trading.position_margin(x)
    assert shown == pytest.approx(want, abs=1e-9) and shown != pytest.approx(raw, abs=1e-9)
    monkeypatch.setattr(LongFlatStrategy, "_funded", lambda self: 0.0)
    engine = SimpleNamespace(_cfg=SimpleNamespace(perp=SimpleNamespace(maintenance_margin=mm)), _entry_px=ENTRY,
                             _entry_fees=x["entry_fees"], instrument=SimpleNamespace(price_increment=0.1),
                             runtime=SimpleNamespace(profile=SimpleNamespace(max_leverage=lev)))
    assert LongFlatStrategy._liq_from_entry(engine, x["cash"], qty) == pytest.approx(shown, abs=1e-9)


@pytest.mark.parametrize("qty", [0.5, -0.5])
def test_the_most_it_can_lose_is_worked_from_the_price_on_the_tick(qty):
    x, lev, _, _, want = _by_hand(qty)
    taker = float(markets.fees_for(SLEEVE.params, __import__("sleeve_fund.venues", fromlist=["x"]).venue(VENUE).fees,
                                   VENUE).taker)
    cap = markets.gap_loss_cap(qty, ENTRY, lev, x["cash"] + qty * ENTRY, taker, want)
    assert cap < x["equity"]  # the cap binds, not the equity
    assert riskops.most_it_can_lose(x) == pytest.approx(cap, abs=1e-9)
