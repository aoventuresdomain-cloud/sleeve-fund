"""BH-102, the Advisor's round-1 blocking item 2 (21 Oct): the paper path. A fully paid 1x long (the conservative
profile's cap, stopless as the PM's rule allows) is liquidated on the runtime/tick path once funding drains it to the
maintenance margin: booked at the bankruptcy price, so the Strategy never shows below zero. And the book shows an open
1x long's liquidation price, where before BH-102 a 1x long had none ("fully paid for"). Expected prices are worked out
here by hand."""
from __future__ import annotations

import dataclasses
import math
from types import SimpleNamespace

import pandas as pd
import pytest
import test_hub_146_qa as qa
from test_hub_146_qa import _probe, flat_prices  # noqa: F401 - _probe registers the probe
from test_liq_funding import _rate

from sleeve_fund import markets, risk
from sleeve_fund.dashboard import trading


@pytest.fixture
def conservative_hourly(monkeypatch):
    """Hourly settlements with the run from 07:50, so one falls at 08:00; the conservative profile's drawdown and
    daily-loss brakes lifted: they would pause it first, and the liquidation is the subject."""
    c = risk.PROFILES["conservative"]
    assert c.max_leverage == 1.0
    monkeypatch.setitem(risk.PROFILES, "conservative", dataclasses.replace(c, max_drawdown=0.99, daily_loss=0.99))
    monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=tuple(range(24))))
    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))


# The probe buys 20% of the 10,000 (the profile's max_position_pct): about 2,000 of margin at 1x, about 8,000 free.
# Funding comes from the free balance first (BH-102 rule 1), so a settlement drains the margin only past that: 4.997x
# the position's value (about 9,994) leaves the margin under its maintenance but above zero; 6x (about 12,000) is more
# than all of the Strategy's equity.
@pytest.mark.parametrize("rate", [4.997, 6.0], ids=["into-maintenance", "past-everything"])
def test_paper_a_funded_1x_long_is_liquidated_at_bankruptcy_and_never_below_zero(conservative_hourly, monkeypatch,
                                                                                rate):
    _rate(monkeypatch, rate)
    run = qa.paper(flat_prices(40), side=1, perp=True, profile="conservative", stop=None, leave=35)
    intents = {o["order_id"]: o["intent"] for o in run.orders}
    fills = sorted(({**f, "intent": intents[f["order_id"]]} for f in run.fills), key=lambda f: f["ts"])
    assert [f["intent"] for f in fills] == ["entry", "liquidation"], [(f["intent"], f["price"]) for f in fills]
    entry = next(o for o in run.orders if o["intent"] == "entry")
    assert entry["signal"]["position_liquidation_px"] is not None  # a 1x long has a liquidation price now
    book = run.store.journal_book("q146", 10_000)
    assert book["qty"] == pytest.approx(0.0, abs=1e-9)
    assert float(book["cash"]) >= -1e-6, book  # the isolated margin is all it can lose: never below zero


def test_the_book_shows_an_open_1x_longs_liquidation_price():
    """A 1x long on the venue's own perpetual: the margin M is the whole notional less the entry fee. Its equity
    M + q (p - E) meets the maintenance on the value at the price (A: mm q p) at p = fee / (q (1 - mm)), and the
    maintenance on the entry value (B: mm q E) at p = fee / q + mm E; the nearer (a long's higher) is the price
    (VS-1-TABLE), rounded up to the 0.1 tick (toward the entry). Before BH-102 a 1x long showed none."""
    pair, venue = "BTC/USDT", "BINANCE"
    sleeve = SimpleNamespace(params={"market": "perp"}, venue=venue, instrument=pair)
    qty, entry, taker = 0.1, 60_000.0, 0.00055
    fee = qty * entry * taker
    cash = 10_000.0 - fee - qty * entry
    x = {"sleeve": sleeve, "qty": qty, "cash": cash, "price": entry, "entry_px": entry, "equity": 10_000.0 - fee,
         "profile": risk.PROFILES["conservative"], "unrealised": 0.0, "entry_fees": fee}
    tier = markets.terms(sleeve.params, venue, pair).maintenance_margin[0]  # the first tier: 6,000 is inside it
    assert tier.amount == 0 and tier.ceiling > qty * entry
    mm = tier.rate
    want = math.ceil(max(fee / (qty * (1 - mm)), fee / qty + mm * entry) * 10) / 10
    _, liq = trading.position_margin(x)
    assert liq is not None and liq == pytest.approx(want, abs=1e-9), (liq, want)
