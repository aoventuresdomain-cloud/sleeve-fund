"""LIQ-FEE on the TIER-DIFF path (DA v3, found at the head-5 merge of LIQ-FEE onto main). The stop's half-way cap
takes the nearer of the priced table's liquidation and the newest pulled table's (_pulled_liq). The priced one already
takes the entry fees off the margin and rounds to the tick toward the entry; the pulled one must too, or a changed
table holds the stop against a price further from the entry than the venue's. Expected values come from
markets.isolated_liquidation with fees= and tick= given here, never read back from the strategy."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from decimal import Decimal as D
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund import markets, venue_terms
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy
from sleeve_fund.venues import venue

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.venue_terms_live]
DATA = Path(__file__).parent / "data" / "fee_live"
PAPER = venue("BINANCE")
PAIR = "BTC/USDT"
T = datetime(2026, 10, 15, 12, tzinfo=UTC)
FEES = 60_000.0 * 0.00055  # the taker entry fee on 1 at 60,000, a money amount
TICK = 0.1


def _changed():
    a = json.loads((DATA / "risk-limit-BTCUSDT-2026-10-15.json").read_text())
    for r in a["result"]["list"]:
        r["maintenanceMargin"] = str(D(r["maintenanceMargin"]) * D("1.5"))
        r["mmDeduction"] = str(D(r["mmDeduction"] or "0") * D("1.5"))
    return a


def _held(store, side):
    s = SimpleNamespace(_backtest=False, _cfg=SimpleNamespace(perp=object()), _entry_px=60_000.0, _entry_side=side,
                        _entry_fees=FEES, instrument=SimpleNamespace(price_increment=TICK),
                        _net_position=lambda: (side * 1.0, 60_000.0),
                        runtime=SimpleNamespace(store=store, name="p1", venue="BINANCE", now=lambda: T))
    return lambda cash, qty: LongFlatStrategy._pulled_liq(s, cash, qty)


@pytest.mark.parametrize("side, cash", [(1, -40_000.0), (-1, 80_000.0)])
def test_a_changed_tables_liquidation_takes_the_entry_fee_off_the_margin_and_rounds_toward_the_entry(
        monkeypatch, side, cash):
    monkeypatch.setattr(LongFlatStrategy, "_funded", lambda self: 0.0)
    monkeypatch.setattr(LongFlatStrategy, "_max_leverage", lambda self: 3.0)
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=20_000, venue="BINANCE", params={"market": "perp"})
    venue_terms.fetch_tiers(store, PAPER, PAIR, get=lambda url: _changed(), now=T)
    table = venue_terms.differing_tiers(store, PAPER, PAIR)
    assert table is not None
    qty = side * 1.0
    want = markets.isolated_liquidation(cash, qty, 60_000.0, 3.0, table, fees=FEES, tick=TICK)
    without = markets.isolated_liquidation(cash, qty, 60_000.0, 3.0, table)
    assert side * (want - without) > 0  # the fee moves it nearer the entry: the cell can tell the two apart
    assert _held(store, side)(cash, qty) == pytest.approx(want, abs=1e-9)
