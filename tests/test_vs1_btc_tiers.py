"""VS-1 tables (HoE 15 Oct, data only): the engine's perp profile keeps the paper venue's published maintenance tiers
for BTC/USDT, hand-checked by the DA (pe1-plans/vs1-tiers.md: 35 tiers, continuous at every ceiling, the last
unbounded). Tier 1 is 0.33% to 300,000 with no deduction; tier 2 is 0.5% with a 510 deduction, so the table meets
at the ceiling. Any other pair still has no table and a perp on it is refused."""
from __future__ import annotations

import math

import pytest

from sleeve_fund import markets
from sleeve_fund.margin import entry_liquidation, flat
from sleeve_fund.venues import venue

pytestmark = pytest.mark.no_maintenance_tables  # the venue as it is, not the tests' flat stand-in


def test_btc_keeps_the_paper_venues_published_tiers_with_their_source():
    perp = venue("BINANCE")
    table = markets.maintenance_for(perp, "BTC/USDT")
    assert markets.maintenance_for(perp, "BTCUSDT-PERP") == table  # by the venue's own symbol too
    assert len(table) == 35 and table[-1].ceiling == math.inf
    assert table[0] == (300_000, 0.0033, 0) and table[1] == (2_000_000, 0.005, 510)
    assert 300_000 * 0.0033 - 0 == pytest.approx(300_000 * 0.005 - 510)  # continuous at tier 1's ceiling
    assert perp.maintenance_basis == "paper venue risk-limit BTCUSDT, 15 Oct 2026, research/venue-data/" \
                                     "paper-venue-risk-limit-BTCUSDT-2026-10-15.json"


@pytest.mark.parametrize("side, at_tiers, at_flat", [(1, 40_132.44, 40_201.01), (-1, 79_736.87, 79_601.99)])
def test_a_3x_btc_position_of_150000_liquidates_where_the_da_worked_it_out(side, at_tiers, at_flat):
    """The DA's worked example: 3x at 60,000, qty 2.5 (150,000 notional, the most the fund's gross cap allows), on a
    100,000 balance. Both ends stay in tier 1; the tiers move the price from the old flat 0.5% stand-in's."""
    table = markets.maintenance_for(venue("BINANCE"), "BTC/USDT")
    liq, _ = entry_liquidation(100_000, 2.5, 60_000, side, 0.0, table, 3)
    assert round(liq, 2) == at_tiers
    assert round(entry_liquidation(100_000, 2.5, 60_000, side, 0.0, flat(0.005), 3)[0], 2) == at_flat


@pytest.mark.parametrize("pair", ["ETH/USDT", "SOL/USDT", "XRP/USDT", "SUI/USDT"])
def test_every_other_pair_is_still_refused_until_its_table_is_kept(pair):
    with pytest.raises(markets.NoMaintenanceTable, match=pair):
        markets.maintenance_for(venue("BINANCE"), pair)
