"""PAPER-TIERS-3 (PM YES 17:54 UK 21 Oct; Advisor plan cleared 18:15 UK): the paper venue's published maintenance tiers,
price tick and lot size for BTC, ETH, SOL and XRP perps are kept in venues.py exactly as the venue's own replies give
them (tests/data/venue_tables/, read in one pass on the server, 21 Oct 2026 21:30 UTC), so research, paper and live read
the same tables; R-28's BTC-tier reuse is gone and a perp on a pair with no table is refused in research too."""
from __future__ import annotations

import json
import math
from decimal import Decimal
from pathlib import Path

import pytest

pytestmark = pytest.mark.no_maintenance_tables  # the venue as it is, not the 0.5% stand-in (conftest)

from sleeve_fund import margin, markets
from sleeve_fund.margin import Tier
from sleeve_fund.venues import venue

DATA = Path(__file__).parent / "data" / "venue_tables"
PAIRS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
VENUE = "BINANCE"  # the profile the paper venue's tables are kept on (venues.py, VS-1)


def _reply(kind: str, pair: str) -> dict:
    raw = json.loads((DATA / f"{kind}-{pair.replace('/', '')}.json").read_text())
    assert raw["retCode"] == 0 and not raw["result"].get("nextPageCursor"), (kind, pair, raw.get("retMsg"))
    return raw["result"]["list"]


def _published_tiers(pair: str) -> tuple[Tier, ...]:
    """The venue's reply as margin.Tier rows: riskLimitValue the ceiling (the last unbounded, its value kept as
    maintenance_limits), maintenanceMargin the rate, mmDeduction the amount (blank is 0)."""
    rows = sorted(_reply("risk-limit", pair), key=lambda r: Decimal(r["riskLimitValue"]))
    out = [Tier(float(r["riskLimitValue"]), float(r["maintenanceMargin"]), float(r["mmDeduction"] or 0)) for r in rows]
    return (*out[:-1], out[-1]._replace(ceiling=math.inf))


def _listing(pair: str) -> dict:
    (row,) = _reply("instruments-info", pair)
    return row


@pytest.mark.parametrize("pair", PAIRS)
def test_each_kept_table_is_the_venues_published_table_exactly(pair):
    profile = venue(VENUE)
    assert tuple(profile.maintenance_tiers[pair]) == _published_tiers(pair)
    assert margin.tiers(profile.maintenance_tiers[pair])  # rising, unbounded at the top, continuous at every ceiling
    top = max(Decimal(r["riskLimitValue"]) for r in _reply("risk-limit", pair))
    assert profile.maintenance_limits[pair] == float(top)


@pytest.mark.parametrize("pair", PAIRS)
def test_each_pairs_tick_and_lot_are_the_venues_listing_exactly(pair):
    profile, row = venue(VENUE), _listing(pair)
    assert row["symbol"] == pair.replace("/", "") and row["contractType"] == "LinearPerpetual"
    assert Decimal(profile.price_ticks[pair]) == Decimal(row["priceFilter"]["tickSize"])
    lot = row["lotSizeFilter"]
    assert tuple(Decimal(x) for x in profile.lot_sizes[pair]) == (Decimal(lot["qtyStep"]), Decimal(lot["minOrderQty"]))


def test_all_four_tables_share_one_read_date_and_say_where_they_came_from():
    profile = venue(VENUE)
    read = (DATA / "read-at.txt").read_text().split()[-1]  # READ_AT_UTC <iso>
    assert profile.maintenance_read_on.isoformat() == read[:10] == "2026-10-21"
    for pair in PAIRS:
        assert pair.replace("/", "") in profile.maintenance_basis
        assert set(_reply("risk-limit", pair)[0]) >= {"riskLimitValue", "maintenanceMargin", "mmDeduction"}


@pytest.mark.parametrize("pair", PAIRS)
@pytest.mark.parametrize("side", [1, -1])
def test_a_known_position_liquidates_where_the_venues_formula_puts_it(pair, side):
    """Advisor's extra cell: 3x isolated, 100,000 of notional at a round entry, no fees. The venue's formula for an
    isolated position (maintenance on the entry value): liq = entry -/+ (IM - MM) / qty. The engine keeps the nearer of
    that and the maintenance on the value at the price (VS-1-TABLE): exactly the venue's for a long, at it or nearer
    for a short. Each pair's own table, never BTC's."""
    table = markets.maintenance_for(venue(VENUE), pair)
    entry = 100.0
    qty = side * 100_000 / entry
    im, t = 100_000 / 3, margin.tier_at(table, 100_000)
    venue_liq = entry - side * (im - (100_000 * t.rate - t.amount)) / abs(qty)
    ours = margin.isolated_liquidation(im - qty * entry, qty, entry, 3.0, table)  # a balance of exactly its margin
    if side > 0:
        assert ours == pytest.approx(venue_liq, rel=1e-12)
    else:
        assert venue_liq * (1 - 1e-12) >= ours > entry


def test_sol_and_xrp_keep_higher_first_tier_maintenance_than_btc():
    """Advisor: the smaller perps carry a higher maintenance rate; a reuse of BTC's (R-28) priced them too far away."""
    first = {p: markets.maintenance_for(venue(VENUE), p)[0].rate for p in PAIRS}
    assert first["SOL/USDT"] > first["BTC/USDT"] and first["XRP/USDT"] > first["BTC/USDT"], first


def test_r28s_research_reuse_of_btcs_table_is_gone():
    assert not hasattr(markets, "research_tiers_pair") and not hasattr(markets, "RESEARCH_TIERS_PAIR")
    for pair in PAIRS:
        assert markets.terms({"market": "perp"}, VENUE, pair).label.find("assumed") == -1


def test_a_perp_on_a_pair_with_no_table_is_refused_in_research_too():
    with pytest.raises(markets.NoMaintenanceTable, match="SUI/USDT"):
        markets.terms({"market": "perp"}, VENUE, "SUI/USDT")


@pytest.mark.parametrize("pair", PAIRS)
def test_venues_py_keeps_the_generators_output_verbatim(pair):
    """HoE (PM: fast, as data): the kept tuples are scripts/venue_tiers.py's output from the reply, never typed."""
    import importlib.util

    path = Path(__file__).resolve().parent.parent / "scripts" / "venue_tiers.py"
    spec = importlib.util.spec_from_file_location("venue_tiers", path)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    from sleeve_fund import venues

    assert gen.tuple_source(pair.replace("/", "")) in Path(venues.__file__).read_text()
