"""VS-1-TABLE (CR review of head 2, MUST 1 and 2): a perp position the journal already holds on a pair whose venue tiers
aren't kept (ETH/USDT here) never takes a page down. Its liquidation reads as unknown; its margin is still the
isolated margin, and the most it can lose is its equity. The refusal's words never name the venue."""

import pytest
from test_dashboard import (  # noqa: F401 - the fixtures
    AUTH,
    _strategy_accounts,
    client,
)
from test_portfolio_page import _book

from sleeve_fund import markets
from sleeve_fund.venues import venue

pytestmark = pytest.mark.no_maintenance_tables  # the venue's real tables: BTC/USDT only


@pytest.mark.parametrize("path", ["/", "/risk", "/trades", "/sleeves/rsi-long", "/sleeves/pp-short"])
def test_every_page_reads_a_perp_with_no_kept_table_without_raising(client, path):  # noqa: F811
    c, store = client
    _book(store)  # rsi-long holds an ETH/USDT perp; ETH has no kept table
    r = c.get(path, auth=AUTH)
    assert r.status_code == 200, r.text[-800:]


def test_the_readers_give_the_liquidation_as_unknown_and_the_margin_as_isolated():
    from types import SimpleNamespace

    from sleeve_fund.dashboard import riskops, trading

    s = SimpleNamespace(params={"market": "perp"}, venue="BINANCE", instrument="ETH/USDT")
    x = {"sleeve": s, "qty": 1.0, "cash": 7_000.0, "price": 2_970.0, "entry_px": 3_000.0, "equity": 9_970.0,
         "profile": "balanced", "unrealised": -30.0}
    margin, liq = trading.position_margin(x)
    assert liq is None and margin == pytest.approx(1_500.0)  # 3,000 at the balanced 2x cap
    assert trading.perp_view(x, None, [])["liq_px"] is None
    assert riskops.most_it_can_lose(x) == 9_970.0  # unknown liquidation: no more than its equity


def test_the_refusal_never_names_the_venue():
    with pytest.raises(markets.NoMaintenanceTable) as e:
        markets.maintenance_for(venue("BINANCE"), "ETH/USDT")
    assert "ETH/USDT has no maintenance table kept" in str(e.value)
    assert venue("BINANCE").label not in str(e.value) and "Binance" not in str(e.value)
