"""Multi-asset spec §7 (start_check.py): the server's start check checks a multi-asset Strategy asset by asset. Every
listed asset has its row, and every journal row is keyed to its own asset (DA 0028's reconcile by (Strategy, asset)).
One failing asset fails the check, named."""

from __future__ import annotations

from sqlalchemy import text

from sleeve_fund import start_check
from sleeve_fund.store import Store
from test_multi_asset_legs import _fund, _make


def _journal(tmp_path):
    url = f"sqlite:///{tmp_path}/ma.db"
    store = _fund(Store(url))
    _make(store)
    return url, store


def test_a_multi_asset_strategy_with_every_asset_in_place_passes_its_section(tmp_path):
    url, store = _journal(tmp_path)
    store.engine.dispose()
    _, page = start_check.run(url, environ={})
    assert "Multi-asset Strategies: PASS" in page, page
    assert "trend-x: BTC/USDT, ETH/USDT, SOL/USDT, each with its row" in page


def test_a_missing_asset_or_a_journal_row_on_the_wrong_asset_fails_and_is_named(tmp_path):
    url, store = _journal(tmp_path)
    store.record_fill("trend-x~ETH", side="BUY", qty=0.5, price=2400.0, fee=0.6, order_id="o1", trade_id="t1")
    with store.engine.begin() as c:
        c.execute(text("UPDATE fills SET instrument = 'SOL/USDT' WHERE sleeve = 'trend-x~ETH'"))
        c.execute(text("DELETE FROM sleeves WHERE name = 'trend-x~SOL'"))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok
    assert "FAIL  trend-x: SOL/USDT has no row" in page, page
    assert "FAIL  fills row" in page and "trend-x~ETH is keyed to SOL/USDT, not its ETH/USDT" in page
