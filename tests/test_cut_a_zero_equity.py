"""CUT-A (Advisor; HoE 14 Oct 22:52 UK, a MUST for Bridge Drop A): after the reset the fund holds the cash and a
Strategy's own equity is 0 plus its P&L. No page may throw or show inf or nan for a Strategy at 0, below it or just
above it; per-Strategy figures read as attribution, never as a share of its own equity."""

from __future__ import annotations

import re

import pytest
from test_dashboard import AUTH, _strategy_accounts, client  # noqa: F401 - the fixtures

PAGES = ["/", "/risk", "/alerts", "/records", "/trades", "/orders", "/setup", "/sleeves/zero",
         "/api/sleeves/zero/equity", "/api/sleeves/zero/equity?days=1", "/api/book/equity"]


@pytest.mark.parametrize("equity", [0.0, -37.5, 12.25])
def test_no_page_throws_or_shows_inf_for_a_strategy_with_no_capital_of_its_own(client, equity):  # noqa: F811
    c, store = client
    store.create_sleeve(name="zero", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=0, params={"market": "perp", "allow_short": True})
    store.record_fill("zero", side="BUY", qty=0.01, price=60_000, fee=0.5, order_id="z-1", trade_id="z-t1")
    store.record_equity("zero", equity=equity, cash=equity - 600, qty=0.01, price=60_000, benchmark=0.0)
    for path in PAGES:
        r = c.get(path, auth=AUTH)
        if r.status_code == 404:
            continue
        assert r.status_code == 200, (path, r.status_code, r.text[-400:])
        text = re.sub(r"<script\b.*?</script>", "", r.text, flags=re.DOTALL)
        bad = [text[max(0, m.start() - 90):m.end() + 30] for m in
               re.finditer(r"(?<![\w-])(-?inf|nan|Infinity|NaN)(?![\w-])", text, flags=re.IGNORECASE)]
        assert not bad, (path, bad)
