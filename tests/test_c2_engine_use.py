"""C2 (ui-v2/compliance-build-spec.md): each test's In use now bar is the engine's own figure (PE2's compliance_use),
the largest asset or group as the headline and the rest on hover; a stale book reads as a dash, never as room."""

from __future__ import annotations

from decimal import Decimal as D

from test_dashboard import AUTH, client  # noqa: F401 - the fixture

from sleeve_fund import supervisor


def _use(**over):
    base = {k: {"used": D(0), "limit": D(1), "share": 0.0, "enabled": True, "version": 1}
            for k in ("asset_stop_loss", "type_stop_loss", "fund_stop_loss", "fund_gross", "asset_net", "fund_margin",
                      "fund_day_loss", "fund_drawdown")}
    base.update(over)
    return base


def test_the_bars_read_the_engines_use_with_the_largest_asset_and_the_rest_on_hover(client, monkeypatch):  # noqa: F811
    c, store = client
    use = _use(asset_stop_loss={"used": D(3600), "limit": D(4000), "share": 0.9, "enabled": True, "version": 1,
                                "by_asset": {"BTC": 0.9, "ETH": 0.25}},
               fund_gross={"used": D(75000), "limit": D(150000), "share": 0.5, "enabled": True, "version": 1})
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: use, raising=False)
    t1 = next(t for t in store.compliance().tests if t["test"] == "asset_stop_loss")["limit"]
    page = c.get("/portfolio-settings", auth=AUTH).text
    row = page.split('data-test="asset_stop_loss"')[1].split("</tr>")[0]
    from sleeve_fund import compliance_rules as rules

    assert f"BTC {rules.show('asset_stop_loss', D('0.9') * D(str(t1)))} " in row  # 0.9 of the limit, as a share of the book
    assert 'title="BTC 90% of its limit · ETH 25% of its limit"' in row and 'class="ps-use warn"' in row
    gross = page.split('data-test="fund_gross"')[1].split("</tr>")[0]
    assert "width:50.0%" in gross


def test_a_stale_book_shows_a_dash_not_room(client, monkeypatch):  # noqa: F811
    c, _ = client
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: {"stale": True}, raising=False)
    page = c.get("/portfolio-settings", auth=AUTH).text
    row = page.split('data-test="fund_gross"')[1].split("</tr>")[0]
    assert '<span class="muted">–</span>' in row and "The fund's book has no mark yet" in page
