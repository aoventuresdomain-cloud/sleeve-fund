"""C2 (ui-v2/compliance-build-spec.md): each test's In use now bar is the engine's own figure (PE2's
supervisor.compliance_use: money used of the money each test allows), the largest asset or group as the headline and
the rest on hover; a book never marked reads as a dash, and one that can't be read says so, never as room."""

from __future__ import annotations

from decimal import Decimal as D

from test_dashboard import AUTH, client  # noqa: F401 - the fixture

from sleeve_fund import compliance_rules as rules
from sleeve_fund import supervisor

KEYS = ("asset_stop_loss", "type_stop_loss", "fund_stop_loss", "fund_gross", "asset_net", "fund_margin",
        "fund_day_loss", "fund_drawdown")


def _use(equity=D(100000), **over):
    tests = {k: {"test": rules.BY_KEY[k].test, "limit": D(1000), "used": D(0)} for k in KEYS}
    for k in ("asset_stop_loss", "asset_net"):
        tests[k]["used"] = {}
    tests["type_stop_loss"]["used"] = {}
    tests.update(over)
    return {"version": 1, "equity": equity, "marked_at": None, "tests": tests}


def _limit(store, key):
    return D(str(next(t for t in store.compliance().tests if t["test"] == key)["limit"]))


def _row(page, key):
    return page.split(f'data-test="{key}"')[1].split("</tr>")[0]


def test_the_bars_read_the_engines_use_with_the_largest_asset_and_the_rest_on_hover(client, monkeypatch):  # noqa: F811
    c, store = client
    use = _use(asset_stop_loss={"test": "T1", "limit": D(4000), "used": {"ETH": D(1000), "BTC": D(3600)}},
               fund_gross={"test": "T4", "limit": D(150000), "used": D(75000)})
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: use, raising=False)
    page = c.get("/portfolio-settings", auth=AUTH).text
    row = _row(page, "asset_stop_loss")
    assert f"BTC {rules.show('asset_stop_loss', D('0.9') * _limit(store, 'asset_stop_loss'))} " in row  # 3,600 of 4,000
    assert 'title="BTC 90% of its limit · ETH 25% of its limit"' in row and 'class="ps-use warn"' in row
    assert "width:90.0%" in row
    assert "width:50.0%" in _row(page, "fund_gross")


def test_net_is_shown_by_its_size_whichever_side(client, monkeypatch):  # noqa: F811
    c, _ = client
    use = _use(asset_net={"test": "T4", "limit": D(50000), "used": {"BTC": D(10000), "ETH": D(-40000)}})
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: use, raising=False)
    row = _row(c.get("/portfolio-settings", auth=AUTH).text, "asset_net")
    assert "ETH " in row and "width:80.0%" in row and 'title="ETH 80% of its limit · BTC 20% of its limit"' in row


def test_an_asset_outside_the_group_map_heads_the_group_test_as_unmapped(client, monkeypatch):  # noqa: F811
    c, _ = client
    use = _use(type_stop_loss={"test": "T2", "limit": D(6000), "used": {"unmapped": D(3000), "small": D(600)}})
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: use, raising=False)
    row = _row(c.get("/portfolio-settings", auth=AUTH).text, "type_stop_loss")
    assert "Unmapped " in row and "width:50.0%" in row and 'title="Unmapped 50% of its limit · Small 10% of its limit"' in row


def test_a_book_never_marked_shows_a_dash_not_room(client, monkeypatch):  # noqa: F811
    c, _ = client
    monkeypatch.setattr(supervisor, "compliance_use", lambda store: _use(equity=None), raising=False)
    page = c.get("/portfolio-settings", auth=AUTH).text
    assert '<span class="muted">–</span>' in _row(page, "fund_gross") and "The fund's book has no mark yet" in page


def test_a_book_the_engine_cant_read_shows_a_dash_not_room(client, monkeypatch):  # noqa: F811
    c, _ = client

    def boom(store):
        raise ValueError("a position has no mark")

    monkeypatch.setattr(supervisor, "compliance_use", boom, raising=False)
    page = c.get("/portfolio-settings", auth=AUTH).text
    assert '<span class="muted">–</span>' in _row(page, "fund_gross")
