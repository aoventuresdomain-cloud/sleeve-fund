"""C2 on a real book (HoE, TEC-FE-C2): the Portfolio Settings rows show what PE2's supervisor.compliance_use returns
for the same book, used of its limit, not the page's own reckoning. SQLite, and Postgres with TEST_DATABASE_URL."""

from __future__ import annotations

from test_drop_a_fund_book import MIN, _entry, _rebased
from test_halt_ref import T0, _make, _mark, _trade, store  # noqa: F401

from sleeve_fund import supervisor
from sleeve_fund.dashboard import compliance


def test_the_rows_show_the_engines_own_use_of_each_limit_on_a_real_book(store, monkeypatch):  # noqa: F811
    _rebased(store)
    _make(store, "x", 0)
    _trade(store, "x", 100, T0 + MIN)
    _mark(store, T0 + MIN)
    assert _entry(store, T0 + MIN).outcome == "approved"  # 0.001 BTC at 60,000: margin 60, risk to its stop 1.2
    real, calls = supervisor.compliance_use, []
    monkeypatch.setattr(supervisor, "compliance_use", lambda s, version=None: calls.append(1) or real(s, version))
    want = real(store)["tests"]
    rows = {r["key"]: r for g in compliance.page(store)["tests"] for r in g["rows"]}
    assert calls  # the page asked the engine
    for key in ("fund_stop_loss", "fund_gross", "fund_margin"):
        t = want[key]
        assert rows[key]["fill"] == float(t["used"] / t["limit"] * 100), key
    t1 = want["asset_stop_loss"]
    assert rows["asset_stop_loss"]["what"] == "BTC"
    assert rows["asset_stop_loss"]["fill"] == float(t1["used"]["BTC"] / t1["limit"] * 100)
    assert rows["asset_stop_loss"]["rest"] == f"BTC {float(t1['used']['BTC'] / t1['limit'] * 100):.0f}% of its limit"
    net = want["asset_net"]
    assert rows["asset_net"]["fill"] == float(abs(net["used"]["BTC"]) / net["limit"] * 100)
