import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    (tmp_path / "trend_filter_x.md").write_text("# Tear sheet\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _new(c, **over):
    form = {"name": "btc-test", "strategy": "trend_filter", "instrument": "BTC/USD",
            "bar_spec": "1-HOUR-LAST-INTERNAL", "starting_balance": "5000", "risk_profile": "balanced",
            "warmup_bars": "0", "p_trend_filter__fast": "10", "p_trend_filter__slow": "30",
            "p_buy_and_hold__ignored": "1", "reason": "first test", **over}
    return c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)


def test_needs_password(client):
    c, _ = client
    assert c.get("/").status_code == 401
    assert c.get("/", auth=("pm", "nope")).status_code == 401
    assert c.get("/healthz").status_code == 200


def test_refuses_to_start_without_password(monkeypatch, tmp_path):
    monkeypatch.delenv("DASHBOARD_PASSWORD", raising=False)
    monkeypatch.delenv("DASHBOARD_INSECURE_DEV", raising=False)
    from sleeve_fund.dashboard.app import create_app

    with pytest.raises(RuntimeError):
        create_app(Store(f"sqlite:///{tmp_path}/t.db"))


def test_create_sleeve_and_control_it(client):
    c, store = client
    r = _new(c)
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/btc-test"
    s = store.sleeve("btc-test")
    assert s.params == {"fast": 10, "slow": 30} and s.desired_state == "running"
    assert store.decisions("btc-test")[0]["reason"] == "first test"

    store.record_equity("btc-test", equity=5100, cash=3000, qty=0.02, price=105000, benchmark=5050)
    assert "5,100.00" in c.get("/", auth=AUTH).text
    page = c.get("/sleeves/btc-test", auth=AUTH)
    assert page.status_code == 200 and "Flatten" in page.text
    assert c.get("/api/sleeves/btc-test/equity", auth=AUTH).json()["equity"] == [5100.0]

    r = c.post("/sleeves/btc-test/command", data={"command": "flatten", "reason": "testing"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and store.pending_commands("btc-test")[0]["command"] == "flatten"
    c.post("/sleeves/btc-test/command", data={"command": "stop", "reason": "done"}, auth=AUTH, headers=SAME)
    assert store.sleeve("btc-test").desired_state == "stopped"


def test_exits_entered_as_percent_and_stats_shown(client):
    c, store = client
    r = _new(c, name="sol-stops", instrument="SOL/USD", stop_loss_pct="8", take_profit_pct="20", risk_per_trade_pct="1")
    assert r.status_code == 303
    p = store.sleeve("sol-stops").params
    assert (p["stop_loss"], p["take_profit"], p["risk_per_trade"]) == (0.08, 0.2, 0.01)
    store.record_fill("sol-stops", side="BUY", qty=1, price=100, fee=0.8, order_id="o1", trade_id="t1")
    store.record_fill("sol-stops", side="SELL", qty=1, price=110, fee=0.88, order_id="o2", trade_id="t2")
    page = c.get("/sleeves/sol-stops", auth=AUTH).text
    assert "Stop-loss" in page and "8.0%" in page and "Closed trades" in page and "100% won" in page
    assert "+8.32%" in page  # the round trip's return after both fees: (110 - 100 - 1.68) / 100
    r = _new(c, name="no-stop", risk_per_trade_pct="1")
    assert "stop_loss" in r.headers["location"]


def test_any_asset_pair_can_be_chosen(client):
    c, store = client
    assert 'list="pairs"' in c.get("/sleeves/new", auth=AUTH).text
    r = _new(c, name="sui-trend", instrument="sui/usd")
    assert r.status_code == 303 and store.sleeve("sui-trend").instrument == "SUI/USD"


@pytest.mark.parametrize(
    "over,msg",
    [({"name": "Bad Name"}, "name"), ({"p_trend_filter__fast": "50", "p_trend_filter__slow": "20"}, "fast"),
     ({"risk_profile": "yolo"}, "risk"), ({"reason": " "}, "reason"), ({"p_trend_filter__fsat": "3"}, "unknown")],
)
def test_bad_sleeve_rejected_with_message(client, over, msg):
    c, store = client
    r = _new(c, **over)
    assert r.status_code == 303 and r.headers["location"].startswith("/sleeves/new?error=")
    assert msg in r.headers["location"].lower()
    assert store.sleeves() == []


def test_duplicate_name_rejected(client):
    c, store = client
    _new(c)
    assert "already+exists" in _new(c).headers["location"]


def test_cross_site_post_blocked(client):
    c, store = client
    _new(c)
    r = c.post("/sleeves/btc-test/command", data={"command": "stop", "reason": "x"}, auth=AUTH,
               headers={"origin": "https://evil.example"})
    assert r.status_code == 403
    assert store.sleeve("btc-test").desired_state == "running"


def test_command_needs_reason(client):
    c, _ = client
    _new(c)
    r = c.post("/sleeves/btc-test/command", data={"command": "pause", "reason": " "}, auth=AUTH, headers=SAME)
    assert r.status_code == 400


def test_research_and_tearsheet(client):
    c, _ = client
    assert "trend_filter_x" in c.get("/research", auth=AUTH).text
    assert "<table>" in c.get("/research/trend_filter_x", auth=AUTH).text
    assert c.get("/research/..%2F..%2Fetc%2Fpasswd", auth=AUTH).status_code == 404
    assert c.get("/decisions", auth=AUTH).status_code == 200
    assert c.get("/sleeves/new", auth=AUTH).status_code == 200


def test_portfolio_shows_book_figures_and_alerts_can_be_acknowledged(client):
    c, store = client
    _new(c, name="eth-book", instrument="ETH/USD")
    store.record_equity("eth-book", equity=10_100, cash=5_000, qty=2, price=2_550, benchmark=10_050)
    store.event("eth-book", "warning", "mark_unavailable", "price feed quiet")
    page = c.get("/", auth=AUTH).text
    for text in ("Book equity", "Month to date", "In the market", "Where the money is", "price feed quiet"):
        assert text in page
    alert = store.alerts()[0]
    r = c.post(f"/alerts/{alert['id']}/ack", data={"note": "seen", "next": "/"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert store.open_alert_count() == 0 and store.alerts(include_acked=True)[0]["ack_note"] == "seen"
    assert "price feed quiet" not in c.get("/", auth=AUTH).text
    assert "seen" in c.get("/alerts?show=all", auth=AUTH).text


def test_ack_rejects_cross_site_and_unknown_alerts(client):
    c, store = client
    _new(c)
    store.event("btc-test", "error", "x", "boom")
    eid = store.alerts()[0]["id"]
    assert c.post(f"/alerts/{eid}/ack", auth=AUTH, headers={"Origin": "https://evil.example"}).status_code == 403
    assert c.post("/alerts/99999/ack", auth=AUTH, headers=SAME).status_code == 404
    store.event("btc-test", "info", "start", "not an alert")
    info_id = store.events("btc-test")[0]["id"]
    assert c.post(f"/alerts/{info_id}/ack", auth=AUTH, headers=SAME).status_code == 404
    r = c.post(f"/alerts/{eid}/ack", data={"next": "https://evil.example"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.headers["location"] == "/alerts"  # no open redirect


def test_book_and_sleeve_chart_data(client):
    c, store = client
    _new(c)
    store.record_equity("btc-test", equity=10_000, cash=10_000, qty=0, price=60_000, benchmark=10_000)
    store.record_fill("btc-test", side="BUY", qty=0.1, price=60_000, fee=48, order_id="o", trade_id="t")
    book = c.get("/api/book/equity", auth=AUTH).json()
    assert book["equity"] and len(book["t"]) == len(book["drawdown"])
    sleeve = c.get("/api/sleeves/btc-test/equity", auth=AUTH).json()
    assert sleeve["res"] == "intraday" and sleeve["fills"][0]["side"] == "BUY"
    assert c.get("/api/sleeves/nope/equity", auth=AUTH).status_code == 404


def test_flatten_asks_for_confirmation_with_its_effect(client):
    c, store = client
    _new(c)
    store.record_equity("btc-test", equity=10_000, cash=4_000, qty=0.1, price=60_000, benchmark=10_000)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert 'id="dlg-flatten"' in page and "Sells 0.1 BTC (about 6,000.00 USD) at market" in page


def test_risk_page_stress_and_limits(client):
    c, store = client
    _new(c, name="eth-risk", instrument="ETH/USD")
    # 60% of a 10,000 sleeve in ETH: a 50% fall costs 3,000, a 30% drawdown, past the balanced 20% limit.
    store.record_equity("eth-risk", equity=10_000, cash=4_000, qty=2, price=3_000, benchmark=10_000)
    store.event("eth-risk", "error", "risk_halt", "drawdown 21% hit the 20% limit")
    page = c.get("/risk", auth=AUTH).text
    assert "Limits by sleeve" in page and "−3,000.00" in page and "eth-risk</span>" in page
    assert "drawdown 21% hit the 20% limit" in page


def test_ops_page_shows_processes_and_safety_nets(client):
    c, store = client
    _new(c)
    page = c.get("/ops", auth=AUTH).text
    assert "Sleeve processes" in page and "btc-test" in page and "Dead man" in page and "Database size" in page


def test_preview_runs_the_form_settings_on_history(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    preview._history.clear()
    monkeypatch.setattr(preview, "fetch_kraken_daily", lambda pair: synthetic_ohlcv(days=400, seed=2, start_price=150))
    q = {"strategy": "trend_filter", "instrument": "sol/usd", "p_trend_filter__fast": "10",
         "p_trend_filter__slow": "40", "stop_loss_pct": "8", "starting_balance": "5000"}
    d = c.get("/api/preview", params=q, auth=AUTH).json()
    assert d["pair"] == "SOL/USD" and d["days"] == 400 and d["equity"][0] == pytest.approx(5000, rel=0.01)
    assert set(d["strategy"]) >= {"sharpe", "max_drawdown"} and d["trades"]["trades"] >= 1
    bad = c.get("/api/preview", params={**q, "p_trend_filter__fast": "50"}, auth=AUTH)  # fast must be < slow
    assert bad.status_code == 422 and "error" in bad.json()
    assert c.get("/api/preview", params={**q, "instrument": "nonsense"}, auth=AUTH).status_code == 422


def test_research_pipeline_and_strategy_pages(client):
    c, _ = client
    _new(c)  # a buy_and_hold sleeve, with no real G1 pass, so it shows as an observation
    page = c.get("/research", auth=AUTH).text
    assert "Pipeline" in page and "observation" in page and "trend filter" in page
    s = c.get("/strategies/trend_filter", auth=AUTH).text
    assert "Exact rules" in s and "Start a sleeve with this" in s
    assert c.get("/strategies/nope", auth=AUTH).status_code == 404
    form = c.get("/sleeves/new?strategy=rsi_pullback", auth=AUTH).text
    assert '<option value="rsi_pullback" selected' in form


def test_fetch_kraken_daily_parses_and_drops_the_open_candle():
    from sleeve_fund.data import fetch_kraken_daily

    day = 86400
    canned = {
        "AssetPairs": {"error": [], "result": {"XXBTZUSD": {"wsname": "XBT/USD"}, "SUIUSD": {"wsname": "SUI/USD"}}},
        "OHLC": {"error": [], "result": {"XXBTZUSD": [[1_700_000_000 + i * day, "100", "110", "95", "105", "102", "7", 3]
                                                      for i in range(5)], "last": 1}},
    }
    seen = []

    def get_json(url):
        seen.append(url)
        return canned["AssetPairs" if "AssetPairs" in url else "OHLC"]

    df = fetch_kraken_daily("BTC/USD", get_json=get_json)
    assert len(df) == 4 and "pair=XXBTZUSD" in seen[-1]
    assert df.index[0].timestamp() == 1_700_000_000 + day  # stamped at the bar's close, not its open
    with pytest.raises(ValueError):
        fetch_kraken_daily("NOPE/USD", get_json=get_json)


def test_decision_log_filters_and_csv_keeps_formulas_as_text(client):
    c, store = client
    _new(c, name="btc-a")
    _new(c, name="eth-b", instrument="ETH/USD", reason="=HYPERLINK(\"https://evil.example\")")
    c.post("/sleeves/btc-a/command", data={"command": "pause", "reason": "news"}, auth=AUTH, headers=SAME)
    page = c.get("/decisions?sleeve=btc-a&action=pause", auth=AUTH).text
    assert "news" in page and "first test" not in page
    assert c.get("/decisions?from=not-a-date", auth=AUTH).status_code == 200
    r = c.get("/decisions.csv?sleeve=eth-b", auth=AUTH)
    assert r.headers["content-disposition"].startswith('attachment; filename="sleeve-fund-decisions-')
    lines = r.text.splitlines()
    assert lines[0] == "ts,actor,action,sleeve,reason" and len(lines) == 2
    assert "'=HYPERLINK" in lines[1]


def test_exports_reports_and_settings(client):
    c, store = client
    _new(c, name="sol-x", instrument="SOL/USD")
    store.record_equity("sol-x", equity=10_000, cash=10_000, qty=0, price=100, benchmark=10_000)
    store.record_fill("sol-x", side="BUY", qty=10, price=100, fee=0.8, order_id="o1", trade_id="t1")
    store.record_fill("sol-x", side="SELL", qty=10, price=110, fee=0.88, order_id="o2", trade_id="t2")
    fills = c.get("/exports/fills.csv?sleeve=sol-x", auth=AUTH).text.splitlines()
    assert len(fills) == 3 and ",BUY," in fills[1] and ",SELL," in fills[2]  # oldest first
    trades = c.get("/exports/trades.csv", auth=AUTH).text.splitlines()
    assert len(trades) == 2 and trades[1].startswith("sol-x,")
    assert c.get("/exports/equity.csv", auth=AUTH).status_code == 200
    assert c.get("/exports/secrets.csv", auth=AUTH).status_code == 404
    assert c.get("/exports/fills.csv?sleeve=nope", auth=AUTH).status_code == 404
    assert c.get("/exports/fills.csv").status_code == 401
    assert "Book by month" in c.get("/reports", auth=AUTH).text
    settings = c.get("/settings", auth=AUTH).text
    assert "Live trading" in settings and "test-pw" not in settings
