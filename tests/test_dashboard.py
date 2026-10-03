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
    for text in ("Book equity", "Month to date", "Gross exposure", "Where the money is", "price feed quiet"):
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


def _round_trip(store, sleeve="sol-x"):
    store.record_order(sleeve, order_id="O-1", side="BUY", qty=10, intent="entry",
                       reason="RSI 28.0 below 30, close 100 above the 200-bar EMA 95", signal={"rsi": 28.0, "close": 100.0})
    store.record_fill(sleeve, side="BUY", qty=10, price=100, fee=0.8, order_id="O-1", trade_id="t1")
    store.record_order(sleeve, order_id="O-2", side="SELL", qty=10, intent="stop_loss",
                       reason="Stop-loss: price 92 is -8.00% from the 100 entry", signal={"move": -0.08})
    store.record_fill(sleeve, side="SELL", qty=10, price=92, fee=0.74, order_id="O-2", trade_id="t2")


def test_trades_page_shows_positions_trades_and_the_journaled_reasons(client):
    c, store = client
    _new(c, name="sol-x", instrument="SOL/USD")
    _round_trip(store)
    store.record_order("sol-x", order_id="O-3", side="BUY", qty=5, intent="entry", reason="RSI 25.1 below 30")
    store.record_fill("sol-x", side="BUY", qty=5, price=90, fee=0.36, order_id="O-3", trade_id="t3")
    store.record_equity("sol-x", equity=9_900, cash=9_450, qty=5, price=99, benchmark=9_950)
    page = c.get("/trades", auth=AUTH).text
    assert "Open positions" in page and "RSI 25.1 below 30" in page  # why the open position was bought
    assert "+45.00" in page  # unrealised: 5 x (99 - 90)
    assert "RSI 28.0 below 30" in page and "Stop-loss: price 92" in page and "Move from entry" in page
    assert "−81.54" in page  # (92 - 100) x 10 - 1.54 fees
    assert c.get("/trades?sleeve=nope", auth=AUTH).status_code == 200  # unknown sleeve shows the book
    sleeve = c.get("/sleeves/sol-x", auth=AUTH).text
    assert "Why it was bought" in sleeve and "Stop-loss: price 92" in sleeve and "Blotter" in sleeve


def test_trades_from_before_the_order_journal_say_so(client):
    c, store = client
    _new(c, name="old", instrument="SOL/USD")
    store.record_fill("old", side="BUY", qty=1, price=100, fee=0.8, order_id="x1", trade_id="t1")
    store.record_fill("old", side="SELL", qty=1, price=110, fee=0.88, order_id="x2", trade_id="t2")
    assert "predates the order journal" in c.get("/trades", auth=AUTH).text


def test_order_blotter_tabs_and_csv(client):
    c, store = client
    _new(c, name="sol-x", instrument="SOL/USD")
    _round_trip(store)
    store.record_order("sol-x", order_id="O-9", side="BUY", qty=3, intent="entry", reason="test reject")
    store.update_order("O-9", status="rejected", message="EOrder:Insufficient funds")
    store.record_order("sol-x", order_id="O-10", side="BUY", qty=3, intent="entry", reason="still working")
    page = c.get("/orders", auth=AUTH).text
    assert "Order blotter" in page and "Stop-loss" in page and "EOrder:Insufficient funds" in page
    rejected = c.get("/orders?status=rejected", auth=AUTH).text
    assert "test reject" in rejected and "still working" not in rejected
    assert "still working" in c.get("/orders?status=open", auth=AUTH).text
    assert "test reject" not in c.get("/orders?status=filled", auth=AUTH).text
    assert c.get("/orders?status=bogus", auth=AUTH).status_code == 200
    csv_text = c.get("/exports/orders.csv?sleeve=sol-x", auth=AUTH).text.splitlines()
    assert csv_text[0].startswith("ts,sleeve,order_id,side") and len(csv_text) == 5
    assert '""rsi"": 28.0' in csv_text[1]
    trades_csv = c.get("/exports/trades.csv", auth=AUTH).text
    assert "stop_loss" in trades_csv and "RSI 28.0 below 30" in trades_csv


@pytest.mark.parametrize("path", ["/", "/trades", "/orders", "/alerts", "/risk", "/reports", "/settings"])
def test_every_page_offers_new_sleeve_and_reaches_every_page(client, path):
    c, _ = client
    page = c.get(path, auth=AUTH).text
    assert 'class="button rail-new" href="/sleeves/new"' in page and 'id="more"' in page
    for href in ("/trades", "/orders", "/alerts", "/risk", "/ops", "/research", "/decisions", "/reports", "/settings"):
        assert f'href="{href}"' in page  # nothing is desktop-only any more; phones reach it through More


def test_backtest_page_shows_every_trade_with_its_reason_and_hands_off_to_a_sleeve(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    preview._history.clear()
    monkeypatch.setattr(preview, "fetch_kraken_daily", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    assert "How testing works" in c.get("/backtest", auth=AUTH).text
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
         "&p_trend_filter__slow=20&starting_balance=5000&period=365")
    page = c.get(q, auth=AUTH).text
    assert "Trend filter on ETH/USD" in page and "Every trade" in page and "Buy and hold" in page
    assert "5-bar average" in page and "above the 20-bar average" in page  # the entry reasons, from the strategy
    assert "365 days" in page
    # The sleeve decides on the daily bars that were tested, warm from its first bar (2 x the slow 20).
    assert ('href="/sleeves/new?instrument=ETH%2FUSD&amp;strategy=trend_filter&amp;p_trend_filter__fast=5'
            '&amp;p_trend_filter__slow=20&amp;starting_balance=5000&amp;bar_spec=1-DAY-LAST-EXTERNAL'
            '&amp;warmup_bars=40&amp;from=backtest"') in page
    assert "33% invested" in page  # the benchmark is held at the balanced profile's cap
    form = c.get("/sleeves/new?instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5&from=backtest"
                 "&bar_spec=1-DAY-LAST-EXTERNAL&warmup_bars=40", auth=AUTH).text
    assert 'value="ETH/USD"' in form and 'name="p_trend_filter__fast" value="5"' in form and "carried over" in form
    assert '<input type="hidden" name="bar_spec" value="1-DAY-LAST-EXTERNAL">' in form
    assert '<select id="bar_spec" disabled>' in form and 'name="warmup_bars" type="number" min="0" max="720" value="40"' in form


def test_sleeve_from_a_backtest_cannot_change_its_interval(client):
    c, _ = client
    r = _new(c, name="bt-hourly", bar_spec="1-HOUR-LAST-INTERNAL", **{"from": "backtest"})
    assert r.status_code == 303 and "interval" in r.headers["location"] and "/sleeves/new" in r.headers["location"]
    r = _new(c, name="bt-daily", bar_spec="1-DAY-LAST-EXTERNAL", warmup_bars="400", **{"from": "backtest"})
    assert r.headers["location"] == "/sleeves/bt-daily"


def test_backtest_sizes_with_the_chosen_risk_profile(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    preview._history.clear()
    monkeypatch.setattr(preview, "fetch_kraken_daily", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&risk_profile=conservative"
                 "&max_notional=1000&period=365", auth=AUTH).text
    assert "20% invested" in page and "sleeve cap" in page  # the 1,000 order cap binds before the 20% cap
    assert "risk profile" in c.get("/backtest?run=1&instrument=ETH/USD&risk_profile=reckless", auth=AUTH).text


@pytest.mark.parametrize("query, msg", [
    ("instrument=nonsense", "BASE/QUOTE"),
    ("instrument=BTC/USD&starting_balance=5", "capital"),
    ("instrument=BTC/USD&p_trend_filter__fast=30&p_trend_filter__slow=10", "fast"),
])
def test_backtest_explains_bad_settings(client, monkeypatch, query, msg):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    preview._history.clear()
    monkeypatch.setattr(preview, "fetch_kraken_daily", lambda pair: synthetic_ohlcv(days=200, seed=3))
    page = c.get(f"/backtest?run=1&strategy=trend_filter&{query}", auth=AUTH).text
    assert "Couldn't run it" in page and msg in page


def test_new_sleeve_errors_keep_what_was_typed(client):
    c, _ = client
    r = _new(c, name="BAD NAME", reason="testing keeps fields")
    assert r.status_code == 303
    form = c.get(r.headers["location"], auth=AUTH).text
    assert "Not saved" in form and 'value="testing keeps fields"' in form


def test_accounts_page_adds_live_accounts_and_shows_key_presence_only(client, monkeypatch):
    from sleeve_fund import accounts
    from sleeve_fund.supervisor import Supervisor

    c, store = client
    page = c.get("/accounts", auth=AUTH).text
    assert "paper" in page and "Connect a Kraken sub-account" in page and "Never tick Withdraw Funds" in page
    r = c.post("/accounts/new", data={"name": "kraken-trend", "kind": "live", "reason": "first live sub-account"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303
    assert "KRAKEN_API_KEY__KRAKEN_TREND=your-api-key" in c.get("/accounts", auth=AUTH).text
    assert "Not checked yet" in c.get("/accounts", auth=AUTH).text
    # The supervisor reports presence; the secret's value never reaches the store or the page.
    monkeypatch.setenv("KRAKEN_API_KEY__KRAKEN_TREND", "k-SECRET-VALUE")
    monkeypatch.setenv("KRAKEN_API_SECRET__KRAKEN_TREND", "s-SECRET-VALUE")
    assert accounts.key_present("kraken-trend")
    Supervisor(store).check_keys()
    page = c.get("/accounts", auth=AUTH).text
    assert "Installed" in page and "SECRET-VALUE" not in page
    assert "Installed</span> for 1 of 1" in c.get("/settings", auth=AUTH).text
    assert any(d["action"] == "create_account" for d in store.decisions())


def test_account_rules(client):
    c, store = client
    bad = c.post("/accounts/new", data={"name": "Bad Name", "kind": "live", "reason": "x"}, auth=AUTH, headers=SAME,
                 follow_redirects=False)
    assert "error=" in bad.headers["location"] and "name=Bad+Name" in bad.headers["location"]
    assert c.post("/accounts/new", data={"name": "x1", "kind": "live", "reason": "x"}, auth=AUTH,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    store.create_account("kraken-live", "live")
    store.create_account("research", "paper")
    form = c.get("/sleeves/new", auth=AUTH).text
    assert 'value="kraken-live" disabled' in form and "live, locked until G2" in form
    r = _new(c, name="on-live", account="kraken-live")
    assert "locked+until+G2" in r.headers["location"]
    assert _new(c, name="on-research", account="research").status_code == 303
    assert store.account_of("on-research") == "research" and store.account_of("btc-nothing") == "paper"
    assert "research</a>" in c.get("/sleeves/on-research", auth=AUTH).text


def test_paper_processes_do_not_inherit_kraken_keys(monkeypatch, tmp_path):
    import subprocess

    from sleeve_fund.supervisor import Proc, Supervisor

    seen = {}
    monkeypatch.setenv("KRAKEN_API_KEY__X", "k")
    monkeypatch.setenv("KRAKEN_API_SECRET__X", "s")
    monkeypatch.setattr(subprocess, "Popen", lambda args, env=None: seen.update(env=env) or type("P", (), {"pid": 1})())
    store = Store(f"sqlite:///{tmp_path}/t.db")
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=1000)
    Supervisor(store)._start("s1", Proc())
    assert not any(k.startswith("KRAKEN_API") for k in seen["env"])


def test_price_chart_marks_fills_with_reasons_and_falls_back_to_marks(client, monkeypatch):
    import pandas as pd

    from sleeve_fund.dashboard import charts

    c, store = client
    _new(c, name="sol-x", instrument="SOL/USD", bar_spec="1-HOUR-LAST-INTERNAL", stop_loss_pct="8")
    store.record_equity("sol-x", equity=10_000, cash=9_000, qty=10, price=100, benchmark=10_000)
    _round_trip(store)
    store.record_order("sol-x", order_id="O-3", side="BUY", qty=10, intent="entry", reason="RSI 25.1 below 30")
    store.record_fill("sol-x", side="BUY", qty=10, price=90, fee=0.72, order_id="O-3", trade_id="t3")
    store.record_equity("sol-x", equity=9_950, cash=9_000, qty=10, price=95, benchmark=9_900)

    def down(pair, minutes):
        raise OSError("no route to Kraken")

    charts._cache.clear()
    monkeypatch.setattr(charts, "fetch_kraken_ohlc", down)
    d = c.get("/api/sleeves/sol-x/candles", auth=AUTH).json()
    assert d["source"] == "marks" and d["chosen"] == "1h" and d["candles"]
    assert [m["text"] for m in d["markers"]] == ["", "Stop", ""]  # only the forced exit is labelled
    notes = list(d["notes"].values())
    assert notes[1]["reason"].startswith("Stop-loss: price 92") and notes[2]["reason"] == "RSI 25.1 below 30"
    assert {ln["title"] for ln in d["lines"]} == {"Entry", "Stop"} and d["lines"][0]["price"] == 90

    now = pd.Timestamp.now(tz="UTC").floor("h")
    kraken = pd.DataFrame({"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [2.0, 2.5],
                           "volume": [10.0, 20.0]}, index=pd.DatetimeIndex([now - pd.Timedelta("4h"), now]))
    charts._cache.clear()
    monkeypatch.setattr(charts, "fetch_kraken_ohlc", lambda pair, minutes: kraken)
    d = c.get("/api/sleeves/sol-x/candles?interval=4h", auth=AUTH).json()
    assert d["source"] == "kraken" and d["interval"] == 240 and len(d["candles"]) == 2 and d["volume"][1]["value"] == 20
    assert c.get("/api/sleeves/nope/candles", auth=AUTH).status_code == 404
    assert 'id="pc"' in c.get("/sleeves/sol-x", auth=AUTH).text


def test_chart_helpers():
    from sleeve_fund.dashboard import charts

    assert charts.default_interval("1-DAY-LAST-EXTERNAL") == "1d"
    assert charts.default_interval("5-MINUTE-LAST-INTERNAL") == "15m"
    assert charts.default_interval("4-HOUR-LAST-EXTERNAL") == "4h"


def test_fetch_kraken_ohlc_keeps_the_forming_candle_and_uses_open_times():
    from sleeve_fund.data import fetch_kraken_ohlc

    def get_json(url):
        if "AssetPairs" in url:
            return {"result": {"SOLUSD": {"wsname": "SOL/USD"}}}
        assert "interval=60" in url
        return {"error": [], "result": {"SOLUSD": [[1700000000, "1", "2", "0.5", "1.5", "1.2", "10", 5],
                                                   [1700003600, "1.5", "2.5", "1", "2", "1.8", "4", 2]], "last": 1}}

    df = fetch_kraken_ohlc("SOL/USD", 60, get_json)
    assert len(df) == 2 and df.index[0].timestamp() == 1700000000 and df["close"].iloc[-1] == 2.0
    with pytest.raises(ValueError):
        fetch_kraken_ohlc("SOL/USD", 7, get_json)


def test_every_headline_figure_explains_itself(client):
    from sleeve_fund.dashboard.glossary import GLOSSARY

    c, store = client
    _new(c)
    for path in ("/", "/sleeves/btc-test", "/trades"):
        page = c.get(path, auth=AUTH).text
        assert 'class="help"' in page and 'aria-label="What does this mean?"' in page
    assert GLOSSARY["drawdown"] in c.get("/", auth=AUTH).text


def test_clone_with_changes_prefills_the_new_sleeve_form(client):
    from urllib.parse import parse_qs, urlparse

    c, store = client
    _new(c, name="sol-stops", instrument="SOL/USD", stop_loss_pct="8", max_notional="500")
    page = c.get("/sleeves/sol-stops", auth=AUTH).text
    href = next(p for p in page.split('"') if p.startswith("/sleeves/new?"))
    q = {k: v[0] for k, v in parse_qs(urlparse(href.replace("&amp;", "&")).query).items()}
    assert q["instrument"] == "SOL/USD" and q["stop_loss_pct"] == "8" and q["max_notional"] == "500"
    assert q["p_trend_filter__fast"] == "10" and q["name"] == "sol-stops-v2" and q["from"] == "clone"
    form = c.get(href.replace("&amp;", "&"), auth=AUTH).text
    assert "Settings copied from" in form and 'value="sol-stops-v2" data-touched=1' in form
    # Submitting the clone unchanged (bar the reason) gives an identical second sleeve.
    form_data = {k: v for k, v in q.items() if k not in ("from", "source")}
    assert _new(c, **form_data, reason="same again").status_code == 303
    assert store.sleeve("sol-stops-v2").params == store.sleeve("sol-stops").params


def test_archive_hides_a_stopped_sleeve_and_restore_brings_it_back(client):
    c, store = client
    _new(c)
    post = lambda action, reason="done with it": c.post(  # noqa: E731
        "/sleeves/btc-test/archive", data={"action": action, "reason": reason}, auth=AUTH, headers=SAME,
        follow_redirects=False)
    assert post("archive").status_code == 400  # still running
    store.set_desired_state("btc-test", "stopped")
    assert post("archive", reason=" ").status_code == 400
    assert c.post("/sleeves/btc-test/archive", data={"action": "archive", "reason": "x"}, auth=AUTH,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    assert post("archive").status_code == 303
    home = c.get("/", auth=AUTH).text
    assert "Archived sleeves (1)" in home and home.count('href="/sleeves/btc-test"') == 1
    assert "Archived: hidden" in c.get("/sleeves/btc-test", auth=AUTH).text
    assert post("restore", reason="back in use").status_code == 303
    assert "Archived sleeves" not in c.get("/", auth=AUTH).text
    assert [d["action"] for d in store.decisions("btc-test")][:2] == ["restore", "archive"]


def test_path_to_live_reports_g2_evidence_and_never_approves(client):
    from datetime import timedelta

    from sleeve_fund.dashboard import gates
    from sleeve_fund.dashboard.book import sleeve_extras
    from sleeve_fund.dashboard.metrics import sleeve_summary
    from sleeve_fund.store import utcnow

    c, store = client
    _new(c)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "Path to live" in page and "0 of 8 G2 conditions met" not in page and "Your G2 approval" in page
    s = store.sleeve("btc-test")
    x = sleeve_extras(store, sleeve_summary(store, s), __import__("pandas").DataFrame())
    rows = {r["label"]: r for r in gates.path_to_live(store, x, "PASS", store.accounts(), utcnow() + timedelta(days=50))}
    assert rows["Strategy passed G1"]["ok"] and rows["Six weeks of paper trading"]["ok"]
    assert not rows["At least 10 closed trades"]["ok"] and rows["Results inside the backtest's range"]["ok"] is None
    assert rows["Your G2 approval"]["ok"] is False  # the checklist can never approve G2 itself
    store.event("btc-test", "error", "reconcile_mismatch", "engine 1 BTC, journal 0")
    rows = {r["label"]: r for r in gates.path_to_live(store, x, None, store.accounts(), utcnow())}
    assert rows["Journal and engine always agreed"]["bad"] and not rows["Strategy passed G1"]["ok"]
    store.create_account("kraken-live", "live")
    store.report_keys({"kraken-live": True})
    rows = {r["label"]: r for r in gates.path_to_live(store, x, None, store.accounts(), utcnow())}
    assert rows["Live Kraken account with its key installed"]["detail"] == "kraken-live"


def test_new_sleeve_form_is_one_step_at_a_time_with_javascript_and_whole_without(client):
    c, _ = client
    page = c.get("/sleeves/new", auth=AUTH).text
    assert page.count('<fieldset class="panel step">') == 5  # all present in the HTML; the script shows one at a time
    js = c.get("/static/console.js", auth=AUTH).text
    assert "function wizard(form)" in js


@pytest.mark.parametrize("path", ["/", "/sleeves/btc-test", "/trades", "/orders", "/alerts", "/risk"])
def test_pages_update_live_without_a_reload(client, path):
    c, store = client
    _new(c)
    page = c.get(path, auth=AUTH).text
    assert '<script src="/static/live.js" defer></script>' in page and 'id="live"' in page
    assert 'data-live="health"' in page and 'data-live="nav-alerts"' in page
    # Regions the script swaps are keyed by the server's own markup, so a fresh render matches.
    assert page.count('aria-labelledby="') + page.count('data-live="') >= 3


def test_sleeve_page_marks_its_status_banners_and_dialogs_live(client):
    c, store = client
    _new(c)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    for key in ('data-live="head"', 'data-live="banners"', 'data-live="dialogs" data-live-forms'):
        assert key in page
    js = c.get("/static/live.js", auth=AUTH).text
    assert "Not updating since" in js and "visibilitychange" in js
