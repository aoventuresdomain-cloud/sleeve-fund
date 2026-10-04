import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store
from sleeve_fund.venues import KRAKEN

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    (tmp_path / "trend_filter_x.md").write_text("# Tear sheet\n\n| a | b |\n| - | - |\n| 1 | 2 |\n")
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
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
    for text in ("Book equity", "Month to date", "Gross exposure", "Allocation", "price feed quiet"):
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
    assert "Limits by strategy" in page and "−3,000.00" in page and "eth-risk</span>" in page
    assert "drawdown 21% hit the 20% limit" in page


def test_ops_page_shows_processes_and_safety_nets(client, tmp_path, monkeypatch):
    import os
    import time

    from sleeve_fund.alerts import Forwarder

    c, store = client
    _new(c)
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "none"))
    page = c.get("/ops", auth=AUTH).text
    assert "Strategy processes" in page and "btc-test" in page and "Database size" in page
    assert "Price watchdog" in page and "Database backup" in page and "None yet" in page

    folder = tmp_path / "backups"
    folder.mkdir()
    (folder / "sleeve_fund-old.dump").write_bytes(b"x" * 10)
    os.utime(folder / "sleeve_fund-old.dump", (time.time() - 3 * 86400,) * 2)
    monkeypatch.setenv("BACKUP_DIR", str(folder))
    store.event(None, "info", "alerts_config", Forwarder(store, environ={}).describe())
    page = c.get("/ops", auth=AUTH).text
    assert "Last 3" in page and '<span class="warn">Last' in page  # three days old: flagged
    assert "Alerts are not set up" in page
    (folder / "sleeve_fund-new.dump").write_bytes(b"x" * 2048)
    store.event(None, "info", "alerts_config",
                Forwarder(store, environ={"ALERT_WEBHOOK_URL": "https://hooks.example.com/T/secret"}).describe())
    page = c.get("/ops", auth=AUTH).text
    assert "2 kept" in page and '<span class="">Last' in page
    assert "Copies on this server only" in page and "Lightsail snapshots cover" not in page
    (folder / "status.json").write_text('{"ok": true, "message": "x", "restored": {"sleeves": 3, "orders": 40, '
                                        '"fills": 41, "events": 900}}')
    assert "Restored into a scratch database and read back (3 strategies, 40 orders, 900 events)" in c.get(
        "/ops", auth=AUTH).text
    (folder / "status.json").write_text('{"ok": false, "message": "pg_dump failed: disk full"}')
    assert "Last run failed: pg_dump failed: disk full." in c.get("/ops", auth=AUTH).text
    assert "Alerts go to hooks.example.com" in page and "secret" not in page


def test_preview_runs_settings_on_history(monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=2, start_price=150))
    d = preview.run("trend_filter", "SOL/USD", {"fast": 10, "slow": 40, "stop_loss": 0.08}, starting=5000)
    assert d["pair"] == "SOL/USD" and d["days"] == 400 and d["equity"][0] == pytest.approx(5000, rel=0.01)
    assert set(d["strategy"]) >= {"sharpe", "max_drawdown"} and d["trades"]["trades"] >= 1


def test_the_new_strategy_form_backtests_its_settings_instead_of_a_look_back(client):
    """Round 4, R4-M8: the Look-back ran other bars and a shorter period than the backtest beside it."""
    c, _ = client
    form = c.get("/sleeves/new", auth=AUTH).text
    assert 'id="bt-these"' in form and "Look-back" not in form
    assert c.get("/api/preview?instrument=BTC/USD", auth=AUTH).status_code == 404


def test_research_pipeline_and_strategy_pages(client):
    c, _ = client
    _new(c)  # a buy_and_hold sleeve, with no real G1 pass, so it shows as an observation
    page = c.get("/research", auth=AUTH).text
    assert "Pipeline" in page and "observation" in page and "trend filter" in page
    s = c.get("/strategies/trend_filter", auth=AUTH).text
    assert "Exact rules" in s and "Start a strategy with this" in s
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
    assert r.headers["content-disposition"].startswith('attachment; filename="fund-decisions-')
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
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    assert "Saved runs" in c.get("/backtest", auth=AUTH).text
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
         "&p_trend_filter__slow=20&starting_balance=5000&period=365")
    page = c.get(q, auth=AUTH).text
    assert "Trend filter on ETH/USD" in page and "Every trade" in page and "Buy and hold" in page
    assert "5-bar average" in page and "above the 20-bar average" in page  # the entry reasons, from the strategy
    assert "365 days" in page
    # The sleeve decides on the daily bars that were tested, warm from its first bar (the slow 20).
    assert ('href="/sleeves/new?instrument=ETH%2FUSD&amp;strategy=trend_filter&amp;p_trend_filter__fast=5'
            '&amp;p_trend_filter__slow=20&amp;starting_balance=5000&amp;bar_spec=1-DAY-LAST-EXTERNAL'
            '&amp;tested_bar_spec=1-DAY-LAST-EXTERNAL&amp;warmup_bars=20&amp;from=backtest"') in page
    assert "33% invested" in page  # the benchmark is held at the balanced profile's cap
    form = c.get("/sleeves/new?instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5&from=backtest"
                 "&bar_spec=1-DAY-LAST-EXTERNAL&warmup_bars=40", auth=AUTH).text
    assert 'value="ETH/USD"' in form and 'name="p_trend_filter__fast" value="5"' in form and "From backtest." in form
    assert '<input type="hidden" name="bar_spec" value="1-DAY-LAST-EXTERNAL">' in form
    assert '<select id="bar_spec" disabled>' in form and 'name="warmup_bars" type="number" min="0" max="50000" value="40"' in form


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
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&risk_profile=conservative"
                 "&max_notional=1000&period=365", auth=AUTH).text
    assert "20% invested" in page and "largest order cap" in page  # the 1,000 order cap binds before the 20% cap
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
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=200, seed=3))
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
    monkeypatch.setattr(KRAKEN, "ohlc_history", down)
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
    monkeypatch.setattr(KRAKEN, "ohlc_history", lambda pair, minutes: kraken)
    d = c.get("/api/sleeves/sol-x/candles?interval=4h", auth=AUTH).json()
    assert d["source"] == "venue" and d["interval"] == 240 and len(d["candles"]) == 2 and d["volume"][1]["value"] == 20
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
    assert "Cloned from" in form and 'value="sol-stops-v2" data-touched=1' in form
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
    assert "Archived strategies (1)" in home and home.count('href="/sleeves/btc-test"') == 1
    assert "Archived.</div>" in c.get("/sleeves/btc-test", auth=AUTH).text
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


def test_maker_first_orders_round_trip_through_the_form_and_clone(client):
    from urllib.parse import parse_qs, urlparse

    c, store = client
    assert _new(c, name="eth-maker", instrument="ETH/USD", execution="maker", maker_wait_minutes="20").status_code == 303
    assert store.sleeve("eth-maker").params["maker_wait_minutes"] == 20
    page = c.get("/sleeves/eth-maker", auth=AUTH).text
    assert "Maker first, the rest at market after 20 min" in page
    href = next(p for p in page.split('"') if p.startswith("/sleeves/new?"))
    q = {k: v[0] for k, v in parse_qs(urlparse(href.replace("&amp;", "&")).query).items()}
    assert q["execution"] == "maker" and q["maker_wait_minutes"] == "20"
    form = c.get(href.replace("&amp;", "&"), auth=AUTH).text
    assert '<option value="maker" selected>' in form and 'name="maker_wait_minutes" type="number"' in form
    # Market orders store nothing extra, so existing strategies are unchanged.
    _new(c, name="eth-market", instrument="ETH/USD", execution="market", maker_wait_minutes="20")
    assert "maker_wait_minutes" not in store.sleeve("eth-market").params
    # The wait must fit inside one decision bar (hourly here).
    r = _new(c, name="eth-slow", execution="maker", maker_wait_minutes="60")
    assert "shorter than one bar" in c.get(r.headers["location"], auth=AUTH).text


def _wavy_minutes(days):
    import numpy as np
    import pandas as pd

    now = pd.Timestamp.now(tz="UTC").floor("1D")
    idx = pd.date_range(now - pd.Timedelta(days=days), periods=days * 1440, freq="1min", tz="UTC")
    t = np.arange(len(idx))
    c = 2_000 * (1 + 0.0004 * t / 1440) + 5 * np.sin(t / 7)  # drifts up, swings a few dollars every few minutes
    return pd.DataFrame({"open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0}, index=idx)


def test_backtest_matches_maker_orders_on_stored_minutes_and_says_so(client, monkeypatch, tmp_path):
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    deep = _wavy_minutes(90).assign(volume=1_000.0)  # the whole order fits in what one minute shows
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", deep, cursor="x")
    preview._history.clear()
    q = "/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&execution=maker&maker_wait_minutes=15"
    page = c.get(q, auth=AUTH).text
    assert "1 of 1 as maker" in page and "Maker fills matched on 1-minute bars" in page
    assert "at most 5% of a bar&#39;s volume per price it trades through" in page
    assert "maker_wait_minutes=15" in page and "execution=maker" in page  # carried to the paper strategy

    preview._history.clear()  # no stored minutes for this one: the page says every maker order paid taker
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=200, seed=3))
    page = c.get(q.replace("ETH/USD", "SOL/USD"), auth=AUTH).text
    assert "0 of 1 as maker" in page and "maker orders assumed to miss, charged the taker fee" in page


def test_backtest_guard_checks_stored_minutes_and_says_how_often(client, monkeypatch, tmp_path):
    """R3-M1: with stored minutes the backtest's risk guard values the book every minute, as paper's
    30-second ticks do, so a daily-loss pause fires near its limit rather than at the day's close."""
    import numpy as np

    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    c, _ = client
    m = _wavy_minutes(70)
    crash = np.ones(len(m))
    crash[60 * 1440:61 * 1440] = np.linspace(1.0, 0.6, 1440)  # day 61 falls 40% minute by minute
    crash[61 * 1440:] = 0.6
    for col in ("open", "high", "low", "close"):
        m[col] = m[col] * crash
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", m, cursor="x")
    preview._history.clear()
    d = preview.run("buy_and_hold", "ETH/USD", {}, risk_profile="aggressive", detail=True)
    assert d["risk"]["checked_minutes"] == 1 and d["risk"]["pauses"] >= 1
    assert d["risk"]["note"].endswith("Guard: 1-min bars.")
    sell = next(f for f in d["fills"] if f["side"] == "SELL")
    assert sell["price"] > 0.8 * 2_000  # out near an 8% loss on a 50% position, not after the whole 40% fall


def test_backtest_page_says_when_the_risk_guard_halted(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv
    from test_backtest import _path

    c, _ = client
    preview._history.clear()
    falling = _path(synthetic_ohlcv(days=200, seed=3), [2_000.0] * 20 + [2_000.0 * 0.98**i for i in range(1, 46)] + [2_000.0 * 0.98**45] * 60)
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: falling)
    q = "/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&risk_profile=conservative"
    page = c.get(q, auth=AUTH).text
    assert "Halted " in page and "(DD " in page
    preview._history.clear()
    page = c.get(q.replace("conservative", "aggressive"), auth=AUTH).text  # a 60% fall at a 50% cap is a 30% drawdown, short of 35%
    assert "Halted " not in page


def test_backtest_charges_the_measured_spread_and_says_where_it_came_from(client, monkeypatch):
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, store = client
    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=200, seed=3))
    q = "/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold"
    page = c.get(q, auth=AUTH).text
    assert "0.10% spread (assumed)" in page and " spread</div>" in page
    store.record_spread("KRAKEN", "ETH/USD", 0.0001, samples=900)
    page = c.get(q, auth=AUTH).text
    assert "0.02% spread (measured " in page


def test_backtest_runs_on_hourly_and_minute_bars_from_the_history_store(client, monkeypatch, tmp_path):
    """Review R2-B2: any interval the paper engine allows, down to 1 minute, from stored minutes."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", _wavy_minutes(20), cursor="x")
    preview._history.clear()
    q = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
         "&p_trend_filter__slow=20&bar_spec=1-HOUR-LAST-INTERNAL")
    page = c.get(q, auth=AUTH).text
    assert "Couldn't run it" not in page
    assert 'value="1-HOUR-LAST-INTERNAL" selected' in page
    assert "bar_spec=1-HOUR-LAST-INTERNAL" in page and "tested_bar_spec=1-HOUR-LAST-INTERNAL" in page
    d = preview.run("trend_filter", "ETH/USD", {"fast": 5, "slow": 20}, minutes=1, detail=True)
    assert d["minutes"] == 1 and d["bars"] > 20 * 1400 and d["trades"]["trades"] > 0
    assert len(d["t"]) <= 21  # equity judged day by day, so Sharpe is annualised as daily
    assert d["chart_minutes"] == 15 and len(d["price"]["candles"]) <= preview.CHART_CANDLES  # 20 days fit at 15 min

    form = c.get("/sleeves/new?from=backtest&bar_spec=1-HOUR-LAST-INTERNAL&tested_bar_spec=1-HOUR-LAST-INTERNAL",
                 auth=AUTH).text
    assert '<input type="hidden" name="tested_bar_spec" value="1-HOUR-LAST-INTERNAL">' in form
    r = _new(c, name="bt-hour", bar_spec="1-HOUR-LAST-INTERNAL", tested_bar_spec="1-HOUR-LAST-INTERNAL",
             **{"from": "backtest"})
    assert r.headers["location"] == "/sleeves/bt-hour"
    r = _new(c, name="bt-day", bar_spec="1-DAY-LAST-EXTERNAL", tested_bar_spec="1-HOUR-LAST-INTERNAL",
             **{"from": "backtest"})
    assert "interval" in r.headers["location"]


def test_minute_backtests_need_the_history_store(client, monkeypatch, tmp_path):
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "empty")
    preview._history.clear()
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&bar_spec=1-MINUTE-LAST-INTERNAL",
                 auth=AUTH).text
    assert "has no stored minute history here, so it can only be backtested on daily bars" in page
    assert "Instruments with stored minutes: none yet" in page


def test_intervals_say_what_is_stored_and_run_all_of_it(client, monkeypatch, tmp_path):
    """R3-M3 and round 4, R4-M2: the backtest form lists the instruments with stored minutes, and a
    1-minute run takes all of the stored history, not the last year."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview
    from sleeve_fund.venues import venue

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", _wavy_minutes(400), cursor="x")
    preview._history.clear()
    form = c.get("/backtest", auth=AUTH).text
    assert "Intraday: ETH/USD (since" in form and "At most" not in form
    df = preview._intraday("ETH/USD", venue("kraken"), 1, None)
    assert len(df) >= 399 * 1440


def test_trade_built_intervals_get_a_warm_up_from_the_store():
    from sleeve_fund.dashboard.app import _warmup_for

    # Venue candles stop at one request (720); bars built from trades load from the history store.
    assert _warmup_for("trend_filter", {"p_trend_filter__slow": "1000"}, "1-HOUR-LAST-EXTERNAL") == 720
    assert _warmup_for("trend_filter", {"p_trend_filter__slow": "1000"}, "1-HOUR-LAST-INTERNAL") > 720


def test_backtest_says_when_its_history_has_long_quiet_stretches(client, monkeypatch, tmp_path):
    """Review R2-M1: a stretch with no trades is held flat, as the venue reported, and said on the page."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    minutes = _wavy_minutes(20)
    gap = (minutes.index >= minutes.index[4980]) & (minutes.index < minutes.index[4980 + 180])
    minutes.loc[gap, "volume"] = 0.0
    minutes.loc[gap, ["open", "high", "low", "close"]] = minutes["close"].iloc[4979]
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", minutes, cursor="x")
    preview._history.clear()
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&bar_spec=1-HOUR-LAST-INTERNAL",
                 auth=AUTH).text
    assert "1 stretch of an hour or more with no trades" in page and "3 hours" in page
    assert "ETH/USD 1-hour candles" in page


def test_backtest_price_chart_marks_every_trade_over_years(monkeypatch):
    """R3-M4: the chart used to stop at the last 720 candles, dropping older trades off it."""
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    preview._history.clear()
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=1800, seed=3, vol=0.03))
    d = preview.run("trend_filter", "ETH/USD", {"fast": 5, "slow": 20}, detail=True, risk_profile="aggressive")
    assert d["chart_label"] == "daily candles" and len(d["price"]["candles"]) == d["bars"] > 720
    assert len(d["price"]["markers"]) == len(d["fills"]) > 0
    first = d["price"]["candles"][0]["time"]
    assert all(m["time"] >= first for m in d["price"]["markers"])


def test_settings_states_fees_as_they_are_charged(client):
    """R3-M5: both rates, which tier and when they were read, and the spread charged apart."""
    c, _ = client
    page = c.get("/settings", auth=AUTH).text
    assert "Every order is charged the taker fee" not in page
    assert "0.40% maker / 0.80% taker (Tier 1, 3 Oct 2026) · spread: measured, else 0.10%" in page


def test_a_new_strategy_warms_up_automatically_and_says_when_it_cannot(client):
    """R3-M7: a strategy started from scratch used to default to no warm-up. Blank now means enough
    for the model's longest look-back; past the most that loads, the strategy says so."""
    c, store = client
    assert _new(c, name="auto-warm", warmup_bars="", p_trend_filter__fast="5", p_trend_filter__slow="20").status_code == 303
    assert store.sleeve("auto-warm").warmup_bars == 20  # the slow average's length
    assert not [e for e in store.events("auto-warm") if e["kind"] == "warmup_short"]
    _new(c, name="short-warm", warmup_bars="10", p_trend_filter__fast="5", p_trend_filter__slow="20")
    assert store.sleeve("short-warm").warmup_bars == 10
    (e,) = [e for e in store.events("short-warm") if e["kind"] == "warmup_short"]
    assert e["level"] == "info" and "needs 20 bars of history but 10 load at start" in e["message"]
    # Past the most the history store loads, the shortfall is an alert.
    _new(c, name="long-warm", warmup_bars="", p_trend_filter__fast="5", p_trend_filter__slow="60000")
    assert store.sleeve("long-warm").warmup_bars == 50_000
    (e,) = [e for e in store.events("long-warm") if e["kind"] == "warmup_short"]
    assert e["level"] == "warning" and "needs 60,000 bars" in e["message"]


def test_the_guard_cadence_is_not_the_chart_spacing(client, monkeypatch, tmp_path):
    """Round 4, NEW-2: over a long run the chart thins its points, and that spacing once overwrote
    how often the note said the guard checked the book."""
    from sleeve_fund import history
    from sleeve_fund.dashboard import preview

    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    import numpy as np

    m = _wavy_minutes(820)
    crash = np.ones(len(m))
    crash[800 * 1440:801 * 1440] = np.linspace(1.0, 0.6, 1440)
    crash[801 * 1440:] = 0.6
    for col in ("open", "high", "low", "close"):
        m[col] = m[col] * crash
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", m, cursor="x")
    preview._history.clear()
    d = preview.run("buy_and_hold", "ETH/USD", {}, risk_profile="aggressive")
    assert len(d["t"]) < d["days"]  # the chart is thinned
    assert d["risk"]["checked_minutes"] == 15 and "Guard: 15-min bars" in d["risk"]["note"]


def test_a_backtest_chart_covers_its_whole_period(client):
    """Review round 5, R5-M6: a 5-year run's price chart showed only the last 720 candles, about 23 of
    60 months at daily candles, beside an equity panel covering all five years."""
    from datetime import datetime, timedelta, timezone

    from sleeve_fund.store import BACKTEST_PREFIX

    c, store = client
    name, t0 = BACKTEST_PREFIX + "r1", datetime(2021, 1, 1, tzinfo=timezone.utc)
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=1000)
    for i in range(5 * 365):
        store.record_equity(name, equity=1000, cash=1000, qty=0, price=100 + i % 50, benchmark=1000,
                            ts=t0 + timedelta(days=i))
    d = c.get(f"/api/sleeves/{BACKTEST_PREFIX}r1/candles?interval=1d", auth=AUTH).json()
    assert len(d["candles"]) == 5 * 365 and d["source"] == "marks"


def test_the_book_kill_switch_flattens_every_running_strategy(client):
    """Review rounds 1 to 6: one button on Risk sells every strategy still trading or holding a position
    to cash and pauses it, each with the PM's reason. A stopped one holding a position is started just
    to sell; a halted, flat one is left alone; a blank reason sells nothing and says why."""
    c, store = client
    _new(c)
    _new(c, name="eth-test", instrument="ETH/USD")
    _new(c, name="sol-test", instrument="SOL/USD")
    store.record_equity("btc-test", equity=5100, cash=3000, qty=0.02, price=105000, benchmark=5050)
    store.record_equity("eth-test", equity=5000, cash=4000, qty=0.3, price=3300, benchmark=5000)
    store.record_equity("sol-test", equity=4000, cash=4000, qty=0.0, price=150, benchmark=5000)
    store.set_desired_state("eth-test", "stopped")
    store.set_status("sol-test", "halted", "drawdown 10.1% hit the 10% limit")
    page = c.get("/risk", auth=AUTH).text
    assert "Flatten everything" in page and "All 2 strategies still trading or holding a position sell" in page
    assert "eth-test is stopped, so it starts just to sell" in page
    r = c.post("/book/flatten", data={"reason": " "}, auth=AUTH, headers=SAME)
    assert "Nothing was sold: the kill switch needs a reason" in r.text and store.pending_commands("btc-test") == []
    r = c.post("/book/flatten", data={"reason": "Market event; standing aside"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303
    for name in ("btc-test", "eth-test"):
        (cmd,) = store.pending_commands(name)
        assert cmd["command"] == "flatten" and cmd["reason"] == "Book kill switch: Market event; standing aside"
    assert store.sleeve("eth-test").desired_state == "running"
    assert store.pending_commands("sol-test") == []
    assert store.decisions()[0]["action"] == "flatten everything" and "(2 strategies)" in store.decisions()[0]["reason"]
    assert c.post("/book/flatten", data={"reason": "x"}, auth=AUTH).status_code in (400, 403)  # same origin only
    # A command still waiting when its strategy is stopped lapses rather than firing on the next start.
    c.post("/sleeves/btc-test/command", data={"command": "stop", "reason": "done for now"}, auth=AUTH, headers=SAME)
    assert store.pending_commands("btc-test") == []
    assert any(d["action"] == "drop flatten" and "lapsed" in d["reason"] for d in store.decisions("btc-test"))


def test_the_backtest_result_page_shows_the_intraday_drawdown(client, monkeypatch, tmp_path):
    """Review round 6, R6-M2: the result page measured drawdown on daily closes, so a fall that
    recovered by the close vanished, and a run read "drawdown 19.3%" beside "halted at 20.0%"."""
    import numpy as np
    import pandas as pd

    from sleeve_fund import history
    from sleeve_fund.dashboard import preview
    from sleeve_fund.research.metrics import returns_from_equity, summary

    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    m = _wavy_minutes(120)
    dip = np.ones(len(m))
    dip[100 * 1440 + 600:100 * 1440 + 700] = 0.85  # 15% down for 100 minutes, back by the close
    for col in ("open", "high", "low", "close"):
        m[col] = m[col] * dip
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", m, cursor="x")
    preview._history.clear()
    keep = {}
    d = preview.run("buy_and_hold", "ETH/USD", {}, risk_profile="aggressive", keep=keep)
    shown = -d["strategy"]["max_drawdown"]
    assert shown == pytest.approx(keep["journal"].max_drawdown(), abs=1e-9)
    daily = summary(returns_from_equity(pd.Series(d["equity"], index=pd.to_datetime(d["t"]))))["max_drawdown"]
    assert shown > 0.05 > -daily  # the dip is in, where daily closes saw almost none
    assert -d["hold"]["max_drawdown"] > 0.05  # the benchmark is measured on every bar too


def test_a_g1_study_runs_from_the_research_page(client, tmp_path, monkeypatch):
    import time
    from urllib.parse import parse_qs, urlparse

    from sleeve_fund import history
    from tests.test_research import _stored_minutes

    c, _ = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore().append("KRAKEN", "ETH/USD", _stored_minutes(130), cursor="x")
    form = {"strategy": "buy_and_hold", "instrument": "eth/usd", "minutes": "240", "risk_profile": "conservative",
            "train_days": "60", "test_days": "30", "holdout_days": "30"}
    assert 'action="/research/run"' in c.get("/research", auth=AUTH).text
    # Another site can't start one, and a bad form says what is wrong without starting anything.
    assert c.post("/research/run", data=form, auth=AUTH, headers={"Origin": "https://evil.example"}).status_code == 403
    bad = c.post("/research/run", data={**form, "minutes": "1"}, auth=AUTH, headers=SAME)
    assert bad.status_code == 200 and "would take hours" in bad.text
    r = c.post("/research/run", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/research?job=")
    job_id = parse_qs(urlparse(r.headers["location"]).query)["job"][0]
    running = c.get(r.headers["location"], auth=AUTH).text
    assert 'id="study-job"' in running and 'value="240" selected' in running  # the form shows what is running
    for _ in range(600):
        j = c.get(f"/api/backtest/jobs/{job_id}", auth=AUTH).json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert j["run_id"] == "buy_and_hold_kraken-ethusd-store-240m"
    assert "conservative risk profile" in c.get(f"/research/{j['run_id']}", auth=AUTH).text
    assert (tmp_path / "idea_ledger.jsonl").exists()  # counted in the server's ledger, not the repository's
    # Missing history is an error on the page, not a crash.
    r = c.post("/research/run", data={**form, "instrument": "SOL/USD"}, auth=AUTH, headers=SAME, follow_redirects=False)
    for _ in range(200):
        j = c.get(f"/api/backtest/jobs/{parse_qs(urlparse(r.headers['location']).query)['job'][0]}", auth=AUTH).json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "error" and "no stored Kraken spot history for SOL/USD" in j["error"]
