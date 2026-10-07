import re

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store, utcnow
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
    # Listed under Portfolio in the rail, marked as the page you're on, with trades and orders as its tabs.
    assert 'href="/sleeves/btc-test" aria-current=page' in page.text and 'data-sub="history"' in page.text
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
    assert 'role="combobox"' in c.get("/sleeves/new", auth=AUTH).text  # the instrument pick-list
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
    assert r.status_code == 200 and "Not done: every command needs a reason." in r.text


def test_research_and_tearsheet(client):
    c, _ = client
    assert "trend_filter_x" in c.get("/research", auth=AUTH).text
    assert "<table>" in c.get("/research/trend_filter_x", auth=AUTH).text
    assert c.get("/research/..%2F..%2Fetc%2Fpasswd", auth=AUTH).status_code == 404
    # Each tear sheet downloads as its Markdown file, to hand to the research thread as it is.
    assert 'href="/research/trend_filter_x/download"' in c.get("/research", auth=AUTH).text
    assert 'href="/research/trend_filter_x/download"' in c.get("/research/trend_filter_x", auth=AUTH).text
    md = c.get("/research/trend_filter_x/download", auth=AUTH)
    assert md.status_code == 200 and md.headers["content-type"].startswith("text/markdown")
    assert md.headers["content-disposition"] == 'attachment; filename="trend_filter_x.md"' and "|" in md.text
    assert c.get("/research/..%2F..%2Fetc%2Fpasswd/download", auth=AUTH).status_code == 404
    assert c.get("/research/nope/download", auth=AUTH).status_code == 404
    # A refresh or Back onto a form's address (a GET) returns to the research page, not "no such tear sheet".
    for path, to in (("/research/history?venue=binance", "/research?venue=binance#history"),
                     ("/research/run", "/research?venue=kraken"), ("/research/history?venue=nope", "/research?venue=kraken#history")):
        r = c.get(path, auth=AUTH, follow_redirects=False)
        assert (r.status_code, r.headers["location"]) == (303, to), path
    assert c.get("/decisions", auth=AUTH).status_code == 200
    assert c.get("/sleeves/new", auth=AUTH).status_code == 200


def test_portfolio_shows_book_figures_and_alerts_can_be_acknowledged(client):
    c, store = client
    _new(c, name="eth-book", instrument="ETH/USD")
    store.record_equity("eth-book", equity=10_100, cash=5_000, qty=2, price=2_550, benchmark=10_050)
    store.event("eth-book", "warning", "mark_unavailable", "price feed quiet")
    page = c.get("/", auth=AUTH).text
    for text in ("Book value", "Month to date", "Allocation and top", "Needs you", "price feed quiet"):
        assert text in page
    alert = store.alerts()[0]
    r = c.post(f"/alerts/{alert['id']}/ack", data={"note": "seen", "next": "/"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/"
    assert store.open_alert_count() == 0 and store.alerts(include_acked=True)[0]["ack_note"] == "seen"
    after = c.get("/", auth=AUTH).text
    assert "price feed quiet" not in after and 'class="needs-you"' not in after  # the bar goes once nothing waits
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


def test_a_stopped_strategy_still_counts_in_the_book(client):
    """Stopping a strategy must not rewrite the book: its cash is still the fund's."""
    c, store = client
    _new(c)
    _new(c, name="eth-test", instrument="ETH/USD")
    store.record_equity("btc-test", equity=5_500, cash=5_500, qty=0, price=60_000, benchmark=5_000)
    store.record_equity("eth-test", equity=4_500, cash=4_500, qty=0, price=3_000, benchmark=5_000)
    before = c.get("/api/book/equity", auth=AUTH).json()["equity"][-1]
    store.set_desired_state("eth-test", "stopped")
    after = c.get("/api/book/equity", auth=AUTH).json()["equity"][-1]
    assert before == after == 10_000
    assert "10,000.00" in c.get("/", auth=AUTH).text


def test_short_ranges_read_every_mark(client):
    c, store = client
    _new(c)
    store.record_equity("btc-test", equity=5_100, cash=5_100, qty=0, price=60_000, benchmark=5_050)
    day = c.get("/api/book/equity?days=1", auth=AUTH).json()
    assert day["res"] == "intraday" and day["equity"][-1] == 5_100 and len(day["t"]) == len(day["drawdown"])
    week = c.get("/api/sleeves/btc-test/equity?days=7", auth=AUTH).json()
    assert week["equity"][-1] == 5_100 and "fills" in week
    assert c.get("/api/book/equity?days=3", auth=AUTH).status_code == 400


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


def _healthy_desk(c, store, tmp_path, monkeypatch, *names):
    """Strategies running and reporting, balances checked, last night's backup on disk: an All clear desk."""
    folder = tmp_path / "fresh-backups"
    folder.mkdir(exist_ok=True)
    (folder / "sleeve_fund-new.dump").write_bytes(b"x" * 100)
    monkeypatch.setenv("BACKUP_DIR", str(folder))
    for name in names:
        _new(c, name=name)
        store.set_status(name, "running")
        store.heartbeat(name)
        store.event(name, "info", "reconcile", "balance matches the venue")


def _status_block(page):
    block = page.split('aria-labelledby="status-h">', 1)[1].split("</section>", 1)[0]
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", block)).strip()


def test_risk_status_word_rules(client, tmp_path, monkeypatch):
    """Risk & health's one word: All clear when nothing is halted or paused, no limit is past 80%, every
    process reports, feeds are fresh and backup and balance checks are fine; otherwise Needs a look (amber)
    or Action needed (red), naming the first item."""
    c, store = client
    _healthy_desk(c, store, tmp_path, monkeypatch, "btc-a", "btc-b")
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status ok"' in page
    assert _status_block(page).startswith("All clear 2 strategies running. Every limit has room. Nothing needs you.")
    assert "Processes 2/2" in page and "Feeds 2/2" in page and "Balances matched" in page and "Backup 0h ago" in page
    assert 'href="/alerts"' in page and "Alerts 0 open" in page

    # A stale price feed: amber, naming it; the feed coming back clears it.
    store.event("btc-b", "warning", "stale_price", "No trade or quote from the venue for 6 minutes")
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status warn"' in page and "Feeds 1/2" in page
    assert _status_block(page).startswith("Needs a look btc-b has had no trade or quote from the venue lately.")
    store.event("btc-b", "info", "price_feed_back", "Market data is arriving again")
    assert 'class="rh-status ok"' in c.get("/risk", auth=AUTH).text

    # Paused: amber.
    store.set_status("btc-a", "paused", "paused by the PM")
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status warn"' in page and "Needs a look btc-a is paused: paused by the PM." in _status_block(page)
    store.set_status("btc-a", "running")

    # A limit more than 80% used: amber, naming the strategy and the limit.
    store.record_equity("btc-a", equity=5000, cash=5000, qty=0, price=100_000, benchmark=5000)
    store.record_equity("btc-a", equity=4150, cash=4150, qty=0, price=100_000, benchmark=5000)  # 17% of a 20% limit
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status warn"' in page and re.search(r"Needs a look btc-a has used \d+% of its \w[\w ]* limit",
                                                          _status_block(page))

    # Halted: red, and it comes first, with the rest listed under it.
    store.set_status("btc-b", "halted", "drawdown 20.4% hit the 20% limit")
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status bad"' in page
    assert _status_block(page).startswith("Action needed btc-b is halted: drawdown 20.4% hit the 20% limit. And 1 more below.")


def test_risk_status_turns_red_for_a_silent_process_a_balance_mismatch_and_amber_for_backups(client, tmp_path,
                                                                                           monkeypatch):
    from datetime import timedelta

    from sleeve_fund.store import sleeves_t

    c, store = client
    _healthy_desk(c, store, tmp_path, monkeypatch, "btc-a")
    store.event("btc-a", "error", "reconcile_mismatch", "venue holds 0.1 BTC, the journal 0")
    page = c.get("/risk", auth=AUTH).text
    assert 'class="rh-status bad"' in page and "Balance mismatch: btc-a" in page
    assert _status_block(page).startswith("Action needed btc-a&#39;s balance doesn&#39;t match the venue&#39;s.")
    store.event("btc-a", "info", "reconcile", "balance matches the venue")
    with store.engine.begin() as conn:  # the process stopped reporting four minutes ago
        conn.execute(sleeves_t.update().values(heartbeat_at=utcnow() - timedelta(minutes=4)))
    page = c.get("/risk", auth=AUTH).text
    assert "Action needed btc-a is not reporting." in _status_block(page) and "Processes 0/1" in page
    store.heartbeat("btc-a")
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "no-backups"))
    page = c.get("/risk", auth=AUTH).text
    assert _status_block(page).startswith("Needs a look No database backup has been written yet.")
    assert "Backup none yet" in page


def test_risk_status_word_names_the_worst_first():
    from sleeve_fund.dashboard.riskops import status_word

    assert status_word([], 0)["word"] == "All clear" and status_word([], 0)["line"].startswith("No strategy running.")
    assert status_word([], 1)["line"].startswith("1 strategy running.")
    warn = status_word([{"level": "warn", "text": "a is paused"}], 2)
    assert (warn["word"], warn["tone"], warn["line"]) == ("Needs a look", "warn", "a is paused.")
    bad = status_word([{"level": "bad", "text": "b is halted"}, {"level": "warn", "text": "a is paused"}], 2)
    assert (bad["word"], bad["tone"], bad["line"]) == ("Action needed", "bad", "b is halted. And 1 more below.")


def test_ops_redirects_to_the_system_tab_of_risk(client):
    c, _ = client
    _new(c)
    r = c.get("/ops", auth=AUTH, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/risk#system"
    assert c.get("/ops", follow_redirects=False).status_code == 401  # still behind the password
    page = c.get("/ops", auth=AUTH).text
    assert 'data-panel="system"' in page and "Technical" in page and "Strategy processes" in page
    assert "Safety nets" in page and "Price watchdog" in page


def test_risk_overview_lists_room_before_halt_and_stops(client):
    c, store = client
    _new(c, name="eth-stop", instrument="ETH/USD", stop_atr="2", atr_bars="14")
    _new(c, name="sol-none", instrument="SOL/USD")
    store.record_equity("sol-none", equity=5000, cash=5000, qty=0, price=100, benchmark=5000)
    store.record_equity("sol-none", equity=4400, cash=1400, qty=30, price=100, benchmark=5000)  # 12% of 20%: amber
    page = c.get("/risk", auth=AUTH).text
    overview = page.split('data-panel="overview"', 1)[1].split('data-panel="limits"', 1)[0]
    assert "Limits by strategy" not in overview and "Limits by strategy" in page  # the full table is on Limits
    assert '<span class="rh-chip ok" title="2.0 simple ATR (14 bars) below entry">2 simple ATR</span>' in overview
    assert '<span class="rh-chip warn">None</span>' in overview
    assert 'class="w" style="width:60.0%"' in overview and "8.0% left" in overview
    assert "If the market moved now" in overview and "Book drawdown, 30 days" in overview


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
    assert "Off-site: Lightsail's automatic daily snapshots" in page
    (folder / "status.json").write_text('{"ts": "2026-10-04T02:00:00Z", "ok": true, "message": "x", "restored": '
                                        '{"sleeves": 3, "orders": 40, "fills": 41, "events": 900}}')
    assert ("Restored into a scratch database and read back (3 strategies, 40 orders, 41 fills, 900 events), "
            in c.get("/ops", auth=AUTH).text)
    (folder / "status.json").write_text('{"ok": false, "message": "pg_dump failed: disk full"}')
    assert '<span class="loss">Last run failed: pg_dump failed: disk full.' in c.get("/ops", auth=AUTH).text
    # A status with no result is a failure, as the alert says, not a silent pass (review round 8, R8-10).
    (folder / "status.json").write_text("{}")
    assert "doesn&#39;t say whether the last run worked" in c.get("/ops", auth=AUTH).text
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
    # The pipeline folds into Development: a card per model, its stage and running strategies in its study.
    assert 'data-plan="buy_and_hold"' in page and "on observation" in page and "Trend filter" in page
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
    assert "Portfolio views" in page and "Stop-loss" in page and "EOrder:Insufficient funds" in page
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
    for href in ("/", "/alerts", "/research", "/backtest", "/risk", "/records", "/setup"):
        assert f'href="{href}"' in page  # nothing is desktop-only any more; phones reach it through More
    assert 'id="strats"' in page  # every strategy is listed under Portfolio (a sheet on phones)
    if path in ("/", "/trades", "/orders"):  # the book-wide blotters are tabs of the portfolio
        assert 'href="/trades"' in page and 'href="/orders"' in page


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
    assert 'name="bar_spec_shown" value="1-DAY-LAST-EXTERNAL" checked disabled' in form and 'name="warmup_bars" type="number" min="0" max="50000" value="40"' in form


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
    r = _new(c, name="BAD NAME", reason_pick="Other", reason_note="testing keeps fields")
    assert r.status_code == 303
    form = c.get(r.headers["location"], auth=AUTH).text
    assert "Not saved" in form and ">testing keeps fields</textarea>" in form and 'value="Other" required checked' in form


def test_accounts_page_adds_live_accounts_and_shows_key_presence_only(client, monkeypatch):
    from sleeve_fund import accounts
    from sleeve_fund.supervisor import Supervisor

    c, store = client
    page = c.get("/accounts", auth=AUTH).text
    assert "paper" in page and "Connect a Kraken sub-account" in page and "Never tick Withdraw Funds" in page
    r = c.post("/accounts/new", data={"name": "kraken-trend", "kind": "live", "venue": "kraken", "reason": "first live sub-account"},
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
    store.create_account("kraken-live", "live", venue="kraken")
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
    assert {ln["title"] for ln in d["lines"]} == {"Entry", "SL"} and d["lines"][0]["price"] == 90

    now = pd.Timestamp.now(tz="UTC").floor("h")
    kraken = pd.DataFrame({"open": [1.0, 2.0], "high": [2.0, 3.0], "low": [0.5, 1.5], "close": [2.0, 2.5],
                           "volume": [10.0, 20.0]}, index=pd.DatetimeIndex([now - pd.Timedelta("4h"), now]))
    charts._cache.clear()
    monkeypatch.setattr(KRAKEN, "ohlc_history", lambda pair, minutes: kraken)
    d = c.get("/api/sleeves/sol-x/candles?interval=4h", auth=AUTH).json()
    assert d["source"] == "venue" and d["interval"] == 240 and len(d["candles"]) == 2 and d["volume"][1]["value"] == 20
    assert c.get("/api/sleeves/nope/candles", auth=AUTH).status_code == 404
    assert 'id="pc"' in c.get("/sleeves/sol-x", auth=AUTH).text

    # Another instrument on the same chart: the venue's candles for it, none of this strategy's trades or exits.
    asked = []
    monkeypatch.setattr(KRAKEN, "ohlc_history", lambda pair, minutes: asked.append(pair) or kraken)
    d = c.get("/api/sleeves/sol-x/candles?interval=4h&pair=eth/usd", auth=AUTH).json()
    assert asked == ["ETH/USD"] and d["pair"] == "ETH/USD" and d["home"] == "SOL/USD" and len(d["candles"]) == 2
    assert d["markers"] == [] and d["lines"] == [] and d["notes"] == {} and d["pairs"][0] == "SOL/USD"
    assert c.get("/api/sleeves/sol-x/candles?pair=ETH", auth=AUTH).status_code == 400

    def unknown(pair, minutes):
        raise ValueError(f"no candles for {pair}")

    monkeypatch.setattr(KRAKEN, "ohlc_history", unknown)
    d = c.get("/api/sleeves/sol-x/candles?pair=ZZZ/USD", auth=AUTH).json()
    assert d["candles"] == [] and d["note"] == "The venue has no candles for ZZZ/USD."
    d = c.get("/api/sleeves/sol-x/candles?interval=4h&pair=SOL/USD", auth=AUTH).json()  # its own: back to the full chart
    assert d["pair"] == d["home"] == "SOL/USD" and d["lines"]


def test_instrument_list_is_the_venues_with_a_fallback(client, monkeypatch):
    from sleeve_fund.dashboard import charts

    listing = {"error": [], "result": {"XXBTZUSD": {"wsname": "XBT/USD"}, "XETHZUSD": {"wsname": "ETH/USD"},
                                       "XDGEUR": {"wsname": "XDG/EUR"}, "ODD": {}}}
    monkeypatch.setattr(charts, "_listed", {})
    assert charts.instruments(lambda url: listing) == ["BTC/USD", "DOGE/EUR", "ETH/USD"]  # venue codes read as the usual names
    assert charts.instruments(lambda url: 1 / 0) == ["BTC/USD", "DOGE/EUR", "ETH/USD"]  # cached for the day

    c, _ = client

    def down(*a, **k):
        raise OSError("no route to the venue")

    monkeypatch.setattr(charts, "instruments", down)
    d = c.get("/api/instruments", auth=AUTH).json()
    assert d["source"] == "fallback" and "BTC/USD" in d["instruments"]
    monkeypatch.setattr(charts, "instruments", lambda: ["ETH/USD", "SOL/USD"])
    assert c.get("/api/instruments", auth=AUTH).json() == {"instruments": ["ETH/USD", "SOL/USD"], "source": "venue"}
    assert c.get("/api/instruments").status_code == 401


def test_position_tab_is_compact_with_reason_folded(client):
    c, store = client
    _new(c, name="sol-x", instrument="SOL/USD", bar_spec="1-HOUR-LAST-INTERNAL", stop_loss_pct="8")
    store.record_order("sol-x", order_id="O-1", side="BUY", qty=10, intent="entry", reason="RSI 25.1 below 30")
    store.record_fill("sol-x", side="BUY", qty=10, price=90, fee=0.72, order_id="O-1", trade_id="t1")
    store.record_equity("sol-x", equity=10_049.28, cash=9_099.28, qty=10, price=95, benchmark=10_000)
    page = c.get("/sleeves/sol-x", auth=AUTH).text
    tab = page[page.index('id="tab-positions"'):page.index('id="tab-activity"')]
    for label in ("Size", "Quantity", "Notional", "Exposure", "Entry", "Stop-loss", "Take-profit", "Unrealised", "Realised", "Total"):
        assert f">{label}<" in tab or f">{label} " in tab, label
    assert "10 SOL" in tab and "950.00 USD" in tab  # quantity in the instrument and notional in the quote
    assert '<details class="why-fold">' in tab and "RSI 25.1 below 30" in tab


def test_chart_helpers():
    from sleeve_fund.dashboard import charts

    assert charts.default_interval("1-DAY-LAST-EXTERNAL") == "1d"
    assert charts.default_interval("5-MINUTE-LAST-INTERNAL") == "5m"
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
    c, store = client
    _new(c)
    for path in ("/sleeves/btc-test", "/trades"):
        page = c.get(path, auth=AUTH).text
        assert 'class="help"' in page and 'aria-label="What does this mean?"' in page
    # Portfolio's tiles are a label and a number; the detail is each tile's hover title (UI v2, PM 5 Oct).
    kpis = c.get("/", auth=AUTH).text.split('aria-label="Book figures">')[1].split("</section>")[0]
    assert kpis.count('<div class="kpi') == kpis.count('title="') == 8 and 'class="s"' not in kpis


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
    store.create_account("kraken-live", "live", venue="kraken")
    store.report_keys({"kraken-live": True})
    rows = {r["label"]: r for r in gates.path_to_live(store, x, None, store.accounts(), utcnow())}
    assert rows["Live spot account with its key installed"]["detail"] == "kraken-live"


def test_g2_key_row_counts_only_a_key_on_the_strategys_own_venue(client):
    """A Binance strategy with only a Kraken key installed has no live account it could trade on."""
    import copy

    from sleeve_fund.dashboard import gates
    from sleeve_fund.dashboard.book import sleeve_extras
    from sleeve_fund.dashboard.metrics import sleeve_summary
    from sleeve_fund.store import utcnow

    c, store = client
    _new(c)
    x = sleeve_extras(store, sleeve_summary(store, store.sleeve("btc-test")), __import__("pandas").DataFrame())
    on_binance = copy.copy(x["sleeve"])
    object.__setattr__(on_binance, "venue", "BINANCE")
    store.create_account("kraken-live", "live", venue="kraken")
    store.report_keys({"kraken-live": True})

    def key_row(sleeve):
        return next(r for r in gates.path_to_live(store, {**x, "sleeve": sleeve}, None, store.accounts(), utcnow())
                    if r["label"].endswith("account with its key installed"))

    row = key_row(on_binance)
    assert row["label"] == "Live perpetual account with its key installed" and row["ok"] is False
    assert row["detail"] == "add one on the Accounts page"
    store.create_account("binance-live", "live", venue="binance")
    store.report_keys({"kraken-live": True, "binance-live": True})
    assert key_row(on_binance)["detail"] == "binance-live" and key_row(x["sleeve"])["detail"] == "kraken-live"


def test_accounts_carry_a_registered_venue_and_paper_names_none(client):
    c, store = client
    with pytest.raises(ValueError, match="needs its venue"):
        store.create_account("no-venue", "live")
    with pytest.raises(ValueError, match="needs its venue"):
        store.create_account("odd-venue", "live", venue="nowhere")
    store.create_account("b-live", "live", venue="BINANCE")
    store.create_account("desk-c", "paper", venue="binance")
    by = {a["name"]: a for a in store.accounts()}
    assert by["b-live"]["venue"] == "binance" and by["desk-c"]["venue"] == "" and by["paper"]["venue"] == ""
    assert "kraken" not in by["paper"]["note"].lower()
    r = c.post("/accounts/new", data={"name": "b-two", "kind": "live", "venue": "binance", "reason": "x"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "error" not in r.headers["location"]
    page = c.get("/setup", auth=AUTH).text
    assert '<option value="binance"' in page and "Binance USD-M perpetuals" in page


def test_the_old_paper_account_loses_its_venue_and_its_default_note_only(client):
    from sleeve_fund.store import accounts_t, insert, utcnow

    _, store = client

    with store.engine.begin() as c:
        c.execute(insert(accounts_t).values(name="paper", kind="paper", venue="kraken",
                                            note="Simulated money at live Kraken prices and fees", created_at=utcnow()))
    paper = next(a for a in store.accounts() if a["name"] == "paper")
    assert paper["venue"] == "" and paper["note"] == "Simulated money at each strategy's own venue prices and fees"
    store.set_account_note("paper", "the PM's own words")
    assert next(a for a in store.accounts() if a["name"] == "paper")["note"] == "the PM's own words"


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


@pytest.mark.usefixtures("maker_on")
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


@pytest.mark.usefixtures("maker_on")
def test_backtest_matches_maker_orders_on_stored_minutes_and_says_so(client, monkeypatch, tmp_path):
    from sleeve_fund import history
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import preview
    from sleeve_fund.data import synthetic_ohlcv

    c, _ = client
    # A 90-day minute run can outlast the page's 8 s wait on a busy runner, which then shows the run's progress
    # instead of its result: wait for it, as test_jobs does (QA: this test failed at random).
    monkeypatch.setattr(app_mod, "BACKTEST_WAIT", 120.0)
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
    assert 'value="1-HOUR-LAST-INTERNAL" checked' in page
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
    # Fewer than the model needs is raised to what it needs (PM, 5 Oct 2026: the bare minimum is required).
    _new(c, name="short-warm", warmup_bars="10", p_trend_filter__fast="5", p_trend_filter__slow="20")
    assert store.sleeve("short-warm").warmup_bars == 20
    assert not [e for e in store.events("short-warm") if e["kind"] == "warmup_short"]
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
    assert "Flatten everything" in page and "Both strategies still trading or holding a position sell" in page
    assert "eth-test is stopped, so it starts just to sell" in page
    r = c.post("/book/flatten", data={"reason": " "}, auth=AUTH, headers=SAME)
    assert "Nothing was sold: the kill switch needs a reason" in r.text and store.pending_commands("btc-test") == []
    r = c.post("/book/flatten", data={"reason": "Market event; standing aside"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/risk?killed=2"
    fired = c.get(r.headers["location"], auth=AUTH).text
    assert "Kill switch fired: 2 strategies are closing out to cash" in fired and "data-once" in fired
    for name in ("btc-test", "eth-test"):
        (cmd,) = store.pending_commands(name)
        assert cmd["command"] == "flatten" and cmd["reason"] == "Book kill switch: Market event; standing aside"
    assert store.sleeve("eth-test").desired_state == "running"
    assert store.pending_commands("sol-test") == []
    assert store.decisions()[0]["action"] == "flatten everything" and "(2 strategies)" in store.decisions()[0]["reason"]
    assert c.post("/book/flatten", data={"reason": "x"}, auth=AUTH).status_code in (400, 403)  # same origin only
    # Firing again sells nothing twice: both are already waiting to sell, named, and the button rests
    # (review round 8, R8-11).
    again = c.get("/risk", auth=AUTH).text
    assert "Waiting to sell: btc-test, eth-test." in again and "Selling to cash…" in again
    assert 'data-open="dlg-kill"' not in again
    logged = len(store.decisions())
    second = c.post("/book/flatten", data={"reason": "again"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert len(store.pending_commands("btc-test")) == 1 and len(store.pending_commands("eth-test")) == 1
    # Round 9, N1: a second fire logged "0 strategies" and its banner said "0 strategies are selling".
    assert second.headers["location"] == "/risk?killed=0" and len(store.decisions()) == logged
    said = c.get(second.headers["location"], auth=AUTH).text
    assert "Nothing more to sell: btc-test, eth-test are already closing out to cash. Nothing was logged." in said
    # A command still waiting when its strategy is stopped lapses rather than firing on the next start.
    c.post("/sleeves/btc-test/command", data={"command": "stop", "reason": "done for now"}, auth=AUTH, headers=SAME)
    assert store.pending_commands("btc-test") == []
    assert any(d["action"] == "drop flatten" and "lapsed" in d["reason"] for d in store.decisions("btc-test"))
    # With one strategy left to act on, the dialog names it.
    store.set_desired_state("eth-test", "stopped")
    store.record_equity("eth-test", equity=5000, cash=5000, qty=0.0, price=3300, benchmark=5000)
    store.set_desired_state("btc-test", "running")
    store.record_equity("btc-test", equity=5100, cash=5100, qty=0.0, price=105000, benchmark=5050)
    assert "btc-test, the one strategy still trading or holding a position, sells" in c.get("/risk", auth=AUTH).text
    from datetime import timedelta

    from sleeve_fund.store import utcnow

    store.set_status("btc-test", "paused", "daily loss 5.2%", utcnow() + timedelta(hours=20))
    assert "btc-test is on a daily-loss pause, which then lasts until you resume too" in c.get("/risk", auth=AUTH).text


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


def test_an_age_never_reads_minus_zero():
    from datetime import timedelta

    from sleeve_fund.dashboard.app import _held

    assert _held(timedelta(seconds=-2)) == "0 min" and _held(timedelta(minutes=5)) == "5 min"


def test_a_g1_study_runs_from_the_research_page(client, tmp_path, monkeypatch):
    import time
    from urllib.parse import parse_qs, urlparse

    from sleeve_fund import history
    from test_research import _stored_minutes

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
    assert 'id="study-job"' in running and 'value="240" checked' in running  # the form shows what is running
    for _ in range(600):
        j = c.get(f"/api/backtest/jobs/{job_id}", auth=AUTH).json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "done", j
    assert re.fullmatch(r"buy_and_hold_kraken-ethusd-store-240m_\d{8}-\d{6}", j["run_id"])
    first = j["run_id"]
    assert "conservative risk profile" in c.get(f"/research/{first}", auth=AUTH).text
    # A re-run with other exits is new evidence beside the old, not a replacement (review round 8, R8-M3).
    r = c.post("/research/run", data={**form, "stop_loss_pct": "5"}, auth=AUTH, headers=SAME, follow_redirects=False)
    for _ in range(200):
        j = c.get(f"/api/backtest/jobs/{parse_qs(urlparse(r.headers['location']).query)['job'][0]}", auth=AUTH).json()
        if j["status"] not in ("queued", "running"):
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["run_id"] != first
    listing = c.get("/research", auth=AUTH).text
    assert f'/research/{first}"' in listing and f'/research/{j["run_id"]}"' in listing
    assert "exits: the signal only" in listing and "exits: stop-loss 5.0% below entry" in listing
    assert (tmp_path / "idea_ledger.jsonl").exists()  # counted in the server's ledger, not the repository's
    # Missing history is said on the page at once with its badge, and starts no job (R8-M6); no Collect
    # button (UI v2, item 10): a core instrument is stored from its listing anyway.
    r = c.post("/research/run", data={**form, "instrument": "SOL/USD"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 200 and "SOL/USD: not stored yet. SOL/USD is on the collector&#39;s core list" in r.text
    assert ">Collect" not in r.text and "collect-study" not in r.text


def test_research_collects_history_for_any_instrument(client, tmp_path, monkeypatch):
    """Review round 8, R8-M6: a study on anything but BTC/USD failed for want of stored history, and
    the only way to get it was to start a paper strategy on the instrument."""
    from sleeve_fund import history
    from test_research import _stored_minutes

    c, store = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore().append("KRAKEN", "ETH/USD", _stored_minutes(3), cursor="x")
    old = _stored_minutes(3)
    history.HistoryStore().append("KRAKEN", "BTC/USD", old.set_axis(old.index - pd.Timedelta(days=30)), cursor="y")
    listed = {"ADA/USD"}

    def check(pair):
        if pair not in listed:
            raise ValueError(f"Kraken does not list {pair}")

    monkeypatch.setattr(KRAKEN, "check_listed", check)
    page = c.get("/research", auth=AUTH).text
    # Data coverage is read-only (UI v2, item 8): what is stored, and how far along it is.
    assert 'data-panel="history"' in page and "Collect another instrument" not in page
    assert re.search(r"<td data-label=\"Instrument\">BTC/USD</td>.*?being filled", page, re.S)
    assert re.search(r"<td data-label=\"Instrument\">ETH/USD</td>.*?stored · last candle \d\d:\d\d · no gaps", page, re.S)
    # Another site can't ask, an unlisted instrument is refused, and a listed one is queued once.
    assert c.post("/research/history", data={"instrument": "ADA/USD"}, auth=AUTH,
                  headers={"Origin": "https://evil.example"}).status_code == 403
    bad = c.post("/research/history", data={"instrument": "FOO/USD"}, auth=AUTH, headers=SAME)
    assert "Kraken does not list FOO/USD" in bad.text and not store.history_requests("KRAKEN")
    ok = c.post("/research/history", data={"instrument": "ada/usd"}, auth=AUTH, headers=SAME)
    assert "Asked the collector for ADA/USD: it backfills from" in ok.text
    assert re.search(r"ADA/USD</td>.*?being filled", ok.text, re.S)
    (req,) = store.history_requests("KRAKEN")
    assert req["instrument"] == "ADA/USD" and 5 * 365 - 2 <= (utcnow() - req["since"]).days <= 5 * 365 + 1
    again = c.post("/research/history", data={"instrument": "ADA/USD"}, auth=AUTH, headers=SAME)
    assert "ADA/USD was already asked for" in again.text and len(store.history_requests("KRAKEN")) == 1
    # Review round 9, N5: a stored pair was asked for again, a mistyped one offered a button that failed the
    # same way, and with the venue unreachable any pair was asked for, to sit "Asked for" for good.
    stored = c.post("/research/history", data={"instrument": "eth/usd"}, auth=AUTH, headers=SAME)
    assert "ETH/USD is already stored, from" in stored.text and "a study can run on it now" in stored.text
    typo = c.post("/research/history", data={"instrument": "btcusd"}, auth=AUTH, headers=SAME)
    assert "BTCUSD isn&#39;t an instrument: write it as base and quote with a slash, like BTC/USD" in typo.text
    assert "Collect BTCUSD history</button>" not in typo.text and "Collect FOO/USD history</button>" not in bad.text

    def unreachable(pair):
        raise ConnectionError("403 Forbidden")

    monkeypatch.setattr(KRAKEN, "check_listed", unreachable)
    down = c.post("/research/history", data={"instrument": "ZZZQ/USD"}, auth=AUTH, headers=SAME)
    assert "couldn&#39;t reach the venue to check it lists ZZZQ/USD, so nothing was asked for" in down.text
    assert len(store.history_requests("KRAKEN")) == 1
    # A core instrument is always stored from its listing: no request, and no "five years back" to mislead.
    core = c.post("/research/history", data={"instrument": "sol/usd"}, auth=AUTH, headers=SAME)
    assert "SOL/USD is on the collector&#39;s core list for the venue: it is stored from its listing" in core.text
    assert len(store.history_requests("KRAKEN")) == 1
    # A study on it before anything is stored says so with its badge, without asking again.
    form = {"strategy": "buy_and_hold", "instrument": "ADA/USD", "minutes": "240", "train_days": "60",
            "test_days": "30", "holdout_days": "0"}
    r = c.post("/research/run", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 200 and "ADA/USD: being filled · the study waits until some is stored." in r.text
    assert len(store.history_requests("KRAKEN")) == 1
    # A study on a listed instrument nobody asked for asks the collector itself: no Collect button (item 10).
    monkeypatch.setattr(KRAKEN, "check_listed", check)
    listed.add("DOT/USD")
    r = c.post("/research/run", data={**form, "instrument": "DOT/USD"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 200 and "DOT/USD: not stored yet. Asked the collector for DOT/USD" in r.text
    assert [q["instrument"] for q in store.history_requests("KRAKEN")] == ["ADA/USD", "DOT/USD"]
    r = c.post("/research/run", data={**form, "instrument": "FOO/USD"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert "FOO/USD: not stored yet, and the collector wasn&#39;t asked: Kraken does not list FOO/USD" in r.text


def test_the_form_and_the_backtest_refuse_a_thin_target_at_the_same_round_trip(client, monkeypatch):
    """Review round 7: the form refused a 1.5% target against a 1.61% round trip and the backtest
    against 1.71%, because only the backtest charged the spread."""
    import re
    from urllib.parse import parse_qs, urlparse

    from sleeve_fund.data import synthetic_ohlcv

    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=2, start_price=150))
    c, _ = client
    r = _new(c, stop_loss_pct="1", take_profit_pct="1.5", risk_per_trade_pct="1")
    form_cost = re.search(r"\(([\d.]+%)\)", parse_qs(urlparse(r.headers["location"]).query)["error"][0]).group(1)
    page = c.get("/backtest?run=1&instrument=ETH/USD&strategy=buy_and_hold&stop_loss_pct=1&take_profit_pct=1.5",
                 auth=AUTH).text
    assert f"doesn&#39;t cover the round trip&#39;s fees and spread ({form_cost})" in page


def test_a_strategy_takes_an_atr_stop_and_a_target_in_multiples_of_it(client):
    c, store = client
    r = _new(c, stop_atr="2.5", atr_bars="20", take_profit_r="3", risk_per_trade_pct="1", warmup_bars="")
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/btc-test", r.headers["location"]
    params = store.sleeve("btc-test").params
    assert params["stop_atr"] == 2.5 and params["atr_bars"] == 20 and params["take_profit_r"] == 3.0
    assert store.sleeve("btc-test").warmup_bars >= 21  # the ATR's bars load at start, like the model's
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "2.5 simple ATR (20 bars) below entry" in page and "3.0R after costs" in page
    clone = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "stop_atr=2.5" in clone and "take_profit_r=3" in clone  # Clone with changes keeps them
    form = c.get("/sleeves/new?stop_atr=2.5&take_profit_r=3", auth=AUTH).text
    assert '<option value="atr" selected' in form and '<option value="r" selected>' in form
    assert "data-costs=" in form
    r = _new(c, name="btc-two", stop_loss_pct="5", stop_atr="2")
    assert "one kind of stop" in r.headers["location"].replace("+", " ")


def _settings(c, name="btc-test", **over):
    form = {"risk_profile": "balanced", "instrument": "BTC/USD", "reason": "tighter risk", **over}
    return c.post(f"/sleeves/{name}/settings", data=form, auth=AUTH, headers=SAME, follow_redirects=False)


@pytest.mark.usefixtures("maker_on")
def test_risk_settings_change_in_place_with_a_reason_and_restart(client):
    c, store = client
    _new(c, stop_loss_pct="8", maker_wait_minutes="20", execution="maker")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert 'id="settings-form"' in page and 'value="8"' in page  # the form starts from the settings as they are
    r = _settings(c, risk_profile="conservative", stop_atr="2", atr_bars="10", take_profit_r="3", max_notional="500")
    assert r.headers["location"].endswith("saved=settings#settings")
    s = store.sleeve("btc-test")
    assert s.risk_profile == "conservative"
    assert s.params == {"fast": 10, "slow": 30, "maker_wait_minutes": 20, "stop_atr": 2.0, "atr_bars": 10,
                        "take_profit_r": 3.0, "max_notional": 500.0}  # the model and order type untouched
    assert s.warmup_bars >= 11  # enough bars for the new stop's average true range
    (d,) = store.decisions("btc-test", action="change_settings")
    assert "Risk profile balanced to conservative" in d["reason"] and "Stop-loss 8% below the entry to 2 simple average" \
        in d["reason"] and d["reason"].endswith("tighter risk")
    assert store.pending_reload("btc-test") is not None
    assert store.last_event("btc-test", ("exits_change",)) is not None
    page = c.get("/sleeves/btc-test?saved=settings", auth=AUTH).text
    assert "Restarting to trade under the changed settings" in page and "1 command" not in page
    # A risk profile on its own leaves the open position's stop alone.
    store.event("btc-test", "info", "marker", "-")
    _settings(c, risk_profile="aggressive", stop_atr="2", atr_bars="10", take_profit_r="3", max_notional="500")
    assert store.last_event("btc-test", ("exits_change", "settings_change", "marker"))["kind"] == "settings_change"


@pytest.mark.parametrize("over, msg", [
    ({"reason": ""}, "reason"),
    ({"risk_profile": "reckless"}, "no+profile"),
    ({"stop_loss_pct": "2", "stop_atr": "2"}, "stop"),
    ({"stop_loss_pct": "8"}, "nothing+changed"),
    ({"take_profit_pct": "0.5", "stop_loss_pct": "8"}, "cover"),
])
def test_bad_settings_changes_are_refused_and_keep_what_was_typed(client, over, msg):
    c, store = client
    _new(c, stop_loss_pct="8")
    r = _settings(c, **{"stop_loss_pct": "8", **over})
    loc = r.headers["location"]
    assert "settings_error=" in loc and msg in loc.lower() and loc.endswith("#settings")
    assert store.sleeve("btc-test").params.get("stop_loss") == 0.08 and store.pending_reload("btc-test") is None
    if over.get("reason") != "":
        assert "Not saved" in c.get(loc.split("#")[0], auth=AUTH).text


def test_loosening_the_stop_on_an_open_position_is_checked_against_its_size(client):
    """Review round 8, M8-1: a 3% to 50% edit on an open position saved, about 17R at risk. A looser stop
    now needs the PM's confirmation, with the risk in money and R; one past the drawdown halt is refused."""
    from urllib.parse import unquote_plus

    from sleeve_fund.paper.runtime import SleeveRuntime

    c, store = client
    _new(c, stop_loss_pct="3")
    rt = SleeveRuntime(store, "btc-test")
    rt.on_order(order_id="E-1", side="BUY", qty=0.06, intent="entry", reason="Signal to be long",
                signal={"stop_frac": 0.03, "risk_amount": 137.28, "stop_cfg": {"stop_loss": 0.03}})
    rt.on_fill(side="BUY", qty=0.06, price=50_000.0, fee=24.0, order_id="E-1", trade_id="T-1")
    store.record_equity("btc-test", equity=4_976.0, cash=1_976.0, qty=0.06, price=50_000.0, benchmark=5_000)
    assert "Accept a looser stop on the open position" in c.get("/sleeves/btc-test", auth=AUTH).text
    loc = unquote_plus(_settings(c, stop_loss_pct="5").headers["location"])
    assert "looser than the 3.0%" in loc and "(1.5R of the entry" in loc
    assert store.sleeve("btc-test").params["stop_loss"] == 0.03
    loc = unquote_plus(_settings(c, stop_loss_pct="50", confirm_looser="1").headers["location"])
    assert "drawdown halt" in loc and store.sleeve("btc-test").params["stop_loss"] == 0.03
    assert "saved=settings" in _settings(c, stop_loss_pct="5", confirm_looser="1").headers["location"]
    assert store.sleeve("btc-test").params["stop_loss"] == 0.05
    assert "saved=settings" in _settings(c, stop_loss_pct="2").headers["location"]  # tighter: no question


def test_the_kill_banner_counts_the_strategies_it_names(client):
    """Review round 9, N1: with one already selling, the banner said "2 strategies" and named 4."""
    c, store = client
    _new(c)
    _new(c, name="eth-test", instrument="ETH/USD")
    store.record_equity("btc-test", equity=5100, cash=3000, qty=0.02, price=105000, benchmark=5050)
    store.record_equity("eth-test", equity=5000, cash=4000, qty=0.3, price=3300, benchmark=5000)
    c.post("/sleeves/btc-test/command", data={"command": "flatten", "reason": "de-risk"}, auth=AUTH, headers=SAME)
    r = c.post("/book/flatten", data={"reason": "Market event"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.headers["location"] == "/risk?killed=1"  # only eth-test was new
    fired = c.get(r.headers["location"], auth=AUTH).text
    assert "Kill switch fired: 2 strategies are closing out to cash at market (btc-test, eth-test)" in fired


def test_a_stopped_strategy_holding_a_position_is_never_stranded(client):
    """Review round 8, M8-2: the journal's position decides, not the desired state. An account a stopped
    strategy still holds a position on can't be retired, and a flatten for a stopped strategy starts it to
    sell rather than waiting for a start that may never come."""
    from sleeve_fund.paper.runtime import SleeveRuntime

    c, store = client
    store.create_account("desk-b", "paper")
    _new(c, account="desk-b")
    rt = SleeveRuntime(store, "btc-test")
    rt.on_order(order_id="E-1", side="BUY", qty=0.01, intent="entry", reason="Signal to be long", signal={})
    rt.on_fill(side="BUY", qty=0.01, price=50_000.0, fee=4.0, order_id="E-1", trade_id="T-1")
    store.set_desired_state("btc-test", "stopped")
    store.record_equity("btc-test", equity=10_000, cash=10_000, qty=0.0, price=50_000, benchmark=10_000)  # a stale mark
    page = c.get("/accounts", auth=AUTH).text
    assert '<a href="/sleeves/btc-test">btc-test</a> is stopped but still holds a position on it' in page
    # Round 9, R9-M1: the page told the PM to flatten it first but offered no Flatten. The journal decides.
    head = c.get("/sleeves/btc-test", auth=AUTH).text
    assert 'data-open="dlg-start">Start' in head and 'data-open="dlg-flatten">Flatten' in head
    assert "It is stopped, so it starts only to sell 0.01 BTC (about 500.00 USD) at market" in head
    r = c.post("/accounts/desk-b/retire", data={"action": "retire", "reason": "tidy"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "still+holds+a+position" in r.headers["location"]
    assert not next(a for a in store.accounts() if a["name"] == "desk-b")["retired_at"]
    r = c.post("/sleeves/btc-test/command", data={"command": "flatten", "reason": "de-risk"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and store.sleeve("btc-test").desired_state == "running"
    assert [x["command"] for x in store.pending_commands("btc-test")] == ["flatten"]
    # Round 10, m5: while it waits, Flatten is off, and a stale page's second one is refused in words.
    assert 'disabled title="A flatten is already waiting' in c.get("/sleeves/btc-test", auth=AUTH).text
    r = c.post("/sleeves/btc-test/command", data={"command": "flatten", "reason": "again"}, auth=AUTH, headers=SAME)
    assert "Not done: a flatten is already waiting for the strategy to act on it." in r.text
    assert len(store.pending_commands("btc-test")) == 1
    _new(c, name="eth-flat", instrument="ETH/USD")
    store.set_desired_state("eth-flat", "stopped")
    assert 'data-open="dlg-flatten"' not in c.get("/sleeves/eth-flat", auth=AUTH).text
    # Round 9, N8: a stale page reaching this got raw JSON; the refusal is said on the page.
    r = c.post("/sleeves/eth-flat/command", data={"command": "flatten", "reason": "x"}, auth=AUTH, headers=SAME)
    assert r.status_code == 200 and "application/json" not in r.headers["content-type"]
    assert "Not done: it is stopped and holds no position, so there is nothing to flatten." in r.text


def test_a_stopped_strategy_takes_new_settings_at_its_next_start(client):
    c, store = client
    _new(c)
    store.set_desired_state("btc-test", "stopped")
    _settings(c, stop_loss_pct="5")
    assert store.sleeve("btc-test").params["stop_loss"] == 0.05 and store.pending_reload("btc-test") is None
    assert "apply when it next starts" in store.events("btc-test")[0]["message"]


def test_strategies_move_between_accounts_only_when_flat(client):
    c, store = client
    _new(c)
    store.create_account("desk-b", "paper")
    store.create_account("kraken-live", "live", venue="kraken")

    def move(account, reason="reorganise"):
        return c.post("/sleeves/btc-test/account", data={"account": account, "reason": reason}, auth=AUTH, headers=SAME,
                      follow_redirects=False).headers["location"]

    store.record_order("btc-test", order_id="O-1", side="BUY", qty=0.1, intent="entry", reason="x")
    store.record_fill("btc-test", side="BUY", qty=0.1, price=100, fee=0.08, order_id="O-1", trade_id="t1")
    store.record_equity("btc-test", equity=5000, cash=4990, qty=0.1, price=100, benchmark=5000)
    assert "flatten+the+strategy+before+moving" in move("desk-b") and store.account_of("btc-test") == "paper"
    assert "disabled title=\"Flatten it first" in c.get("/sleeves/btc-test", auth=AUTH).text
    store.set_desired_state("btc-test", "stopped")  # stopped still holds the position: it can't move (review round 8)
    assert "flatten+the+strategy+before+moving" in move("desk-b") and store.account_of("btc-test") == "paper"
    store.record_order("btc-test", order_id="O-2", side="SELL", qty=0.1, intent="exit", reason="x")
    store.record_fill("btc-test", side="SELL", qty=0.1, price=100, fee=0.08, order_id="O-2", trade_id="t2")
    assert "locked+until+G2" in move("kraken-live")
    assert move("desk-b").endswith("saved=account#settings") and store.account_of("btc-test") == "desk-b"
    (d,) = store.decisions("btc-test", action="move_account")
    assert d["reason"] == "from paper to desk-b: reorganise"
    assert "already+on" in move("desk-b")


def test_account_notes_and_retiring(client):
    c, store = client
    store.create_account("desk-b", "paper", "old note")
    _new(c, account="desk-b")

    def post(path, **data):
        return c.post(f"/accounts/desk-b/{path}", data=data, auth=AUTH, headers=SAME,
                      follow_redirects=False).headers["location"]

    assert "saved=desk-b" in post("note", note="Second desk", reason="clearer")
    assert next(a for a in store.accounts() if a["name"] == "desk-b")["note"] == "Second desk"
    page = c.get("/accounts", auth=AUTH).text
    assert "btc-test still runs on it" in page and 'id="dlg-retire-desk-b"' in page
    assert "still+runs+on+desk-b" in post("retire", action="retire", reason="tidy")
    store.set_desired_state("btc-test", "stopped")
    assert "saved=desk-b" in post("retire", action="retire", reason="tidy")
    assert "Retired" in c.get("/accounts", auth=AUTH).text
    # Nothing starts on, or moves onto, a retired account.
    r = c.post("/sleeves/btc-test/command", data={"command": "start", "reason": "go"}, auth=AUTH, headers=SAME)
    assert "Not done: its account desk-b is retired" in r.text and store.sleeve("btc-test").desired_state != "running"
    assert "retired" in _new(c, name="eth-new", account="desk-b").headers["location"]
    assert "saved=desk-b" in post("retire", action="reinstate", reason="back in use")
    assert c.post("/sleeves/btc-test/command", data={"command": "start", "reason": "go"}, auth=AUTH, headers=SAME,
                  follow_redirects=False).status_code == 303
    actions = [d["action"] for d in store.decisions()]
    assert {"account_note", "retire_account", "reinstate_account"} <= set(actions)
    assert "be+retired" in c.post("/accounts/paper/retire", data={"action": "retire", "reason": "x"}, auth=AUTH,
                                        headers=SAME, follow_redirects=False).headers["location"]
    assert c.post("/accounts/desk-b/note", data={"note": "x", "reason": "x"}, auth=AUTH,
                  headers={"Origin": "https://evil.example"}).status_code == 403


def test_saved_backtests_have_no_settings_to_change(client):
    c, store = client
    store.create_sleeve(name="bt:abc", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=1000)
    assert "backtest" in _settings(c, name="bt:abc", stop_loss_pct="5").headers["location"]


@pytest.mark.parametrize("stop", [{"stop_atr": "2", "atr_bars": "10"}, {"stop_swing_bars": "20"}, {"stop_loss_pct": "5"}])
def test_every_page_renders_with_each_kind_of_stop(client, stop):
    """Review round 8 blocker: /risk returned 500 for an ATR or swing-low stop, which hid the kill switch."""
    c, store = client
    assert _new(c, **stop).status_code == 303
    store.record_equity("btc-test", equity=5000, cash=4000, qty=0.01, price=100_000, benchmark=5000)
    for path in ("/", "/risk", "/sleeves/btc-test", "/orders", "/trades", "/accounts", "/decisions", "/reports", "/ops",
                 "/settings", "/research", "/alerts"):
        r = c.get(path, auth=AUTH)
        assert r.status_code == 200, path
    risk = c.get("/risk", auth=AUTH).text
    assert "Flatten everything" in risk and ("ATR (10 bars)" in risk or "lowest low of 20 bars" in risk or "5.0% below" in risk)


def test_raw_labels_read_as_words(client):
    """Review round 8, R8-4 and R8-9: decisions showed "Move_account" and alerts "maker fill above tape"."""
    c, store = client
    _new(c)
    store.decide("pm", "move_account", "from paper to desk-b: grouping", "btc-test")
    store.decide("pm", "change_settings", "Risk profile balanced to conservative. tighter", "btc-test")
    store.event("btc-test", "warning", "maker_fill_above_tape", "Post-only order O-1 has filled more")
    page = c.get("/decisions", auth=AUTH).text
    assert "<strong>Moved account</strong>" in page and "<strong>Changed settings</strong>" in page
    assert 'data-kind="changed"' in page  # both drawn as settings changes in the Records log
    assert "Move_account" not in page and "Change_settings" not in page
    assert "Maker fill ahead of the tape" in c.get("/alerts", auth=AUTH).text


def test_a_run_whose_strategy_raised_is_flagged_and_not_offered_for_paper(client):
    """Review round 8, R8-9: a run with handler errors still offered "Start a paper strategy with these
    settings", and neither Saved runs nor its strategy screen said anything was wrong."""
    from sleeve_fund.paper.journal import MemoryJournal
    from sleeve_fund.strategies.base import handler_error_words

    def journal():
        j = MemoryJournal()
        j.create_sleeve(name="bt", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000)
        return j

    c, store = client
    words = handler_error_words("on_order_filled", "ZeroDivisionError('float division by zero')")
    assert words == "handling an order filled: float division by zero (ZeroDivisionError)"
    assert handler_error_words("on_quote", ValueError("bad quote")) == "handling a quote: bad quote (ValueError)"
    _new(c)
    j = journal()
    result = {"errors": f"The strategy hit 1 error during this run, the first {words}.", "pair": "BTC/USD"}
    name = store.save_backtest(j, run_id="err1", key="k", title="Errored run", query="", result=result)
    store.save_backtest(journal(), run_id="ok1", key="k2", title="Clean run",
                        query="", result={"pair": "BTC/USD"})
    runs = {r["id"]: r for r in store.backtests()}
    assert runs["err1"]["errored"] and not runs["ok1"]["errored"]
    assert store.strategy_errors(name) == 1
    screen = c.get(f"/sleeves/{name}", auth=AUTH).text
    assert "The strategy's code raised an error 1 time during this run" in screen
    assert "raised an error" not in c.get("/sleeves/btc-test", auth=AUTH).text


def test_round_10_ui_minors(client):
    """Round 10: a halted run says so in Saved runs (m7); a lost job says so (m9); a stopped strategy whose
    process hasn't stopped yet reads Stopping, and a waiting flatten can't be queued twice (m5)."""
    from sleeve_fund.paper.journal import MemoryJournal

    c, store = client
    j = MemoryJournal()
    j.create_sleeve(name="bt", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                    starting_balance=10_000)
    name = store.save_backtest(j, run_id="h1", key="k", title="Halted run", query="", result={"pair": "BTC/USD"})
    store.set_status(name, "halted", "drawdown 21% past the 20% limit")
    assert {r["id"]: r for r in store.backtests()}["h1"]["halted"]
    assert '<span class="chip halted">Halted</span>' in c.get("/backtest", auth=AUTH).text
    for url in ("/backtest?job=gone", "/research?job=gone"):
        assert "that run is no longer known" in c.get(url, auth=AUTH).text
    assert "no longer known" not in c.get("/backtest", auth=AUTH).text

    _new(c)
    store.set_status("btc-test", "running")
    store.set_desired_state("btc-test", "stopped")
    head = c.get("/sleeves/btc-test", auth=AUTH).text
    assert '<span class="chip stopping">Stopping</span>' in head
    from sleeve_fund.paper.runtime import SleeveRuntime

    rt = SleeveRuntime(store, "btc-test")  # a stopped strategy still holding: Flatten is offered once
    rt.on_order(order_id="E-1", side="BUY", qty=0.01, intent="entry", reason="Signal to be long", signal={})
    rt.on_fill(side="BUY", qty=0.01, price=50_000.0, fee=4.0, order_id="E-1", trade_id="T-1")
    assert 'data-open="dlg-flatten">Flatten' in c.get("/sleeves/btc-test", auth=AUTH).text
    store.command("btc-test", "flatten", "cash out")
    head = c.get("/sleeves/btc-test", auth=AUTH).text
    assert 'data-open="dlg-flatten" disabled title="A flatten is already waiting' in head


STOP_KINDS = {"pct": {"stop_loss": 0.03, "take_profit": 0.06},
              "atr": {"stop_atr": 2.0, "atr_bars": 14, "take_profit_r": 2.0},
              "swing": {"stop_swing_bars": 10, "take_profit_r": 1.5}}
STATES = ("running", "paused", "halted", "stopped", "holding")  # holding: stopped with a position


def test_every_page_renders_for_every_stop_type_and_strategy_state(client, monkeypatch):
    """Review round 8, m8-T: the Risk page crashed for any strategy with an ATR or swing-low stop, and
    the kill switch lives there. Every page renders with each kind of stop in each state, with and
    without an open position, its stop edited or not."""
    from datetime import timedelta

    from sleeve_fund.dashboard import charts
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.store import utcnow

    monkeypatch.setattr(charts, "candles", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    c, store = client
    names = []
    for kind, exits in STOP_KINDS.items():
        for state in STATES:
            name = f"{kind}-{state}"
            names.append(name)
            store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                                starting_balance=5_000, params={"fast": 10, "slow": 30, **exits},
                                risk_profile="balanced")
            if state in ("running", "paused", "halted", "holding"):
                rt = SleeveRuntime(store, name)
                basis = {"atr": "2 x the 14-bar simple average true range (1,000)",
                         "swing": "at the lowest low of the last 10 bars (48,500)"}.get(kind)
                sig = {"stop_frac": 0.03, "tp_frac": 0.07, "risk_amount": 70.0, "planned_r": 2.0,
                       "stop_cfg": {k: v for k, v in exits.items() if k.startswith(("stop", "atr"))},
                       **({"stop_basis": basis} if basis else {})}
                rt.on_order(order_id=f"{name}-E", side="BUY", qty=0.03, intent="entry", reason="Signal to be long",
                            signal=sig)
                rt.on_fill(side="BUY", qty=0.03, price=50_000.0, fee=12.0, order_id=f"{name}-E", trade_id=f"{name}-T")
                store.record_equity(name, equity=4_950.0, cash=3_488.0, qty=0.03, price=48_700.0, benchmark=4_990.0)
                if kind == "atr":  # this one had its stop edited while open
                    store.set_exit_plan(name, f"{name}-E", kind="edit", stop_frac=0.02, tp_frac=0.07,
                                        risk_amount=70.0, planned_r=2.4)
            if state == "paused":
                store.set_status(name, "paused", "daily loss 5.1% hit the 5% limit", paused_until=utcnow() + timedelta(hours=8))
            elif state == "halted":
                store.set_status(name, "halted", "drawdown 20.4% hit the 20% limit")
            elif state in ("stopped", "holding"):
                store.set_desired_state(name, "stopped")
                store.set_status(name, "stopped", "process stopped")
            else:
                store.set_status(name, "running")
    pages = ["/", "/trades", "/orders", "/alerts", "/risk", "/reports", "/settings", "/ops", "/accounts", "/decisions",
             "/research", "/sleeves/new"]
    for n in names:
        pages += [f"/sleeves/{n}", f"/api/sleeves/{n}/equity", f"/api/sleeves/{n}/candles", f"/trades?sleeve={n}"]
    for path in pages:
        r = c.get(path, auth=AUTH)
        assert r.status_code == 200, (path, r.text[:300])
    risk = c.get("/risk", auth=AUTH).text
    assert "Flatten everything" in risk
    swing = c.get("/sleeves/swing-holding", auth=AUTH).text
    assert "48,500" in swing and "Accept a looser stop on the open position" in swing
    edited = c.get("/sleeves/atr-running", auth=AUTH).text
    assert "49,000.00" in edited and "edited by the PM" in edited  # the edited plan: 2% under the 50,000 entry


def test_the_stop_distance_is_the_drop_from_the_last_price():
    """Review round 9, N3: "Stop-loss 3,746.20 (80.9% away)" with the price at 6,775.98 divided by the stop;
    the drop from here is 44.7%."""
    from types import SimpleNamespace

    from sleeve_fund.dashboard.app import _risk_view

    x = {"price": 6775.98, "exposure": 0.5, "day_ret": 0.0,
         "profile": SimpleNamespace(max_position_pct=1.0, daily_loss=0.05)}
    view = _risk_view(x, {"stop_px": 3746.20, "target_px": None})
    assert round(view["to_stop"], 3) == 0.447


def test_the_research_form_knows_which_holdouts_are_spent(client, tmp_path):
    """Review round 10, M10-1: a holdout opened on daily bars was offered again for a 60-minute study."""
    import html
    import json

    from sleeve_fund.research.ledger import IdeaLedger

    c, _ = client
    IdeaLedger(tmp_path / "idea_ledger.jsonl").record(idea="trend_filter", family="trend", params={},
                                                       dataset="kraken-btcusd-store", stage="holdout", sharpe=0.4)
    page = c.get("/research?strategy=trend_filter&venue=kraken", auth=AUTH).text
    spent = json.loads(html.unescape(re.search(r'data-spent="([^"]*)"', page).group(1)))
    assert list(spent) == ["trend_filter|kraken-btcusd-store"]
    assert spent["trend_filter|kraken-btcusd-store"].endswith(", on daily bars")
    # The page script keys a spent holdout by the venue picked in the study, which switches in place.
    assert 'name="venue" value="kraken" data-pl-venue' in page and "${venue()}-${pair}-store" in page


def test_sub_dollar_numbers_read_in_full(client):
    """Review round 10, m3: DOGE read "0.0674" and "2.335e+04 DOGE", and a filled order looked partial."""
    from sleeve_fund.dashboard.app import _qty

    c, store = client
    assert [_qty(x) for x in (23350.12345, 0.0765165, 9926.15, 5.0, 0.0)] == ["23,350.1", "0.0765165", "9,926.15", "5", "0"]
    _new(c)
    store.record_order("btc-test", order_id="O-1", side="BUY", qty=9926.15, intent="entry", reason="Signal", signal={})
    store.record_fill("btc-test", side="BUY", qty=9926.149999999998, price=0.0765, fee=0.6, order_id="O-1", trade_id="T-1")
    store.update_order("O-1", fill_qty=9926.149999999998, fill_px=0.0765, fee=0.6)
    page = c.get("/orders", auth=AUTH).text
    assert "9,926.15" in page and "filled</div>" not in page
    assert "0.076500" in page  # the average price at the decimals the Why text uses


def test_the_home_page_opens_after_a_strategy_reset_with_no_clean_slate_ever(client):
    """QA on #164 (pre-existing on main): the first per-strategy Reset, with no clean slate ever made, left the
    home page a 500, as the Previous book list named the new book's start date and there was none."""
    from sleeve_fund.supervisor import Supervisor

    c, store = client
    _new(c)
    store.request_reset("btc-test", "Test finished")
    Supervisor(store, python="true").reset_pending()
    assert store.book_start() is None and store.reset_runs()
    r = c.get("/", auth=AUTH)
    assert r.status_code == 200
    assert "Previous book (1)" in r.text and "put away by a strategy reset;" in r.text
    assert "new book began" not in r.text


def test_a_clean_slate_starts_a_new_book_and_keeps_the_old_one_viewable(client, tmp_path):
    """PM, 4 Oct 2026: the book's equity starts again from the new strategies; the strategies the clean
    slate put away keep their history, their pages and a "Previous book" list, and can be brought back."""
    from sleeve_fund.supervisor import clear

    c, store = client
    store.create_sleeve(name="old-one", strategy="trend_filter", instrument="BTC/USD",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=7000)
    store.record_equity("old-one", equity=6500, cash=6500, qty=0, price=1, benchmark=7000)
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "new book"\n')
    clear(store, str(path))
    store.create_sleeve(name="new-one", strategy="ping_pong", instrument="BTC/USD",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10000)
    assert set(store.previous_book()) == {"old-one"}
    home = c.get("/", auth=AUTH).text
    assert "10,000.00" in home and "17,000.00" not in home  # the book starts from the new strategy only
    assert "Previous book (1)" in home and 'href="/sleeves/old-one"' in home
    assert c.get("/sleeves/old-one", auth=AUTH).status_code == 200
    store.unarchive("old-one")  # brought back: it rejoins the book
    assert store.previous_book() == {}


def test_the_5_oct_clean_slate_leaves_a_fresh_book_of_the_two_binance_strategies(client, tmp_path):
    """PM, 5 Oct 2026: the book's equity, P&L and drawdown carried the old test strategies' history and read as
    made up. After the 5 Oct slate the book starts from the two Binance strategies' capital alone."""
    import glob

    from sleeve_fund.supervisor import clear, seed

    c, store = client
    path = tmp_path / "clear.toml"
    path.write_text('[[clear]]\nid = "2026-10-04"\nreason = "first slate"\n')
    clear(store, str(path))
    seed(store, sorted(glob.glob("configs/sleeves/*.toml")))
    store.record_equity("rsi-bands-ls-test", equity=8123.45, cash=8123.45, qty=0, price=1, benchmark=10000)
    store.record_fill("ping-pong-ls-test", side="BUY", qty=0.01, price=60_000, fee=0.5, order_id="O-1", trade_id="T-1")
    store.record_fill("ping-pong-ls-test", side="SELL", qty=0.01, price=59_000, fee=0.5, order_id="O-2", trade_id="T-2")
    clear(store, "configs/clear.toml")
    book = c.get("/", auth=AUTH).text
    assert "20,000.00" in book and "8,123.45" not in book
    assert 'href="/sleeves/ping-pong-ls-binance"' in book and 'href="/sleeves/rsi-bands-ls-binance"' in book
    curve = c.get("/api/book/equity", auth=AUTH).json()
    assert all(v == 20_000 for v in curve.get("equity", [])) and not any(curve.get("drawdown", []))


def test_a_strategy_a_clean_slate_put_away_still_holding_stays_in_the_book_until_flat(client, tmp_path):
    """5 Oct 2026: the 4 Oct clean slate archived btc-trend-smoke and btc-trend-daily while they were still
    long, and the book then left them out, so their positions sat in no exposure, risk page or kill switch.
    One still holding stays in the book, the kill switch reaches it, and it leaves once it is flat."""
    c, store = client
    store.create_sleeve(name="old-long", strategy="trend_filter", instrument="BTC/USD",
                        bar_spec="1-HOUR-LAST-INTERNAL", starting_balance=7000)
    store.record_fill("old-long", side="BUY", qty=0.05, price=60_000, fee=1.2, order_id="O-1", trade_id="T-1")
    store.record_equity("old-long", equity=6998.8, cash=3998.8, qty=0.05, price=60_000, benchmark=7000)
    store.set_desired_state("old-long", "stopped")
    from sleeve_fund.store import sleeve_archive_t, utcnow

    with store.engine.begin() as conn:  # archived with its position, as the clear did before #86
        conn.execute(sleeve_archive_t.insert().values(sleeve="old-long", archived_at=utcnow()))
    store.decide("system", "clear", "2026-10-04: new book (1 put away)")  # as the 4 Oct clear left it
    assert "old-long" not in store.previous_book()
    risk = c.get("/risk", auth=AUTH).text
    assert 'href="/sleeves/old-long"' in risk and "Flatten everything" in risk
    store.record_fill("old-long", side="SELL", qty=0.05, price=60_000, fee=1.2, order_id="O-2", trade_id="T-2")
    assert set(store.previous_book()) == {"old-long"}
    assert 'href="/sleeves/old-long"' not in c.get("/risk", auth=AUTH).text


def test_the_strategy_page_shows_how_old_its_price_feed_is(client):
    """PM, 5 Oct 2026: next to the Live badge, the seconds since the strategy's venue last sent it a trade or
    quote, so a quiet feed is seen at once. Stale past a minute; off while the strategy is stopped."""
    from datetime import timedelta

    from sleeve_fund.paper.runtime import SleeveRuntime

    c, store = client
    _new(c)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert 'data-live="feed"' in page and "Prices: waiting for the first trade" in page
    now = [utcnow()]
    rt = SleeveRuntime(store, "btc-test", now=lambda: now[0])
    rt.market_seen()
    now[0] += timedelta(seconds=1)
    rt.market_seen()  # within the write interval: not written again
    assert store.last_feed("btc-test") == now[0] - timedelta(seconds=1)
    store.feed_seen("btc-test", utcnow() - timedelta(seconds=7))
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "Prices: 7 s ago" in page or "Prices: 8 s ago" in page
    assert '<span class="ok">Prices' in page
    store.feed_seen("btc-test", utcnow() - timedelta(seconds=150))
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert '<span class="bad-dot">Prices: 2 min ago' in page
    store.set_desired_state("btc-test", "stopped")
    assert "Prices: off while stopped" in c.get("/sleeves/btc-test", auth=AUTH).text
    home = c.get("/", auth=AUTH).text
    assert "Prices: live venue feeds" in home and "Kraken live feed" not in home


def test_maker_first_orders_are_switched_off_by_default(client, prices, instrument):
    """PM, 4 Oct 2026: market orders only until a strategy proves it needs maker fills. The form offers no
    order type, a hand-made maker request is refused in words, and so is a backtest asking for one."""
    from sleeve_fund.research.runner import run_backtest

    c, store = client
    form = c.get("/sleeves/new?strategy=trend_filter", auth=AUTH).text
    assert 'name="execution" value="market"' in form and "Maker first" not in form
    r = _new(c, execution="maker", maker_wait_minutes="20")
    assert r.status_code in (200, 303, 400) and "btc-test" not in [s.name for s in store.sleeves()]
    with pytest.raises(ValueError, match="switched off"):
        run_backtest("trend_filter", prices.iloc[:50], instrument, {"fast": 5, "slow": 20, "maker_wait_minutes": 10})


def test_a_resume_on_a_running_strategy_is_refused_in_words(client):
    """Review round 10, m10-3: there is nothing to resume on a running strategy."""
    c, store = client
    _new(c, name="eth-run", instrument="ETH/USD")
    store.set_status("eth-run", "running", "")
    r = c.post("/sleeves/eth-run/command", data={"command": "resume", "reason": "again"}, auth=AUTH, headers=SAME)
    assert "Not done: it is already running, so there is nothing to resume." in r.text
    assert store.pending_commands("eth-run") == []
    store.set_status("eth-run", "paused", "paused by PM")
    c.post("/sleeves/eth-run/command", data={"command": "resume", "reason": "carry on"}, auth=AUTH, headers=SAME)
    assert [x["command"] for x in store.pending_commands("eth-run")] == ["resume"]


def test_the_g2_checklist_says_a_long_short_strategy_has_no_g1_yet(client):
    """Round 12, M12-U3: the G1 row explains why a perp or long/short strategy can't pass yet."""
    from sleeve_fund.dashboard import gates
    from sleeve_fund.dashboard.book import sleeve_extras
    from sleeve_fund.dashboard.metrics import sleeve_summary

    c, store = client
    store.create_sleeve(name="rsi-ls", strategy="rsi_bands", instrument="BTC/USD", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True})
    x = sleeve_extras(store, sleeve_summary(store, store.sleeve("rsi-ls")), pd.DataFrame())
    row = next(r for r in gates.path_to_live(store, x, None, store.accounts(), utcnow()) if "G1" in r["label"])
    assert not row["ok"] and "perpetual or long/short" in row["detail"]


def test_the_audit_csv_has_every_fill_with_its_pnl_reason_and_indicator_values(client):
    """PM, 5 Oct 2026: every trade checkable outside the dashboard, in a spreadsheet or against a chart: time,
    instrument, side, price, size, fee, the P&L each fill realised, and the indicator values behind the decision."""
    import csv
    import io

    c, store = client
    store.create_sleeve(name="rsi", strategy="rsi_bands", instrument="BTC/USD", bar_spec="15-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000)
    store.record_order("rsi", order_id="o1", side="BUY", qty=0.1, intent="entry",
                       reason="RSI 29.5 at or below 30: long until it reaches 55", signal={"rsi": 29.5, "close": 60_000.0})
    store.record_fill("rsi", side="BUY", qty=0.1, price=60_000.0, fee=4.8, order_id="o1", trade_id="t1")
    store.record_order("rsi", order_id="o2", side="SELL", qty=0.1, intent="exit",
                       reason="RSI 55.2 reached 55: the long leg ends", signal={"rsi": 55.2, "close": 60_600.0})
    store.record_fill("rsi", side="SELL", qty=0.04, price=60_600.0, fee=1.94, order_id="o2", trade_id="t2")
    store.record_fill("rsi", side="SELL", qty=0.06, price=60_500.0, fee=2.9, order_id="o2", trade_id="t3")
    r = c.get("/exports/audit.csv?sleeve=rsi", auth=AUTH)
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.text)))
    assert list(rows[0])[:15] == ["ts", "strategy", "venue", "instrument", "side", "qty", "price", "notional", "fee",
                                  "realised_pnl", "position_after", "intent", "reason", "order_id", "trade_id"]
    assert [(x["side"], x["rsi"], x["venue"]) for x in rows] == [("BUY", "29.5", "KRAKEN"), ("SELL", "55.2", "KRAKEN"),
                                                                 ("SELL", "55.2", "KRAKEN")]
    # The buy realises only its fee; each sell its share of the move off the 60,000 entry, after its fee.
    assert [float(x["realised_pnl"]) for x in rows] == pytest.approx([-4.8, 0.04 * 600 - 1.94, 0.06 * 500 - 2.9])
    assert [float(x["position_after"]) for x in rows] == pytest.approx([0.1, 0.06, 0.0])
    assert rows[1]["reason"] == "RSI 55.2 reached 55: the long leg ends"
    page = c.get("/sleeves/rsi", auth=AUTH).text
    assert 'href="/exports/audit.csv?sleeve=rsi"' in page
    assert 'href="/exports/audit.csv"' in c.get("/reports", auth=AUTH).text


def test_the_audit_trail_books_a_turn_from_long_to_short_at_the_turning_price():
    from types import SimpleNamespace

    from sleeve_fund.dashboard.trading import audit_rows

    s = SimpleNamespace(name="x", instrument="BTC/USD")
    fills = [{"ts": 1, "side": "BUY", "qty": 1.0, "price": 100.0, "fee": 0.0},
             {"ts": 2, "side": "SELL", "qty": 3.0, "price": 110.0, "fee": 0.0},  # closes 1 (+10), opens 2 short at 110
             {"ts": 3, "side": "BUY", "qty": 2.0, "price": 105.0, "fee": 0.0}]  # covers 2 at 105: +10
    rows, keys = audit_rows(s, fills, {}, "KRAKEN")
    assert [r["realised_pnl"] for r in rows] == pytest.approx([0.0, 10.0, 10.0]) and keys == []
    assert [r["position_after"] for r in rows] == pytest.approx([1.0, -2.0, 0.0])


def test_research_backtest_and_new_strategy_pages_offer_the_venue(client, tmp_path, monkeypatch):
    """Binance as a venue on every page that picks one (PM, 5 Oct 2026): the research page shows the chosen
    venue's stored history and asks its collector, the backtest page tests its perpetual, and a new strategy
    keeps its venue. A perpetual venue has no spot, so spot there is refused with the reason."""
    from sleeve_fund import history
    from sleeve_fund.dashboard.app import _backtest_args
    from sleeve_fund.venues import venue
    from test_research import _stored_minutes

    c, store = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore().append("KRAKEN", "ETH/USD", _stored_minutes(3), cursor="x")
    history.HistoryStore().append("BINANCE", "SOL/USDT", _stored_minutes(3), cursor="y")
    monkeypatch.setattr(venue("binance"), "check_listed", lambda pair: None)
    # The study's venue is picked in place (no reload); History lists every venue's stored instruments.
    page = c.get("/research?venue=binance", auth=AUTH).text
    assert 'name="venue" value="binance" data-pl-venue' in page and "data-reload" not in page
    assert re.search(r'<td data-label="Instrument">SOL/USDT</td>\s*<td[^>]*>perpetual</td>', page)
    assert re.search(r'<td data-label="Instrument">ETH/USD</td>\s*<td[^>]*>spot</td>', page)
    assert "A perpetual is studied long only" in page and "Binance" not in page.split('id="tab-history"')[1].split("</section>")[0]
    kraken = c.get("/research?venue=kraken", auth=AUTH).text
    assert 'name="venue" value="kraken" data-pl-venue' in kraken and "Spot is studied long only" in kraken
    asked = c.post("/research/history", data={"instrument": "doge/usdt", "venue": "binance"}, auth=AUTH, headers=SAME)
    assert "Asked the collector for DOGE/USDT" in asked.text and not store.history_requests("KRAKEN")
    assert [r["instrument"] for r in store.history_requests("BINANCE")] == ["DOGE/USDT"]
    none = c.post("/research/run", data={"strategy": "buy_and_hold", "instrument": "BTC/USDT", "venue": "binance",
                                         "minutes": "1440", "risk_profile": "balanced"}, auth=AUTH, headers=SAME)
    assert "BTC/USDT: not stored yet." in none.text

    bt = c.get("/backtest?venue=binance", auth=AUTH).text
    assert 'name="venue" value="binance" data-pl-venue' in bt and 'value="BTC/USDT"' in bt
    q = {"strategy": "buy_and_hold", "instrument": "BTC/USDT", "venue": "binance", "bar_spec": "1-DAY-LAST-EXTERNAL"}
    with pytest.raises(ValueError, match="perpetuals only"):
        _backtest_args(q)
    args = _backtest_args({**q, "market": "perp"})
    assert args["venue"] == "BINANCE" and "(perpetual)" in args["title"] and "Binance" not in args["title"]

    spot = _new(c, name="bn-spot", instrument="BTC/USDT", venue="binance")
    assert "perpetuals+only" in spot.headers["location"] and "venue=binance" in spot.headers["location"]
    ok = _new(c, name="bn-perp", instrument="BTC/USDT", venue="binance", market="perp")
    assert ok.headers["location"] == "/sleeves/bn-perp" and store.sleeve("bn-perp").venue == "BINANCE"
    assert _new(c, name="kr").status_code == 303 and store.sleeve("kr").venue is None
    shown = c.get("/sleeves/bn-perp", auth=AUTH).text
    # The header's line: model · instrument and market · candle · profile; never the venue's name (QA U8).
    assert "BTC/USDT perpetual ·" in shown and "Binance USD-M" not in shown and "venue=binance" in shown  # clone keeps it


def test_risk_and_health_reads_a_feed_as_fresh_from_its_venues_latest_trade():
    from datetime import timedelta

    from sleeve_fund.dashboard import riskops
    from sleeve_fund.store import Store, utcnow

    store = Store.in_memory()
    store.create_sleeve(name="bn", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={}, venue="binance")
    x = {"sleeve": store.sleeve("bn"), "healthy": True}
    assert riskops.feed_fresh(store, x)  # no trade seen yet and no stale-price warning
    store.feed_seen("bn", utcnow() - timedelta(seconds=20))
    assert riskops.feed_fresh(store, x)
    store.feed_seen("bn", utcnow() - timedelta(seconds=riskops.FEED_FRESH_SECONDS + 5))
    assert not riskops.feed_fresh(store, x)
    assert not riskops.feed_fresh(store, {**x, "healthy": False})




def test_reset_strategy_spells_out_what_it_closes_and_queues_it_for_the_supervisor(client):
    """PM, 5 Oct 2026: a quick reset while testing that closes the position, puts the run away and starts
    again at the starting capital. The confirm says exactly what happens; the supervisor carries it out."""
    c, store = client
    store.create_sleeve(name="bn-ls", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True},
                        venue="binance")
    store.record_fill("bn-ls", side="BUY", qty=0.076, price=86_000.0, fee=3.27, order_id="o1", trade_id="t1")
    page = c.get("/sleeves/bn-ls", auth=AUTH).text
    assert 'data-open="dlg-reset"' in page and 'action="/sleeves/bn-ls/reset"' in page
    assert "Closes the long of 0.076" in page and "and the copy on the demo account" in page and "Bybit" not in page
    assert "Puts this run away under Previous book" in page and "Starts bn-ls again at 10,000.00" in page
    assert "isolated margin at 2×" in page and "Demo copy" in page and "out of line" in page
    r = c.post("/sleeves/bn-ls/reset", data={"reason_pick": "Test finished; starting a clean run", "reason_note": ""},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and store.pending_reset("bn-ls")["restart"] == 1
    assert store.decisions("bn-ls")[0]["action"] == "reset"
    assert "Resetting…" in c.get("/sleeves/bn-ls", auth=AUTH).text
    r = c.post("/sleeves/bn-ls/reset", data={"reason": "again"}, auth=AUTH, headers=SAME, follow_redirects=False)
    assert "already+under+way" in r.headers["location"]
    _new(c)
    setup = c.get("/setup", auth=AUTH).text
    assert 'data-open="dlg-reset-all"' in setup and 'action="/book/reset"' in setup and "Applies to: " in setup
    r = c.post("/book/reset", data={"reason_pick": "Other", "reason_note": "short"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "reset_error=" in r.headers["location"] and not store.pending_reset("btc-test")
    r = c.post("/book/reset", data={"reason_pick": "Test finished; starting a clean run"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/setup?reset=1" and {x["sleeve"] for x in store.pending_resets()} == {"bn-ls", "btc-test"}
    assert "Demo copy <span" not in c.get("/sleeves/btc-test", auth=AUTH).text  # not copied to Bybit Demo


def test_room_to_halt_is_measured_from_the_peak(client):
    """Round 13, U13-3: room to halt was (limit - drawdown) x equity, which understates it by equity / peak
    once equity is below the peak. It is equity - peak x (1 - limit), as the stop check measures it."""
    from sleeve_fund.dashboard.metrics import sleeve_summary

    c, store = client
    _new(c, name="btc-room", starting_balance="10000")
    store.record_equity("btc-room", equity=11_000, cash=11_000, qty=0, price=60_000, benchmark=10_000)
    store.record_equity("btc-room", equity=10_000, cash=10_000, qty=0, price=60_000, benchmark=10_000)
    x = sleeve_summary(store, store.sleeve("btc-room"))
    assert x["room"] == pytest.approx(10_000 - 11_000 * (1 - 0.20))  # 1,200.00 on the balanced 20% limit
    assert "1,200.00" in c.get("/", auth=AUTH).text and "1,200.00" in c.get("/risk", auth=AUTH).text


VENUE_NAME = re.compile(r"\b(?:Bybit|Binance|Kraken|Deribit|BYBIT|BINANCE|KRAKEN|DERIBIT)\b(?!_)")


def _visible(html: str) -> str:
    """The words a reader sees and hears: text, hovers and screen-reader labels, without scripts or markup."""
    html = re.sub(r"(?s)<(script|style)\b.*?</\1>", " ", html)
    shown = re.findall(r'\b(?:title|aria-label|placeholder|alt)="([^"]*)"', html)
    return " ".join([re.sub(r"<[^>]+>", " ", html), *shown])


def test_the_pm_pages_name_no_venue_even_in_old_messages(client):
    """P1-U18: mirror reasons, alerts and the decision log name no venue, including rows stored before the
    wording changed; only Setup, Accounts does."""
    c, store = client
    assert _new(c).status_code == 303
    old = {
        "mirror_skipped": "Demo mirror didn't copy the buy of 0.0001 to the demo account: 0.0001 (6.00 USDT) is under "
                          "the smallest order Bybit takes (0.001, 5 USDT). The paper book is unaffected.",
        "mirror_failed": "Demo mirror couldn't copy the sell of 0.01 at 60,000.00 to the demo account: Bybit Demo "
                         "Trading /v5/order/create: insufficient balance (110007). The paper book is unaffected.",
        "mirror_resync": "Demo copy resynced (test): BTCUSDT: paper +0.01 at 2x isolated; Bybit Demo before +0; "
                         "after +0.01",
    }
    for kind, message in old.items():
        store.event("btc-test", "warning", kind, message)
    store.event(None, "error", "mirror_drift", "Deribit testnet holds -5 USD of BTC-PERPETUAL; KRAKEN feed stale")
    store.decide("pm", "create_account", "live account qa-live on Kraken spot: QA walk")
    store.decide("pm", "create_account", "live account qa-perp on Binance USD-M perpetuals: QA walk")
    for path in ("/", "/alerts", "/risk", "/trades", "/orders", "/records", "/sleeves/btc-test"):
        r = c.get(path, auth=AUTH)
        assert r.status_code == 200, path
        text = _visible(r.text)
        assert not VENUE_NAME.findall(text), (path, VENUE_NAME.findall(text))
    alerts = _visible(c.get("/alerts", auth=AUTH).text)
    assert "under the smallest order the demo account takes" in alerts
    assert "to the demo account: refused (/v5/order/create): insufficient balance" in alerts
    records = _visible(c.get("/records", auth=AUTH).text)
    assert "live account qa-live on the spot venue" in records
    assert "live account qa-perp on the perpetual venue" in records
    # Setup, Accounts is where a venue is named on purpose: the filter leaves it alone.
    assert "Kraken" in c.get("/setup", auth=AUTH).text


def test_no_venues_keeps_account_and_key_names():
    from sleeve_fund.wording import no_venues

    assert no_venues("kraken-live: BYBIT_DEMO_API_KEY missing") == "kraken-live: BYBIT_DEMO_API_KEY missing"
    assert no_venues("no Bybit demo account set up") == "the demo account isn't set up"
    assert no_venues("A Kraken order on Binance's book") == "The spot venue order on the perpetual venue's book"
    assert no_venues("BTCUSDT-PERP.BINANCE: the feed missed 2 minutes") == "BTCUSDT-PERP: the feed missed 2 minutes"
    assert no_venues("BTC/USD.KRAKEN warm-up ready.") == "BTC/USD warm-up ready."
    # P1-U21: old sentences read naturally once filtered.
    assert (no_venues("didn't copy the buy: no Bybit demo perpetual set up for BTC/USDT.")
            == "didn't copy the buy: the demo account has no perpetual set up for BTC/USDT.")
    assert (no_venues("to the demo account: Deribit testnet private/buy: not_enough_funds ().")
            == "to the demo account: refused (private/buy): not_enough_funds ().")
    assert no_venues("Bybit's API said position exists") == "the demo account's API said position exists"
    assert no_venues(None) is None and no_venues("") == ""


def test_the_spot_cap_bar_reads_on_the_exposure_basis(client):
    """P1-U17: on spot the limit bar uses the same basis as the Exposure figure (value at today's price against
    the entry cap), so its hover and screen-reader text agree with it after the price has moved."""
    c, store = client
    _new(c)
    store.record_fill("btc-test", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="o1", trade_id="t1")
    store.record_equity("btc-test", equity=5_600, cash=2_000, qty=0.05, price=72_000, benchmark=5_000)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    shown, cap = map(float, re.search(r"<dt[^>]*>Exposure</dt><dd>([\d.]+)× equity <span class=\"faint\">· cap "
                                      r"([\d.]+)×", page).groups())
    bar = re.search(r'<div class="exposure-cap" title="([^"]+)"><span class="k">Exposure of cap</span>'
                    r'<span class="meter[^"]*" role="img" aria-label="(\d+)% of the entry cap used"', page)
    assert bar, "the spot limit bar is on the exposure basis"
    assert f"Exposure {shown:.2f}× equity, against an entry cap of {cap:.2f}×" in bar.group(1)
    assert int(bar.group(2)) == round(3_600 / 5_600 / cap * 100)
    assert "Isolated margin of cap" not in page and "Position of cap" not in page


def test_an_atr_stop_says_it_is_the_simple_atr_and_the_chart_draws_it(client):
    """atr-149 A2: the models' ATR stops use the simple ATR, the chart's ATR is Wilder's, so wherever the
    position's stop is shown its basis says "simple ATR", and the chart draws the stop level itself."""
    c, store = client
    _new(c, stop_atr="2", atr_bars="14")
    store.record_order("btc-test", order_id="E-1", side="BUY", qty=0.05, intent="entry", reason="trend up",
                       signal={"stop_frac": 0.04, "stop_basis": "2 x the 14-bar average true range (1,200)",
                               "stop_cfg": {"stop_atr": 2.0, "atr_bars": 14}})
    store.record_fill("btc-test", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-1", trade_id="t1")
    store.record_equity("btc-test", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "57,600" in page and "· 2 simple ATR (14 bars) at entry" in page
    risk = c.get("/risk", auth=AUTH).text
    assert "from its entry, 2 simple ATR (14 bars) at entry" in risk
    lines = c.get("/api/sleeves/btc-test/candles", auth=AUTH).json()["lines"]
    assert {"price": 57_600.0, "title": "SL · simple ATR", "kind": "stop"} in lines


def test_a_fixed_stop_has_no_atr_label_and_the_stress_note_names_3x_gap_loss(client):
    c, store = client
    _new(c, stop_loss_pct="3")
    store.record_order("btc-test", order_id="E-1", side="BUY", qty=0.05, intent="entry", reason="trend up",
                       signal={"stop_frac": 0.03, "stop_cfg": {"stop_loss": 0.03}})
    store.record_fill("btc-test", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-1", trade_id="t1")
    store.record_equity("btc-test", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    assert "simple ATR" not in c.get("/sleeves/btc-test", auth=AUTH).text
    lines = c.get("/api/sleeves/btc-test/candles", auth=AUTH).json()["lines"]
    assert {"price": 58_200.0, "title": "SL", "kind": "stop"} in lines
    assert "at 3x, a gap through liquidation loses the whole position margin" in c.get("/risk", auth=AUTH).text


def test_the_last_demo_resync_result_names_no_venue(client):
    """P1-U19: a resync result stored before #160 reads "demo account", not the venue, on the Demo copy card."""
    c, store = client
    store.create_sleeve(name="bn-ls", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params={"market": "perp", "demo_mirror": True}, venue="binance")
    store.request_resync("bn-ls", "PM asked")
    store.finish_resync(store.pending_resyncs()[0]["id"],
                        "BTCUSDT: paper +0.1 at 2x isolated; Bybit Demo before +0, cross 10x; after +0.1")
    page = c.get("/sleeves/bn-ls", auth=AUTH).text
    assert "the demo account before +0, cross 10x" in page and "Bybit" not in page


def test_a_trailing_stop_says_its_level_is_not_shown_rather_than_guess_it(client):
    """atr-149 A2 (HoE/Advisor ruling): rsi_pullback trails its stop inside the model and doesn't journal the
    level yet, so the pages say so and the chart draws no stop line, never an estimate."""
    c, store = client
    store.create_sleeve(name="rp", strategy="rsi_pullback", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5_000, params={"atr_mult": 2.5})
    store.record_order("rp", order_id="E-1", side="BUY", qty=0.05, intent="entry", reason="RSI 28 in an up-trend")
    store.record_fill("rp", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-1", trade_id="t1")
    store.record_equity("rp", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    page = c.get("/sleeves/rp", auth=AUTH).text
    assert "Trailing stop · Trail level not shown" in page
    assert 'title="Trailing stop, 2.5 simple ATR below the highest close since entry"' in page
    risk = c.get("/risk", auth=AUTH).text
    assert ">Trailing</span>" in risk and "trail level not shown" in risk
    lines = c.get("/api/sleeves/rp/candles", auth=AUTH).json()["lines"]
    assert [ln["kind"] for ln in lines] == ["entry"]


def test_a_trailing_stop_is_not_unbounded_and_open_risk_says_what_it_leaves_out(client):
    """P1-U24: a trailing stop bounds the loss, so its risk to stop reads "trailing · level not shown", never
    "unbounded"; "unbounded" is for a position with no stop of any kind. Open risk says which it leaves out."""
    c, store = client
    store.create_sleeve(name="rp", strategy="rsi_pullback", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5_000, params={"atr_mult": 2.5})
    store.record_order("rp", order_id="E-1", side="BUY", qty=0.05, intent="entry", reason="RSI 28 in an up-trend")
    store.record_fill("rp", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-1", trade_id="t1")
    store.record_equity("rp", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    trailing = "trailing · level not shown"
    for url in ("/risk", "/trades", "/"):
        page = c.get(url, auth=AUTH).text
        assert "unbounded" not in page.replace("so unbounded", ""), url
    assert trailing in c.get("/risk", auth=AUTH).text and trailing in c.get("/trades", auth=AUTH).text
    assert "rp: trailing stop, level not shown, not counted" in c.get("/risk", auth=AUTH).text
    assert "1 with a trailing stop, level not shown, not counted" in c.get("/", auth=AUTH).text
    # A position with no stop of any kind is still unbounded, and the hover lists both.
    store.create_sleeve(name="nostop", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5_000, params={})
    store.record_order("nostop", order_id="E-2", side="BUY", qty=0.05, intent="entry", reason="start")
    store.record_fill("nostop", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-2", trade_id="t2")
    store.record_equity("nostop", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    risk = c.get("/risk", auth=AUTH).text
    assert "nostop: no stop, so unbounded · rp: trailing stop, level not shown, not counted" in risk


def test_open_risk_tile_with_only_a_trailing_position_shows_plus_warn_and_count(client):
    """P1-U24-1 (QA probe, Advisor S9): whenever a position is left out of the Open risk sum (no stop, or a
    trailing stop whose level isn't shown) the tile carries the "+", the warn colour and a visible count; a
    hover-only note isn't enough. Home, Risk and Trades agree."""
    c, store = client
    store.create_sleeve(name="rp", strategy="rsi_pullback", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5_000, params={"atr_mult": 2.5})
    store.record_order("rp", order_id="E-1", side="BUY", qty=0.05, intent="entry", reason="RSI 28 in an up-trend")
    store.record_fill("rp", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-1", trade_id="t1")
    store.record_equity("rp", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)

    def tiles():
        out = {}
        for url in ("/", "/risk", "/trades"):
            html = c.get(url, auth=AUTH).text
            m = re.search(r'<div class="k">Open risk</div><div class="v([^"]*)">([^<]*)', html)
            out[url] = (m.group(1), m.group(2)) if m else None
        return out

    for url, got in tiles().items():
        assert got is not None, url
        assert "warn" in got[0], url
        assert got[1].strip().endswith("0.00+ · 1 not counted"), (url, got)
    # A stopless position joins it: both are left out, so the count is 2.
    store.create_sleeve(name="nostop", strategy="buy_and_hold", instrument="BTC/USD", bar_spec="1-HOUR-LAST-INTERNAL",
                        starting_balance=5_000, params={})
    store.record_order("nostop", order_id="E-2", side="BUY", qty=0.05, intent="entry", reason="start")
    store.record_fill("nostop", side="BUY", qty=0.05, price=60_000, fee=2.4, order_id="E-2", trade_id="t2")
    store.record_equity("nostop", equity=5_000, cash=2_000, qty=0.05, price=60_500, benchmark=5_000)
    for url, got in tiles().items():
        assert "warn" in got[0] and got[1].strip().endswith("+ · 2 not counted"), (url, got)


def test_resuming_after_a_liquidation_says_it_stays_halted(client):
    """Advisor 6 Oct 17:57: a strategy halted because its position margin was lost stays halted through a
    resume until the PM resets it after liquidation; other halts keep their wording. Read from the journal,
    so a Stop/Start that halts it again on drawdown doesn't hide the liquidation (QA P1-U22)."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    stays = ("Its position margin was lost (liquidated), so it stays halted: resuming doesn&#39;t restart it. It "
             "trades again only after you use Reset after liquidation, which asks for an incident note.")
    normal = "The strategy trades again on its next signal. Its drawdown reference resets to today"
    store.set_status("btc-test", "halted", "drawdown 21% hit the 20% limit")
    store.event("btc-test", "error", "risk_halt", "Drawdown 21% hit the 20% limit: flattened, PM must resume")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert normal in page and "Reset after liquidation" not in page
    store.event("btc-test", "error", "liquidation", "Liquidated: the price 50,000 gapped through 51,000")
    store.event("btc-test", "error", "risk_halt", "Position margin lost (liquidated): 1,000.00, 20% of strategy equity")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert stays in page and normal not in page
    # Stop, then Start: the new runtime halts it again with a fresh drawdown reason. Still liquidated.
    store.set_status("btc-test", "stopped")
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    store.event("btc-test", "error", "risk_halt", "drawdown 96.2% hit the 20% limit; flattened, PM must resume")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert stays in page and normal not in page
    # Only a reset after liquidation ends it; a later ordinary halt then reads as one.
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it: the half-liquidation stop gapped")
    store.event("btc-test", "error", "risk_halt", "Drawdown 22% hit the 20% limit: flattened, PM must resume")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert normal in page and "it stays halted" not in page


def test_a_liquidation_order_alone_marks_the_halt_as_liquidated(client):
    """The paper guard's liquidation journals an order with intent "liquidation"; that counts too."""
    c, store = client
    _new(c)
    store.record_order("btc-test", order_id="L-1", side="SELL", qty=0.05, intent="liquidation",
                       reason="Liquidated: gapped through the liquidation price")
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    assert "it stays halted" in c.get("/sleeves/btc-test", auth=AUTH).text


def test_a_resume_is_refused_while_liquidated_and_the_page_points_to_the_reset(client):
    """QA P1-U25 and U26: the server refuses a Resume while liquidated (a stale page or a direct post), the
    dialog offers no Resume button, and the halted banner points to Reset after liquidation."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    store.event("btc-test", "error", "liquidation", "Liquidated: the price 50,000 gapped through 51,000")
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "Nothing trades until you resume" not in page
    assert "nothing trades until you use Reset after liquidation, which asks for an incident note" in page
    dialog = page.split('id="dlg-resume"')[1].split("</dialog>")[0]
    assert 'value="resume"' not in dialog and ">Resume trading<" not in dialog and ">Close<" in dialog
    r = c.post("/sleeves/btc-test/command", data={"command": "resume", "reason": "carry on"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and "command_error" in r.headers["location"]
    assert not any(cmd["command"] == "resume" for cmd in store.pending_commands("btc-test"))
    assert "Reset after liquidation" in c.get(r.headers["location"], auth=AUTH).text
    # After the reset, an ordinary halt resumes as before.
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it after liquidation")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "Nothing trades until you resume" in page
    assert 'value="resume"' in page.split('id="dlg-resume"')[1].split("</dialog>")[0]
    r = c.post("/sleeves/btc-test/command", data={"command": "resume", "reason": "carry on"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert "command_error" not in r.headers["location"]
    assert any(cmd["command"] == "resume" for cmd in store.pending_commands("btc-test"))


def test_a_reset_is_refused_while_liquidated_and_points_to_the_reset_after_liquidation(client):
    """Advisor 6 Oct 20:41 (QA P1-U27): an ordinary per-strategy Reset would put the liquidation away
    unanswered, so it is refused, halted or not, and the Reset button is off, until Reset after liquidation."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    store.event("btc-test", "error", "liquidation", "Liquidated: the price 50,000 gapped through 51,000")
    store.set_status("btc-test", "stopped")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    button = page.split('data-open="dlg-reset"')[1].split(">")[0]
    assert "disabled" in button and "Reset after liquidation" in button
    r = c.post("/sleeves/btc-test/reset", data={"reason": "Test finished"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and "command_error" in r.headers["location"]
    assert store.pending_reset("btc-test") is None
    assert "Reset after liquidation" in c.get(r.headers["location"], auth=AUTH).text
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it after liquidation")
    button = c.get("/sleeves/btc-test", auth=AUTH).text.split('data-open="dlg-reset"')[1].split(">")[0]
    assert "disabled" not in button
    r = c.post("/sleeves/btc-test/reset", data={"reason": "Test finished"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" not in r.headers["location"] and store.pending_reset("btc-test") is not None


@pytest.mark.parametrize("status", ["stopped", "halted"])
def test_start_resume_and_reset_are_all_refused_while_liquidated_whatever_the_status(client, status):
    """QA P1-U31 and U27: one guard for the three restart routes. Stop then Start must not restart a liquidated
    strategy, and the page says so for a stopped one too."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    store.event("btc-test", "error", "liquidation", "Liquidated: the price 50,000 gapped through 51,000")
    store.set_desired_state("btc-test", "stopped")
    store.set_status("btc-test", status, "drawdown 96.2% hit the 20% limit" if status == "halted" else "")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "until you use Reset after liquidation" in page
    dialog = page.split('id="dlg-start"')[1].split("</dialog>")[0]
    assert 'value="start"' not in dialog and "can&#39;t start" in dialog
    for command in ("start", "resume"):
        r = c.post("/sleeves/btc-test/command", data={"command": command, "reason": "carry on"}, auth=AUTH,
                   headers=SAME, follow_redirects=False)
        assert "command_error" in r.headers["location"], command
    r = c.post("/sleeves/btc-test/reset", data={"reason": "Test finished"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "command_error" in r.headers["location"]
    s = store.sleeve("btc-test")
    assert s.desired_state == "stopped" and store.pending_reset("btc-test") is None
    assert not store.pending_commands("btc-test")
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it after liquidation")
    page = c.get("/sleeves/btc-test", auth=AUTH).text
    assert "until you use Reset after liquidation" not in page
    r = c.post("/sleeves/btc-test/command", data={"command": "start", "reason": "carry on"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert "command_error" not in r.headers["location"] and store.sleeve("btc-test").desired_state == "running"


def test_a_book_reset_skips_a_liquidated_strategy_and_names_it(client):
    """Code review on #164 (HoE): Setup's Reset book must not put a liquidation away unanswered either. The
    liquidated strategy is left for Reset after liquidation and named; the others reset as before."""
    c, store = client
    _new(c)
    _new(c, name="btc-other")
    store.event("btc-test", "error", "liquidation", "Liquidated: the price 50,000 gapped through 51,000")
    r = c.post("/book/reset", data={"reason_pick": "Test finished; starting a clean run"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert "not_reset=btc-test" in r.headers["location"]
    assert {x["sleeve"] for x in store.pending_resets()} == {"btc-other"}
    setup = c.get(r.headers["location"], auth=AUTH).text
    assert "Not reset: btc-test. Its position margin was lost (liquidated)" in setup
    assert "Reset after liquidation" in setup and "Reset asked for every other strategy" in setup


@pytest.mark.parametrize("words", ["position margin lost (liquidated): 1.00", "POSITION MARGIN LOST (LIQUIDATED)",
                                   " Position margin lost (liquidated)", "Position  margin lost (liquidated)",
                                   "Position\u00a0margin lost (liquidated)"])
def test_the_liquidated_halt_matches_in_any_case_and_spacing(client, words):
    """QA P1-U28a."""
    c, store = client
    _new(c)
    store.event("btc-test", "error", "risk_halt", words)
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    assert "it stays halted" in c.get("/sleeves/btc-test", auth=AUTH).text


def test_a_liquidation_order_in_the_same_instant_as_the_reset_wins_the_tie(client):
    """QA P1-U28b: orders and events share no id, so a liquidation order with the reset's timestamp counts,
    unless the reset answered a liquidation event of that same instant."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    at = utcnow()
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it after liquidation", ts=at)
    store.record_order("btc-test", order_id="L-1", side="SELL", qty=0.05, intent="liquidation",
                       reason="Liquidated", ts=at)
    assert "it stays halted" in c.get("/sleeves/btc-test", auth=AUTH).text


def test_a_reset_answering_a_liquidation_of_the_same_instant_clears_it(client):
    """The engine journals the event with the order; a reset of that instant, written after both, clears it."""
    from sleeve_fund.store import LIQUIDATION_RESET

    c, store = client
    _new(c)
    at = utcnow()
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    store.record_order("btc-test", order_id="L-1", side="SELL", qty=0.05, intent="liquidation",
                       reason="Liquidated", ts=at)
    store.event("btc-test", "error", "liquidation", "Liquidated: gapped through the liquidation price", ts=at)
    store.event("btc-test", "info", LIQUIDATION_RESET, "PM reset it after liquidation", ts=at)
    assert "it stays halted" not in c.get("/sleeves/btc-test", auth=AUTH).text


def test_a_liquidated_halt_far_back_in_the_journal_still_counts(client):
    """Code review on #164: the liquidation is found however many events came after it."""
    c, store = client
    _new(c)
    store.event("btc-test", "error", "risk_halt", "Position margin lost (liquidated): 900.00, 18% of strategy equity")
    for i in range(600):
        store.event("btc-test", "error", "tick_failed", f"tick {i} failed")
    store.set_status("btc-test", "halted", "drawdown 96.2% hit the 20% limit")
    assert "it stays halted" in c.get("/sleeves/btc-test", auth=AUTH).text
