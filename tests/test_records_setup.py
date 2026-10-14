"""Records (/records) and Setup (/setup): the redirects from the old pages, the records filters, and the
path-to-live state."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.store import Store, utcnow

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    monkeypatch.setenv("BACKUP_DIR", str(tmp_path / "backups"))
    monkeypatch.delenv("DEMO_MIRROR", raising=False)
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _new(c, **over):
    form = {"name": "btc-test", "strategy": "trend_filter", "instrument": "BTC/USD",
            "bar_spec": "1-HOUR-LAST-INTERNAL", "starting_balance": "5000", "risk_profile": "balanced",
            "warmup_bars": "0", "p_trend_filter__fast": "10", "p_trend_filter__slow": "30", "reason": "first test", **over}
    r = c.post("/sleeves/new", data=form, auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303, r.text[:300]
    return r


def _loc(c, path):
    r = c.get(path, auth=AUTH, follow_redirects=False)
    return r.status_code, r.headers.get("location")


# --- redirects -----------------------------------------------------------------------------------------------

def test_old_pages_redirect_with_their_query(client):
    c, _ = client
    assert _loc(c, "/reports") == (303, "/records")
    assert _loc(c, "/reports?sleeve=btc-a") == (303, "/records?sleeve=btc-a")
    assert _loc(c, "/decisions") == (303, "/records#log")
    assert _loc(c, "/decisions?sleeve=btc-a&action=pause") == (303, "/records?sleeve=btc-a&action=pause#log")
    assert _loc(c, "/settings") == (303, "/setup#settings")
    assert _loc(c, "/accounts") == (303, "/setup#accounts")
    assert _loc(c, "/accounts?error=x") == (303, "/setup?error=x#accounts")
    assert c.get("/reports").status_code == 401  # still behind the password
    for path in ("/records", "/setup"):
        assert c.get(path, auth=AUTH).status_code == 200
    # Downloads keep their addresses.
    for path in ("/decisions.csv", "/exports/trades.csv", "/exports/audit.csv", "/exports/orders.csv",
                 "/exports/equity.csv", "/exports/fills.csv"):
        assert c.get(path, auth=AUTH).status_code == 200, path


def test_account_forms_land_on_setup_with_their_result(client):
    c, store = client
    ok = c.post("/accounts/new", data={"name": "kraken-trend", "kind": "live", "venue": "kraken", "reason": "first"}, auth=AUTH,
                headers=SAME, follow_redirects=False)
    assert ok.headers["location"] == "/setup?saved=kraken-trend#acct-kraken-trend"
    page = c.get(ok.headers["location"], auth=AUTH).text
    assert "Saved kraken-trend." in page and 'id="acct-kraken-trend"' in page
    bad = c.post("/accounts/new", data={"name": "Bad Name", "kind": "live", "reason": "x"}, auth=AUTH, headers=SAME,
                 follow_redirects=False)
    assert bad.headers["location"].startswith("/setup?error=") and bad.headers["location"].endswith("#add")
    page = c.get(bad.headers["location"], auth=AUTH).text
    assert "Not saved:" in page and 'value="Bad Name"' in page
    note = c.post("/accounts/kraken-trend/note", data={"note": "trend", "reason": "  "}, auth=AUTH, headers=SAME,
                  follow_redirects=False)
    assert note.headers["location"].startswith("/setup?error=") and note.headers["location"].endswith("#accounts")


# --- records -------------------------------------------------------------------------------------------------

def test_records_period_and_strategy_filter_the_figures_log_and_exports(client, monkeypatch):
    import sleeve_fund.store as store_mod

    c, store = client
    _new(c, name="btc-a")
    _new(c, name="eth-b", instrument="ETH/USD")
    now = utcnow()
    old, recent = now - timedelta(days=100), now - timedelta(days=3)
    for days, eq_a, eq_b in ((120, 5000, 5000), (100, 5200, 4900), (40, 5300, 4800), (3, 5500, 4700)):
        ts = now - timedelta(days=days)
        store.record_equity("btc-a", equity=eq_a, cash=eq_a, qty=0, price=1, benchmark=5000, ts=ts)
        store.record_equity("eth-b", equity=eq_b, cash=eq_b, qty=0, price=1, benchmark=5000, ts=ts)
    store.record_fill("btc-a", side="BUY", qty=1, price=100, fee=1.5, order_id="o1", trade_id="t1", ts=old)
    store.record_fill("btc-a", side="SELL", qty=1, price=110, fee=2.5, order_id="o2", trade_id="t2", ts=recent)
    with monkeypatch.context() as m:
        m.setattr(store_mod, "utcnow", lambda: old)
        store.decide("PM", "pause", "an old pause", "btc-a")
        m.setattr(store_mod, "utcnow", lambda: recent)
        store.decide("PM", "change_settings", "tighter stop", "eth-b")

    page = c.get("/records", auth=AUTH).text  # All: every decision, both strategies
    assert "an old pause" in page and "tighter stop" in page and "Book by month" in page
    assert 'data-tab="performance"' in page and 'data-tab="log"' in page
    assert "quote currency" in page and "Strategies by month" in page
    assert 'href="/exports/trades.csv"' in page and 'href="/decisions.csv"' in page

    one = c.get("/records?period=1m", auth=AUTH).text
    assert "tighter stop" in one and "an old pause" not in one
    assert "2.50" in one and "1 fill<" in one  # fees and fills inside the month only
    assert f"/decisions.csv?from={(now - timedelta(days=30)).strftime('%Y-%m-%d')}" in one  # the log's export follows

    a = c.get("/records?sleeve=btc-a", auth=AUTH).text
    assert "an old pause" in a and "tighter stop" not in a
    assert 'href="/exports/trades.csv?sleeve=btc-a"' in a and 'href="/exports/audit.csv?sleeve=btc-a"' in a
    assert "/decisions.csv?sleeve=btc-a" in a
    assert "+10.0%" in a  # btc-a: 5,000 to 5,500

    # A month chip filters to that month, and links back to the period.
    key = recent.strftime("%Y-%m")
    assert f'href="/records?month={key}"' in page
    month = c.get(f"/records?month={key}", auth=AUTH).text
    assert recent.strftime("%b %Y") + "</strong> only" in month and "an old pause" not in month
    assert c.get("/records?period=nope&month=2026-13&sleeve=nobody", auth=AUTH).status_code == 200

    # An old Decisions link keeps its filters on the log.
    log = c.get("/decisions?sleeve=btc-a&action=pause", auth=AUTH).text
    assert "an old pause" in log and "first test" not in log


def test_decision_kinds_and_days():
    from sleeve_fund.dashboard import reports

    assert [reports.decision_kind(a) for a in ("start", "create", "stop", "archive", "pause", "flatten everything",
                                               "change_settings", "retire_account", "approve_g2")] == \
        ["started", "started", "stopped", "stopped", "risk", "risk", "changed", "changed", "approval"]
    now = utcnow()
    days = reports.timeline([{"ts": now, "action": "start"}, {"ts": now - timedelta(days=1), "action": "stop"},
                             {"ts": now - timedelta(days=9), "action": "pause"}], now)
    assert [d["label"] for d in days][:2] == ["Today", "Yesterday"] and len(days) == 3
    assert days[2]["items"][0]["kind"] == "risk"
    w = reports.window("6m", "2026-09", now)
    assert (w["start"].month, w["end"].month) == (9, 10) and w["period"] == "6m"
    assert reports.window("bogus", "", now)["period"] == reports.DEFAULT_PERIOD


# --- setup ---------------------------------------------------------------------------------------------------

def _here(page):
    import re

    m = re.search(r'<li class="jn here" aria-current=step>.*?<b>([^<]+)</b>', page)
    return m.group(1) if m else None


def test_path_to_live_follows_the_real_state(client, monkeypatch):
    c, store = client
    page = c.get("/setup", auth=AUTH).text
    assert _here(page) == "Research" and "Step 1 of 5" in page
    _new(c, name="btc-a")
    page = c.get("/setup", auth=AUTH).text
    assert _here(page) == "Paper" and "Step 2 of 5" in page and "You are here" in page  # G2 not approved: Paper
    # A strategy asks for the mirror, but the mirror hasn't named a demo account: still Paper.
    _new(c, name="eth-perp", instrument="ETH/USDT", venue="binance", market="perp", demo_mirror="1",
         risk_profile="conservative")  # a stopless perp runs at 1x at most (check_perp_stop)
    assert store.sleeve("eth-perp").params.get("demo_mirror")
    store.event("eth-perp", "info", "mirror_start", "Demo mirror on: each new fill is copied to its demo account, "
                "once that is set up (BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET), and the paper book stays the record")
    page = c.get("/setup", auth=AUTH).text
    assert _here(page) == "Paper" and "asked, not running" in page
    # The mirror copied a fill: Demo check is in progress.
    store.record_mirror("eth-perp", fill_id=1, status="filled", instrument="ETHUSDT", amount=0.1, price=3000)
    page = c.get("/setup", auth=AUTH).text
    assert _here(page) == "Demo check" and "Step 3 of 5" in page and "mirroring" in page


def test_path_to_live_stages():
    from types import SimpleNamespace

    from sleeve_fund.dashboard import setup_view

    on, off = {"on": True}, {"on": False}
    locked = {"mode": "paper", "live_locked": True}
    assert setup_view.stage(locked, [], off) == 0
    assert setup_view.stage(locked, [SimpleNamespace()], off) == 1
    assert setup_view.stage(locked, [SimpleNamespace()], on) == 2
    assert setup_view.stage({"mode": "paper", "live_locked": False}, [SimpleNamespace()], on) == 4
    assert setup_view.stage({"mode": "live", "live_locked": False}, [SimpleNamespace()], off) == 4
    assert [s["state"] for s in setup_view.path(1)] == ["done", "here", "todo", "todo", "todo"]


def test_mirror_on_from_its_environment_switch(client, monkeypatch):
    from sleeve_fund.dashboard import setup_view

    c, store = client
    _new(c, name="eth-perp", instrument="ETH/USDT", venue="binance", market="perp", demo_mirror="1",
         risk_profile="conservative")  # a stopless perp runs at 1x at most (check_perp_stop)
    sleeves = store.sleeves()
    assert not setup_view.mirror_state(store, sleeves, environ={})["on"]
    assert setup_view.mirror_state(store, sleeves, environ={"DEMO_MIRROR": "on"})["on"]
    store.set_desired_state("eth-perp", "stopped")  # a stopped strategy isn't being mirrored
    assert not setup_view.mirror_state(store, store.sleeves(), environ={"DEMO_MIRROR": "on"})["on"]


def test_setup_cards_and_detail(client):
    c, store = client
    page = c.get("/setup", auth=AUTH).text
    for card in ("Accounts", "Venues and costs", "Risk profiles", "Outside alerts", "Backups", "Access"):
        assert f"<b>{card}</b>" in page, card
    assert 'class="scard att" href="#outside-alerts"' in page and "Not set up" in page  # amber until set up
    assert "halt at 10%" in page and "halt at 20%" in page and "halt at 35%" in page
    assert "Kraken spot" in page and "0.8% taker" in page
    # Today's Accounts and Settings content, unchanged, as the detail.
    assert "Connect a Kraken sub-account" in page and "How to connect" in page and "Never tick Withdraw Funds" in page
    assert "Live trading" in page and "Pause at daily loss" in page and "DASHBOARD_PASSWORD" in page
    assert 'id="tab-accounts"' in page and 'id="tab-settings"' in page and "test-pw" not in page
    store.event(None, "info", "alerts_config", "Alerts go to api.telegram.org; uptime pings go to hc-ping.com.")
    page = c.get("/setup", auth=AUTH).text
    assert 'class="scard att" href="#outside-alerts"' not in page and "Telegram" in page
    assert "Sent to api.telegram.org" in page
