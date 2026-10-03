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
    assert "stop loss 8.0%" in page and "Closed trades" in page and "100%" in page
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
