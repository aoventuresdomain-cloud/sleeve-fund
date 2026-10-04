"""Review round 3, R3-B1 and R3-B2: a backtest journals in memory (fast), is saved to the real
journal under its own name, opens in the paper screens, and runs in the background with progress."""

import time

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.store import Store
from sleeve_fund.venues import KRAKEN

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}
RUN = ("/backtest?run=1&instrument=ETH/USD&strategy=trend_filter&p_trend_filter__fast=5"
       "&p_trend_filter__slow=20&risk_profile=aggressive")


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.dashboard import preview

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(KRAKEN, "daily_history", lambda pair: synthetic_ohlcv(days=400, seed=3, vol=0.03))
    preview._history.clear()
    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def test_a_backtest_is_saved_and_opens_in_the_paper_screens(client):
    c, store = client
    r = c.get(RUN, auth=AUTH, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"].startswith("/backtest/")
    run_id = r.headers["location"].rsplit("/", 1)[1]
    name = f"bt:{run_id}"
    page = c.get(r.headers["location"], auth=AUTH).text
    assert "Saved run:" in page and f"/orders?sleeve=bt%3A{run_id}" in page

    orders = store.orders(name, limit=1000)
    assert orders and all(o["reason"] and o["order_id"].startswith(run_id) for o in orders)
    assert {f["order_id"] for f in store.fills(name, limit=1000)} <= {o["order_id"] for o in orders}
    assert store.equity_series(name, limit=10)  # marks for the strategy screen's curve

    blotter = c.get(f"/orders?sleeve={name}", auth=AUTH).text
    assert "Showing a saved backtest" in blotter and orders[0]["reason"].split()[0] in blotter
    trades = c.get(f"/trades?sleeve={name}", auth=AUTH).text
    assert "Showing a saved backtest" in trades
    screen = c.get(f"/sleeves/{name}", auth=AUTH).text
    assert "Back to the backtest" in screen and 'id="dlg-flatten"' not in screen
    assert c.get(f"/exports/orders.csv?sleeve={name}", auth=AUTH).status_code == 200

    # It never runs, takes no commands, and stays out of the book, the blotter and the alerts.
    assert name not in [s.name for s in store.sleeves()]
    assert all(o["sleeve"] != name for o in store.orders(limit=1000))
    assert all(e["sleeve"] != name for e in store.alerts(limit=1000))
    r = c.post(f"/sleeves/{name}/command", data={"command": "pause", "reason": "x"}, auth=AUTH, headers=SAME)
    assert r.status_code == 400
    r = c.post(f"/sleeves/{name}/command", data={"command": "start", "reason": "x"}, auth=AUTH, headers=SAME)
    assert r.status_code == 400

    # The trade table is rebuilt from the journal, with size and fees.
    assert "Size</th>" in page and "Fees</th>" in page
    # Asking again for the same settings shows the saved run instead of making a copy.
    again = c.get(RUN, auth=AUTH, follow_redirects=False)
    assert again.headers["location"] == f"/backtest/{run_id}"


def test_a_long_backtest_shows_progress_and_then_its_result(client, monkeypatch):
    from sleeve_fund.dashboard import app as app_mod

    c, store = client
    monkeypatch.setattr(app_mod, "BACKTEST_WAIT", 0.0)
    page = c.get(RUN + "&starting_balance=20000", auth=AUTH).text
    assert 'id="bt-job"' in page and "Running:" in page
    job_id = page.split('data-job="', 1)[1].split('"', 1)[0]
    for _ in range(200):
        j = c.get(f"/api/backtest/jobs/{job_id}", auth=AUTH).json()
        if j["status"] != "queued" and j["status"] != "running":
            break
        time.sleep(0.05)
    assert j["status"] == "done" and j["progress"] == 1.0
    assert store.backtest(j["run_id"])["title"].startswith("Trend filter on ETH/USD")


def test_old_backtests_are_pruned_with_their_journals(client, monkeypatch):
    from sleeve_fund.dashboard import app as app_mod

    c, store = client
    monkeypatch.setattr(app_mod, "BACKTEST_KEEP", 2)
    ids = []
    for capital in (1000, 2000, 3000):
        r = c.get(RUN + f"&starting_balance={capital}", auth=AUTH, follow_redirects=False)
        ids.append(r.headers["location"].rsplit("/", 1)[1])
    assert [b["id"] for b in store.backtests()] == ids[:0:-1]
    assert store.orders(f"bt:{ids[0]}") == [] and c.get(f"/backtest/{ids[0]}", auth=AUTH).status_code == 404


def test_the_memory_journal_matches_the_database_journal():
    """Buffering changes speed, not results: the same run on either journal sends the same orders."""
    from sleeve_fund.paper.runtime import SleeveRuntime
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    prices = synthetic_ohlcv(days=300, seed=5, vol=0.03)
    inst = venue("kraken").instrument("ETH", "USD")

    def orders(runtime):
        run_backtest("trend_filter", prices, inst, params={"fast": 5, "slow": 20}, runtime=runtime)
        return [(o["side"], o["intent"], round(o["qty"], 8), o["status"], o["ts"])
                for o in runtime.store.orders("backtest", limit=10_000)]

    fast = SleeveRuntime.for_backtest(strategy="trend_filter", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                                      starting_balance=10_000, risk_profile="aggressive")
    db = Store.in_memory()
    db.create_sleeve(name="backtest", strategy="trend_filter", instrument="ETH/USD", bar_spec="1-DAY-LAST-EXTERNAL",
                     starting_balance=10_000, risk_profile="aggressive")
    slow = SleeveRuntime(db, "backtest", tick_seconds=86_400)
    got = orders(fast)
    assert got and got == orders(slow)


def test_a_backtest_screen_shows_no_paper_only_panels(client):
    """Round 4, R4-M3 and R4-M6: no paper chip, Today, heartbeat, reconcile, account, path to live or
    Clone on a saved run, and its dates carry the year."""
    c, store = client
    r = c.get(RUN + "&bar_spec=1-DAY-LAST-EXTERNAL", auth=AUTH, follow_redirects=False)
    run_id = r.headers["location"].rsplit("/", 1)[1]
    screen = c.get(f"/sleeves/bt:{run_id}", auth=AUTH).text
    assert '<span class="chip">Backtest</span>' in screen and "Not in the book." in screen
    for gone in ("since midnight", "Heartbeat", "Reconciled", "Path to live", "Clone with changes",
                 'href="/accounts#acct-', "on Kraken"):
        assert gone not in screen, gone
    first = store.fills(f"bt:{run_id}", limit=1_000_000)[-1]["ts"]
    assert first.strftime("%d %b %Y") in screen  # synthetic history starts years ago, so the year shows
    candles = c.get(f"/api/sleeves/bt:{run_id}/candles", auth=AUTH).json()
    assert candles["note"] == "Candles built from the run's price marks."


def test_backtests_stay_out_of_accounts_counts_and_alerts(client):
    """Round 4, R4-M5: accounts, the journal counts and alert acknowledgements ignore saved runs, and
    an acknowledgement made before that rule can't block pruning."""
    from sleeve_fund.store import acks_t, utcnow

    c, store = client
    r = c.get(RUN, auth=AUTH, follow_redirects=False)
    name = "bt:" + r.headers["location"].rsplit("/", 1)[1]
    assert all(name not in a["sleeves"] for a in store.accounts())
    sizes = store.table_sizes()
    assert sizes["sleeves"] == 0 and sizes["fills"] == 0 and sizes["backtests"] == 1
    assert "Saved backtests" in c.get("/ops", auth=AUTH).text

    store.event(name, "warning", "risk_pause", "paused in the replay")
    ev = store.events(name, limit=1)[0]
    with pytest.raises(KeyError):
        store.ack(ev["id"], "PM")
    with store.engine.begin() as conn:  # as an older build could have written it
        conn.execute(acks_t.insert().values(event_id=ev["id"], ts=utcnow(), actor="PM", note=""))
    assert store.prune_backtests(keep=0) == 1 and store.backtests() == []


def test_saved_runs_are_listed_and_keep_the_chosen_interval(client, monkeypatch, tmp_path):
    """Round 4, R4-M4 and B4-5: an hourly run on bars built from trades is saved as that, not as the
    venue's hourly candles its replay used."""
    from sleeve_fund import history
    from test_dashboard import _wavy_minutes

    c, store = client
    monkeypatch.setattr(history, "DEFAULT_ROOT", tmp_path / "hist")
    history.HistoryStore(tmp_path / "hist").append("KRAKEN", "ETH/USD", _wavy_minutes(20), cursor="x")
    r = c.get(RUN + "&bar_spec=1-HOUR-LAST-INTERNAL", auth=AUTH, follow_redirects=False)
    run_id = r.headers["location"].rsplit("/", 1)[1]
    assert store.sleeve(f"bt:{run_id}").bar_spec == "1-HOUR-LAST-INTERNAL"
    page = c.get("/backtest", auth=AUTH).text
    assert "Saved runs" in page and f'href="/backtest/{run_id}"' in page and "Trend filter on ETH/USD" in page
