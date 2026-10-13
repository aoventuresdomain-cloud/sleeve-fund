"""CCY-8DP-UI (HoE 13 Oct): a Strategy on a named test account (supervisor.is_test_currency_account) is marked in the
PM's views and left out of the book's figures, display only: it is never Strategy evidence, the kill switch still acts
on it, and what the engine's limits read is unchanged."""

from fastapi.testclient import TestClient

from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setattr(app_mod, "TEARSHEETS", tmp_path)
    monkeypatch.setattr(app_mod, "LEDGER", tmp_path / "idea_ledger.jsonl")
    store = Store(f"sqlite:///{tmp_path}/t.db")
    # one Strategy on the Strategy paper venue, one on the hub-fed test account (paper, at the default venue)
    store.create_sleeve(name="real", strategy="trend_filter", instrument="BTC/USDT", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=5000, params={"fast": 10, "slow": 30}, venue="BINANCE")
    store.create_sleeve(name="tst", strategy="trend_filter", instrument="BTC/USD", bar_spec="1-HOUR-LAST-EXTERNAL",
                        starting_balance=3000, params={"fast": 10, "slow": 30})
    store.record_equity("real", equity=5100, cash=3000, qty=0.02, price=105000, benchmark=5050)
    store.record_equity("tst", equity=2500, cash=500, qty=0.02, price=100000, benchmark=3000)
    return TestClient(app_mod.create_app(store)), store


def test_a_test_account_strategy_is_marked_and_listed_apart_on_home(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    home = c.get("/", auth=AUTH).text
    group = home[home.index('id="test-accounts"'):].split("</details>")[0]
    assert "Test accounts (1)" in group and 'href="/sleeves/tst"' in group and "Test account</span>" in group
    assert 'href="/sleeves/real"' not in group
    assert "test</span>" in home.split('data-live="nav-strategies"')[1].split("</div>")[0]  # marked in the rail too


def test_the_books_figures_leave_a_test_account_strategy_out(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    curve = c.get("/api/book/equity", auth=AUTH).json()
    assert curve["start"] == 5000 and curve["equity"][-1] == 5100.0  # the real Strategy's alone, not 8,000 / 7,600
    risk = c.get("/risk", auth=AUTH).text
    assert "tst runs on a test account: left out of the figures on this page" in risk


def test_its_path_to_live_says_it_is_not_strategy_evidence(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    page = c.get("/sleeves/tst", auth=AUTH).text
    assert "Runs on a Strategy account" in page and "its results are not Strategy evidence" in page
    assert "Test account</span>" in page
    assert "Runs on a Strategy account" not in c.get("/sleeves/real", auth=AUTH).text


def test_the_kill_switch_still_acts_on_a_test_account_strategy(tmp_path, monkeypatch):
    c, store = _client(tmp_path, monkeypatch)
    r = c.post("/book/flatten", data={"reason": "Market event; standing aside"}, auth=AUTH, headers=SAME,
               follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/risk?killed=2"
    for name in ("real", "tst"):
        (cmd,) = store.pending_commands(name)
        assert cmd["command"] == "flatten"


def test_the_limits_inputs_are_unchanged_by_the_display_filter(tmp_path, monkeypatch):
    """HoE's boundary: leaving test accounts out of the PM's totals changes nothing the engine's or supervisor's limits
    and halts read (open_risk.book_open_risk, the gate's own figure, and each Strategy's equity)."""
    from sleeve_fund import open_risk
    from sleeve_fund.dashboard import book as bookm

    _, store = _client(tmp_path, monkeypatch)
    marked = (open_risk.book_open_risk(store, {}), [store.last_equity(s.name) for s in store.sleeves()])
    monkeypatch.setattr(bookm, "is_test_account", lambda store, sleeve: False)
    assert (open_risk.book_open_risk(store, {}), [store.last_equity(s.name) for s in store.sleeves()]) == marked
    assert [s.name for s in store.sleeves()] == ["real", "tst"]
