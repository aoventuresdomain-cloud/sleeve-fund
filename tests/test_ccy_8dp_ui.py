"""CCY-8DP-UI (HoE 13 Oct; F229-6 A4, 14 Oct): a Strategy outside the fund's book (supervisor.outside_fund_book: under
8 places on a named test account) is marked in the PM's views and left out of the book's figures, as the gate and
fund_equity leave it out. It is never Strategy evidence, and what the engine's limits read is unchanged."""

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
    history = c.get("/history", auth=AUTH).text  # B-16 part 2 (TEC): the panel moved from Home to History
    group = history[history.index('id="test-accounts"'):].split("</details>")[0]
    assert "Test accounts (1)" in group and 'href="/sleeves/tst"' in group and "Test account</span>" in group
    assert 'href="/sleeves/real"' not in group
    assert "test</span>" in home.split('data-live="nav-strategies"')[1].split("</div>")[0]  # marked in the rail too


def test_the_books_figures_leave_a_test_account_strategy_out(tmp_path, monkeypatch):
    c, _ = _client(tmp_path, monkeypatch)
    curve = c.get("/api/book/equity", auth=AUTH).json()
    assert curve["start"] == 5000 and curve["equity"][-1] == 5100.0  # the real Strategy's alone, not 8,000 / 7,600



def test_its_position_shows_marked_as_not_counted_and_the_home_equity_is_the_funds(tmp_path, monkeypatch):
    """F229-6 A4: mark, don't hide. The position row stays on the Positions tab, not counted in the open risk (outside
    the fund's book); the book's equity is the in-book Strategies' only."""
    from sleeve_fund.dashboard import trading

    c, _ = _client(tmp_path, monkeypatch)
    seen = []
    real = trading.book_positions
    monkeypatch.setattr(trading, "book_positions", lambda st, summaries: seen.append(
        sorted(x["sleeve"].name for x in summaries)) or real(st, summaries))
    home = c.get("/", auth=AUTH).text
    assert seen == [["real", "tst"]]  # both listed
    assert "outside the fund's book" in home and "incl. test accounts" not in home


def test_the_test_account_mark_is_the_gates_predicate(tmp_path, monkeypatch):
    from sleeve_fund import supervisor
    from sleeve_fund.dashboard import book as bookm

    _, store = _client(tmp_path, monkeypatch)
    for s in store.sleeves():
        assert bookm.is_test_account(store, s) == supervisor.outside_fund_book(store, s)
    assert [s.name for s in store.sleeves() if bookm.is_test_account(store, s)] == ["tst"]


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


def test_its_fills_and_funding_show_marked_and_the_funding_total_leaves_them_out(tmp_path, monkeypatch):
    """F273-1: Home lists what a test-account Strategy did on every tab, as its order and position are, marked; the
    book's funding total, and so its costs, leave it out."""
    c, store = _client(tmp_path, monkeypatch)
    for name, inst, venue in (("realp", "BTC/USDT", "BINANCE"), ("tstp", "BTC/USD", None)):
        store.create_sleeve(name=name, strategy="trend_filter", instrument=inst, bar_spec="1-HOUR-LAST-EXTERNAL",
                            starting_balance=3000, params={"fast": 10, "slow": 30, "market": "perp"},
                            **({"venue": venue} if venue else {}))
        store.record_fill(name, side="BUY", qty=0.01, price=100000, fee=0.5, order_id=f"o-{name}",
                          trade_id=f"t-{name}")
        store.record_equity(name, equity=3000, cash=2000, qty=0.01, price=100000, benchmark=3000)
    store.record_funding("realp", qty=0.01, price=100000, rate=0.0001, amount=-0.1)
    store.record_funding("tstp", qty=0.01, price=100000, rate=0.0001, amount=-7.0)
    home = c.get("/", auth=AUTH).text
    fills = home[home.index('id="fills-h"'):].split("</section>")[0]
    funding = home[home.index('id="funding-h"'):].split("</section>")[0]
    for tab in (fills, funding):
        assert 'href="/sleeves/realp"' in tab and 'href="/sleeves/tstp"' in tab
        row = tab[tab.index('href="/sleeves/tstp"'):].split("</td>")[0]
        assert "Test account</span>" in row  # marked where it is listed
        assert "Test account</span>" not in tab[tab.index('href="/sleeves/realp"'):].split("</td>")[0]
    assert "−7.00" in funding or "-7.00" in funding  # its payment is listed
    total = funding[funding.index("<tfoot>"):].split("</tfoot>")[0]
    assert "test accounts not counted" in total and "0.10" in total and "7.10" not in total
