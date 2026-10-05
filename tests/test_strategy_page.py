"""The strategy page (combined build F2-F4): its tabs, the reason every action asks for (its own list plus
Other with a note, stored as "<picked>" or "<picked>: <note>"), and the stop-while-holding flow: flatten
first, and a stop never drops a flatten that is still waiting."""

import re
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient

from sleeve_fund.dashboard import reasons
from sleeve_fund.store import Store

AUTH = ("pm", "test-pw")
SAME = {"origin": "http://testserver"}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    from sleeve_fund.dashboard import app as app_mod

    store = Store(f"sqlite:///{tmp_path}/t.db")
    return TestClient(app_mod.create_app(store)), store


def _sleeve(store, name="btc-x", held=0.0, params=None):
    store.create_sleeve(name=name, strategy="rsi_bands", instrument="BTC/USD", bar_spec="15-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=params or {})
    if held:
        store.record_order(name, order_id="E-1", side="BUY", qty=held, intent="entry", reason="RSI 28 at or below 30",
                           signal={})
        store.record_fill(name, side="BUY", qty=held, price=60_000, fee=4.8, order_id="E-1", trade_id="T-1")
    store.record_equity(name, equity=10_000, cash=10_000 - held * 60_000, qty=held, price=60_000, benchmark=10_000)
    store.heartbeat(name)  # its process is reporting


def _post(c, name, **data):
    return c.post(f"/sleeves/{name}/command", data=data, auth=AUTH, headers=SAME, follow_redirects=False)


# --- the reason format -------------------------------------------------------------------------------------

def test_reason_is_the_pick_or_the_pick_and_its_note():
    assert reasons.compose("pause", "Market event ahead") == "Market event ahead"
    assert reasons.compose("pause", "Market event ahead", "  payrolls   at 13:30 ") == "Market event ahead: payrolls at 13:30"
    assert reasons.compose("stop", "Other", "the venue is retiring this pair") == "Other: the venue is retiring this pair"


@pytest.mark.parametrize("action, pick, note, words", [
    ("pause", "", "", "pick a reason"),
    ("pause", "Other", "", "at least 10 characters"),
    ("pause", "Other", "too   short", "at least 10 characters"),  # 9 characters once spaces are tidied
    ("stop", "Market event ahead", "", "isn't one of the reasons"),  # pause's reason, not stop's
])
def test_reason_refuses_an_incomplete_pick(action, pick, note, words):
    with pytest.raises(ValueError, match=words):
        reasons.compose(action, pick, note)


def test_each_action_has_its_own_list():
    lists = reasons.ACTION_REASONS
    assert [r for r, _ in lists["pause"]] == ["Market event ahead", "Data or venue problem", "Reviewing a recent trade",
                                              "Settings change coming", "Too much risk across the book"]
    assert [r for r, _ in lists["resume"]] == ["Issue resolved", "Event passed", "Review done", "Settings changed"]
    assert [r for r, _ in lists["stop"]] == ["Test finished", "Kill rule hit", "Replaced by a new version",
                                             "Not behaving like its backtest", "Long-lasting venue or data problem"]
    assert [r for r, _ in lists["flatten"]] == ["Stop or risk limit reached", "Market event", "Reducing exposure",
                                                "Position looks wrong", "Data or venue problem"]
    assert lists["close"] == lists["flatten"] == lists["book_flatten"]
    for action in ("start", "archive", "save", "move"):
        assert lists[action]


def test_a_picked_reason_reaches_the_decision_log(client):
    c, store = client
    _sleeve(store)
    r = _post(c, "btc-x", command="pause", reason_pick="Reviewing a recent trade", reason_note="the 14:00 fill")
    assert r.status_code == 303
    assert store.pending_commands("btc-x")[0]["reason"] == "Reviewing a recent trade: the 14:00 fill"
    assert store.decisions("btc-x")[0]["reason"] == "Reviewing a recent trade: the 14:00 fill"
    # The toast after confirming repeats the reason.
    done = parse_qs(urlparse(r.headers["location"]).query)["done"][0]
    assert done == "Pause sent. Reason: Reviewing a recent trade: the 14:00 fill."
    assert 'class="toast" role="status" data-once>Pause sent. Reason: Reviewing' in c.get(r.headers["location"], auth=AUTH).text


def test_other_without_a_long_enough_note_is_refused(client):
    c, store = client
    _sleeve(store)
    r = c.post("/sleeves/btc-x/command", data={"command": "pause", "reason_pick": "Other", "reason_note": "hmm"},
               auth=AUTH, headers=SAME)
    assert "Not done: Other needs a note of at least 10 characters." in r.text
    assert store.pending_commands("btc-x") == []
    _post(c, "btc-x", command="pause", reason_pick="Other", reason_note="funding looks wrong on the venue")
    assert store.pending_commands("btc-x")[0]["reason"] == "Other: funding looks wrong on the venue"


def test_settings_archive_and_book_flatten_take_the_picked_reason(client):
    c, store = client
    _sleeve(store, held=0.01)
    r = c.post("/sleeves/btc-x/settings", data={"instrument": "BTC/USD", "risk_profile": "conservative",
                                                "reason_pick": "Change the risk level"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert "saved=settings" in r.headers["location"]
    assert store.decisions("btc-x", action="change_settings")[0]["reason"].endswith("Change the risk level")
    r = c.post("/book/flatten", data={"reason_pick": "Market event", "reason_note": "rate decision"}, auth=AUTH,
               headers=SAME, follow_redirects=False)
    assert "killed=1" in r.headers["location"]
    assert [x["reason"] for x in store.pending_commands("btc-x") if x["command"] == "flatten"] == [
        "Book kill switch: Market event: rate decision"]
    store.set_desired_state("btc-x", "stopped")
    r = c.post("/sleeves/btc-x/archive", data={"action": "archive", "reason_pick": "Superseded by a new version"},
               auth=AUTH, headers=SAME, follow_redirects=False)
    assert r.status_code == 303 and store.decisions("btc-x", action="archive")[0]["reason"] == "Superseded by a new version"


def test_every_dialog_lists_its_own_reasons_and_waits_for_one(client):
    c, store = client
    _sleeve(store, held=0.01)
    page = c.get("/sleeves/btc-x", auth=AUTH).text

    def dialog(cmd):
        return page[page.index(f'id="dlg-{cmd}"'):page.index("</dialog>", page.index(f'id="dlg-{cmd}"'))]

    assert "Market event ahead" in dialog("pause") and "Kill rule hit" not in dialog("pause")
    assert "Kill rule hit" in dialog("stop") and "Reducing exposure" in dialog("flatten")
    assert "Reducing exposure" in dialog("close") and 'name="reason_for" value="close"' in dialog("close")
    assert "Separate the books" in dialog("move")
    for cmd in ("pause", "stop", "flatten", "close", "move"):
        d = dialog(cmd)
        assert 'value="Other"' in d and 'name="reason_note"' in d
        assert re.search(r"<button[^>]*data-needs-reason disabled", d), cmd  # Confirm waits for a reason
    # Close position sends today's flatten, which pauses, and says so.
    assert 'value="flatten"' in dialog("close") and "the strategy pauses once it has sent the order" in dialog("close")
    # A position with no stop: Pause warns that nothing will exit it.
    assert "This position has no stop." in dialog("pause")


# --- stop while holding --------------------------------------------------------------------------------------

def test_stop_while_holding_flattens_first_then_asks_to_stop(client):
    c, store = client
    _sleeve(store, held=0.01)
    stop = c.get("/sleeves/btc-x", auth=AUTH).text
    stop = stop[stop.index('id="dlg-stop"'):stop.index("</dialog>", stop.index('id="dlg-stop"'))]
    assert "The position stays open." in stop and 'value="flatten-stop"' in stop and ">Flatten and stop<" in stop

    # Flatten and stop: the flatten goes now, with the Stop dialog's reason; the strategy keeps running.
    r = _post(c, "btc-x", command="flatten-stop", reason_pick="Test finished")
    assert [(x["command"], x["reason"]) for x in store.pending_commands("btc-x")] == [("flatten", "Test finished")]
    assert store.sleeve("btc-x").desired_state == "running"
    assert parse_qs(urlparse(r.headers["location"]).query) == {"then_stop": ["Test finished"]}
    waiting = c.get(r.headers["location"], auth=AUTH).text
    assert "Flattening first. Once it is flat, this page asks you to stop it." in waiting
    assert "Flat now. Stop it?" not in waiting

    # A stop now would drop the waiting flatten: refused, and the flatten stays.
    r = c.post("/sleeves/btc-x/command", data={"command": "stop", "reason_pick": "Test finished"}, auth=AUTH,
               headers=SAME)
    assert "Not done: a flatten is still waiting for the strategy to act on it" in r.text
    assert [x["command"] for x in store.pending_commands("btc-x")] == ["flatten"]
    assert store.sleeve("btc-x").desired_state == "running"
    page = c.get("/sleeves/btc-x", auth=AUTH).text
    stop = page[page.index('id="dlg-stop"'):page.index("</dialog>", page.index('id="dlg-stop"'))]
    assert "A flatten is still waiting." in stop and re.search(r'value="stop"[^>]*data-needs-reason disabled data-blocked', stop)

    # The strategy acts on it and is flat: the page offers the stop with the same reason.
    store.mark_applied(store.pending_commands("btc-x")[0]["id"])
    store.record_fill("btc-x", side="SELL", qty=0.01, price=60_100, fee=4.8, order_id="X-1", trade_id="T-2")
    flat = c.get("/sleeves/btc-x?then_stop=Test+finished", auth=AUTH).text
    assert "Flat now. Stop it?" in flat and '<input type="hidden" name="reason" value="Test finished">' in flat
    r = _post(c, "btc-x", command="stop", reason="Test finished")
    assert r.status_code == 303 and store.sleeve("btc-x").desired_state == "stopped"
    assert store.decisions("btc-x", action="stop")[0]["reason"] == "Test finished"


def test_a_stop_is_not_refused_when_the_process_is_not_there_to_flatten(client):
    """A flatten waiting on a process that has stopped reporting can't be acted on, so it doesn't lock the PM
    out of stopping the strategy (the stop lapses it, as before)."""
    from datetime import timedelta

    from sleeve_fund.store import sleeves_t, utcnow

    c, store = client
    _sleeve(store, held=0.01)
    _post(c, "btc-x", command="flatten", reason_pick="Market event")
    with store.engine.begin() as conn:
        conn.execute(sleeves_t.update().values(heartbeat_at=utcnow() - timedelta(minutes=10)))
    assert _post(c, "btc-x", command="stop", reason_pick="Long-lasting venue or data problem").status_code == 303
    assert store.sleeve("btc-x").desired_state == "stopped"


# --- the page ------------------------------------------------------------------------------------------------

def test_strategy_page_tabs_in_order_with_seven_figures(client):
    c, store = client
    _sleeve(store, held=0.01)
    page = c.get("/sleeves/btc-x", auth=AUTH).text
    nav = page[page.index("data-tabs>"):page.index("</nav>", page.index("data-tabs>"))]
    tabs = re.findall(r'data-tab="([a-z]+)"', nav)
    assert tabs == ["overview", "signals", "positions", "trades", "orders", "activity", "path", "settings"]
    assert ">Path to live<" in nav
    figures = page[page.index('aria-label="Strategy figures"'):page.index("</section>", page.index('aria-label="Strategy figures"'))]
    assert [k.strip() for k in re.findall(r'<div class="k">([^<{]+)', figures)] == [
        "Equity", "Today", "Since start", "In the market", "Unrealised", "Realised", "Fees paid"]
    overview = page[page.index('id="tab-overview"'):page.index('id="tab-signals"')]
    assert 'data-mode-to="price"' in overview and 'data-mode-to="equity"' in overview and 'id="pc"' in overview
    assert "Why it bought" in overview and "RSI 28 at or below 30" in overview and 'data-open="dlg-close"' in overview
    path = page[page.index('id="tab-path"'):page.index('id="tab-settings"')]
    assert "Research" in path and "Demo check" in path and "G2 approval" in path and "Your G2 approval" in path
    activity = page[page.index('id="tab-activity"'):page.index('id="tab-path"')]
    assert [k for k in re.findall(r'data-chip="([a-z]+)"', activity)] == ["all", "decisions", "risk", "system"]


def test_settings_save_opens_a_confirm_with_the_changes(client):
    c, store = client
    _sleeve(store)
    page = c.get("/sleeves/btc-x", auth=AUTH).text
    form = page[page.index('id="settings-form"'):page.index("</form>", page.index('id="settings-form"'))]
    assert "data-diff-open" in form and 'id="dlg-save"' in form and "data-diff-rows" in form
    assert "Research recommendation" in form and "Risk profile change." in form
    assert 'data-limits="Conservative: halts at' in form
