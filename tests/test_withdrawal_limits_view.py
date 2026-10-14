"""F-C3B-A-UI (HoE, 14 Oct; Advisor on W10): putting a Strategy away takes its equity out of the fund's book, so the
same positions use more of each portfolio limit. The confirm view shows the book after the withdrawal and the limits in
use against it; the page after shows them against the book as it now is. Display only: every figure is the entry
gate's (its last mark less what Store._put_away takes, and its book's gross, net, margin and open risk)."""

from datetime import timedelta
from decimal import Decimal as D
from html import unescape

import pytest
from test_halt_ref import T0, _make, _mark, store  # noqa: F401

from sleeve_fund import supervisor
from sleeve_fund.dashboard.withdrawal import limits_in_use, words

AUTH, HDR = ("pm", "pw"), {"Origin": "http://testserver"}


@pytest.fixture
def held(store, monkeypatch):
    """a: 10,000 holding 6 BTC long at 1,000 (spot, no stop); b: 10,000 flat. The book 20,000, marked."""
    monkeypatch.setattr("sleeve_fund.open_risk.history_atr_pct", lambda *a: 0.02)  # the stopless measure's ATR
    _make(store, "a", 10_000)
    _make(store, "b", 10_000)
    store.record_fill("a", side="BUY", qty=6.0, price=1_000.0, fee=0.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("a", equity=10_000, cash=4_000, qty=6.0, price=1_000.0, benchmark=10_000, ts=T0)
    _mark(store, T0)
    return store


@pytest.fixture
def client(held, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    return TestClient(dash.create_app(held))


def _get(client, path):
    return unescape(client.get(path, auth=AUTH).text)


def _used(view):
    return {x["key"]: x["used"] for x in view["limits"]}


def test_the_preview_is_what_the_gate_reads_once_the_strategy_is_archived(held):
    now = limits_in_use(held)
    assert now["source"] == "gate" and now["after"] == 20_000 and _used(now)["gross"] == pytest.approx(0.3)
    preview = limits_in_use(held, ["b"])
    assert (preview["book"], preview["taken"], preview["after"]) == (20_000, 10_000, 10_000)
    assert _used(preview)["gross"] == pytest.approx(0.6) and _used(preview)["net"] == pytest.approx(0.6)
    assert _used(preview)["margin"] == pytest.approx(0.6)  # spot at its full notional
    held.archive("b")
    after = limits_in_use(held)  # waiting for the poll's mark (F286-CR1), shown as gone
    assert after["after"] == preview["after"] and _used(after) == pytest.approx(_used(preview))
    _mark(held, T0 + timedelta(seconds=5))
    after = limits_in_use(held)
    assert after["after"] == preview["after"] and _used(after) == pytest.approx(_used(preview))
    assert held.withdrawn() == {"b": D(10_000)}
    assert limits_in_use(held, ["b"])["taken"] == 0  # already out: nothing more to take


def test_a_limit_at_or_over_its_share_reads_red_and_says_entries_are_blocked_not_closed(held):
    preview = limits_in_use(held, ["b"])
    tones = {x["key"]: x["tone"] for x in preview["limits"]}
    assert tones["margin"] == "loss" and tones["gross"] == "" and "Margin" in preview["over"]  # 60% of 50%
    net = next(x for x in preview["limits"] if x["key"] == "net")
    assert net["what"] == "BTC" and net["tone"] == "loss"  # 60% of 50%, in one underlying
    line = words(preview)
    assert line.startswith("The fund's book is now 10,000.00. Limits in use: Total size 60.0% of 150%")
    assert "New entries that add to it are blocked until it is back under; nothing open is closed." in line
    assert "BINANCE" not in line.upper()


def test_emptying_the_book_or_no_mark_yet_says_so_instead_of_shares(store):
    _make(store, "solo", 5_000)
    assert limits_in_use(store, ["solo"])["source"] == "unmarked"
    _mark(store, T0)
    view = limits_in_use(store, ["solo"])
    assert view["empty"] and view["after"] == 0 and view["limits"] == []
    assert "no capital, so nothing can trade" in words(view)


def test_the_archive_dialog_previews_the_book_and_the_page_after_shows_it(held, client):
    page = _get(client, "/sleeves/b")
    dialog = page.split('id="dlg-archive"')[1].split("</dialog>")[0]
    assert "its equity of 10,000.00 leaves the fund's book, which isn't a loss" in dialog
    assert "The fund's book would be <b>10,000.00</b>, 10,000.00 less. Limits in use against it:" in dialog
    assert '<b class="loss">60.0%</b> <span class="faint">of 50%</span>' in dialog
    assert "The withdrawal still goes through and closes nothing" in dialog
    assert 'class="limits-after' not in _get(client, "/sleeves/a")  # holding: Archive isn't offered
    r = client.post("/sleeves/b/archive", data={"action": "archive", "reason": "done with it"}, auth=AUTH,
                    headers=HDR, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/?archived=b"
    home = _get(client, r.headers["location"])
    assert "Archived b: its equity left the fund's book, which isn't a loss." in home
    assert "The fund's book is now <b>10,000.00</b>. Limits in use against it:" in home
    assert "their equity is out of the fund's book</span>" in home
    assert "Archived b:" not in _get(client, "/")  # only right after
    r = client.post("/sleeves/b/archive", data={"action": "restore", "reason": "back"}, auth=AUTH, headers=HDR,
                    follow_redirects=False)
    assert r.headers["location"] == "/sleeves/b?put_away=restored"
    page = _get(client, r.headers["location"])
    assert "Restored: its equity is back in the fund's book, which isn't a gain." in page
    assert "The fund's book is now <b>20,000.00</b>" in page


def test_an_archive_from_before_the_withdrawal_is_said_to_still_count(held, client):
    from test_capital_withdrawal import _archived_before

    held.set_desired_state("b", "stopped")
    _archived_before(held, "b")
    assert "their equity is out of the fund's book, except 1 still in it<" in _get(client, "/")


def test_a_retire_shows_the_book_after_it(held, client):
    r = client.post("/sleeves/b/lifecycle", data={"action": "retire", "reason": "done"}, auth=AUTH, headers=HDR,
                    follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/sleeves/b?put_away=retired"
    page = _get(client, r.headers["location"])
    assert "Retired: its equity left the fund's book, which isn't a loss." in page
    assert "The fund's book is now <b>10,000.00</b>" in page
    r = client.post("/sleeves/a/lifecycle", data={"action": "wind_down", "reason": "rotate"}, auth=AUTH,
                    headers=HDR, follow_redirects=False)
    assert r.headers["location"] == "/sleeves/a"  # not put away: no line


def test_withdraw_archived_prints_the_book_after_it(held, monkeypatch, capfd):
    from test_capital_withdrawal import _archived_before

    _archived_before(held, "b")
    monkeypatch.setattr(supervisor, "Store", lambda **k: held)
    assert supervisor.main(["withdraw-archived", "--actor", "operator", "--commit", "abc1234"]) in (0, None)
    out = capfd.readouterr().out
    assert "queued for the supervisor's next mark: b" in out and "The fund's book is now 10,000.00. Limits in use: Total size 60.0% of 150%" in out


def test_an_archive_that_would_empty_the_book_says_so_with_no_shares(store, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    _make(store, "solo", 5_000)
    _mark(store, T0)
    dialog = _get(TestClient(dash.create_app(store)), "/sleeves/solo").split('id="dlg-archive"')[1].split("</dialog>")[0]
    assert "The fund's book would be 0.00:</b> no capital, so nothing can trade until some is added." in dialog
    assert "otherwise it is refused, with the reason" in dialog and "Limits in use against it" not in dialog


def test_a_strategy_not_flat_takes_nothing_out_and_shows_as_waiting_to_go_flat(held, client):
    """Advisor 14 Oct 10:45 UK: archived while it holds, it stays in the book in full until flat, then withdraws."""
    held.record_fill("b", side="BUY", qty=1.0, price=1_000.0, fee=0.0, order_id="b1", trade_id="b1", ts=T0)
    held.record_equity("b", equity=10_000, cash=9_000, qty=1.0, price=1_000.0, benchmark=10_000, ts=T0)
    preview = limits_in_use(held, ["b"])
    assert preview["waiting"] == {"b": held.withdrawal_blocker("b")} and preview["waiting"]["b"].startswith("it holds")
    assert (preview["taken"], preview["after"]) == (0, 20_000) and _used(preview)["gross"] == pytest.approx(0.35)
    held.archive("b")
    assert held.withdrawal_pending() == ["b"] and "b" not in held.withdrawn()
    home = _get(client, "/")
    assert "waiting to go flat: it holds" in home and "except 1 still in it (1 waiting to go flat)" in home
    page = _get(client, "/sleeves/b")
    assert "Archived, waiting to go flat: it holds" in page and "It stays in the fund's book in full until then." in page


def test_a_strategy_with_nothing_to_withdraw_says_so_and_stays_in_the_book(store, monkeypatch):
    """Advisor 14 Oct 11:10 UK: settled at 0 or less, nothing is withdrawn; it stays in the book at that figure."""
    from datetime import timedelta

    from fastapi.testclient import TestClient
    from test_capital_withdrawal import _close, _open, _poll

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    _make(store, "a", 20_000)
    _make(store, "b", 1_000)
    _open(store, "b", qty=1.0)
    _mark(store, T0)
    store.archive("b")
    _close(store, "b", qty=1.0, price=1.0, fee=5.0, ts=T0 + timedelta(minutes=1))  # 1,000 - 999 - 5: -4
    _poll(store, T0 + timedelta(minutes=1))
    assert store.nothing_withdrawn() == {"b": D(-4)}
    client = TestClient(dash.create_app(store))
    home = _get(client, "/")
    assert "nothing to withdraw (equity -4.00)" in home and "except 1 still in it<" in home
    assert "Archived, nothing to withdraw (equity -4.00). It stays in the fund's book at that figure." in _get(
        client, "/sleeves/b")
