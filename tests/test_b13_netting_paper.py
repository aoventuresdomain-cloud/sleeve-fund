"""B-13 (Head 6): netting across Strategies in shadow on paper (PM 13 Oct "shadow first"; Advisor specs-p2-3 §2).
Read from the paper journal: what one account per venue would have sent had each decision bar's intents in an
instrument gone out as one net order. It writes nothing and moves no Strategy's figures.

Every expected figure is worked by hand below from the rows written, never read back from the shadow."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import func, select

from sleeve_fund.portfolio import netting_paper as npr
from sleeve_fund.store import Store

TAKER = D("0.00055")
T1 = datetime(2026, 10, 20, 12, 0, tzinfo=timezone.utc)  # a decision bar's close
T2 = T1 + timedelta(minutes=1)
INST = "BTC/USDT"


def _ns(t):
    return int(t.timestamp()) * 1_000_000_000


@pytest.fixture(params=["sqlite", "postgres"])
def store(request):
    """Both engines: SQLite in memory, and Postgres where CI (or a local box) sets TEST_DATABASE_URL."""
    if request.param == "sqlite":
        store = Store.in_memory()
    else:
        import os

        from sleeve_fund.store import make_engine, metadata

        url = os.environ.get("TEST_DATABASE_URL")
        if not url:
            pytest.skip("Postgres: CI runs these against it (TEST_DATABASE_URL)")
        engine = make_engine(url)
        metadata.drop_all(engine)
        store = Store(engine=engine)
    for name in ("a", "b", "c"):
        store.create_sleeve(name=name, strategy="ping_pong", instrument=INST, bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=0, params={"market": "perp", "allow_short": True}, venue="binance")
    yield store
    store.engine.dispose()


def _trade(store, sleeve, oid, side, qty, px, close=None, bar=None, at=None, intent="entry", rate=TAKER):
    qty, px = D(qty), D(px)
    fee = (qty * px * rate).quantize(D("1e-8"))
    at = at or (bar or T1) + timedelta(seconds=2)
    timing = {"bar_close": _ns(bar), "bar_recv": _ns(bar), "decided": _ns(bar)} if bar is not None else None
    store.record_order(sleeve, order_id=oid, side=side, qty=float(qty), intent=intent, reason="test",
                       signal={"close": float(close)} if close is not None else {}, ts=at, timing=timing)
    store.record_fill(sleeve, side=side, qty=qty, price=px, fee=fee, order_id=oid, trade_id=f"t-{oid}", ts=at)
    return fee


def _run(store, members=lambda n: n in ("a", "b"), equity=D("100000")):
    fills, funding = npr.journal_rows(store, members)
    return npr.shadow(fills, funding, equity)


def test_b13a_opposite_intents_on_one_bar_net_to_one_order_saving_the_crossed_spread_and_fee(store):
    fa = _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)  # a buys 0.3, half spread 0.05% up
    fb = _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)  # b sells 0.1, 0.05% down
    [venue] = (out := _run(store))
    n = out[venue]
    # Alone: -0.3 x 60,030 + 0.1 x 59,970 = -12,012.00 in price; fees 9.904950 + 3.298350 = 13.203300.
    assert fa + fb == D("13.2033")
    # One account: net +0.2 at the mark moved by that bar's half spread, 60,000 x 1.0005 = 60,030; fee 6.6033.
    # Crossed saving: -12,006 - (-12,012) = 6; fee saving 13.2033 - 6.6033 = 6.60.
    assert n["saving"] == {"crossed": D("6"), "fee": D("6.6")}
    assert n["account_cash"] == D("-12012.6033") and n["sub_ledger_total"] == D("-12025.2033")
    assert n["sub_ledgers"] == {"a": D("-18018.90495"), "b": D("5993.70165")}  # each its own step A, untouched
    assert n["residual"]["amount"] == 0 and n["residual"]["block"] is False
    assert (n["orders_sent"], n["orders_alone"], n["bars_shared"]) == (1, 2, 1)
    assert n["crossed_notional"] == D("6000")  # 0.1 matched inside the account at the 60,000 mark


def test_b13b_intents_on_different_bars_or_one_way_save_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T2)  # the next bar: nothing to cross
    _trade(store, "b", "b2", "BUY", "0.2", "60030", close="60000", bar=T1 + timedelta(minutes=5))
    _trade(store, "a", "a2", "BUY", "0.1", "60030", close="60000", bar=T1 + timedelta(minutes=5))  # same side
    [n] = _run(store).values()
    assert n["saving"] == {} and n["orders_sent"] == 4 - 1  # the same-side bar is one order, at step A's price
    assert n["account_cash"] == n["sub_ledger_total"] and n["residual"]["amount"] == 0


def test_b13c_a_watched_exit_with_no_decision_bar_is_its_own_order(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    # b's target fired on a trade two seconds after T1's close: no decision bar, so it nets with nothing
    _trade(store, "b", "b1", "SELL", "0.1", "59970", at=T1 + timedelta(seconds=2), intent="take_profit")
    [n] = _run(store).values()
    assert n["saving"] == {} and n["orders_sent"] == 2


def test_b13d_fully_crossed_intents_send_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.2", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.2", "59970", close="60000", bar=T1)
    [n] = _run(store).values()
    # Nothing goes out: the account moves 0; alone they paid 0.2 x 60 in spread (12) and 6.6033 + 6.5967 in fees.
    assert n["orders_sent"] == 0 and n["account_cash"] == 0
    assert n["saving"] == {"crossed": D("12"), "fee": D("13.2")}


def test_b13e_only_the_funds_strategies_net_together(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "c", "c1", "SELL", "0.1", "59970", close="60000", bar=T1)  # c is outside the fund's book here
    [n] = _run(store).values()
    assert n["saving"] == {} and set(n["sub_ledgers"]) == {"a"}


def test_b13f_opposed_exposure_per_pair(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    [n] = _run(store).values()
    # Just after T1: a +0.3, b -0.1; cancelling 2 x 0.1 of a gross 0.4 = 0.5 (prices cancel out)
    assert n["opposed_exposure"] == {("a", "b"): pytest.approx(0.5)}


def test_b13g_it_writes_nothing(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    from sleeve_fund.store import metadata

    def counts():
        with store.engine.connect() as c:
            return {t.name: c.execute(select(func.count()).select_from(t)).scalar() for t in metadata.sorted_tables}

    before, books = counts(), {n: store.journal_book(n, 0) for n in ("a", "b")}
    _run(store)
    assert counts() == before and {n: store.journal_book(n, 0) for n in ("a", "b")} == books


def test_b13h_one_candles_fills_at_two_taker_rates_are_refused(store):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1, rate=D("0.0006"))
    with pytest.raises(ValueError, match="same taker fee"):
        _run(store)


def test_b13l_each_candle_is_priced_at_the_rate_its_fills_paid_then_never_todays(store):
    """HEAD6 1: inputs as journaled. The rate changes between two crossing candles: each nets at its own."""
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    _trade(store, "a", "a2", "SELL", "0.3", "59970", close="60000", bar=T2, rate=D("0.0004"))
    _trade(store, "b", "b2", "BUY", "0.1", "60030", close="60000", bar=T2, rate=D("0.0004"))
    [n] = _run(store).values()
    # T1 as in b13a: fee saved 6.60. T2 at 0.04%: alone 0.3 x 59,970 x 0.0004 = 7.1964 and 0.1 x 60,030 x 0.0004 =
    # 2.4012; net -0.2 at 60,000 x 0.9995 = 59,970, fee 4.7976; saved 9.5976 - 4.7976 = 4.80. Spread: 6 again.
    assert n["saving"] == {"crossed": D("12"), "fee": D("11.40")}
    assert _run(store) == _run(store)  # the same journal gives the same figures on every read


def test_b13m_strategies_on_two_accounts_never_net(store):
    store.create_account("second", kind="paper")
    store.assign_account("b", "second")
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    out = _run(store)
    assert len(out) == 2 and all(n["saving"] == {} for n in out.values())


def test_b13n_funding_nets_nothing_and_corrections_reconcile(store):
    """HEAD6 3: no funding saving line; the account's funding equals the Strategies' (corrections included)."""
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    settle = T1 + timedelta(hours=4)
    # 0.01% on a 60,000 mark: a's long 0.3 pays 1.80, b's short 0.1 receives 0.60
    store.record_funding("a", qty=0.3, price=60_000, rate=0.0001, amount=-1.8, ts=settle)
    store.record_funding("b", qty=-0.1, price=60_000, rate=0.0001, amount=0.6, ts=settle)
    # a later true-up of the rate to 0.02%: a pays 1.80 more, b receives 0.60 more
    store.record_funding("a", qty=0.3, price=60_000, rate=0.0002, amount=-1.8, ts=settle, kind="true_up",
                         arrived_at=settle + timedelta(hours=1))
    store.record_funding("b", qty=-0.1, price=60_000, rate=0.0002, amount=0.6, ts=settle, kind="true_up",
                         arrived_at=settle + timedelta(hours=1))
    [n] = _run(store).values()
    assert "funding" not in n["saving"] and n["corrections"] == 2
    # the account's net +0.2 pays 0.0001 x 0.2 x 60,000 = 1.20, then 1.20 more: -2.40, the Strategies' -3.60 + 1.20
    assert n["funding"]["account"] == D("-2.4") and n["funding"]["sub_ledgers"] == D("-2.4")
    assert n["funding"]["gap"] == 0 and n["residual"]["amount"] == 0


def test_b13o_the_nightly_check_alerts_a_residual_over_the_line_and_blocks_nothing(store, monkeypatch):
    from sleeve_fund.portfolio import netting as nt

    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    members = lambda n: n in ("a", "b")  # noqa: E731
    assert npr.over_line(store, members, D("100000")) == []
    real = nt.reconcile
    monkeypatch.setattr(nt, "reconcile", lambda s, b, a, e: real(s, b, a + D("60"), e))  # 6 bp of 100,000 unexplained
    [line] = npr.over_line(store, members, D("100000"))
    assert line.startswith("netting shadow (trading unaffected): the account of a, b") and "6.0 bp" in line


# ---- the page: its own route and partial, shadow, not evidence ------------------------------------------------------

def _page(store, monkeypatch, equity=D("100000")):
    import re

    from fastapi.testclient import TestClient

    import sleeve_fund.supervisor as sup
    from sleeve_fund.dashboard import app as dash
    from sleeve_fund.dashboard import netting_view

    def fund(_store):
        if isinstance(equity, Exception):
            raise equity
        return equity

    monkeypatch.setattr(sup, "fund_equity", fund)
    monkeypatch.setattr(sup, "_in_book", lambda store, s: s.name in ("a", "b"))
    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    r = TestClient(dash.create_app(store)).get("/netting-shadow", auth=("pm", "pw"))
    assert r.status_code == 200
    return " ".join(re.sub(r"<[^>]+>", " ", r.text).split())


def test_b13i_the_page_shows_the_saving_labelled_shadow_not_evidence(store, monkeypatch):
    from sleeve_fund.wording import no_venues

    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    words = _page(store, monkeypatch)
    assert "Shadow, not evidence" in words and "Account 1: a, b" in words
    assert "Saved in fees 6.60" in words and "Saved in spread (opposite orders matched) 6.00" in words
    assert "Total saved 12.60" in words and "Orders: combined against alone 1 against 2" in words
    assert "Unexplained difference 0.00 (0.0 bp)" in words and "a / b 50.0%" in words
    assert no_venues(words) == words  # no venue named on a PM page


def test_b13j_a_fund_that_cant_be_marked_is_said_never_guessed(store, monkeypatch):
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    words = _page(store, monkeypatch, equity=RuntimeError("a holds 0.3 with no mark, so the fund can't be marked"))
    assert "Can't work out the shadow right now: a holds 0.3 with no mark" in words and "Total saved" not in words


def test_b13p_the_night_fails_on_a_residual_over_the_line_and_passes_otherwise(store, monkeypatch):
    import sleeve_fund.supervisor as sup
    from sleeve_fund import nightly_recon
    from sleeve_fund.portfolio import netting as nt
    from sleeve_fund.start_check import FAIL, PASS

    monkeypatch.setattr(sup, "fund_equity", lambda s: D("100000"))
    monkeypatch.setattr(sup, "_in_book", lambda s, row: row.name in ("a", "b"))
    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    assert [st for st, _ in nightly_recon.netting_shadow(store).lines] == [PASS]
    real = nt.reconcile
    monkeypatch.setattr(nt, "reconcile", lambda s, b, a, e: real(s, b, a + D("60"), e))
    [(st, what)] = nightly_recon.netting_shadow(store).lines
    assert st == FAIL and "trading unaffected" in what


def test_b13w_the_portfolio_tabs_reach_the_shadow_page(store, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    c = TestClient(dash.create_app(store))
    for path, here in (("/netting-shadow", True), ("/trades", False)):
        tabs = c.get(path, auth=("pm", "pw")).text.split('class="tabs desk-tabs"')[1].split("</nav>")[0]
        assert '<a href="/netting-shadow" ' + ("aria-current=page" if here else "") in tabs
        assert ">Netting (shadow)</a>" in tabs and tabs.count("aria-current=page") == 1


def test_b13x_the_netting_path_places_changes_and_cancels_no_order_and_writes_nothing(store, monkeypatch):
    """HoE Done-when (B-13 SHADOW): the shadow, its page and its nightly section place, change and cancel no order
    (no call reaches an order sender or the order client) and write nothing to the journal (no statement but a read
    reaches the database)."""
    from sqlalchemy import event

    import sleeve_fund.supervisor as sup
    from sleeve_fund import nightly_recon
    from sleeve_fund.paper import exec_client
    from sleeve_fund.strategies.base import LongFlatStrategy

    _trade(store, "a", "a1", "BUY", "0.3", "60030", close="60000", bar=T1)
    _trade(store, "b", "b1", "SELL", "0.1", "59970", close="60000", bar=T1)
    calls = []
    for cls in (LongFlatStrategy,):  # every Model's order sender (the framework's own are compiled: shadowed here)
        for meth in ("submit_order", "submit_order_list", "modify_order", "cancel_order", "cancel_orders",
                     "cancel_all_orders", "close_position", "close_all_positions"):
            if hasattr(cls, meth):
                monkeypatch.setattr(cls, meth, lambda *a, _m=meth, **k: calls.append(_m), raising=False)
    monkeypatch.setattr(exec_client.Limiter, "take", lambda *a, **k: calls.append("order client"))
    for meth in ("record_order", "update_order", "record_fill", "merge_order_signal", "event"):
        monkeypatch.setattr(type(store), meth, lambda *a, _m=meth, **k: calls.append(_m))
    writes = []

    def seen(conn, cursor, statement, *a):
        if not statement.lstrip().upper().startswith(("SELECT", "WITH", "PRAGMA", "SHOW", "BEGIN", "COMMIT",
                                                       "ROLLBACK", "SAVEPOINT", "RELEASE")):
            writes.append(statement.split()[0:3])

    event.listen(store.engine, "before_cursor_execute", seen)
    try:
        _run(store)
        monkeypatch.setattr(sup, "fund_equity", lambda s: D("100000"))
        monkeypatch.setattr(sup, "_in_book", lambda s, row: row.name in ("a", "b"))
        nightly_recon.netting_shadow(store)
        words = _page(store, monkeypatch)
    finally:
        event.remove(store.engine, "before_cursor_execute", seen)
    assert "Total saved 12.60" in words  # the path ran in full
    assert calls == [] and writes == [], (calls, writes)
