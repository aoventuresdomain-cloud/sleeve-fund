"""START CHECKLIST (2): the read-only server check. Dashboard against the journal, no live key, only paper venues,
and a proof that it writes nothing and calls nothing."""

import hashlib
import os
import socket
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import insert, text

from sleeve_fund import start_check
from sleeve_fund.store import Store, account_keys_t

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _journal(tmp_path, *, mark_qty=0.1, mark_cash=4_000.0 - 3.0):
    """One Strategy that bought 0.1 at 60,000 (fee 3), marked after the fill at mark_qty / mark_cash."""
    url = f"sqlite:///{tmp_path}/j.db"
    store = Store(url)
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE")
    store.record_fill("s1", side="BUY", qty=0.1, price=60_000.0, fee=3.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("s1", equity=mark_cash + mark_qty * 60_000.0, cash=mark_cash, qty=mark_qty, price=60_000.0,
                        benchmark=10_000.0, ts=T0 + timedelta(minutes=1))
    store.engine.dispose()
    return url, store


def test_a_matching_paper_journal_with_no_keys_passes_on_one_page(tmp_path):
    url, _ = _journal(tmp_path)
    ok, page = start_check.run(url, environ={})
    assert ok, page
    assert page.startswith("Start check: PASS")
    assert "s1: position 0.1 vs journal 0.1; cash 3,997.00 vs journal 3,997.00" in page
    assert len(page.splitlines()) < 60  # one page


@pytest.mark.parametrize("qty, cash", [(0.2, 3_997.0), (0.1, 3_990.0)], ids=["position", "cash"])
def test_a_dashboard_that_disagrees_with_the_journal_fails(tmp_path, qty, cash):
    url, _ = _journal(tmp_path, mark_qty=qty, mark_cash=cash)
    ok, page = start_check.run(url, environ={})
    assert not ok and "Dashboard against the journal: FAIL" in page and "  FAIL  s1:" in page


def test_a_live_key_fails_and_is_named_never_shown(tmp_path):
    url, _ = _journal(tmp_path)
    ok, page = start_check.run(url, environ={"BINANCE_API_KEY__DESK": "sekret-value", "PATH": "/bin"})
    assert not ok and "BINANCE_API_KEY__DESK" in page and "sekret-value" not in page
    ok, page = start_check.run(url, environ={"BYBIT_DEMO_API_KEY": "d", "BYBIT_DEMO_API_SECRET": "d"})
    assert ok and "demo mirror keys set" in page and "d," not in page  # demo keys are allowed


def test_a_key_the_supervisor_reports_for_an_account_fails(tmp_path):
    url, store = _journal(tmp_path)
    store.create_account("desk", "live", venue="BINANCE")
    with store.engine.begin() as c:
        c.execute(insert(account_keys_t).values(account="desk", present=1, checked_at=T0))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "reports a live key for account desk" in page


def test_a_live_account_or_a_strategy_on_one_fails(tmp_path):
    url, store = _journal(tmp_path)
    store.create_account("desk", "live", venue="BINANCE")
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "account desk is live on binance: only paper may exist" in page
    store.assign_account("s1", "desk")
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert "Strategy s1 is assigned to account desk, not paper" in page


def test_it_is_strictly_read_only_no_writes_and_no_network(tmp_path, monkeypatch):
    """The journal's bytes are unchanged after a run, its engine refuses a write, and no socket is ever opened (no
    order can be placed without one)."""
    url, _ = _journal(tmp_path)
    path = url[len("sqlite:///"):]
    def digest():
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    before = digest()

    def no_network(*a, **k):
        raise AssertionError("the start check opened a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    ok, _ = start_check.run(url, environ={})
    assert ok
    assert digest() == before
    engine = start_check.read_only_engine(url)
    with pytest.raises(Exception, match="readonly|read-only|read only"), engine.begin() as c:
        c.execute(text("DELETE FROM fills"))
    engine.dispose()


def test_a_check_that_cannot_run_fails(tmp_path, capfd):
    assert start_check.main(["--db", f"sqlite:///{tmp_path}/missing.db"]) == 1
    assert capfd.readouterr().out.startswith("Start check: FAIL")


def test_an_archived_strategy_still_in_the_book_is_checked(tmp_path):
    """FE MUST 1: archived but still holding, it stays in the dashboard's book (previous_book keeps it until flat), so
    its disagreeing mark fails rather than being skipped."""
    url, store = _journal(tmp_path, mark_qty=0.2)
    store.set_desired_state("s1", "stopped")
    store.archive("s1")
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "  FAIL  s1: position 0.2 vs journal 0.1" in page


def test_a_mark_whose_equity_disagrees_fails_though_position_and_cash_match(tmp_path):
    """FE MUST 2: Book, P&L and Return show the mark's equity column, so it must equal the journal's cash plus its
    position at the mark's price."""
    url, store = _journal(tmp_path)
    store.record_equity("s1", equity=4_000.0 - 3.0 + 6_000.0 + 1.0, cash=3_997.0, qty=0.1, price=60_000.0,
                        benchmark=10_000.0, ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "but equity 9,998.00 vs journal 9,997.00" in page


def test_a_wiped_out_isolated_margin_mark_floored_at_zero_is_a_note(tmp_path):
    url, store = _journal(tmp_path)
    store.record_equity("s1", equity=0.0, cash=3_997.0, qty=0.1, price=-50_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert ok and "  NOTE  s1: position 0.1" in page and "equity 0.00, floored from -1,003.00" in page


def test_nothing_compared_is_a_fail_not_a_vacuous_pass(tmp_path):
    """FE FOLLOW-UP: every Strategy with fills had a fill after its latest mark, so nothing was checked."""
    url, store = _journal(tmp_path)
    store.record_fill("s1", side="SELL", qty=0.1, price=61_000.0, fee=3.0, order_id="o2", trade_id="t2",
                      ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "  NOTE  s1: a fill landed after its latest mark" in page
    assert "  FAIL  nothing was compared: s1 had a fill after the latest mark" in page


@pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"), reason="Postgres only: TEST_DATABASE_URL")
def test_postgres_read_only_engine_refuses_a_write():
    """HoQA F-SC-1: the Postgres path (default_transaction_read_only) pinned in CI."""
    engine = start_check.read_only_engine(os.environ["TEST_DATABASE_URL"])
    with pytest.raises(Exception, match="read-only|ReadOnlySqlTransaction"), engine.begin() as c:
        c.execute(text("CREATE TABLE start_check_probe (x int)"))
    engine.dispose()


def test_a_zero_mark_when_the_journal_is_not_underwater_fails(tmp_path):
    """HoE: the floor is a NOTE only when the journal side really is underwater; at journal equity >= 0 a mark of 0
    is wrong."""
    url, store = _journal(tmp_path)
    store.record_equity("s1", equity=0.0, cash=3_997.0, qty=0.1, price=60_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "but equity 0.00 vs journal 9,997.00" in page
