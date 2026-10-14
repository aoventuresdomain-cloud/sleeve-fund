"""START CHECKLIST (2): the read-only server check. Dashboard against the journal, no live key, only paper venues,
and a proof that it writes nothing and calls nothing."""

import hashlib
import os
import socket
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy import insert, text

from sleeve_fund import start_check
from sleeve_fund.store import Store, account_keys_t

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)


def _journal(tmp_path, *, mark_qty=0.1, mark_cash=4_000.0 - 3.0, params=None):
    """One Strategy that bought 0.1 at 60,000 (fee 3), marked after the fill at mark_qty / mark_cash."""
    url = f"sqlite:///{tmp_path}/j.db"
    store = Store(url)
    store.create_sleeve(name="s1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, venue="BINANCE", **({"params": params} if params else {}))
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
    out = capfd.readouterr().out
    assert out.startswith("Start check: FAIL") and "FileNotFoundError" in out
    assert str(tmp_path) not in out  # the error's text (a path, a host) never reaches the public log


def test_the_page_names_the_journal_kind_never_its_address(tmp_path, capfd):
    """The workflow's log is public: no host, user, database name or path."""
    url, _ = _journal(tmp_path)
    assert start_check.main(["--db", url]) == 0
    out = capfd.readouterr().out
    assert "SQLite journal, read-only" in out and str(tmp_path) not in out and "j.db" not in out
    assert start_check.journal_kind("postgresql://sleeve:pw@db:5432/sleeve_fund") == "Postgres"


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


@pytest.mark.parametrize("market", ["perp", "spot"])
def test_a_wiped_out_isolated_margin_mark_floored_at_zero_is_a_note_on_a_perp_only(tmp_path, market):
    """The floor is the perp's isolated margin (Strategy._mark); a spot mark is never floored (FE), so there it fails."""
    url, store = _journal(tmp_path, params={"market": market})
    store.record_equity("s1", equity=0.0, cash=3_997.0, qty=0.1, price=-50_000.0, benchmark=10_000.0,
                        ts=T0 + timedelta(minutes=2))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    if market == "perp":
        assert ok and "  NOTE  s1: position 0.1" in page and "equity 0.00, floored from -1,003.00" in page
    else:
        assert not ok and "but equity 0.00 vs journal -1,003.00" in page


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


@pytest.mark.parametrize("cash", ["funding", "true_up", "insurance"])
def test_funding_or_insurance_booked_after_the_latest_mark_is_a_note_not_a_fail(tmp_path, cash):
    """CR F276-1: a settlement, a true-up arriving or an insurance-fund cover moves the journal's cash after the latest
    mark; the mark catches up at the next one, as with a late fill. Booked before the mark, it is compared."""
    url, store = _journal(tmp_path)
    after = T0 + timedelta(minutes=2)
    if cash == "funding":
        store.record_funding("s1", qty=0.1, price=60_000.0, rate=0.0001, amount=-0.6, ts=after)
    elif cash == "true_up":
        store.record_funding("s1", qty=0.1, price=60_000.0, rate=0.0001, amount=-0.6, ts=T0, kind="true_up",
                             arrived_at=after)
    else:
        store.record_insurance("s1", price=60_000.0, amount=5.0, order_id="o1", trade_id="t1", ts=after)
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert f"  NOTE  s1: {'an insurance-fund cover' if cash == 'insurance' else 'a funding payment'} landed after " \
           "its latest mark" in page
    assert "  FAIL  nothing was compared" in page and not ok  # s1 was the only one: nothing was checked


def test_funding_booked_before_the_latest_mark_is_compared(tmp_path):
    url, store = _journal(tmp_path, mark_cash=4_000.0 - 3.0 - 0.6)
    store.record_funding("s1", qty=0.1, price=60_000.0, rate=0.0001, amount=-0.6, ts=T0 + timedelta(seconds=30))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert ok and "landed after" not in page


def test_a_key_the_supervisor_reports_fails_on_its_own(tmp_path):
    """F276-2: a key the supervisor reports present fails the keys section by itself, with only the paper account and no
    credential in the environment, so the venues section passes; reported absent, it passes."""
    url, store = _journal(tmp_path)
    store.accounts()  # the paper account's row, which the key row refers to
    with store.engine.begin() as c:
        c.execute(insert(account_keys_t).values(account="paper", present=0, checked_at=T0))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert ok and "Live API keys: PASS" in page, page
    with store.engine.begin() as c:
        c.execute(account_keys_t.update().values(present=1))
    store.engine.dispose()
    ok, page = start_check.run(url, environ={})
    assert not ok and "Live API keys: FAIL" in page and "Venues and accounts: PASS" in page
    assert "  FAIL  the supervisor reports a live key for account paper" in page
    assert "Dashboard against the journal: PASS" in page


@pytest.mark.parametrize("flat", [Decimal("0E-18"), Decimal("-0E-18"), -0.0, 0.0])
def test_a_flat_position_shows_as_0(flat):
    """F276-4: Postgres NUMERIC hands a flat position back as 0E-18, which formats as 0e-18, and pads the rest."""
    assert f"{start_check._qty(flat):.12g}" == "0"
    assert f"{start_check._qty(Decimal('0.100000000000000000')):.12g}" == "0.1"


def test_a_flat_position_on_the_page_is_0(tmp_path, monkeypatch):
    url, store = _journal(tmp_path, mark_qty=0.0, mark_cash=9_994.0)
    store.record_fill("s1", side="SELL", qty=0.1, price=60_000.0, fee=3.0, order_id="o2", trade_id="t2",
                      ts=T0 + timedelta(seconds=30))
    store.engine.dispose()
    book, last = Store.journal_book, Store.last_equity
    monkeypatch.setattr(Store, "journal_book", lambda *a: {**book(*a), "qty": Decimal("0E-18")})
    monkeypatch.setattr(Store, "last_equity", lambda *a: {**last(*a), "qty": Decimal("-0E-18")})
    ok, page = start_check.run(url, environ={})
    assert ok and "s1: position 0 vs journal 0; cash 9,994.00" in page, page
    assert "e-18" not in page
