"""P2-1b W1: the portfolio gate on a paper journal (HoE rulings, 8 Oct). The gate is in force by the journal's mode
(Store.portfolio_gate), never by its state row existing: a paper journal whose row is gone blocks every entry as
stale, and only a backtest's journal opts out."""

from datetime import datetime, timedelta, timezone
from decimal import Decimal as D

import pytest
from sqlalchemy import delete, select

from sleeve_fund.paper.runtime import entry_blocked
from sleeve_fund.portfolio.gate import check_order
from sleeve_fund.risk import PORTFOLIO
from sleeve_fund.store import Store, events_t, portfolio_state_t
from test_gate_ledger import CellLedger, _drop_schemas, _store, _strategies  # noqa: F401
from test_portfolio_gate import _buy
from test_p21b_wiring_qa import EQ, _flattens, _hold, portfolio_pass

T0 = datetime(2026, 10, 8, 9, 0, 3, tzinfo=timezone.utc)


def test_a_paper_journal_whose_state_row_is_gone_refuses_every_entry():
    store = _store()
    _strategies(store)
    assert store.portfolio_gate
    portfolio_pass(store, T0, D(20_000))
    assert not entry_blocked(store, "a", T0 + timedelta(seconds=1))[0]
    with store.engine.begin() as c:
        c.execute(delete(portfolio_state_t))
    blocked, why = entry_blocked(store, "a", T0 + timedelta(seconds=2))
    assert blocked and why.code == "portfolio_stale" and "never" in why
    c = check_order(CellLedger(D(20_000), store=store), "a", _buy("0.1"), PORTFOLIO, T0 + timedelta(seconds=2))
    assert (c.decision.outcome, c.decision.limit_hit, c.reservation) == ("rejected", "portfolio_state_stale", None)


def test_only_a_backtests_journal_opts_out_and_says_so_by_its_mode():
    backtest = Store.in_memory()
    _strategies(backtest)
    assert not backtest.portfolio_gate and not entry_blocked(backtest, "a", T0)[0]  # never marked, never gated
    paper = Store(engine=backtest.engine, portfolio_gate=True)  # the same tables, opened as a paper journal
    blocked, why = entry_blocked(paper, "a", T0)
    assert blocked and why.code == "portfolio_stale"


def test_advisor_a_missing_state_row_blocks_entries_and_raises_one_alert():
    store = Store.in_memory(portfolio_gate=True)
    _strategies(store)
    for s in range(3):
        blocked, why = entry_blocked(store, "a", T0 + timedelta(seconds=s))
        assert blocked and why.code == "portfolio_stale"
    with store.engine.connect() as c:
        kinds = [k for (k,) in c.execute(select(events_t.c.kind).where(events_t.c.sleeve.is_(None)))]
    assert kinds == ["portfolio_state_stale"]  # once for the spell, not once per read


def test_advisor_paper_startup_refuses_a_journal_without_the_portfolio_gate(monkeypatch):
    from sleeve_fund import supervisor
    from sleeve_fund.paper.safety import PaperSafetyError, assert_portfolio_gate

    assert_portfolio_gate(Store.in_memory(portfolio_gate=True))
    with pytest.raises(PaperSafetyError, match="portfolio gate"):
        assert_portfolio_gate(Store.in_memory())
    opened = []
    monkeypatch.setattr(supervisor, "Store", lambda **k: opened.append(k) or Store.in_memory())  # the gate dropped
    monkeypatch.setattr(supervisor.Supervisor, "run", lambda self: pytest.fail("supervised without the gate"))
    with pytest.raises(PaperSafetyError):
        supervisor.main(["run"])
    assert opened == [{"portfolio_gate": True}]  # the supervisor asks for it; the refusal catches a journal without


def test_the_dashboard_opens_the_servers_journal_with_the_portfolio_gate(monkeypatch):
    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_INSECURE_DEV", "1")
    opened = []
    monkeypatch.setattr(dash, "Store", lambda **k: opened.append(k) or Store.in_memory(**k))
    dash.create_app()
    assert opened == [{"portfolio_gate": True}]


def test_choke_reads_the_portfolio_without_the_gate_lock_and_fails_closed_when_it_cant(monkeypatch):
    """CR F229-2: CHOKE runs on every tick, fill check and dashboard read, so it reads the state row without the gate
    lock (taken only to tell a stale spell once), and a state it can't read blocks entries as stale, never raises."""
    from sleeve_fund.gate_ledger import DbLedger

    store = _store()
    _strategies(store)
    portfolio_pass(store, T0, EQ)
    monkeypatch.setattr(DbLedger, "lock", lambda self: pytest.fail("CHOKE took the gate lock on a marked book"))
    assert not entry_blocked(store, "a", T0 + timedelta(seconds=1))[0]
    monkeypatch.setattr(DbLedger, "state", lambda self: (_ for _ in ()).throw(OSError("db gone")))
    blocked, why = entry_blocked(store, "a", T0 + timedelta(seconds=2))
    assert blocked and why.code == "portfolio_stale" and "couldn't be read" in why and "db gone" in why


def test_a_halt_tells_a_holder_a_failed_command_skipped_on_a_later_poll_and_the_rest_at_once(monkeypatch):
    """CR F229-3: the halt is in the state row from the poll that halted, so a holder whose flatten couldn't be sent
    then is told on a later poll; one failure never stops the others, and a waiting flatten is never sent twice."""
    store = _store()
    _strategies(store)
    _hold(store, "a")
    _hold(store, "b")
    real, failed = store.command, []

    def command(name, *a, **k):
        if name == "a" and not failed:
            failed.append(name)
            raise OSError("db blip")
        return real(name, *a, **k)

    monkeypatch.setattr(store, "command", command)
    portfolio_pass(store, T0, EQ)
    portfolio_pass(store, T0 + timedelta(seconds=5), EQ * D("0.84"))  # the 15% halt
    assert failed == ["a"] and not _flattens(store, "a") and len(_flattens(store, "b")) == 1
    with store.engine.connect() as c:
        errs = list(c.execute(select(events_t.c.message).where(events_t.c.sleeve == "a",
                                                               events_t.c.kind == "supervisor_error")))
    assert len(errs) == 1 and "portfolio halt" in errs[0][0]
    for s in (10, 15, 80):
        portfolio_pass(store, T0 + timedelta(seconds=s), EQ * D("0.80"))
    assert len(_flattens(store, "a")) == 1 and len(_flattens(store, "b")) == 1


# Entry points that open the sleeve journal but never decide an entry: research and tooling, the portfolio gate's
# opt-out (Advisor 06:10 UK). Anything else that opens a journal must run under the gate (HoE 8 Oct).
RESEARCH_ALLOWLIST = {
    "sleeve_fund/__main__.py": "the research CLI: reads fees and spreads from the journal, never trades",
    "sleeve_fund/history.py": "the candle store: writes only alerts to the journal's inbox",
    "sleeve_fund/mirror.py": "the demo mirror: copies paper fills to a demo account, never decides an entry",
}
GATED = ("assert_portfolio_gate(", "Store(portfolio_gate=True)")


def test_every_entry_point_that_opens_a_journal_runs_under_the_gate_or_is_allowlisted_research():
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    opens = re.compile(r"(?<![A-Za-z_])Store\(")
    entry = re.compile(r"if __name__ == .__main__.|^def create_app\(", re.M)
    found = {}
    for path in sorted((root / "sleeve_fund").rglob("*.py")):
        src = path.read_text()
        if entry.search(src) and opens.search(src):
            found[path.relative_to(root).as_posix()] = any(g in src for g in GATED)
    assert {"sleeve_fund/supervisor.py", "sleeve_fund/paper/node.py", "sleeve_fund/dashboard/app.py"} <= set(found)
    ungated = sorted(p for p, gated in found.items() if not gated and p not in RESEARCH_ALLOWLIST)
    assert not ungated, f"opens the sleeve journal without the portfolio gate and isn't allowlisted research: {ungated}"
    assert not [p for p in RESEARCH_ALLOWLIST if found.get(p)], "an allowlisted entry point now runs the gate: drop it"
