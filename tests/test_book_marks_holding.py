"""F-BM-1: the supervisor's book_marks row is written while the fund holds. Its holdings are the paper gate's own
(paper.portfolio_gate.fund_holdings), so the book's history records the book entries are checked against. Before,
supervisor.fund_holdings raised whenever anything held and the row was skipped (HoQA's condition: a w1b-style cell
with a holding fund, 14 Oct)."""

from __future__ import annotations

from datetime import timedelta
from decimal import Decimal as D

import pytest
from test_p21b_wiring_qa import (  # noqa: F401 - the fixtures
    EQ,
    T0,
    _alerts,
    _drop_schemas,
    _marks,
    portfolio_pass,
    store,
)

from sleeve_fund import supervisor
from sleeve_fund.money import money
from sleeve_fund.paper import portfolio_gate


def _hold(store, name="a", qty=0.1, px=60000.0, stop_frac=0.02):  # noqa: F811
    """A filled entry whose plan carries its stop, so the holding is measured to the stop (no ATR read)."""
    store.record_order(name, order_id=f"{name}-held", side="BUY", qty=qty, intent="entry", reason="qa",
                       signal={"stop_frac": stop_frac})
    store.book_fill(name, side="BUY", qty=qty, price=px, fee=1.0, order_id=f"{name}-held", trade_id="t1", ts=T0)


def test_fbm1_a_holding_fund_writes_one_book_mark_per_minute_with_its_holdings(store):  # noqa: F811
    _hold(store)
    t = T0
    for _ in range(25):  # 5 s passes over 2 minutes
        portfolio_pass(store, t, EQ)
        t += timedelta(seconds=5)
    rows = _marks(store)
    assert len(rows) == len({r.ts.replace(second=0, microsecond=0) for r in rows}) == 3
    held = portfolio_gate.fund_holdings(store, T0)
    assert len(held) == 1 and held[0].notional == D(6000)  # 0.1 at 60,000, marked at its entry
    for r in rows:  # the row's figures are the gate's book: 6,000 long on 20,000
        assert money(r.gross) == D(6000) and money(r.net_max) == D(6000) and r.net_underlying == "BTC"
        # To the 2% stop, and the taker fee on the stop's notional at the rate its fills pay (FEE-LIVE 5): this
        # Strategy is spot at the default venue, 0.80%, so 120 + 0.1 x 58,800 x 0.008 = 167.04.
        assert money(r.open_risk) == held[0].risk == D(120) + D("0.1") * 58_800 * D("0.008")
        assert money(r.equity) == EQ and r.status == "ok"
    assert not _alerts(store, "supervisor_error")


def test_fbm1_supervisor_fund_holdings_is_the_gates(store):  # noqa: F811
    _hold(store)
    assert supervisor.fund_holdings(store, T0) == portfolio_gate.fund_holdings(store, T0)


def test_fbm1_holdings_that_cant_be_read_skip_the_row_say_so_once_and_never_crash_the_pass(store, monkeypatch):  # noqa: F811
    _hold(store)
    portfolio_pass(store, T0, EQ)  # one row
    real = portfolio_gate.fund_holdings

    def unknown_atr(*a, **k):
        raise ValueError("a stopless position's daily ATR isn't known")

    monkeypatch.setattr(portfolio_gate, "fund_holdings", unknown_atr)
    monkeypatch.setattr(supervisor, "fund_equity", lambda *a, **k: EQ)
    sup = supervisor.Supervisor(store)  # one supervisor across the passes: the alert is said once per spell
    for m in (1, 2, 3):
        sup.mark_portfolio(T0 + timedelta(minutes=m))  # never raises out of the pass
    assert len(_marks(store)) == 1
    said = _alerts(store, "book_mark_skipped")
    assert len(said) == 1 and "book_marks" in said[0].message and "ATR" in said[0].message
    assert said[0].level == "warning"
    monkeypatch.setattr(portfolio_gate, "fund_holdings", real)
    sup.mark_portfolio(T0 + timedelta(minutes=4))
    assert len(_marks(store)) == 2  # written again once the holdings read


@pytest.mark.parametrize("held", [False, True])
def test_fbm1_a_first_pass_writes_its_book_mark_held_or_not(store, held):  # noqa: F811
    if held:
        _hold(store)
    portfolio_pass(store, T0, EQ)
    assert len(_marks(store)) == 1


def test_fbm1_a_stopless_holder_with_no_atr_yet_records_no_error_event(store, monkeypatch):  # noqa: F811
    """The stack e2e's case: a fresh stack's stopless position has no daily ATR yet. The cache row waits with a
    warning; no error-level event is recorded (the e2e counts those as failures)."""
    from sleeve_fund import open_risk

    _hold(store, stop_frac=None)
    monkeypatch.setattr(open_risk, "history_atr_pct", lambda *a, **k: None)
    portfolio_pass(store, T0, EQ)
    assert not _marks(store)
    assert not [e for e in _alerts(store, "book_mark_skipped") if e.level == "error"]
    assert len(_alerts(store, "book_mark_skipped")) == 1 and not _alerts(store, "supervisor_error")
