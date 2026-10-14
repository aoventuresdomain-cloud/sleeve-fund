"""CUT-A (Advisor; HoE Drop A) with SIZE-FUND Q4/Q5: inside the fund no decision reads a Strategy's own equity.

- Q4: the Strategy's own drawdown halt and daily-loss pause are gone in the fund (the fund's brakes, T5, act). Its
  liquidation still halts it. A backtest's single Strategy is the whole fund and keeps its own.
- Q5: an entry sizes off the fund's last portfolio mark; a stale or unreadable mark is refused, never sized off.
- End to end (test_cut_a_e2e): a Strategy with a starting balance of 0 sizes off the fund through the paper engine.

Red first on 3a272149 (the per-Strategy halt in force; paper sized off its own equity)."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from sleeve_fund import supervisor as _sup
from sleeve_fund.gate_ledger import DbLedger
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.portfolio.gate import PortfolioState
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import LongFlatStrategy

# Red first until PE2's fund-held book is on the head (supervisor.sizing_equity, the simulated account's virtual
# funding and the replay's netting of it); strict, so the merged head must pass them and drop nothing.
PE2_BUILT = pytest.mark.xfail(not hasattr(_sup, "sizing_equity"), strict=True,
                              reason="needs PE2's fund-held book (sizing_equity, virtual funding)")


T0 = datetime(2025, 10, 3, 9, 0, tzinfo=timezone.utc)
HELD = {"cash": 0.0, "qty": 1.0}


def _store(tmp_path, gate: bool) -> Store:
    url = os.environ.get("TEST_DATABASE_URL")
    if url:
        from sleeve_fund.store import make_engine, metadata

        engine = make_engine(url)
        metadata.drop_all(engine)
        return Store(engine=engine, portfolio_gate=gate)
    return Store(f"sqlite:///{tmp_path}/c.db", portfolio_gate=gate)


def _runtime(store, t, balance=10_000):
    if "s1" not in {s.name for s in store.sleeves()}:
        store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                            starting_balance=balance, risk_profile="balanced")
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    return rt


# ---- Q4: no per-Strategy drawdown halt or daily pause inside the fund ------------------------------------------------
@pytest.mark.parametrize("gate", [True, False])
def test_a_strategy_down_20_percent_keeps_trading_inside_the_fund_and_halts_outside_it(tmp_path, gate):
    t = [T0]
    store = _store(tmp_path, gate)
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(days=1, hours=1)  # a new day: the daily pause is not what acts
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(minutes=1)
    sent = rt.tick(equity=7_900, price=7_900, **HELD)  # its contribution down 21%
    if gate:
        assert sent is None and store.sleeve("s1").status == "running" and rt.can_open()
    else:  # a backtest's or a tool's journal: the Strategy is the whole fund and keeps its own halt
        assert sent == "flatten" and store.sleeve("s1").status == "halted"


def test_a_days_loss_past_the_profiles_limit_never_pauses_a_strategy_inside_the_fund(tmp_path):
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(hours=10)
    assert rt.tick(equity=9_450, price=9_450, **HELD) is None  # down 5.5% on the day
    assert store.sleeve("s1").status == "running"


def test_a_strategy_marked_at_zero_inside_the_fund_is_never_halted_as_invalid_equity(tmp_path):
    """Its own equity is its P&L from a starting balance of 0: zero is a fact, not bad data."""
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t, balance=0)
    assert rt.tick(equity=0.0, price=60_000, cash=0.0, qty=0.0) is None
    assert store.sleeve("s1").status == "running"


def test_its_liquidation_still_halts_it_inside_the_fund(tmp_path):
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(minutes=1)
    assert rt.tick(equity=0.0, price=5_000, ruined="liquidated at 5,000", **HELD) == "flatten"
    assert store.sleeve("s1").status == "halted"


# ---- Q5: sized off the fund's last mark; stale is refused ------------------------------------------------------------
def _reader(tmp_path, marked_ago: timedelta | None, equity=D("19887.78")):
    store = _store(tmp_path, True)
    now = T0
    if marked_ago is not None:
        DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=now - marked_ago, equity=equity, hwm=equity,
                                                               day=now.date(), day_start_equity=equity))
    s = SimpleNamespace(runtime=SimpleNamespace(store=store, now=lambda: now), log=SimpleNamespace(warning=print))
    return LongFlatStrategy._paper_fund_equity(s)


@pytest.mark.parametrize("ago, base", [(timedelta(seconds=5), 19_887.78), (timedelta(seconds=59), 19_887.78),
                                       (timedelta(seconds=61), None), (None, None)])
def test_the_fund_figure_sizes_only_while_its_mark_is_fresh(tmp_path, ago, base):
    got = _reader(tmp_path, ago)
    assert got == (pytest.approx(base) if base is not None else None)


def test_a_fund_held_strategy_never_falls_back_to_its_own_equity():
    """Paper inside the fund with a stale book: refused (None), not its own equity."""
    s = SimpleNamespace(fund_equity=None, _fund_held=lambda: True, _paper_fund_equity=lambda: None,
                        _entry_px=None, _entry_qty=0.0)
    assert LongFlatStrategy._sizing_base(s, 10_000.0) is None
    s._paper_fund_equity = lambda: 19_887.78
    assert LongFlatStrategy._sizing_base(s, 0.0) == 19_887.78


@PE2_BUILT
def test_a_gain_not_yet_taken_anywhere_in_the_fund_never_sizes_an_entry_up(tmp_path):
    """TEC-QD-CUTA (HoE): base = deposits net of withdrawals + realised + min(0, unrealised). A Strategy holding 10 from
    1,000, marked at 1,200, adds 2,000 of unrealised gain to the fund's mark; the next entry anywhere sizes off the
    fund without it (20,000 less the entry's fee), not 22,000."""
    from sleeve_fund.supervisor import fund_equity

    store = _store(tmp_path, True)
    store.create_sleeve(name="a", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=20_000, risk_profile="balanced")
    store.record_fill("a", side="BUY", qty=10, price=1_000, fee=5.0, order_id="o1", trade_id="t1", ts=T0)
    store.record_equity("a", equity=21_995, cash=9_995, qty=10, price=1_200, benchmark=20_000, ts=T0)
    marked = fund_equity(store)
    DbLedger(store, lambda: None).set_state(PortfolioState(mark_ts=T0, equity=marked, hwm=marked, day=T0.date(),
                                                           day_start_equity=marked))
    s = SimpleNamespace(runtime=SimpleNamespace(store=store, now=lambda: T0), log=SimpleNamespace(warning=print))
    assert LongFlatStrategy._paper_fund_equity(s) == pytest.approx(19_995.0)


def test_inside_the_fund_the_strategys_own_open_risk_check_never_refuses_an_entry():
    """CUT-A (d): the per-Strategy interim open-risk refusal read the Strategy's own book; inside the fund the gate's
    whole-fund open-risk limit (T3) judges the entry, under its own name."""
    s = SimpleNamespace(_stop_frac=0.02, pair=None, _backtest=False, _fund_held=lambda: True)
    assert LongFlatStrategy._open_risk_refusal(s, 1, 10.0, 60_000.0, 0.0, 0) is None  # equity 0: nothing of its own


def test_the_fund_figure_is_pe2s_reader_once_it_is_there(tmp_path, monkeypatch):
    """PE2's supervisor.sizing_equity (the book less every position's gain not yet taken, None when stale) replaces the
    raw mark once merged; None from it is refused."""
    import sleeve_fund.supervisor as sup

    s = SimpleNamespace(runtime=SimpleNamespace(store=None, now=lambda: T0), log=SimpleNamespace(warning=print))
    monkeypatch.setattr(sup, "sizing_equity", lambda store, now=None: D("18000"), raising=False)
    assert LongFlatStrategy._paper_fund_equity(s) == 18_000.0
    monkeypatch.setattr(sup, "sizing_equity", lambda store, now=None: None, raising=False)
    assert LongFlatStrategy._paper_fund_equity(s) is None
