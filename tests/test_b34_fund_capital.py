"""B-34 (QD capital audit, 15 Oct 2026) with the Advisor's B34-CAPITAL-PREDICATE ruling (20 Oct):

- One predicate: a Strategy is inside the fund once the PM's re-base deposit has landed (supervisor.bridge_start).
  Before it the Strategy still carries its own capital, so its own halt and its own sizing stay on, portfolio gate on
  or not (the gap between gate-on and the bridge had no halt at all).
- Rows 1-3: a target-weight Model inside the fund weighs against the fund, never against its own P&L from 0. A new
  entry's weight is of the fund's sizing base; a rebalance's target and its held weight both read the fund's marked
  equity.
- Row 20: an account's open-risk book inside the fund is the fund's book, not a sum of Strategies' P&L marks.

Runs on Postgres too with TEST_DATABASE_URL."""

from __future__ import annotations

from datetime import timedelta

import pytest
from test_cut_a_fund_equity import HELD, T0, _store
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.paper.runtime import SleeveRuntime


def _runtime(store, t, profile):
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=10_000, risk_profile=profile)
    rt = SleeveRuntime(store, "s1", now=lambda: t[0])
    rt.on_start(0.0005)
    return rt


def _down(rt, t, to):
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(days=1, hours=1)
    rt.tick(equity=10_000, price=10_000, **HELD)
    t[0] += timedelta(minutes=1)
    return rt.tick(equity=to, price=to, **HELD)


def test_gate_on_before_the_bridge_a_strategy_down_16_percent_still_halts(tmp_path):
    """Advisor's cell: the gate on, no re-base deposit yet, a Strategy at -16% under a 10% halt (conservative)."""
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t, "conservative")
    assert not rt.fund_held
    assert _down(rt, t, 8_400) == "flatten" and store.sleeve("s1").status == "halted"


def test_once_the_bridge_lands_its_own_halt_is_off_and_the_funds_brakes_act(tmp_path):
    t = [T0]
    store = _store(tmp_path, True)
    rt = _runtime(store, t, "conservative")
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-34: the fund's capital", rebase=True,
                                      now=T0), T0)
    assert rt.fund_held
    assert _down(rt, t, 8_400) is None and store.sleeve("s1").status == "running"


# ---- rows 1 and 2: a target-weight Model on paper inside the fund --------------------------------------------------

def _weights(tmp_path, monkeypatch, windows):
    import pandas as pd
    import test_exposure_gate_xfails as h
    from test_f_trim_min import _store as fund_store

    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.supervisor import Supervisor

    monkeypatch.setitem(REGISTRY, "egx", (h.EgxW, h.EgxWConfig))
    store = fund_store(tmp_path)
    params = {"market": "spot", "allow_short": False, "rebalance_band": 0.25, "stop_loss": None}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-34: the fund's capital", rebase=True,
                                      now=before), before)
    sup = Supervisor(store)
    sup.mark_portfolio(h.STD_T0.to_pydatetime())
    plan = h.Plan(t0=h.STD_T0, minutes=10, balance=200_000.0, params=params, tag="b34",
                  windows=[(h.M(h.STD_T0, a), h.M(h.STD_T0, b), w) for a, b, w in windows],
                  hooks=[(h.M(h.STD_T0, m), lambda rt, kw: sup.mark_portfolio(rt.now())) for m in range(1, 10)])
    h.run(tmp_path, store, plan, monkeypatch)
    return [o for o in h._orders(store) if (o["filled_qty"] or 0) > 0], h._events(store)


def test_row_1_a_second_entry_after_a_loss_is_weighed_against_the_fund_not_its_own_pnl(tmp_path, monkeypatch):
    """0.3 of the fund twice, flat between: the round trip's fees leave the Strategy's own P&L below 0, which on
    274ecd88 made its target weight limit negative and skipped the second buy."""
    orders, events = _weights(tmp_path, monkeypatch, [(2, 4, 0.3), (6, 9, 0.3)])
    buys = [o for o in orders if o["side"] == "BUY"]
    assert len(buys) == 2, ([(o["ts"], o["side"], o["qty"]) for o in orders],
                            [e["message"] for e in events if "skipped" in (e["message"] or "")])
    first, second = buys[0]["filled_qty"], buys[1]["filled_qty"]
    assert abs(second - first) <= 0.01 * first, (first, second)  # both a weight of the same fund, give or take fees
    assert buys[0]["signal"].get("sized_by") == "target weight", buys[0]["signal"]


def test_row_2_a_lower_target_weight_sells_down_to_it_of_the_fund_not_everything(tmp_path, monkeypatch):
    """0.3 of the fund, then 0.15: about half is sold. On 274ecd88 the target was 0.15 of its own P&L (below 0), so
    the rebalance sold the whole position."""
    orders, _ = _weights(tmp_path, monkeypatch, [(2, 5, 0.3), (5, 9, 0.15)])
    buy = next(o for o in orders if o["side"] == "BUY")
    sells = [o for o in orders if o["side"] == "SELL" and o["intent"] == "rebalance"]
    assert len(sells) == 1, [(o["ts"], o["side"], o["intent"], o["filled_qty"]) for o in orders]
    share = sells[0]["filled_qty"] / buy["filled_qty"]
    assert 0.45 <= share <= 0.55, (share, buy["filled_qty"], sells[0]["filled_qty"])


def test_row_20_an_accounts_open_risk_book_inside_the_fund_is_the_funds_book(tmp_path):
    """The Open risk tile's book: on 274ecd88 the sum of Strategies' marks, here its P&L of -500, not the fund."""
    from sleeve_fund import open_risk
    from sleeve_fund.supervisor import fund_book

    t = [T0]
    store = _store(tmp_path, True)
    store.create_sleeve(name="s1", strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-DAY-LAST-EXTERNAL",
                        starting_balance=0, risk_profile="balanced")
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-34: the fund's capital", rebase=True,
                                      now=T0), T0)
    store.record_equity("s1", equity=-500.0, cash=-500.0, qty=0.0, price=60_000.0, benchmark=0, ts=t[0])
    book = open_risk.account_equity(store, store.account_of("s1"))
    assert book == float(fund_book(store).total) and book > 90_000, book
