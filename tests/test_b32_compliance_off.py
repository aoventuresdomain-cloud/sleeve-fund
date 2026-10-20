"""B-32 (PM run 15 Oct 2026, 21:02 UK): a compliance test the PM switches off is OFF. It has no cap, binds nothing and
is left out of the decision's figures; it never stops the gate from running. On 274ecd88 switching off any T4 row
(gross, net per asset, margin) made limits.decide scale a None ratio, so the gate refused every entry with "the
portfolio check couldn't run (ValueError ...)".

T3 and T5 are locked (no save may switch them off), but the gate and the brakes treat a None the same way, so B-31's
locked half is only the settings switch (HoE, 20 Oct). Each T1, T2 and T4 row is also checked at its minimum and its
maximum (compliance_rules bounds).

Expected sizes are worked out by hand from the inputs below, never read back from the code under test. Runs on
Postgres too with TEST_DATABASE_URL, on an empty schema (the end-to-end cells)."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal as D

import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.portfolio.compliance import SEEDED
from sleeve_fund.compliance_rules import BY_KEY
from sleeve_fund.portfolio.limits import Book, Holding, Intent, book_breach, decide, decide_pair

T4 = ("gross", "net_instrument", "margin")
KEY = {"gross": "fund_gross", "net_instrument": "asset_net", "margin": "fund_margin"}

# Every row loose (T4 at its maximum, the PM's workaround), T1 and T2 off: only the row a cell tightens can bind.
LOOSE = replace(SEEDED, gross=3.0, net_instrument=1.0, margin=0.8, open_risk=0.10, asset_open_risk=None,
                group_open_risk=None)
# The tight value for each T4 row, and the size it trims the one-BTC order below to:
#   book 100,000; 1 BTC at 50,000 (0.5x gross, 0.5x net); margin at 2x, 25,000 a unit; risk 500 a unit.
#   gross 0.25x -> 25,000 / 50,000 = 0.5; net 0.25x -> the same 0.5; margin 10% -> 10,000 / 25,000 = 0.4.
TIGHT = {"gross": (0.25, D("0.5")), "net_instrument": (0.25, D("0.5")), "margin": (0.10, D("0.4"))}
BOOK = Book(D(100_000))
STEP = D("0.001")


def _btc(qty="1", side=1) -> Intent:
    return Intent("BTC", side, D(qty), D(50_000), D(25_000), D(500), STEP, STEP)


def _eth(qty="10", side=-1) -> Intent:
    # 10 ETH at 2,500 = 25,000 (0.25x); margin at 2x, 1,250 a unit; risk 25 a unit
    return Intent("ETH", side, D(qty), D(2_500), D(1_250), D(25), STEP, STEP)


@pytest.mark.parametrize("row", T4)
def test_a_t4_row_binds_when_on_so_the_off_cells_below_prove_something(row):
    limit, trimmed_to = TIGHT[row]
    d = decide(BOOK, _btc(), replace(LOOSE, **{row: limit}))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", trimmed_to, row), d


@pytest.mark.parametrize("row", T4)
def test_a_t4_row_switched_off_alone_binds_nothing_and_the_gate_still_runs(row):
    d = decide(BOOK, _btc(), replace(LOOSE, **{row: None}))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("approved", D(1), None), d
    assert row not in d.figures and "open_risk" in d.figures, d.figures  # off: no figure, never a 0 cap


@pytest.mark.parametrize("row", T4)
def test_a_t4_row_off_leaves_the_rows_still_on_binding(row):
    """The off row is off; every other row still trims. Its tight neighbour binds at its own hand-worked size."""
    other = next(r for r in T4 if r != row)
    limit, trimmed_to = TIGHT[other]
    d = decide(BOOK, _btc(), replace(LOOSE, **{row: None, other: limit}))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", trimmed_to, other), d


def test_every_t4_row_off_together_approves_the_whole_order_and_t1_still_trims():
    off = dict.fromkeys(T4)
    d = decide(BOOK, _btc(), replace(LOOSE, **off))
    assert (d.outcome, d.approved_qty) == ("approved", D(1)), d
    assert set(d.figures) == {"open_risk"}, d.figures
    # T1 at 0.25% of 100,000 = 250 of loss at the stops: 250 / 500 = 0.5 BTC
    d = decide(BOOK, _btc(), replace(LOOSE, **off, asset_open_risk=0.0025))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", D("0.5"), "asset_open_risk"), d


def test_a_short_with_net_off_is_not_judged_on_net_even_with_a_long_already_held():
    held = Book(D(100_000), (Holding("BTC", D(40_000), D(13_333), D(400)),))
    d = decide(held, _btc(side=-1), replace(LOOSE, net_instrument=None))
    assert (d.outcome, d.approved_qty) == ("approved", D(1)), d


@pytest.mark.parametrize("row", T4)
def test_a_two_leg_entry_with_a_t4_row_off_is_decided_and_takes_both_legs_whole(row):
    a, b = decide_pair(BOOK, _btc(), _eth(), replace(LOOSE, **{row: None}))
    assert (a.outcome, a.approved_qty, b.outcome, b.approved_qty) == ("approved", D(1), "approved", D(10)), (a, b)
    assert not any(k == row or k.startswith("net:") and row == "net_instrument" for k in a.figures), a.figures


def test_a_two_leg_entry_with_every_t4_row_off_still_trims_on_a_row_left_on():
    # T1 on at 0.25% (250): BTC leg risk 1 x 500 = 500 binds, so the share is 250 / 500 = 50%: 0.5 BTC and 5 ETH
    a, b = decide_pair(BOOK, _btc(), _eth(), replace(LOOSE, **dict.fromkeys(T4), asset_open_risk=0.0025))
    assert (a.outcome, a.approved_qty, b.approved_qty) == ("trimmed", D("0.5"), D(5)), (a, b)


def test_t3_off_binds_nothing_and_has_no_figure():
    d = decide(BOOK, _btc(), replace(LOOSE, open_risk=None))
    assert (d.outcome, d.approved_qty) == ("approved", D(1)), d
    assert "open_risk" not in d.figures, d.figures
    a, b = decide_pair(BOOK, _btc(), _eth(), replace(LOOSE, open_risk=None))
    assert (a.approved_qty, b.approved_qty) == (D(1), D(10)) and "open_risk" not in a.figures, (a, b)


@pytest.mark.parametrize("row", ("asset_open_risk", "group_open_risk"))
def test_t1_and_t2_off_bind_nothing_as_before(row):
    held = Book(D(100_000), (Holding("BTC", D(10_000), D(5_000), D(4_000)),))  # 4% already at the stops in BTC
    on = replace(LOOSE, asset_open_risk=0.04, group_open_risk=0.04, groups={"BTC": "large"})
    assert decide(held, _btc(), on).outcome == "rejected"  # both on: no room left
    other = "group_open_risk" if row == "asset_open_risk" else "asset_open_risk"
    d = decide(held, _btc(), replace(on, **{row: None, other: None}))
    assert (d.outcome, d.approved_qty) == ("approved", D(1)) and row not in d.figures, d


# (profile field, rule key, its minimum's trimmed size, a book holding first): book 100,000; 1 BTC at 50,000, margin
# 25,000 and risk 500 a unit. At its maximum no row binds this order.
#   gross 0.1x -> 10,000 / 50,000 = 0.2;  net 0.05x -> 5,000 / 50,000 = 0.1;  margin 10% -> 10,000 / 25,000 = 0.4;
#   T1 0.25% -> 250 / 500 = 0.5;  T2 0.5% with 250 already at the stops in the group -> (500 - 250) / 500 = 0.5
BOUNDS = [("gross", "fund_gross", D("0.2"), ()), ("net_instrument", "asset_net", D("0.1"), ()),
          ("margin", "fund_margin", D("0.4"), ()), ("asset_open_risk", "asset_stop_loss", D("0.5"), ()),
          ("group_open_risk", "type_stop_loss", D("0.5"), (Holding("ETH", D(1_000), D(500), D(250)),))]


@pytest.mark.parametrize("row, key, at_min, holdings", BOUNDS, ids=[b[0] for b in BOUNDS])
def test_each_row_at_its_minimum_trims_to_the_hand_worked_size_and_at_its_maximum_binds_nothing(row, key, at_min,
                                                                                               holdings):
    book = Book(D(100_000), holdings)
    groups = {"BTC": "large", "ETH": "large"}
    low, high = (float(BY_KEY[key].low), float(BY_KEY[key].high))
    d = decide(book, _btc(), replace(LOOSE, groups=groups, **{row: low}))
    assert (d.outcome, d.approved_qty, d.limit_hit) == ("trimmed", at_min, row), d
    d = decide(book, _btc(), replace(LOOSE, groups=groups, **{row: high}))
    assert (d.outcome, d.approved_qty) == ("approved", D(1)), d


def test_a_t5_brake_switched_off_never_fires_and_the_other_still_does():
    """book 80 against a reference and day start of 100: a 20% drawdown, and a 20% loss on the day."""
    on = replace(LOOSE, drawdown=0.15, daily_loss=0.03)
    assert book_breach(on, D(80), D(100), D(100)).action == "halt"
    assert book_breach(replace(on, drawdown=None), D(80), D(100), D(100)).action == "pause"
    assert book_breach(replace(on, daily_loss=None), D(80), D(100), D(100)).action == "halt"
    assert book_breach(replace(on, drawdown=None, daily_loss=None), D(80), D(100), D(100)) is None
    # figures that can't be used still halt, brakes off or not (fail closed)
    assert book_breach(replace(on, drawdown=None, daily_loss=None), D(0), D(100), D(100)).action == "halt"


# ----------------------------------------------------------------------------------------------------------------
# End to end: the PM's saved tests, a fund-held store (re-base deposit, the Strategy made with 0), the paper engine
# and its gate, as tonight's run.
# ----------------------------------------------------------------------------------------------------------------

def _run(tmp_path, monkeypatch, off: tuple[str, ...]):
    import pandas as pd
    import test_exposure_gate_xfails as h
    from test_f_trim_min import _store
    from test_portfolio_paper_entry import END

    from sleeve_fund.supervisor import Supervisor

    store = _store(tmp_path)
    c = store.compliance()
    tests = [dict(t) for t in c.tests]
    for t in tests:
        if t["test"] in {KEY[r] for r in off}:
            t["enabled"] = False
    store.add_compliance_version(tests, c.asset_types, expected_version=c.version, note="B-32 cell: T4 off")
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params={**h.PERP, "stop_loss": h.STOP})
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-32: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    plan = h.Plan(t0=h.STD_T0, minutes=8, windows=[(h.M(h.STD_T0, 2), h.M(h.STD_T0, 8), 1)], balance=200_000.0,
                  params={}, tag="b32")
    h.run(tmp_path, store, plan, monkeypatch)
    return store, h


@pytest.mark.parametrize("off", [("gross",), ("net_instrument",), ("margin",), T4],
                         ids=["gross_off", "net_off", "margin_off", "all_t4_off"])
def test_the_paper_gate_sends_the_entry_with_t4_rows_switched_off(tmp_path, monkeypatch, off):
    store, h = _run(tmp_path, monkeypatch, off)
    events = h._events(store)
    failed = [e for e in events if "couldn't run" in (e.get("message") or "")]
    assert failed == [], failed
    entries = [o for o in h._orders(store) if o["intent"] == "entry"]
    assert len(entries) == 1 and entries[0]["qty"] > 0, entries
    fills = h._fills(store)
    assert fills and sum(f["qty"] for f in fills if f["side"] == "BUY") > 0, fills
