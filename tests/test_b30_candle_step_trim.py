"""B-30 (PM run 15 Oct 2026, 20:54-21:06 UK): candle_step fixed Q at the size it ASKED for at step 0, before the
portfolio gate's trim, so "sell half" sold held - Q_asked/2: 40% of the trimmed entry, not half (0.606 - 0.3635 =
0.242 tonight, where half is 0.303). Q must be the quantity actually sent, and only once it is sent.

Paper, through the real gate on a fund-held store (re-base deposit, the Strategy made with 0), with the PM's seeded
T4 net-per-asset limit (0.5x) trimming the 60% size share. Runs on Postgres too with TEST_DATABASE_URL."""

from __future__ import annotations

from decimal import ROUND_DOWN
from decimal import Decimal as D

import pytest
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.strategies.candle_step import CandleStep, CandleStepConfig

LOT = D("0.001")


class _Config(CandleStepConfig):
    def __init__(self, *, windows=(), rest=(), **kw):  # the harness's Egx keys: candle_step reads neither
        super().__init__(**kw)


def _run(tmp_path, monkeypatch, *, size_share=0.6, minutes=9):
    import pandas as pd
    import test_exposure_gate_xfails as h
    from test_f_trim_min import _store
    from test_portfolio_paper_entry import END

    from sleeve_fund.strategies import REGISTRY
    from sleeve_fund.supervisor import Supervisor

    monkeypatch.setitem(REGISTRY, "egx", (CandleStep, _Config))
    store = _store(tmp_path)  # compliance version 1: the PM's seeded tests, T4 net per asset 0.5x
    params = {**h.PERP, "stop_loss": h.STOP, "size_share": size_share}
    store.create_sleeve(name=h.NAME, strategy="egx", instrument=h.PAIR, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, risk_profile="balanced", params=params)
    before = h.STD_T0.to_pydatetime() - pd.Timedelta(hours=1)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-30: the fund's capital", rebase=True,
                                      now=before), before)
    Supervisor(store).mark_portfolio(END)
    plan = h.Plan(t0=h.STD_T0, minutes=minutes, balance=200_000.0, params={"size_share": size_share}, tag="b30")
    h.run(tmp_path, store, plan, monkeypatch)
    rows = [o for o in store.orders(h.NAME, limit=1000) if "cycle" in (o["signal"] or {})]
    return sorted(rows, key=lambda o: (o["ts"], o["id"])), h


def _first_cycle(rows):
    first = next(o for o in rows if o["intent"] == "entry" and o["signal"].get("step") == 0)
    return first, [o for o in rows if o["signal"]["cycle"] == first["signal"]["cycle"]]


def test_a_step_0_open_trimmed_by_t4_sells_half_of_what_was_sent(tmp_path, monkeypatch):
    rows, _ = _run(tmp_path, monkeypatch)
    entry, cycle = _first_cycle(rows)
    sent = D(str(entry["qty"]))
    asked = D(str(entry["signal"]["portfolio_trimmed_from"]))  # the gate trimmed it: this cell needs that
    assert sent < asked, (sent, asked)
    assert D(str(entry["filled_qty"])) == sent, entry
    half = next(o for o in cycle if o["signal"]["step"] == 1)
    # Step 1 is "sell half": held (= sent) less Q/2, with Q = sent, rounded down to the lot
    want = (sent - sent / 2).quantize(LOT, rounding=ROUND_DOWN)
    assert (half["side"], half["intent"], D(str(half["qty"]))) == ("SELL", "exit", want), (half, sent, asked)


def test_q_journaled_on_every_later_step_is_the_sent_size_so_a_restart_reads_the_same_q(tmp_path, monkeypatch):
    """resume_cycle takes Q from a later step's journaled q, or else from step 0's sent qty: both must be one figure,
    or a restart mid-cycle changes Q."""
    rows, _ = _run(tmp_path, monkeypatch)
    entry, cycle = _first_cycle(rows)
    later = [o for o in cycle if o["signal"]["step"] > 0]
    assert later, cycle
    assert {D(str(o["signal"]["q"])) for o in later} == {D(str(entry["qty"]))}, \
        [(o["signal"]["step"], o["signal"]["q"]) for o in later]


def test_restart_reads_the_sent_q_from_the_journal(tmp_path, monkeypatch):
    """A restart inside the cycle (resume_cycle on this journal) carries on with Q = the size sent at step 0."""
    rows, h = _run(tmp_path, monkeypatch)
    entry, _ = _first_cycle(rows)

    class _Rt:  # the runtime's two reads resume_cycle makes
        backtest = False
        name = h.NAME

    rt = _Rt()
    store_rows = [o for o in rows if o["signal"]["cycle"] == entry["signal"]["cycle"]]
    rt.store = type("S", (), {"orders": staticmethod(lambda name, limit=100: list(reversed(store_rows)))})()
    m = CandleStep.__new__(CandleStep)
    m.runtime, m._started, m._resumed, m._last_q = rt, None, None, None
    m.resume_cycle()
    assert m._resumed is not None and m._resumed[1] == D(str(entry["qty"])), (m._resumed, entry["qty"])

