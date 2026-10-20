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


_LAST: dict = {}  # the last _run's store


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
    _LAST["store"] = store
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



@pytest.mark.parametrize("how", ["not_submitted", "under_the_venue_minimum", "gate_allows_under_a_lot"])
def test_a_step_0_that_is_never_sent_sets_no_q_and_the_cycle_opens_nothing(tmp_path, monkeypatch, how):
    """CR follow-up on B-30: an entry refused at _submit (CHOKE) or left under the venue minimum after the gate tells
    the model nothing. Were Q set anyway, step 2 ("sell the rest and short Q") would open a short of Q on a book that
    never went long. The next cycle sizes afresh from flat."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    refused: list[int] = []
    if how == "not_submitted":
        real = LongFlatStrategy._submit

        def submit(self, side, qty, intent, reason, signal, *a, **k):
            if intent == "entry" and signal.get("step") == 0 and not refused:
                refused.append(signal["cycle"])
                return False  # as a CHOKE refusal: nothing sent
            return real(self, side, qty, intent, reason, signal, *a, **k)

        monkeypatch.setattr(LongFlatStrategy, "_submit", submit)
    elif how == "gate_allows_under_a_lot":
        # The common case (CR): the gate approves less than one lot (a T4 trim with the book at its limit), the
        # "Entry skipped: the portfolio limits allowed X" branch.
        from sleeve_fund.paper.portfolio_gate import PaperGate

        real_gate = PaperGate.__call__

        class _UnderALot:
            def __init__(self, gated):
                self._g = gated

            def __getattr__(self, name):
                return getattr(self._g, name)

            @property
            def approved_qty(self):
                return D("0.0001")  # the lot is 0.001

        def gate(self, strategy, intent, ts_ns):
            out = real_gate(self, strategy, intent, ts_ns)
            if out is None or refused or intent.get("what") != "entry":
                return out
            refused.append(None)  # the cycle is read from the journal below
            return _UnderALot(out)

        monkeypatch.setattr(PaperGate, "__call__", gate)
    else:
        real = LongFlatStrategy._under_min_notional

        def under(self, approved, intent):
            if not refused:
                refused.append(self._plan.cycle)
                return (D(1), D(100))  # as F-TRIM-MIN: what is left is under the venue's smallest order value
            return real(self, approved, intent)

        monkeypatch.setattr(LongFlatStrategy, "_under_min_notional", under)
    told: list[tuple[int, D]] = []  # every Q the model was told: (cycle, qty)
    real_told = CandleStep.entry_sent

    def entry_sent(self, qty, values):
        told.append((values.get("cycle"), qty))
        return real_told(self, qty, values)

    monkeypatch.setattr(CandleStep, "entry_sent", entry_sent)
    rows, h = _run(tmp_path, monkeypatch, minutes=15)
    assert refused, "set-up: no step 0 was refused"
    if refused[0] is None:  # the skip's cycle: its bar closed in the minute the skip was noted
        from sleeve_fund.strategies.candle_step import step_at

        skipped = [e for e in h._events(_LAST["store"]) if "Entry skipped" in (e["message"] or "")]
        assert len(skipped) == 1, ("set-up: the gate's under-a-lot answer didn't reach the skip once", skipped)
        refused[0] = step_at(skipped[0]["ts"], 1)[0]
    assert [c for c, _ in told if c == refused[0]] == [], told  # never told a Q for the refused step 0
    in_cycle = [(o["signal"]["step"], o["intent"], o["side"]) for o in rows if o["signal"]["cycle"] == refused[0]]
    assert in_cycle == [], in_cycle  # no Q: steps 1-5 of that cycle ask for nothing, so no short on a flat book
    later = [o for o in rows if o["signal"]["cycle"] > refused[0]]
    assert later and later[0]["intent"] == "entry" and later[0]["signal"]["step"] == 0, \
        [(o["signal"]["cycle"], o["signal"]["step"], o["intent"]) for o in rows]




def _model(at_step, journal=(), last_q=None):
    """A CandleStep (no engine) at the close of a candle that is step `at_step`, its journal holding `journal`'s
    orders of that cycle: (step, intent, filled, extra signal). With a journal it is restarted on it (resume_cycle)."""
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace as NS

    from nautilus_trader.model import BarType

    from sleeve_fund.strategies.candle_step import step_at

    t = datetime(2025, 1, 1, tzinfo=UTC)
    while step_at(t, 1)[1] != at_step:
        t += timedelta(minutes=1)
    cycle = step_at(t, 1)[0]
    rows = [{"ts": t - timedelta(minutes=at_step - k, seconds=-1), "intent": intent, "qty": filled,
             "filled_qty": filled, "signal": {"cycle": cycle, "step": k, "q": 1.0, **extra}}
            for k, intent, filled, extra in reversed(journal)]
    s = CandleStep.__new__(CandleStep)
    s._cfg = NS(bar_type=BarType.from_str("BTCUSDT-PERP.BYBIT-1-MINUTE-LAST-INTERNAL"))
    s.c = NS(run_for_hours=None, size_share=0.6)
    s._plan = s._started = s._resumed = None
    s._last_q = last_q
    s._resumed_legs = {}
    s.runtime = NS(backtest=False, name="cs", store=NS(orders=lambda name, limit: rows))
    if journal:
        s.resume_cycle()
    bar = lambda k: NS(ts_event=int((t + timedelta(minutes=k - at_step)).timestamp() * 1e9))
    return s, bar


def test_a_restart_recomputes_the_same_cover_half_target():
    """B-30 (Advisor B30-HELD-SIZE): step 3's target is the short as its open filled (499.8875) less half of it, an
    absolute target: restarted before step 3 it covers 249.94375; the step-3 candle seen again after the cover (its
    leg journaled with it) sends nothing; never down to -Q/2."""
    flip = [(0, "entry", 500.0, {}), (1, "exit", 250.0, {"leg": 500.0}), (2, "exit", 250.0, {}),
            (2, "entry", 499.8875, {})]
    before, bar = _model(3, flip)
    assert before.step_orders(bar(3), D("-499.8875"))[0] == [("reduce", 1, D("249.94375"))]
    again, bar = _model(3, [*flip, (3, "exit", 249.94375, {"leg": 499.8875})])
    assert again.step_orders(bar(3), D("-249.94375"))[0] == []


@pytest.mark.parametrize("topped_up", [True, False], ids=["top_up_filled", "no_top_up"])
def test_a_long_carried_in_under_q_halves_what_step_0_left_held_restart_or_not(topped_up):
    """CR v4 MUST: T4 trims step 4's buy, so the next cycle carries a long under Q (0.8 of Q 1.0); step 0 tops it up
    (0.2 filled) or doesn't. Step 1 sells half of what step 0 left held, with or without a restart between: 0.5 after
    the top-up, 0.4 without; a restart after step 1 sends nothing more."""
    held = D("1.0") if topped_up else D("0.8")
    live, bar = _model(0, last_q=D("1.0"))
    live.step_orders(bar(0), D("0.8"))  # carried in: Q 1.0, the top-up asked
    if topped_up:
        live._plan.leg_filled(0, held)  # what on_order_filled records as the top-up fills
    orders, _, values = live.step_orders(bar(1), held)
    assert orders == [("reduce", -1, held / 2)]
    assert values["leg"] == float(held)  # journaled with the step, which is what a restart reads back
    if topped_up:  # a restart between the top-up and step 1: the leg is what is held, as without one
        restarted, bar = _model(1, [(0, "entry", 0.2, {})])
        assert restarted.step_orders(bar(1), held)[0] == [("reduce", -1, held / 2)]
    after, bar = _model(1, [*([(0, "entry", 0.2, {})] if topped_up else []), (1, "exit", float(held / 2),
                                                                               {"leg": float(held)})])
    assert after.step_orders(bar(1), held / 2)[0] == []


def test_the_cover_rounds_down_to_the_lot_and_a_retry_sends_nothing():
    """Advisor B30-HELD-SIZE (1): half the leg rounds DOWN to the lot, never over-closing (0.0015 of 0.003 covers
    0.001), and the target it leaves (0.002) is absolute: the step asked again sends nothing."""
    from sleeve_fund.strategies.candle_step import Plan

    plan = Plan(0, D("0.003"), legs={0: D("0.003")})
    assert plan.orders(1, D("0.003"), lot=D("0.001")) == [("reduce", -1, D("0.001"))]
    assert plan.orders(1, D("0.002"), lot=D("0.001")) == []
