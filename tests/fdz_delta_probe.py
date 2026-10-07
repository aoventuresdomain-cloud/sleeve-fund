"""QA adversarial probes, FUNDING-DEADLINE delta round (PE1, #179). Not a master: QA's own probes beyond it.

Run beside the master (test_funding_deadline_xfails.py), o17_harness.py and test_hub_146_qa.py in tests/. Each probe
asserts what the rulings want (Advisor 7 Oct 00:40: funding is owed on the position held at the settlement instant;
the 15-minute bound runs from the settlement; the reversal is its own row). A red probe is a finding.
"""
from __future__ import annotations

import dataclasses
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

import test_hub_146_qa as qa
from test_funding_deadline_xfails import (  # noqa: F401  (fixtures are used by name)
    BASELINE, BOOKED, DAY, SETTLE, SLACK, WAIT, _booked, _start, backtest_funding, booked_at, charges, sparse, utc)
from o17_harness import _o17win, binance  # noqa: F401
from test_hub_146_qa import _probe  # noqa: F401

M, S = qa.M, qa.S
DIAG = "funding_charged_while_flat"
EVENT = "funding_deadline_booked"


def _dt(start, minutes):
    return datetime.fromtimestamp((start + int(round(minutes * 60)) * S) / 1e9, tz=timezone.utc)


def _cadence(monkeypatch, hours):
    from sleeve_fund import markets

    if hours is not None:
        monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=hours))


def rows_at(run, ts):
    return sorted((r for r in run.store.funding("q146") if utc(r["ts"]) == ts), key=lambda r: r["id"])


def every(n_minutes, step, offset=0, after_min=0.0):
    """Seconds with no trade from `after_min` on: one trade every `step` s at `offset`."""
    return frozenset(s for s in range(int(after_min * 60), n_minutes * 60) if (s - offset) % step)


def restarted(monkeypatch, *, back_min, heartbeat_min, n=60, held_qty=0.05, fills=(), funding=(), seen_min=None,
              trades="sparse", prices=None, hours=None, orders=(), prior=False, **kw):
    """Process B starts at `back_min` (minutes after 07:50). Its journal holds a long of held_qty entered at 07:55,
    then `fills` = [(side, qty, minute)] journaled by process A at those times (a fraction of a minute allowed, so
    minute 10 is 08:00:00 exactly), and `funding` = [(qty, minute-of-settlement)] rows A booked. Trades and live hub
    minutes from back_min on (sparse: at :10/:40 only); the store holds the minutes up to then."""
    from sleeve_fund.store import Store

    _cadence(monkeypatch, hours)
    start = _start(monkeypatch, "07:50")
    p = qa.flat_prices(n) if prices is None else prices
    fill, book, order = Store.record_fill, Store.record_funding, Store.record_order

    def seeded_fill(self, name, **kwargs):
        out = fill(self, name, **kwargs)
        if kwargs.get("trade_id") == "T-held":
            for i, (side, q, m) in enumerate(fills):
                order(self, name, order_id=f"O-s{i}", side=side, qty=q, intent="exit" if side == "SELL" else "entry",
                      reason="seeded", signal={"stop_frac": 0.01}, ts=_dt(start, m))
                fill(self, name, side=side, qty=q, price=float(p[int(m * 60)]), fee=0.0, order_id=f"O-s{i}",
                     trade_id=f"T-s{i}", ts=_dt(start, m))
            for q, m in funding:
                px = float(p[int(m * 60)])
                book(self, name, qty=q, price=px, rate=BASELINE, amount=round(-q * px * BASELINE, 8), ts=_dt(start, m))
            for oid, side, q, intent, m, sig in orders:  # journaled orders with no fill (sent, not filled)
                order(self, name, order_id=oid, side=side, qty=q, intent=intent, reason="seeded", signal=sig,
                      ts=_dt(start, m))
            if prior:  # an earlier settlement booked by A (00:00), so the sleeve has a funding row before 08:00
                book(self, name, qty=held_qty, price=float(p[0]), rate=BASELINE,
                     amount=round(-held_qty * float(p[0]) * BASELINE, 8), ts=utc(f"{DAY} 00:00").to_pydatetime())
            if seen_min is not None:
                self.feed_seen(name, _dt(start, seen_min))
        return out

    monkeypatch.setattr(Store, "record_fill", seeded_fill)
    gone = frozenset(range(0, int(back_min * 60)))
    if trades == "sparse":
        gone |= sparse(n, back_min)
    run = qa.paper(p, side=1, perp=True, profile="aggressive", gone=gone,
                   lost={start + k * M for k in range(0, int(back_min) + 1)},
                   held=(held_qty, float(p[300]), start + 5 * M), heartbeat=start + int(heartbeat_min * M),
                   history=qa.stored_history(p, start + int(back_min * M)), **kw)
    return run, start


# ============================================================ (a) a fill exactly at the settlement instant


def test_a_live_exit_filled_at_the_settlement_instant_is_charged_on_the_long(monkeypatch):
    """Bars arrive on their close (lag 0) and a trade lands at 08:00:00, so the exit decided on the 08:00 bar fills at
    08:00:00 (journaled 08:00:00). A fill at the instant is after it: 08:00 is owed by the long, paid once."""
    start = _start(monkeypatch, "07:50")
    p = qa.flat_prices(60)
    gone = every(60, 30, 0, 5)  # trades at :00 and :30 from 07:55
    run = qa.paper(p, side=1, perp=True, profile="aggressive", leave=10, gone=gone, lag=0)
    ex = [f for f in run.fills if f["side"] == "SELL"]
    assert ex and utc(ex[0]["ts"]) == SETTLE, ("set-up: the exit fills at 08:00:00", run.fills)
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1 and got[0]["qty"] > 0 and got[0]["amount"] < 0, run.store.funding("q146")
    assert min(booked_at(SETTLE)) <= SETTLE + WAIT + SLACK


@pytest.mark.parametrize("prior", [False, True], ids=["first-settlement", "funding-row-at-0000"])
def test_a_restart_after_an_exit_journaled_at_the_settlement_instant_still_charges_the_long(prior, monkeypatch):
    """A held 0.05 long from 07:55 and closed it at 08:00:00 exactly; A stopped at 08:05 before its 08:15 deadline
    booking (sparse market); B starts at 08:07, flat. The journal says the long was held at 08:00 (a fill at the
    instant is after it), so B must charge 08:00 on 0.05, once."""
    run, _ = restarted(monkeypatch, back_min=17, heartbeat_min=15, fills=[("SELL", 0.05, 10)], leave=10, prior=prior)
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1, f"08:00 owed by the long closed at 08:00:00 is never charged after the restart: {got}"
    assert got[0]["qty"] == pytest.approx(0.05) and got[0]["amount"] < 0, got


def test_a_restart_after_an_exit_journaled_one_second_before_the_settlement_charges_nothing(monkeypatch):
    """Control: the long closed at 07:59:59; B (08:07) must not charge 08:00."""
    run, _ = restarted(monkeypatch, back_min=17, heartbeat_min=15, fills=[("SELL", 0.05, 9 + 59 / 60)], leave=10)
    assert charges(run.store.funding("q146"), SETTLE) == [], run.store.funding("q146")


def test_a_restart_after_an_entry_journaled_at_the_settlement_instant_charges_nothing_for_it(monkeypatch):
    """B's journal: the 07:55 long was closed at 07:57, and a new long opened at exactly 08:00:00. Flat at 08:00:
    nothing is owed for it."""
    run, _ = restarted(monkeypatch, back_min=17, heartbeat_min=15, fills=[("SELL", 0.05, 7), ("BUY", 0.05, 10)])
    assert charges(run.store.funding("q146"), SETTLE) == [], run.store.funding("q146")


# ============================================ (b) a partial reduce and a reversal between the settlement and restart


@pytest.mark.parametrize("prior", [False, True], ids=["first-settlement", "funding-row-at-0000"])
def test_a_restart_after_a_reduce_after_the_settlement_charges_the_quantity_held_at_it(prior, monkeypatch):
    """0.25 long from 07:55, 0.10 sold at 08:03 (after 08:00); A stopped at 08:05, B starts at 08:07 (sparse, so
    nothing was booked yet). 08:00 is owed on 0.25 (the journal: now 0.15 + the 0.10 sold since)."""
    run, _ = restarted(monkeypatch, back_min=17, heartbeat_min=15, held_qty=0.25, fills=[("SELL", 0.10, 13)],
                       prior=prior)
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1, f"08:00 never charged after the restart: {run.store.funding('q146')}"
    assert got[0]["qty"] == pytest.approx(0.25) and got[0]["amount"] < 0, got


@pytest.mark.parametrize("prior", [False, True], ids=["first-settlement", "funding-row-at-0000"])
def test_a_restart_after_a_reduce_and_a_reversal_after_the_settlement_charges_the_long_held_at_it(prior, monkeypatch):
    """0.25 long from 07:55; 0.10 sold at 08:03, 0.30 sold at 08:06 (now short 0.15); B starts at 08:08 and wants
    the short. 08:00 is owed by the 0.25 long: paid, on 0.25."""
    run, _ = restarted(monkeypatch, back_min=18, heartbeat_min=16, held_qty=0.25,
                       fills=[("SELL", 0.10, 13), ("SELL", 0.30, 16)], leave=16, extra={"after": -1}, prior=prior)
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1, f"08:00 never charged after the restart: {run.store.funding('q146')}"
    assert got[0]["qty"] == pytest.approx(0.25) and got[0]["amount"] < 0, got


# ============================================== (c) two settlements inside one outage, refills after both


def long_outage(monkeypatch, *, stop_min=None, trades="sparse"):
    """8-hourly settlements. Long from 07:55. Trades and hub away 07:56-16:25 (over 08:00 and 16:00); refills land
    at 16:25:05; afterwards trades at :10/:40 (sparse) or every second. stop_min: the price goes 2% through the 1%
    stop for 20 s at that minute (between the settlements)."""
    start = _start(monkeypatch, "07:50")
    n = 8 * 60 + 50  # to 16:40
    p = qa.flat_prices(n)
    if stop_min is not None:
        p = qa.shape(p, stop_min + 0.5, stop_min + 50 / 60, qa.adverse(1, 0.02))
    gone = frozenset(range(6 * 60, 515 * 60)) | (sparse(n, 515) if trades == "sparse" else frozenset())
    back = start + 515 * M + 5 * S
    run = qa.paper(p, side=1, perp=True, profile="aggressive", leave=n - 5, gone=gone,
                   away=(start + 6 * M, start + 515 * M), back_at=back)
    return run, _ts(back)


def _ts(ns):
    return pd.Timestamp(ns, unit="ns", tz="UTC")


S16 = utc(f"{DAY} 16:00")


@pytest.mark.parametrize("trades", ["sparse", "dense"])
def test_two_settlements_in_one_outage_with_a_stop_between_charge_only_the_first(trades, monkeypatch):
    run, back = long_outage(monkeypatch, stop_min=250, trades=trades)  # stop at 12:00:30
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    rows = run.store.funding("q146")
    assert len(charges(rows, SETTLE)) == 1 and charges(rows, SETTLE)[0]["amount"] < 0, rows
    assert charges(rows, S16) == [], rows
    assert not run.kinds(DIAG), run.kinds(DIAG)
    assert all(c >= back for c, _ in BOOKED if c is not None), ("booked before the refills landed", BOOKED)


@pytest.mark.parametrize("trades", ["sparse", "dense"])
def test_two_settlements_in_one_outage_without_a_stop_charge_both_once_after_the_refills(trades, monkeypatch):
    run, back = long_outage(monkeypatch, trades=trades)
    rows = run.store.funding("q146")
    assert len(charges(rows, SETTLE)) == 1 and len(charges(rows, S16)) == 1, rows
    assert all(c >= back for c, _ in BOOKED if c is not None), ("booked before the refills landed", BOOKED)
    assert max(c for c, _ in BOOKED) <= back + timedelta(seconds=75), BOOKED
    assert not run.kinds(DIAG)


# ============================== (d) the deadline booking, a restart, then a late refill showing a stop before 08:00


STOP = (8.5, 8 + 50 / 60)  # 07:58:30-07:58:50


def test_deadline_booking_then_restart_whose_replay_finds_the_stop_reverses_once(monkeypatch):
    """A booked 08:00 at its 08:15 deadline (journaled) and stopped at 08:16 with its market data last at 07:57;
    B starts at 08:18 and its replay of the stored minutes from 07:57 shows the stop at 07:58: exactly one
    funding_charged_while_flat and one reversal row, netting to zero; no deadline event from B."""
    p = qa.shape(qa.flat_prices(60), *STOP, qa.adverse(1, 0.02))
    run, _ = restarted(monkeypatch, back_min=28, heartbeat_min=7, held_qty=0.05, funding=[(0.05, 10)], prices=p,
                       leave=55)
    rows = rows_at(run, SETTLE)
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    assert len(rows) == 2 and sum(r["amount"] for r in rows) == pytest.approx(0, abs=1e-8), rows
    assert len(run.kinds(DIAG)) == 1, run.kinds(DIAG)
    assert not run.kinds(EVENT), run.kinds(EVENT)


def test_deadline_booking_then_restart_then_late_refill_showing_the_stop(monkeypatch):
    """A booked 08:00 at its deadline; B starts at 08:18 with the store holding the minutes only up to 07:57 (the
    minutes to 07:58 and 07:59 never reached it); the hub refills those two at 08:20:05 and they show the stop at
    07:58. Wanted: one diagnostic, one reversal row (net zero)."""
    p = qa.shape(qa.flat_prices(60), *STOP, qa.adverse(1, 0.02))
    start = _start(monkeypatch, "07:50")
    from sleeve_fund.store import Store

    fill, book = Store.record_fill, Store.record_funding

    def seeded_fill(self, name, **kw):
        out = fill(self, name, **kw)
        if kw.get("trade_id") == "T-held":
            px = float(p[600])
            book(self, name, qty=0.05, price=px, rate=BASELINE, amount=round(-0.05 * px * BASELINE, 8),
                 ts=SETTLE.to_pydatetime())
        return out

    monkeypatch.setattr(Store, "record_fill", seeded_fill)
    gone = frozenset(range(0, 28 * 60)) | sparse(60, 28)
    lost = {start + k * M for k in range(0, 29)} - {start + 8 * M, start + 9 * M}
    run = qa.paper(p, side=1, perp=True, profile="aggressive", leave=55, gone=gone, lost=lost,
                   away=(start + 7 * M, start + 9 * M), back_at=start + 30 * M + 5 * S,
                   held=(0.05, float(p[300]), start + 5 * M), heartbeat=start + 7 * M,
                   history=qa.stored_history(p, start + 7 * M))
    rows = rows_at(run, SETTLE)
    print("D2", run.sequence(), rows, [e["kind"] for e in run.events])
    assert qa.first_exit(run.sequence())[0] == "stop_loss", ("the late refill's stop is never found", run.sequence())
    assert len(rows) == 2 and sum(r["amount"] for r in rows) == pytest.approx(0, abs=1e-8), rows
    assert len(run.kinds(DIAG)) == 1, run.kinds(DIAG)


# ============================== (e) sparse trades: a short, a short reversed to long, and flat then open at 08:00:00


def test_sparse_short_reversed_to_long_before_the_bound_charges_the_short(monkeypatch):
    """Short 07:55-08:05, then long: 08:00 is the short's (qty < 0), booked once by its bound."""
    _start(monkeypatch, "07:50")
    run = qa.paper(qa.flat_prices(60), side=-1, perp=True, profile="aggressive", leave=15, gone=sparse(60),
                   extra={"after": 1})
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1 and got[0]["qty"] < 0, run.store.funding("q146")
    assert abs(got[0]["amount"]) == pytest.approx(abs(got[0]["qty"]) * got[0]["price"] * BASELINE, rel=1e-6), got
    assert min(booked_at(SETTLE)) <= SETTLE + WAIT + SLACK


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_sparse_flat_then_opened_at_08_00_00_is_not_charged_for_08_00(side, monkeypatch):
    """Hourly settlements. Flat until the 08:00 bar (on its close, lag 0); the entry fills at 08:00:00 exactly on
    the :00 trade (trades at :00/:30). Not held at 08:00, so nothing for it; 09:00 charged once by its bound."""
    _cadence(monkeypatch, tuple(range(24)))
    _start(monkeypatch, "07:50")
    run = qa.paper(qa.flat_prices(90), side=side, perp=True, profile="aggressive", enter=10, leave=85,
                   gone=every(90, 30, 0), lag=0)
    ent = [f for f in run.fills]
    assert ent and utc(ent[0]["ts"]) == SETTLE, ("set-up: the entry fills at 08:00:00", run.fills)
    rows = run.store.funding("q146")
    assert charges(rows, SETTLE) == [], rows
    s9 = utc(f"{DAY} 09:00")
    got = charges(rows, s9)
    assert len(got) == 1 and (got[0]["qty"] > 0) == (side > 0), rows
    assert min(booked_at(s9)) <= s9 + WAIT + SLACK


# ============================== (f) funding_deadline_booked: once per deadline booking, never on time


def test_deadline_event_once_per_deadline_booking_on_an_hourly_cadence(monkeypatch):
    """Hourly, sparse, long 08:55-10:25: 09:00 and 10:00 each booked at its bound, each with one event naming it."""
    _cadence(monkeypatch, tuple(range(24)))
    _start(monkeypatch, "08:50")
    run = qa.paper(qa.flat_prices(100), side=1, perp=True, profile="aggressive", leave=95, gone=sparse(100))
    ev = run.kinds(EVENT)
    rows = run.store.funding("q146")
    assert len(rows) == 2, rows
    assert len(ev) == 2 and "09:00" in ev[0]["message"] and "10:00" in ev[1]["message"], ev


@pytest.mark.parametrize("published, step, want", [("08:10", 5, 0), ("never", 5, 0), ("08:10", 30, 1),
                                                     ("never", 30, 1)])
def test_deadline_event_only_when_the_trade_hold_ran_out(published, step, want, tmp_path, monkeypatch, binance):
    """Binance's own rate published at 08:10 or never. Dense trades (5 s): the wait is the rate's (FUNDING_WAIT),
    not the trade hold, so no deadline event. Sparse (30 s): exactly one, naming 08:00."""
    from test_funding_deadline_xfails import native_run

    out = native_run(tmp_path, monkeypatch, binance, step=step, published=published)
    assert len(charges(out["funding"], SETTLE)) == 1, out["funding"]
    ev = [e for e in out["events"] if e["kind"] == EVENT]
    assert len(ev) == want, ev
    assert all("08:00" in e["message"] for e in ev), ev


def test_no_deadline_event_for_the_reversal_row(monkeypatch):
    """Dense trades; 08:00 charged on time, then a late refill finds the stop at 07:58: the reversal row is no
    deadline booking."""
    from test_funding_deadline_xfails import refill_run

    run = refill_run(monkeypatch, trades="dense", stop=True)
    assert len(run.kinds(DIAG)) == 1
    assert not run.kinds(EVENT), run.kinds(EVENT)


# ============================== (g) paper vs backtest parity on a sparse day


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_paper_matches_the_backtest_over_a_sparse_day_hourly(side, monkeypatch):
    """Hourly settlements over a whole day of trades at :10/:40: 00:05 to 23:55, 23 settlements."""
    from test_funding_deadline_xfails import hub_run

    hours = tuple(range(24))
    bt = backtest_funding(monkeypatch, hhmm="00:00", minutes=24 * 60, leave=24 * 60 - 5, side=side, hours=hours)
    run = hub_run(monkeypatch, hhmm="00:00", minutes=24 * 60, side=side, leave=24 * 60 - 5, trades="sparse",
                  hours=hours)
    pp = sorted(run.store.funding("q146"), key=lambda r: r["ts"])
    assert len(bt) == 23, bt
    assert [utc(r["ts"]) for r in pp] == [utc(r["ts"]) for r in bt], (pp, bt)
    for a, b in zip(pp, bt):
        assert a["qty"] == pytest.approx(b["qty"]) and a["rate"] == pytest.approx(b["rate"]), (a, b)
        assert a["amount"] == pytest.approx(b["amount"], rel=1e-3), (a, b)
    for r in pp:
        ts = utc(r["ts"])
        assert min(booked_at(ts)) <= ts + WAIT + SLACK, (ts, booked_at(ts))


# ============================== CR (1): the deadline never books a settlement younger than its own bound


CR1 = [pytest.param(tuple(range(24)), 75, "09:00", id="1h"),  # 08:00 and 09:00; back at 09:05
       pytest.param(tuple(range(0, 24, 4)), 255, "12:00", id="4h"),  # 08:00 and 12:00; back at 12:05
       pytest.param((0, 8, 16), 495, "16:00", id="8h")]  # 08:00 and 16:00; back at 16:05


@pytest.mark.parametrize("hours, back, later", CR1)
def test_a_restart_never_books_the_later_settlement_before_its_own_bound(hours, back, later, monkeypatch):
    """Down from 07:58 to `later` + 5 min, sparse trades after: 08:00 is past its bound and booked at once; the later
    settlement is only 5 minutes old and must wait for its own bound (settlement + 15 min)."""
    run, _ = restarted(monkeypatch, back_min=back, heartbeat_min=8, n=back + 30, hours=hours, leave=back + 25)
    t = utc(f"{DAY} {later}")
    when = booked_at(t)
    assert len(charges(run.store.funding("q146"), SETTLE)) == 1, run.store.funding("q146")
    assert when, ("set-up: the later settlement is booked", run.store.funding("q146"))
    assert min(when) >= t + WAIT, f"the {later} settlement was booked at {min(when)}, before its bound {t + WAIT}"


def test_an_outage_never_books_the_later_settlement_before_its_own_bound(monkeypatch):
    """Hourly. Away 07:56-09:05 (refills at 09:05:05), sparse after: 09:00 must wait to 09:15:00."""
    _cadence(monkeypatch, tuple(range(24)))
    start = _start(monkeypatch, "07:50")
    n = 100
    gone = frozenset(range(6 * 60, 75 * 60)) | sparse(n, 75)
    run = qa.paper(qa.flat_prices(n), side=1, perp=True, profile="aggressive", leave=95, gone=gone,
                   away=(start + 6 * M, start + 75 * M), back_at=start + 75 * M + 5 * S)
    t = utc(f"{DAY} 09:00")
    when = booked_at(t)
    assert when and min(when) >= t + WAIT, f"09:00 booked at {when}, before its bound {t + WAIT}"


def test_a_settlement_first_seen_at_08_10_is_booked_at_08_15_not_later_nor_earlier(monkeypatch):
    """Away 07:56-08:10 (refills at 08:10:05), sparse after: the bound runs from the settlement, not from when the
    process first saw it: booked in [08:15:00, 08:16:01]."""
    start = _start(monkeypatch, "07:50")
    gone = frozenset(range(6 * 60, 20 * 60)) | sparse(60, 20)
    run = qa.paper(qa.flat_prices(60), side=1, perp=True, profile="aggressive", leave=55, gone=gone,
                   away=(start + 6 * M, start + 20 * M), back_at=start + 20 * M + 5 * S)
    when = booked_at(SETTLE)
    assert len(charges(run.store.funding("q146"), SETTLE)) == 1
    assert SETTLE + WAIT <= min(when) <= SETTLE + WAIT + SLACK, when


@pytest.mark.parametrize("now, due", [("08:14:59", False), ("08:15:00", True)])
def test_funding_overdue_boundary_white_box(now, due):
    """White box: with a trade from after 08:00 in, _funding_overdue says 08:00 is due at 08:15:00, not at 08:14:59."""
    from types import SimpleNamespace

    from sleeve_fund import markets
    from sleeve_fund.strategies.base import LongFlatStrategy

    ns = SimpleNamespace(_funding_since=utc(f"{DAY} 07:55").to_pydatetime(), _backtest=False,
                         _trade_ns=int(utc(f"{DAY} 08:00:10").value), FUNDING_DEFER_MAX=LongFlatStrategy.FUNDING_DEFER_MAX)
    ns._settlements = lambda terms, since, until: (markets.settlement_times(since, until, terms.funding_hours, None), None)
    terms = markets.LOW_FEE_PERP
    assert LongFlatStrategy._funding_overdue(ns, terms, utc(f"{DAY} {now}").to_pydatetime()) is due


def test_funding_overdue_does_not_judge_a_later_settlement_by_the_first(monkeypatch):
    """White box, hourly: since 07:55, now 09:05, a trade at 09:04: 08:00 is overdue; whatever books then must not
    include 09:00 (5 minutes old). Reports which settlements the deadline path would take."""
    from types import SimpleNamespace

    from sleeve_fund import markets
    from sleeve_fund.strategies.base import LongFlatStrategy

    terms = dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=tuple(range(24)))
    since, now = utc(f"{DAY} 07:55").to_pydatetime(), utc(f"{DAY} 09:05").to_pydatetime()
    ns = SimpleNamespace(_funding_since=since, _backtest=False, _trade_ns=int(utc(f"{DAY} 09:04").value),
                         FUNDING_DEFER_MAX=LongFlatStrategy.FUNDING_DEFER_MAX)
    ns._settlements = lambda t, s, u: (markets.settlement_times(s, u, t.funding_hours, None), None)
    assert LongFlatStrategy._funding_overdue(ns, terms, now)
    times = markets.settlement_times(since, now, terms.funding_hours, None)
    print("DEADLINE-WOULD-BOOK", [str(t) for t in times])


# ============================== CR (2): the replayed close survives a restart before the deadline booking


@pytest.mark.parametrize("row", [True, False], ids=["exit-order-journaled", "exit-order-row-never-written"])
def test_a_replayed_close_before_08_00_survives_a_restart_before_the_deadline_booking(row, monkeypatch):
    """A was away 07:56-08:05; its refills showed the venue's stop closed the long at 07:58 (outage_exit), and A
    stopped at 08:06 before its own exit order filled and before any 08:00 booking (sparse market). A's heartbeat and
    market data run to 08:06. B starts at 08:08 and restores the long. 08:00 was flat (the venue closed it at 07:58):
    no charge, and so no funding_charged_while_flat either."""
    p = qa.shape(qa.flat_prices(60), *STOP, qa.adverse(1, 0.02))
    sig = {"price_source": "replay_model", "replayed_close": utc(f"{DAY} 07:58").isoformat(), "stop_frac": 0.01}
    orders = [("O-rx", "SELL", 0.05, "stop_loss", 15 + 5 / 60, sig)] if row else []
    run, _ = restarted(monkeypatch, back_min=18, heartbeat_min=16, seen_min=16, prices=p, leave=55, orders=orders,
                       prior=True)
    rows = run.store.funding("q146")
    print("CR2", run.sequence(), rows, [e["kind"] for e in run.events])
    assert charges(rows, SETTLE) == [] or sum(r["amount"] for r in rows_at(run, SETTLE)) == pytest.approx(0), rows
    assert not run.kinds(DIAG), run.kinds(DIAG)


def test_a_plain_exit_after_the_settlement_then_restart_charges_08_00_once(monkeypatch):
    """Control for CR (2), corrected 7 Oct ~04:10 (HoQA; was "..._charges_nothing", which passed only through FD-F7):
    a PLAIN exit (no replayed_close) filled at 08:05:05 before A stopped, so the long was held at 08:00 and owes it
    (Advisor 00:40). B (after the restart) is flat, yet 08:00 is charged exactly once, with no while-flat diagnostic.
    The replayed-close case (no charge) is test_a_replayed_close_before_08_00_survives_... above and PE1's pin."""
    p = qa.shape(qa.flat_prices(60), *STOP, qa.adverse(1, 0.02))
    run, _ = restarted(monkeypatch, back_min=18, heartbeat_min=16, seen_min=16, prices=p,
                       fills=[("SELL", 0.05, 15 + 5 / 60)], leave=55)
    assert len(charges(run.store.funding("q146"), SETTLE)) == 1, run.store.funding("q146")
    assert not run.kinds(DIAG)


# ============================== (h) the missed-minutes hold on a sparse market whose first trade beats the bar


@pytest.mark.parametrize("step, offset, lag", [(30, 1, 2), (60, 1, 2), (20, 1, 2), (30, 1, 1.5), (30, 3, 5)])
def test_sparse_trades_landing_before_the_minute_bar_still_book_by_the_bound(step, offset, lag, monkeypatch):
    """Trades every `step` s at second `offset` of the cycle; hub bars arrive `lag` s after their close, so each
    minute's first trade beats its bar. Long 07:55-08:45: 08:00 is charged once by 08:16:01."""
    _start(monkeypatch, "07:50")
    run = qa.paper(qa.flat_prices(60), side=1, perp=True, profile="aggressive", leave=55,
                   gone=every(60, step, offset), lag=int(lag * S))
    rows = run.store.funding("q146")
    assert len(charges(rows, SETTLE)) == 1, f"08:00 never charged: {rows}"
    assert min(booked_at(SETTLE)) <= SETTLE + WAIT + SLACK, booked_at(SETTLE)


# ============================== new head 2cc3c3e: _position_at's cache of the journal's fills


def test_position_at_cache_is_invalidated_by_a_late_or_out_of_order_fill_white_box():
    """White box on a real Store: _position_at(08:00) after each journal change. Fills arriving later with an earlier
    time (a held exit row written late) must still count; a fill at 08:00:00 exactly counts as after 08:00."""
    from types import SimpleNamespace

    from sleeve_fund.store import Store
    from sleeve_fund.strategies.base import LongFlatStrategy

    st = Store.in_memory()
    st.create_sleeve(name="c", strategy="probe", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                     starting_balance=10_000, risk_profile="aggressive", params={})
    ns = SimpleNamespace(runtime=SimpleNamespace(store=st, name="c"), _fills_read=None, _replayed_close=None,
                         _lot=lambda: 0.0001)
    at = lambda hms: utc(f"{DAY} {hms}").to_pydatetime()  # noqa: E731
    pos = lambda cur: LongFlatStrategy._position_at(ns, SETTLE.to_pydatetime(), cur)  # noqa: E731
    k = iter(range(100))

    def fill(side, q, hms):
        i = next(k)
        st.record_order("c", order_id=f"O{i}", side=side, qty=q, intent="entry", reason="", ts=at(hms))
        st.record_fill("c", side=side, qty=q, price=60_000.0, fee=0.0, order_id=f"O{i}", trade_id=f"T{i}", ts=at(hms))

    fill("BUY", 0.25, "07:55:00")
    assert pos(0.25) == pytest.approx(0.25)
    fill("SELL", 0.10, "08:03:00")  # a reduce after the settlement
    assert pos(0.15) == pytest.approx(0.25)
    fill("SELL", 0.05, "08:01:00")  # written after the 08:03 one, earlier in time
    assert pos(0.10) == pytest.approx(0.25)
    fill("BUY", 0.05, "07:59:30")  # a fill before the settlement written late: held at 08:00 then
    assert pos(0.15) == pytest.approx(0.30)
    fill("SELL", 0.30, "08:00:00")  # exactly at the settlement: after it
    assert pos(-0.15) == pytest.approx(0.30)
    ns._fills_read = None  # the same answers with no cache
    assert pos(-0.15) == pytest.approx(0.30)


def test_restart_journal_path_with_an_exit_filled_while_the_rate_is_awaited_charges_the_long(monkeypatch):
    """B restarts at 08:02 (A held 0.05 from 07:55, no fill since; 08:00 not booked), dense trades. The rate is not
    in until 08:10, so each tick works out 08:00's position from the journal (cached) and waits. B's own exit fills at
    08:05. When the rate arrives 08:00 must still be charged on the 0.05 held at 08:00. (Found: on the exit's fill,
    _apply_funding runs with the position already flat while the journal does not yet hold the fill, so 08:00 reads
    flat and _funding_since moves past it.)"""
    from sleeve_fund.strategies.base import LongFlatStrategy

    ready = utc(f"{DAY} 08:10").to_pydatetime()
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate",
                        lambda self, terms, ts, now, *_: None if now < ready else BASELINE)
    run, _ = restarted(monkeypatch, back_min=12, heartbeat_min=11, trades="dense", leave=15, prior=True)
    ex = [f for f in run.fills if f["side"] == "SELL"]
    assert ex and utc(ex[0]["ts"]) < utc(f"{DAY} 08:10"), ("set-up: B exits before the rate arrives", run.fills)
    got = charges(run.store.funding("q146"), SETTLE)
    assert len(got) == 1 and got[0]["qty"] == pytest.approx(0.05), run.store.funding("q146")


# ============================== a replayed stop in the minute that ends at the settlement


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_replayed_stop_in_the_minute_ending_at_the_settlement_matches_the_backtest(side, monkeypatch):
    """Dense trades; hub and trades away 07:56-08:05 (refills 08:05:05); the price goes 2% through the stop at
    07:59:30-07:59:50, in the minute to 08:00. The same minutes as a backtest: paper's 08:00 rows equal its."""
    start = _start(monkeypatch, "07:50")
    n = 40
    p = qa.shape(qa.flat_prices(n), 9.5, 9 + 50 / 60, qa.adverse(side, 0.02))
    m = qa.minutes_of(p)
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    res = run_backtest("probe", bars, inst, params={"enter": 5, "leave": 35, "side": side, "stop_loss": 0.01,
                                                     "market": "perp", "allow_short": True},
                       starting_capital=10_000, risk_profile="aggressive", bar_minutes=1,
                       half_spread=qa.SPREAD / 2 / qa.BASE)
    bt = [r for r in res.journal.funding_ if utc(r["ts"]) == SETTLE]
    run = qa.paper(p, side=side, perp=True, profile="aggressive", leave=35, gone=frozenset(range(6 * 60, 15 * 60)),
                   away=(start + 6 * M, start + 15 * M), back_at=start + 15 * M + 5 * S)
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    pp = rows_at(run, SETTLE)
    print("STOP-IN-LAST-MINUTE", side, "paper", pp, "backtest", bt)
    assert len(charges(pp, SETTLE)) == len(bt), ("paper", pp, "backtest", bt)
