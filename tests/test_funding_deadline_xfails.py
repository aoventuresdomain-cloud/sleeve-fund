"""FUNDING-DEADLINE (start gate 6, owner PE1): #146's funding deferral can withhold a perp's funding for good. Strict
xfails written by QA BEFORE the build, on main 0a5cd8f. The engineer makes each pass and removes its mark at hand-over;
the plain tests pin what already holds and must keep holding.

The mechanism on main (sleeve_fund/strategies/base.py):
- _apply_funding, lines 2505-2511 (added by #146, fac1a93, for QA P1-L19): on paper it returns before
  `_funding_since` moves on whenever `_awaiting` is set or the last trade is more than UNSEEN_GAP_NS (15 s, line 192)
  older than the clock. Nothing bounds that hold.
- _note_trade, lines 1585-1595: every trade more than 15 s after the one before it, while in a position, re-arms
  `_awaiting` for 90 s. It is cleared only by the next bar (lines 1638-1639), and by then the last trade is again more
  than 15 s old. A market trading every 30 s (the Advisor's ":10/:40" case) therefore holds every call, from the bar
  (line 1785) and from the tick (line 2884), for as long as it trades that way.
- The settlement is then lost for good: the hold lifts only when trades come faster, and the charge is worked out on
  the position at that moment (lines 2513-2518), so a position closed in the meantime pays nothing; a restart takes
  `_funding_since` from the later of the last funding row and the last fill (lines 709-713), which is past it.
- _funding_rate's own wait (FUNDING_WAIT, 15 min, lines 2538-2573, older than #146) is bounded: after it the
  baseline is charged. It is the "existing fallback": the venue's stored rate, else (paper) a direct venue request,
  else wait until settlement + 15 min, else the terms' fixed rate (0.01%) with one "funding_fallback" warning a run.
  A simulated perp (funding_venue None) always charges the fixed rate.

Rulings (advisor-rulings.md, 00:40 7 Oct, "funding deferral bound"): funding is owed on the position held at the
settlement instant; the 15-minute bound (FUNDING_DEFER_MAX) runs from the settlement and needs a post-settlement trade
to have arrived; it holds while the feed is away. Gate: the bound plus the diagnostic "funding_charged_while_flat"
when a replay finds an exit before a settlement already charged. The automatic reversal (its own journaled
correction, never an edit of the original row) may follow, before paper results are used as G2 evidence.
Pins named there: the 30 s trades at :10/:40, charged exactly once, long and short; an outage spanning the settlement
holds the charge until the refills land; a late refill with a pre-settlement stop yields the diagnostic (then the
reversal); never charged twice when the replay leaves the position unchanged.

Harnesses: test_hub_146_qa.py (on main in tests/: the hub-fed paper node, one trade a second unless `gone`, hub
away/refill, restart with a journaled position) on a simulated perp (Kraken, fixed 0.01%); o17_harness.py (copy it
from quant-review/o17-xfails/ beside this file: Binance's own settled rates with a publication time each).

ASSUMED INTERFACES (adapt the names, never the assertions):
- The booking time is read by wrapping Store.record_funding (any signature). If the build books through another Store
  method, add it to BOOKING_METHODS.
- EVENT ("funding_deadline_booked", PROVISIONAL: PE1's ca36307 journals no such event, so the name is QA's until PE1
  picks one): one event per settlement booked because its hold ran out, naming the settlement time (HH:MM).
- DIAG ("funding_charged_while_flat", the Advisor's 00:40 name, as PE1 built it): one event, naming the amount. The
  bound is PE1's LongFlatStrategy.FUNDING_DEFER_MAX (15 min); no test reads the constant.
- A correction row (the reversal; any true-up) carries a `kind` in CORRECTION_KINDS, or is told apart by being the
  second row on the settlement; a charge row is any other row.
- The sign a simulated perp's short pays is O17a's (baseline adverse to both sides), so short-side amounts here are
  pinned by size and by backtest == paper, never by sign.
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta

import pandas as pd
import pytest

import test_hub_146_qa as qa
from o17_harness import (  # noqa: F401  (fixtures are used by name)
    BASELINE,
    NEVER,
    _o17win,
    binance,
    kinds,
    utc,
    win,
)
from o17_harness import paper as o17_paper
from test_hub_146_qa import _probe  # noqa: F401 - registers the probe strategy

GATE = pytest.mark.xfail(strict=True, raises=AssertionError,
                         reason="QA FUNDING-DEADLINE (start gate 6, PE1): not built yet")
REVERSAL = pytest.mark.xfail(strict=True, raises=AssertionError,
                             reason="QA FUNDING-DEFER reversal (Advisor 00:40): may follow the gate; must land before "
                                    "paper results are used as G2 evidence")
TRUE_UP = pytest.mark.xfail(strict=True, raises=AssertionError,
                            reason="QA O17b true-up (rulings 76/82, with 00:40's never-edit rule): a rate arriving "
                                   "after the deadline booking is trued up by its own journaled row; not gate 6")

EVENT = "funding_deadline_booked"  # PROVISIONAL name: PE1 to confirm
DIAG = "funding_charged_while_flat"  # the Advisor's name (00:40)
CORRECTION_KINDS = {"true_up", "correction", "reversal", "funding_correction", "funding_reversal"}
BOOKING_METHODS = ("record_funding",)

DAY = "2025-10-03"
SETTLE = utc(f"{DAY} 08:00")
WAIT = timedelta(minutes=15)  # the bound: settlement + 15 min
SLACK = timedelta(seconds=61)  # one bar (or two 30 s ticks) after the bound
M, S = qa.M, qa.S

BOOKED: list[tuple[pd.Timestamp | None, dict]] = []  # (strategy clock when booked, the booking's keywords)


@pytest.fixture(autouse=True)
def _booked(monkeypatch):
    """Record the strategy's clock at each funding booking."""
    from sleeve_fund.store import Store
    from sleeve_fund.strategies.base import LongFlatStrategy

    BOOKED.clear()
    live: dict = {}
    on_start = LongFlatStrategy.on_start

    def started(self):
        live["s"] = self
        return on_start(self)

    monkeypatch.setattr(LongFlatStrategy, "on_start", started)
    for name in BOOKING_METHODS:
        orig = getattr(Store, name, None)
        if orig is None:
            continue

        def booking(self, *a, _orig=orig, **kw):
            s = live.get("s")
            BOOKED.append((utc(s.clock.utc_now()) if s is not None else None, kw))
            return _orig(self, *a, **kw)

        monkeypatch.setattr(Store, name, booking)


# ------------------------------------------------------------------------------------------------------- helpers


def _is_correction(r: dict) -> bool:
    return str(r.get("kind") or "").lower() in CORRECTION_KINDS


def charges(rows, ts) -> list[dict]:
    """The charge rows on the settlement at ts: not a correction by its `kind`, nor (with no `kind` column) a later
    row that exactly cancels an earlier one (a full reversal; a second charge has the same sign, never the opposite)."""
    out: list[dict] = []
    for r in sorted((r for r in rows if utc(r["ts"]) == ts), key=lambda r: (r.get("id") or 0)):
        if _is_correction(r):
            continue
        if r["amount"] and any(abs(c["amount"] + r["amount"]) <= 1e-8 for c in out):
            continue
        out.append(r)
    return out


def booked_at(ts) -> list[pd.Timestamp]:
    return [c for c, kw in BOOKED if c is not None and kw.get("ts") is not None and utc(kw["ts"]) == ts
            and not _is_correction(kw)]


def assert_once_by(rows, ts, by) -> dict:
    """The settlement at ts is charged exactly once, and booked no later than `by` (strategy clock)."""
    got = charges(rows, ts)
    assert len(got) == 1, (f"the {ts:%H:%M} settlement is charged {len(got)} times (main: the #146 hold never lifts, "
                           f"so it is never booked): {rows}")
    when = booked_at(ts)
    assert when, f"no booking time seen for {ts:%H:%M} (add the build's Store method to BOOKING_METHODS)"
    assert min(when) <= by, f"the {ts:%H:%M} settlement was booked at {min(when)}, after the bound {by}"
    return got[0]


def sparse(n_minutes: int, after_min: float = 0) -> frozenset:
    """Seconds with no trade: from `after_min` on, a trade only at :10 and :40 of each minute (30 s apart)."""
    return frozenset(s for s in range(int(after_min * 60), n_minutes * 60) if s % 60 not in (10, 40))


def _start(monkeypatch, hhmm: str) -> int:
    start = int(utc(f"{DAY} {hhmm}").value)
    monkeypatch.setattr(qa, "START", start)
    return start


def _ts(ns: int) -> pd.Timestamp:
    return pd.Timestamp(ns, unit="ns", tz="UTC")


def _cadence(monkeypatch, hours) -> None:
    from sleeve_fund import markets

    if hours is not None:
        monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=hours))


def hub_run(monkeypatch, hhmm="07:50", minutes=60, side=1, leave=55, trades="sparse", hours=None, **kw):
    """The hub-fed paper node on the simulated perp (fixed 0.01%), entered 5 minutes after the start."""
    _cadence(monkeypatch, hours)
    _start(monkeypatch, hhmm)
    p = kw.pop("prices", None)
    p = qa.flat_prices(minutes) if p is None else p
    gone = frozenset(kw.pop("gone", frozenset())) | (sparse(minutes) if trades == "sparse" else frozenset())
    return qa.paper(p, side=side, perp=True, profile="aggressive", leave=leave, gone=gone, **kw)


def entry_qty(run) -> float:
    return sum(f["qty"] for f in run.fills if any(o["order_id"] == f["order_id"] and o["intent"] == "entry"
                                                  for o in run.orders))


TRADES = [pytest.param("sparse", id="sparse"), pytest.param("dense", id="dense")]
SIDES = [pytest.param(1, id="long"), pytest.param(-1, id="short")]


# ============================================================ 1. the :10/:40 case: charged exactly once, by the bound


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("trades", TRADES)
def test_the_settlement_is_charged_once_by_settlement_plus_15_minutes(trades, side, monkeypatch):
    """Held 07:55-08:45; trades at :10 and :40 of each minute (sparse) or every second (dense, the control). Main,
    sparse: 08:00 is never charged, and the position closes at 08:45 with it unpaid."""
    run = hub_run(monkeypatch, side=side, trades=trades)
    row = assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)
    qty = entry_qty(run)
    assert abs(row["qty"]) == pytest.approx(qty), row
    assert abs(row["amount"]) == pytest.approx(qty * qa.BASE * BASELINE, rel=2e-3), row
    if side > 0:
        assert row["amount"] < 0, row  # a long pays a positive rate
    assert len(run.kinds("funding")) == 1, [e["kind"] for e in run.events]


@pytest.mark.parametrize("trades", TRADES)
def test_the_position_held_at_the_settlement_is_charged_though_it_closes_before_the_bound(trades, monkeypatch):
    """Held 07:55-08:02: the 08:00 settlement is owed on the long held at 08:00, though the position is flat by the
    bound (Advisor 00:40: owed on the position held at the settlement instant). Main, sparse: lost for good."""
    run = hub_run(monkeypatch, leave=12, trades=trades)
    assert [s[0] for s in run.sequence()] == ["entry", "exit"], run.sequence()
    row = assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)
    assert row["qty"] == pytest.approx(entry_qty(run)), row
    assert row["amount"] == pytest.approx(-entry_qty(run) * qa.BASE * BASELINE, rel=2e-3), row


@pytest.mark.parametrize("trades", TRADES)
def test_a_reversal_between_the_settlement_and_the_bound_charges_the_side_held_at_the_settlement(trades, monkeypatch):
    """Long 07:55-08:05, then short from 08:05 (a reversal): 08:00 is owed by the long, so it is paid, on the long's
    quantity. Found on PE1's ca36307 (sparse): booked at the bound on the short, as a credit."""
    run = hub_run(monkeypatch, leave=15, trades=trades, extra={"after": -1})
    assert [s[:2] for s in run.sequence()] == [("entry", "BUY"), ("exit", "SELL"), ("entry", "SELL")], run.sequence()
    row = assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)
    assert row["qty"] > 0 and row["amount"] < 0, row
    assert row["amount"] == pytest.approx(-row["qty"] * qa.BASE * BASELINE, rel=2e-3), row


# ======================================================================= 2. every cadence the engine can be given


CADENCES = [pytest.param((0, 8, 16), "07:50", 60, 55, id="8h"),
            pytest.param(tuple(range(0, 24, 4)), "11:50", 60, 55, id="4h"),  # 12:00 is no 8-hourly time
            pytest.param(tuple(range(24)), "08:50", 90, 85, id="1h")]  # 09:00 and 10:00


@pytest.mark.parametrize("hours, hhmm, minutes, leave", CADENCES)
@pytest.mark.parametrize("trades", TRADES)
def test_each_settlement_of_the_cadence_is_charged_once_by_its_own_bound(trades, hours, hhmm, minutes, leave,
                                                                         monkeypatch):
    """PerpTerms.funding_hours set to 8-, 4- or 1-hourly. Each settlement the long is held over is charged once,
    by its own settlement + 15 min, and no other time is charged."""
    from sleeve_fund import markets

    run = hub_run(monkeypatch, hhmm=hhmm, minutes=minutes, leave=leave, trades=trades, hours=hours)
    start = utc(f"{DAY} {hhmm}")
    want = [utc(t) for t in markets.funding_times((start + timedelta(minutes=5)).to_pydatetime(),
                                                  (start + timedelta(minutes=leave)).to_pydatetime(), hours)]
    assert want, "the window must hold a settlement"
    rows = run.store.funding("q146")
    for t in want:
        assert_once_by(rows, t, t + WAIT + SLACK)
    assert sorted({utc(r["ts"]) for r in rows}) == want, rows


# ============================================================================================== 3. restarts


def restart_run(monkeypatch, *, down: float, back: float, trades: str, seeded: bool = False):
    """The previous process held a long from 07:55 and sent its last heartbeat at `down` minutes after 07:50; the new
    one gets trades and live hub minutes from `back` on, and the store's minutes up to then. seeded: the journal
    already holds the 08:00 charge (the previous process booked it at its bound)."""
    from sleeve_fund.store import Store

    start = _start(monkeypatch, "07:50")
    n = 60
    p = qa.flat_prices(n)
    if seeded:
        create = Store.create_sleeve
        book = Store.record_funding

        def create_and_seed(self, *a, **kw):
            out = create(self, *a, **kw)
            book(self, "q146", qty=0.05, price=float(p[600]), rate=BASELINE,
                 amount=round(-0.05 * float(p[600]) * BASELINE, 8), ts=SETTLE.to_pydatetime())
            return out

        monkeypatch.setattr(Store, "create_sleeve", create_and_seed)
    gone = frozenset(range(0, int(back * 60))) | (sparse(n, back) if trades == "sparse" else frozenset())
    run = qa.paper(p, side=1, perp=True, profile="aggressive", leave=55, gone=gone,
                   lost={start + k * M for k in range(0, int(back) + 1)}, held=(0.05, float(p[300]), start + 5 * M),
                   heartbeat=start + int(down * M), history=qa.stored_history(p, start + int(back * M)))
    return run, _ts(start + int(back * M))


@pytest.mark.parametrize("trades", TRADES)
def test_a_restart_inside_the_hold_window_still_books_by_the_bound(trades, monkeypatch):
    """Last heartbeat 08:05 (settled, not yet booked), back at 08:07: 08:00 is booked once by 08:16."""
    run, _ = restart_run(monkeypatch, down=15, back=17, trades=trades)
    row = assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)
    assert row["qty"] == pytest.approx(0.05), row


@pytest.mark.parametrize("trades", TRADES)
def test_a_restart_after_the_bound_books_on_its_first_trades(trades, monkeypatch):
    """Down 07:58 to 08:25, past the bound: 08:00 is booked once, within 75 s of the restart."""
    run, back = restart_run(monkeypatch, down=8, back=35, trades=trades)
    assert_once_by(run.store.funding("q146"), SETTLE, back + timedelta(seconds=75))


@pytest.mark.parametrize("trades", ["sparse", "dense"])
def test_a_restart_after_the_deadline_booking_does_not_book_it_again(trades, monkeypatch):
    """Already true on main; guards the build: the journal holds the 08:00 charge, heartbeat 08:16, back 08:20."""
    run, _ = restart_run(monkeypatch, down=26, back=30, trades=trades, seeded=True)
    assert len(charges(run.store.funding("q146"), SETTLE)) == 1, run.store.funding("q146")


# ============================================================================ 4. a data outage spanning the bound


def outage_run(monkeypatch, *, trades: str, stop: bool):
    """Trades and hub minutes away 07:56-08:25 (across 08:00 and its 08:15 bound); minutes refilled at 08:25:05;
    then trades every second (dense) or at :10/:40 (sparse). stop: the price goes 2% through the long's 1% stop at
    07:57:30-07:57:50, inside the outage."""
    start = _start(monkeypatch, "07:50")
    n = 60
    p = qa.flat_prices(n)
    if stop:
        p = qa.shape(p, 7.5, 7 + 50 / 60, qa.adverse(1, 0.02))
    gone = frozenset(range(6 * 60, 35 * 60)) | (sparse(n, 35) if trades == "sparse" else frozenset())
    back = start + 35 * M + 5 * S
    run = qa.paper(p, side=1, perp=True, profile="aggressive", leave=55, gone=gone,
                   away=(start + 6 * M, start + 35 * M), back_at=back)
    return run, _ts(back)


@pytest.mark.parametrize("trades", TRADES)
def test_an_outage_across_the_bound_holds_the_charge_then_books_it_once_on_return(trades, monkeypatch):
    """Advisor 00:40: the bound holds while the feed is away. Nothing is booked before the refills land at 08:25:05;
    08:00 is booked once within 75 s of them."""
    run, back = outage_run(monkeypatch, trades=trades, stop=False)
    early = [c for c in booked_at(SETTLE) if c < back]
    assert not early, f"08:00 booked while the feed was away, at {early}"
    assert_once_by(run.store.funding("q146"), SETTLE, back + timedelta(seconds=75))


@pytest.mark.parametrize("trades", ["sparse", "dense"])
def test_a_stop_inside_the_outage_before_the_settlement_is_never_charged_by_the_bound(trades, monkeypatch):
    """Already true on main (P1-L19); guards the build: the bound must not book 08:00 during the outage, because the
    refills show the venue's stop closed the long at 07:58."""
    run, _ = outage_run(monkeypatch, trades=trades, stop=True)
    assert qa.first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    assert charges(run.store.funding("q146"), SETTLE) == [], run.store.funding("q146")
    assert not run.kinds(DIAG)


# ================================================== 5. a late refill (Advisor 00:40): once, the diagnostic, reversal


def refill_run(monkeypatch, *, trades: str, stop: bool):
    """No trade 07:57:20-07:58:00 and hub minutes to 07:58 and 07:59 refilled only at 08:20:05 (past the bound);
    trades otherwise every second, or at :10/:40 from 07:58 (sparse). stop: 2% through the 1% stop at
    07:57:30-07:57:50, in the minutes the refill brings."""
    start = _start(monkeypatch, "07:50")
    n = 60
    p = qa.flat_prices(n)
    if stop:
        p = qa.shape(p, 7.5, 7 + 50 / 60, qa.adverse(1, 0.02))
    gone = frozenset(range(7 * 60 + 20, 8 * 60)) | (sparse(n, 8) if trades == "sparse" else frozenset())
    return qa.paper(p, side=1, perp=True, profile="aggressive", leave=55, gone=gone,
                    away=(start + 7 * M, start + 9 * M), back_at=start + 30 * M + 5 * S)


@pytest.mark.parametrize("trades", TRADES)
def test_a_late_refill_that_leaves_the_position_unchanged_never_charges_twice(trades, monkeypatch):
    run = refill_run(monkeypatch, trades=trades, stop=False)
    rows = run.store.funding("q146")
    assert_once_by(rows, SETTLE, SETTLE + WAIT + SLACK)
    assert len([r for r in rows if utc(r["ts"]) == SETTLE]) == 1, rows  # no correction either


def _charged_then_found_flat(monkeypatch):
    run = refill_run(monkeypatch, trades="dense", stop=True)
    ex = qa.first_exit(run.sequence())
    assert ex[0] == "stop_loss" and run.kinds("outage_exit"), run.sequence()
    assert "07:58" in run.kinds("outage_exit")[0]["message"], run.kinds("outage_exit")
    row = assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)  # booked before the refill came
    return run, row


def test_a_late_refill_finding_a_stop_before_a_charged_settlement_journals_the_diagnostic(monkeypatch):
    """The 08:00 charge is booked by its bound; the refill at 08:20:05 shows the stop closed the long in the minute to
    07:58. One DIAG event names the amount; the charge row is never edited. Main: no diagnostic."""
    run, row = _charged_then_found_flat(monkeypatch)
    got = run.kinds(DIAG)
    assert len(got) == 1, f"no single {DIAG} event: {[e['kind'] for e in run.events]}"
    assert f"{abs(row['amount']):,.2f}" in got[0]["message"], got[0]
    assert row["amount"] == pytest.approx(-row["qty"] * row["price"] * BASELINE, rel=1e-6), row


def test_a_late_refill_finding_a_stop_before_a_charged_settlement_reverses_it_in_full(monkeypatch):
    """Option (b), Advisor 00:40: the charge is reversed in full as its own journaled row; the original stays."""
    run, row = _charged_then_found_flat(monkeypatch)
    rows = [r for r in run.store.funding("q146") if utc(r["ts"]) == SETTLE]
    assert len(rows) == 2, f"no reversal row beside the 08:00 charge: {rows}"
    fix = next(r for r in rows if r is not row and r.get("id") != row.get("id"))
    assert fix["amount"] == pytest.approx(-row["amount"], abs=1e-8), rows
    assert sum(r["amount"] for r in rows) == pytest.approx(0, abs=1e-8), rows


# ======================================================================================== 6. backtest == paper


def backtest_funding(monkeypatch, *, hhmm, minutes, leave, side, hours):
    from sleeve_fund.instruments import BOOK_SHARE
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    _cadence(monkeypatch, hours)
    _start(monkeypatch, hhmm)
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {"enter": 5, "leave": leave, "side": side, "stop_loss": 0.01, "market": "perp", "allow_short": True}
    m = qa.minutes_of(qa.flat_prices(minutes))
    bars = m.set_axis(m.index + pd.Timedelta(minutes=1))
    bars["volume"] = 60 / BOOK_SHARE
    res = run_backtest("probe", bars, inst, params=params, starting_capital=10_000, risk_profile="aggressive",
                       bar_minutes=1, half_spread=qa.SPREAD / 2 / qa.BASE)
    return list(res.journal.funding_)


@pytest.mark.parametrize("hours, hhmm, minutes, leave", [CADENCES[0], CADENCES[2]])
@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("trades", TRADES)
def test_paper_books_the_settlements_the_backtest_books(trades, side, hours, hhmm, minutes, leave, monkeypatch):
    """The same minutes as a backtest and as paper: the same settlements, quantities and rates, amounts within
    0.1% (paper books at its bound's price)."""
    bt = backtest_funding(monkeypatch, hhmm=hhmm, minutes=minutes, leave=leave, side=side, hours=hours)
    assert bt, "the backtest must charge a settlement"
    run = hub_run(monkeypatch, hhmm=hhmm, minutes=minutes, side=side, leave=leave, trades=trades, hours=hours)
    pp = sorted((r for r in run.store.funding("q146") if not _is_correction(r)), key=lambda r: r["ts"])
    assert [utc(r["ts"]) for r in pp] == [utc(r["ts"]) for r in bt], f"paper {pp} != backtest {bt}"
    for a, b in zip(pp, bt):
        assert a["qty"] == pytest.approx(b["qty"]) and a["rate"] == pytest.approx(b["rate"]), (a, b)
        assert a["amount"] == pytest.approx(b["amount"], rel=1e-3), (a, b)


# ============================================== 7. the venue's own rates: the best rate available by the bound


RATE = 0.0003
PUBLISHED = [pytest.param("08:00", RATE, id="published-at-settlement"),
             pytest.param("08:10", RATE, id="published-0810"),
             pytest.param("08:30", BASELINE, id="published-after-the-bound"),
             pytest.param("never", BASELINE, id="never-published")]
O17_TRADES = [pytest.param(30, id="sparse"), pytest.param(5, id="dense")]


# Stop safety (#182): the open-risk limit refuses a stopless 3x perp entry (aggressive profile, as o17_harness runs it).
# These cells are about funding, not liquidation, so they run guarded: a 1 % stop, inside half the distance to
# liquidation, open risk about 300 on the 10,000 book (under 5 %), and never reached on the harness's rising ramp.
GUARDED_STOP = 0.01


def guarded(params: dict) -> dict:
    return {**params, "stop_loss": GUARDED_STOP}


def native_run(tmp_path, monkeypatch, binance, *, step, published):
    """Binance's own perp, long 07:55-08:45; the 08:00 rate is 0.03%, published when `published` says; a trade every
    `step` s from 07:50:10 to 08:50 (30 s: at :10 and :40)."""
    pub = {f"{DAY} 08:00": NEVER if published == "never" else utc(f"{DAY} {published}")}
    return o17_paper(tmp_path, monkeypatch, binance, guarded(win((f"{DAY} 07:55", f"{DAY} 08:45", 1))), f"{DAY} 07:50:10",
                     60, published=pub, rates={f"{DAY} 08:00": RATE}, step=step)


@pytest.mark.parametrize("published, want", PUBLISHED)
@pytest.mark.parametrize("step", O17_TRADES)
def test_the_bound_books_the_best_rate_available_by_then_exactly_once(step, published, want, tmp_path, monkeypatch,
                                                                      binance):
    """The existing fallback, unchanged: the venue's rate if it is there by settlement + 15 min, else the 0.01%
    baseline. A rate published after the bound (08:30) books no second charge (whether it corrects the first is
    the true-up test at the end)."""
    out = native_run(tmp_path, monkeypatch, binance, step=step, published=published)
    row = assert_once_by(out["funding"], SETTLE, SETTLE + WAIT + SLACK)
    assert row["rate"] == pytest.approx(want), row
    assert row["amount"] < 0, row  # a long, a positive rate either way


MULTI = {f"{DAY} 00:00": 0.0003, f"{DAY} 08:00": -0.0002, f"{DAY} 16:00": 0.0001}


@pytest.mark.parametrize("step", [pytest.param(30, id="30s-apart"), pytest.param(60, id="60s-apart"),
                                  pytest.param(10, id="10s-apart")])
def test_trades_30_or_60_s_apart_book_every_settlement_once_by_its_bound(step, tmp_path, monkeypatch, binance):
    """HoE / DA addition: on main the 15 s unseen-trade hold withholds ALL funding when trades come 30 s apart (and
    60 s apart). Binance's own perp, long 23:55 to 16:45 over the 00:00, 08:00 and 16:00 settlements (+0.03%, -0.02%,
    +0.01%, each published at its settlement); a trade every `step` s from 23:50:10. Each settlement is charged exactly
    once, at its own rate, booked no later than its settlement + 15 min (+ 61 s). 10 s apart is the control."""
    out = o17_paper(tmp_path, monkeypatch, binance, guarded(win((f"2025-10-02 23:55", f"{DAY} 16:45", 1))),
                    "2025-10-02 23:50:10", 16 * 60 + 55, rates=MULTI, step=step)
    assert out["fills"] and out["fills"][0][1] == "BUY", out["fills"]
    qty = out["fills"][0][2]
    assert sorted({r["ts"] for r in out["funding"]}) == sorted(utc(t) for t in MULTI), (
        f"settlements charged {sorted({str(r['ts']) for r in out['funding']})} of {sorted(MULTI)} "
        f"(main: the #146 hold withholds every one): {out['funding']}")
    for t, rate in MULTI.items():
        row = assert_once_by(out["funding"], utc(t), utc(t) + WAIT + SLACK)
        assert row["rate"] == pytest.approx(rate) and row["qty"] == pytest.approx(qty), row
        assert row["amount"] == pytest.approx(-qty * row["price"] * rate, rel=1e-6), row


# ============================================================================================ 8. the journal event


def test_a_booking_made_at_the_bound_is_journaled_by_name(monkeypatch):
    """One EVENT naming the 08:00 settlement, beside the usual "funding" event. The name is provisional."""
    run = hub_run(monkeypatch, trades="sparse")
    assert_once_by(run.store.funding("q146"), SETTLE, SETTLE + WAIT + SLACK)
    got = run.kinds(EVENT)
    assert len(got) == 1, f"no single {EVENT} event: {sorted({e['kind'] for e in run.events})}"
    assert "08:00" in got[0]["message"], got[0]
    assert len(run.kinds("funding")) == 1, [e["kind"] for e in run.events]


def test_a_booking_made_on_time_journals_no_deadline_event(monkeypatch):
    """Already true on main; guards the build: a settlement booked straight away is not a deadline booking."""
    run = hub_run(monkeypatch, trades="dense")
    assert len(charges(run.store.funding("q146"), SETTLE)) == 1
    assert not run.kinds(EVENT), run.kinds(EVENT)


# ======================================================= 9. the true rate after the deadline booking (was Needs-Advisor)


@TRUE_UP
def test_a_true_rate_arriving_after_the_deadline_booking_is_trued_up_by_its_own_row(tmp_path, monkeypatch, binance):
    """The 08:00 rate (0.03%) is published at 08:30, after the bound booked the 0.01% baseline. Rulings 76/82 (O17):
    the baseline is trued up to the actual rate by a journaled correction, its own row on the settlement it corrects,
    at the same qty and mark, amount = actual - baseline; 00:40: never by editing the original row. Main (dense
    trades) books the baseline at 08:15 and does nothing at 08:30."""
    out = native_run(tmp_path, monkeypatch, binance, step=5, published="08:30")
    rows = sorted((r for r in out["funding"] if r["ts"] == SETTLE), key=lambda r: r["id"])
    assert rows, out["funding"]
    row = rows[0]  # the deadline booking
    assert row["rate"] == pytest.approx(BASELINE), row
    when = booked_at(SETTLE)
    assert when and min(when) <= SETTLE + WAIT + SLACK, when
    assert len(rows) == 2, f"no true-up row after the 08:30 rate (left as booked): {rows}"
    fix = rows[1]
    assert (fix["qty"], fix["price"]) == (row["qty"], row["price"]), rows
    assert fix["amount"] == pytest.approx(-row["qty"] * row["price"] * (RATE - BASELINE), rel=1e-6), rows
    assert sum(r["amount"] for r in rows) == pytest.approx(-row["qty"] * row["price"] * RATE, rel=1e-6), rows
