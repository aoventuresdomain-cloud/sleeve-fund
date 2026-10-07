"""O17b, missing funding rate on paper: block entries and adds once the 15-minute venue check fails (exits always run),
flat strategies included; when the block lifts; the journalled true-up at the settlement-time mark; the never-arriving
rate ("estimated", incident after 7 days). Owner: Platform Engineer 2 (board split, HoE 17:52). Strict xfails written
by QA BEFORE the build on main c7a73f1; the engineer makes each pass and removes its mark before hand-over. Plain tests
pin what already holds and must keep holding.

Rulings pinned (advisor-rulings.md; the latest wins where they differ):
- line 76 (Advisor 6 Oct 17:52): "Paper: baseline on held positions (adverse), block new entries/adds on that perp
  after the 15-min venue check fails; true-up to actual rate via journalled correction".
- lines 81-82 (O17 QA points ~18:30 and O17 open points 18:18, to QA; same content): a flat strategy is blocked on that
  perp too; the block lifts when the missing rate is STORED and the 15-minute check passes (the true-up follows); true-up
  after close OK, = actual amount - baseline amount (short: +n(r+b)); the correction and the re-based baseline use the
  settlement-time mark; a rate that never arrives stays baseline, marked "estimated", and an incident opens after 7 days.
- line 80 (O17a review ~18:25): the paper entry block and journalled true-up are a precondition for any perp strategy on
  paper (a deployment gate: not testable here, see README).
- line 65 (5 Oct): "an ENTRY can be skipped, an EXIT cannot ... stops, TPs and exits keep working"; line 63: "A
  reversal does the close part only."

Today (c7a73f1, strategies/base.py _funding_rate ~1892): paper asks the venue for up to FUNDING_WAIT (15 min), then
charges the fixed 0.01% at the price of the tick that charges it (a long pays, a short is credited) with one
"funding_fallback" warning per run. Nothing is blocked, the baseline is never corrected, and nothing opens an incident.

Adds: on c7a73f1 a perp strategy is all or nothing (target weights are refused on a perp, base.py ~291), so there is no
add to drive here. The reversal test stands in for "entries and adds"; when adds reach perps (P2-1 fraction steps) the
same gate must cover them.

ASSUMED INTERFACES (adapt the names, never the assertions):
- Funding rows carry `kind` ("settled" | "baseline" | "true_up"), as in test_o17a_xfails.py. The baseline row is
  charged at the settlement-time mark (the last trade at or before the settlement). The true-up is its own row,
  stamped with the settlement it corrects, at the same qty and mark, amount = actual amount - baseline amount; the
  baseline row stays (the journal is append-only).
- Events: "funding_true_up" (one per correction); "funding_entry_blocked" (a warning when the gate holds an entry,
  once per episode, as base.py's _note); "incident" (level error, so it reaches the alerts inbox; once, 7 days after a settlement whose rate
  never arrived). A baseline still standing is "estimated": its "funding" event says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from o17_harness import (  # noqa: F401  (fixtures are used by name)
    BASELINE,
    NEVER,
    PAIR,
    _no_swallowed_strategy_errors,
    _o17win,
    binance,
    journal,
    kinds,
    mark_at,
    paper,
    utc,
    win,
)

REASON = "QA O17b: not built yet (Advisor 17:52)"
xfail = pytest.mark.xfail(strict=True, reason=REASON)

START = "2025-10-03 07:50"
S = utc("2025-10-03 08:00")  # the settlement whose rate goes missing
CHECK = utc("2025-10-03 08:15")  # when paper's 15-minute venue check gives up


def _entries_after(out, t0, t1=None):
    """Fills that open or add (the signal's own entries) between t0 and t1: after a close, any fill is an entry."""
    return [f for f in out["fills"] if f[0] > utc(t0) and (t1 is None or f[0] < utc(t1))]


# ---------------------------------------------------------------------------------------------------------------
# 1. Entries blocked after the check fails; exits run
# ---------------------------------------------------------------------------------------------------------------

@xfail
@pytest.mark.strategy_errors
@pytest.mark.parametrize("how", ["never-published", "nan-from-the-store"])
def test_paper_blocks_a_new_entry_after_the_venue_check_fails_but_the_exit_runs(tmp_path, monkeypatch, binance, how,
                                                                                capfd):
    """Long 07:52 -> 08:20, then long again 08:25 -> 08:40. The 08:00 rate never arrives (or is kept as NaN, which is
    missing too). The 08:20 exit runs; the 08:25 entry is held back, and the journal says why. (Today a NaN rate is
    charged as NaN, and on Postgres the NaN in the journal also breaks _on_tick: checked here, not by the guard.)"""
    kw = ({"published": {S: NEVER}} if how == "never-published"
          else {"rates": {S: float("nan")}, "published": {S: "2025-10-03 08:05"}})
    out = paper(tmp_path, monkeypatch, binance,
                win(("2025-10-03 07:52", "2025-10-03 08:20", 1), ("2025-10-03 08:25", "2025-10-03 08:40", 1)),
                START, 50, **kw)
    sides = [(t, s) for t, s, _ in out["fills"]]
    assert sides[:2] == [(utc("2025-10-03 07:52"), "BUY"), (utc("2025-10-03 08:20"), "SELL")]  # the exit runs
    assert _entries_after(out, "2025-10-03 08:20") == [], "an entry went through after the check failed"
    (blocked, *_) = kinds(out["events"], "funding_entry_blocked")
    assert blocked["level"] == "warning" and CHECK <= blocked["ts"] <= utc("2025-10-03 08:26")
    assert not any(f[2] != f[2] for f in out["fills"]) and all(r["amount"] == r["amount"] for r in out["funding"])
    logged = capfd.readouterr()
    assert "strategy handler " not in logged.out + logged.err


@xfail
def test_paper_does_only_the_close_part_of_a_reversal(tmp_path, monkeypatch, binance):
    """Long 07:52 -> 08:20, then short 08:20 -> 08:40; 08:00 never published. The long is closed at 08:20; the short
    (a new entry) is not opened."""
    out = paper(tmp_path, monkeypatch, binance,
                win(("2025-10-03 07:52", "2025-10-03 08:20", 1), ("2025-10-03 08:20", "2025-10-03 08:40", -1)),
                START, 50, published={S: NEVER})
    sides = [(t, s) for t, s, _ in out["fills"]]
    assert sides[:2] == [(utc("2025-10-03 07:52"), "BUY"), (utc("2025-10-03 08:20"), "SELL")]
    assert out["fills"][1][2] == pytest.approx(out["fills"][0][2])  # the close is the whole long, no more
    assert len(out["fills"]) == 2, f"a short was opened: {out['fills'][2:]}"


@xfail
def test_a_stop_still_exits_while_entries_are_blocked(tmp_path, monkeypatch, binance):
    """Short 07:52 with a 0.5% stop; 08:00 never published; the price jumps 1% at 08:22, through the stop. The stop
    exit runs while entries are blocked; the 08:30 re-entry (after the signal has gone flat at 08:28) does not."""
    params = win(("2025-10-03 07:52", "2025-10-03 08:28", -1), ("2025-10-03 08:30", "2025-10-03 08:45", -1))
    jump = 32 * 60  # 08:22:00
    out = paper(tmp_path, monkeypatch, binance, {**params, "stop_loss": 0.005}, START, 55, published={S: NEVER},
                price=lambda s: 60_000 + s * 0.01 + (600 if s >= jump else 0))
    assert out["fills"][0][:2] == (utc("2025-10-03 07:52"), "SELL")
    stop_t, side, _ = out["fills"][1]
    assert side == "BUY" and utc("2025-10-03 08:22") <= stop_t < utc("2025-10-03 08:23")  # the stop closed it
    assert _entries_after(out, stop_t) == [], "the 08:30 re-entry went through"


def test_paper_does_not_block_when_the_venue_publishes_within_15_minutes(tmp_path, monkeypatch, binance):
    """Plain: the 08:00 rate (0) arrives at 08:10, inside the check. Nothing is blocked: the 08:25 entry fills."""
    out = paper(tmp_path, monkeypatch, binance,
                win(("2025-10-03 07:52", "2025-10-03 08:20", 1), ("2025-10-03 08:25", "2025-10-03 08:40", 1)),
                START, 55, rates={S: 0.0}, published={S: "2025-10-03 08:10"})
    assert [(t, s) for t, s, _ in out["fills"]] == [
        (utc("2025-10-03 07:52"), "BUY"), (utc("2025-10-03 08:20"), "SELL"),
        (utc("2025-10-03 08:25"), "BUY"), (utc("2025-10-03 08:40"), "SELL")]
    assert not kinds(out["events"], "funding_entry_blocked")


# ---------------------------------------------------------------------------------------------------------------
# 2. The true-up: journalled, and P&L moves by exactly the difference
# ---------------------------------------------------------------------------------------------------------------

CASES = [(1, 0.0003), (1, -0.0002), (-1, 0.0003), (-1, -0.0002), (-1, 0.0)]


@xfail
@pytest.mark.parametrize("side, actual", CASES, ids=["long+3bp", "long-2bp", "short+3bp", "short-2bp", "short-zero"])
def test_the_true_up_is_journalled_and_moves_pnl_by_exactly_the_difference(tmp_path, monkeypatch, binance, side, actual):
    """Held 07:52 -> 08:50 on `side`; the venue publishes 08:00's `actual` rate at 08:30 (store and venue), after the
    baseline was charged at the 08:15 check. Against the same run where it is never published:
    - the baseline row is at the settlement-time mark (08:00:00's trade, 60,006.0), not the 08:15 price;
    - one true_up row for 08:00 at the same qty and mark, amount = (-qty x mark x actual) - (-|qty| x mark x 0.01%):
      for a long -(actual - baseline) x notional, for a short +(actual + baseline) x notional;
    - the 08:00 rows sum to what the actual rate owes at the settlement-time mark;
    - one funding_true_up event; the last equity mark differs from the never-published run by exactly the correction."""
    params = win(("2025-10-03 07:52", "2025-10-03 08:50", side))
    late = paper(tmp_path, monkeypatch, binance, params, START, 65, rates={S: actual},
                 published={S: "2025-10-03 08:30"})
    never = paper(tmp_path, monkeypatch, binance, params, START, 65, rates={S: actual}, published={S: NEVER})
    mark = mark_at(S, START)
    assert len([r for r in late["funding"] if r["ts"] == S]) == 2, f"no correction journalled: {late['funding']}"
    (base,) = [r for r in late["funding"] if r["kind"] == "baseline"]
    assert base["ts"] == S and np.sign(base["qty"]) == side
    assert base["price"] == pytest.approx(mark, abs=0.15), "the baseline is not at the settlement-time mark"
    assert base["amount"] == pytest.approx(-abs(base["qty"]) * mark * BASELINE, abs=1e-5)  # adverse, either side
    owed = -base["qty"] * mark * actual
    (fix,) = [r for r in late["funding"] if r["kind"] == "true_up"]
    assert fix["ts"] == S and fix["qty"] == base["qty"] and fix["price"] == pytest.approx(mark, abs=0.15)
    assert fix["amount"] == pytest.approx(owed - base["amount"], abs=1e-5)
    assert sum(r["amount"] for r in late["funding"] if r["ts"] == S) == pytest.approx(owed, abs=1e-5)
    assert len(kinds(late["events"], "funding_true_up")) == 1
    assert [r["kind"] for r in never["funding"]] == ["baseline"]
    assert late["equity"]["equity"] - never["equity"]["equity"] == pytest.approx(fix["amount"], abs=1e-6)


@xfail
def test_the_true_up_still_comes_after_the_position_is_closed(tmp_path, monkeypatch, binance):
    """Short 07:52 -> 08:20; baseline charged at 08:15; flat from 08:20; the rate (+0.03%) is published at 08:30. The
    short was owed +0.03% and was charged -0.01%: the correction, +n(r + b) at the settlement-time mark, is booked
    although nothing is held any more."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 08:20", -1)), START, 45,
                rates={S: 0.0003}, published={S: "2025-10-03 08:30"})
    assert len(out["funding"]) == 2, f"no correction journalled: {out['funding']}"
    (base,) = [r for r in out["funding"] if r["kind"] == "baseline"]
    (fix,) = [r for r in out["funding"] if r["kind"] == "true_up"]
    n = abs(base["qty"]) * mark_at(S, START)
    assert fix["amount"] == pytest.approx(n * (0.0003 + BASELINE), abs=1e-5)


# ---------------------------------------------------------------------------------------------------------------
# 3. When the block lifts: the rate stored AND the 15-minute check passing; a flat strategy is blocked too
# ---------------------------------------------------------------------------------------------------------------

@xfail
def test_a_flat_strategy_is_blocked_on_that_perp_too(tmp_path, monkeypatch, binance):
    """Flat over 08:00, whose rate never arrives; the strategy wants in at 08:20 (after the 08:15 check failed). The
    entry is held back although the strategy itself paid nothing (18:18)."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 08:20", "2025-10-03 08:40", 1)), START, 45,
                published={S: NEVER})
    assert out["fills"] == [], f"a flat strategy entered while 08:00 was missing: {out['fills']}"
    assert kinds(out["events"], "funding_entry_blocked")


@xfail
def test_the_block_lifts_once_the_rate_is_stored_and_trued_up(tmp_path, monkeypatch, binance):
    """Long 07:52 -> 08:20, long 08:22 -> 08:26 (blocked: 08:00 still missing), long 08:40 -> 08:50 (08:00 stored and
    published at 08:30, so the check passes and the true-up follows: allowed)."""
    out = paper(tmp_path, monkeypatch, binance,
                win(("2025-10-03 07:52", "2025-10-03 08:20", 1), ("2025-10-03 08:22", "2025-10-03 08:26", 1),
                    ("2025-10-03 08:40", "2025-10-03 08:50", 1)),
                START, 60, rates={S: 0.0002}, published={S: "2025-10-03 08:30"})
    assert _entries_after(out, "2025-10-03 08:20", "2025-10-03 08:30") == []
    assert (utc("2025-10-03 08:40"), "BUY") in [(t, s) for t, s, _ in out["fills"]]
    assert [r["kind"] for r in out["funding"]] == ["baseline", "true_up"]


@xfail
def test_the_venue_answering_is_not_enough_until_the_rate_is_stored(tmp_path, monkeypatch, binance):
    """The venue answers for 08:00 from 08:30 but the collector stores it only at 08:50. The 08:40 entry is still held
    back (not stored); the 08:55 entry goes through (stored, and the check passes)."""
    out = paper(tmp_path, monkeypatch, binance,
                win(("2025-10-03 07:52", "2025-10-03 08:20", 1), ("2025-10-03 08:40", "2025-10-03 08:45", 1),
                    ("2025-10-03 08:55", "2025-10-03 09:05", 1)),
                START, 80, rates={S: 0.0002}, published={S: "2025-10-03 08:30"}, stored={S: "2025-10-03 08:50"})
    assert _entries_after(out, "2025-10-03 08:20", "2025-10-03 08:50") == [], "entered before the rate was stored"
    assert (utc("2025-10-03 08:55"), "BUY") in [(t, s) for t, s, _ in out["fills"]]


# ---------------------------------------------------------------------------------------------------------------
# 4. A rate that never arrives: "estimated" for good, and an incident after 7 days
# ---------------------------------------------------------------------------------------------------------------

@xfail
def test_a_rate_that_never_arrives_stays_estimated_and_opens_an_incident_after_7_days(tmp_path, monkeypatch, binance):
    """Long from 07:56 for 7 days and 3 hours; every settlement's rate arrives on time except 08:00 on 3 Oct, which
    never does. A trade every 5 minutes and a 5-minute tick keep the replay short. The 08:00 charge stays baseline and
    its "funding" event says "estimated"; one incident event (kind "incident", level error, naming funding; ~19:20), opened 7 days after 08:00 (within the hour),
    and none before."""
    end = S + pd.Timedelta(days=7, hours=3)
    rates = {t: 0.0001 for t in pd.date_range(S, end, freq="8h")}
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2026-01-01", 1)), START,
                int((end - utc(START)).total_seconds() // 60), rates=rates, published={S: NEVER}, step=300,
                tick_seconds=300)
    rows = [r for r in out["funding"] if r["ts"] == S]
    assert len(rows) == 1 and len(out["funding"]) >= 20  # setup: every later settlement was charged
    assert rows[0].get("kind") == "baseline"
    charged = [e for e in kinds(out["events"], "funding") if e["ts"] == S]
    assert charged and "estimated" in charged[-1]["message"].lower()
    (incident,) = kinds(out["events"], "incident")
    assert incident["level"] == "error" and "funding" in incident["message"].lower()
    assert S + pd.Timedelta(days=7) <= incident["ts"] <= S + pd.Timedelta(days=7, hours=1)


# ---------------------------------------------------------------------------------------------------------------
# 5. Adds, and the venue shortening the funding interval mid-run (Advisor 18:39): entries and adds blocked on that perp
#    until DA-11 lands; exits untouched; the same alert (funding_stale), once per instrument per episode
# ---------------------------------------------------------------------------------------------------------------

DAY = "2025-10-03"
# Stored and published on time: the 8-hour schedule up to 08:00, then the venue settles at 12:00 too (every 4 hours).
EIGHT_THEN_FOUR = {"2025-10-02 16:00": 0.0001, f"{DAY} 00:00": 0.0001, f"{DAY} 08:00": 0.0001,
                   f"{DAY} 12:00": 0.0002}
EIGHT_ONLY = {k: v for k, v in EIGHT_THEN_FOUR.items() if not k.endswith("12:00")}
LATE = "2025-10-03 11:50"  # runs start here: the last step before the run is 8 hours, so the start is allowed (18:35)


@xfail
def test_a_shortened_interval_blocks_the_next_entry_and_the_exit_still_runs(tmp_path, monkeypatch, binance):
    """Long 11:52 -> 12:20, long again 12:35 -> 12:45. The venue settles at 12:00, four hours after 08:00. The 12:20 exit
    runs; the 12:35 entry is held back (the block holds until DA-11, so for the rest of the run)."""
    out = paper(tmp_path, monkeypatch, binance,
                win((f"{DAY} 11:52", f"{DAY} 12:20", 1), (f"{DAY} 12:35", f"{DAY} 12:45", 1)),
                LATE, 60, rates=EIGHT_THEN_FOUR)
    sides = [(t, s) for t, s, _ in out["fills"]]
    assert sides[:2] == [(utc(f"{DAY} 11:52"), "BUY"), (utc(f"{DAY} 12:20"), "SELL")]  # the exit runs
    assert _entries_after(out, f"{DAY} 12:20") == [], "an entry went through after the interval shortened"
    assert kinds(out["events"], "funding_entry_blocked")


def test_no_block_while_the_interval_stays_at_eight_hours(tmp_path, monkeypatch, binance):
    """Plain control for the test above: the same run with no 12:00 settlement re-enters at 12:35."""
    out = paper(tmp_path, monkeypatch, binance,
                win((f"{DAY} 11:52", f"{DAY} 12:20", 1), (f"{DAY} 12:35", f"{DAY} 12:45", 1)),
                LATE, 60, rates=EIGHT_ONLY)
    assert (utc(f"{DAY} 12:35"), "BUY") in [(t, s) for t, s, _ in out["fills"]]
    assert not kinds(out["events"], "funding_entry_blocked") and not kinds(out["events"], "funding_stale")


ADD_CASES = {
    # cause: (start, minutes, add at, rates, published) - the add comes after the block begins
    "interval-shortened": (LATE, 60, f"{DAY} 12:35", EIGHT_THEN_FOUR, {}),
    "missing-rate": (START, 45, f"{DAY} 08:25", {}, {S: NEVER}),
}


@xfail
@pytest.mark.parametrize("cause", list(ADD_CASES))
def test_an_add_is_blocked_but_would_go_through_without_the_block(tmp_path, monkeypatch, binance, cause):
    """Half the weight from the first minute, all of it from the add time (an ADD to a held long). Without the cause the
    add fills; with it, the position stays at its first size. On c7a73f1 a perp cannot add at all (E13-6: a weight-sized
    model is refused on a perp until P2-1b), which is what fails first."""
    from sleeve_fund.strategies import check_perp_sizing

    start, minutes, add_at, rates, published = ADD_CASES[cause]
    first = (utc(start) + pd.Timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M")
    params = win((first, add_at, 0.5), (add_at, "2026-01-01", 1.0))
    try:
        check_perp_sizing("o17weight", params)
    except ValueError as exc:
        pytest.fail(f"a perp cannot add at this commit, so the add gate cannot be shown: {exc}")
    clean = paper(tmp_path, monkeypatch, binance, params, start, minutes, strategy="o17weight", name="clean",
                  rates={**rates, **{k: 0.0001 for k in published}})
    assert [s for _, s, _ in clean["fills"]] == ["BUY", "BUY"], f"control: the add did not fill: {clean['fills']}"
    out = paper(tmp_path, monkeypatch, binance, params, start, minutes, strategy="o17weight", rates=rates,
                published=published)
    assert [s for _, s, _ in out["fills"]] == ["BUY"], f"the add went through: {out['fills']}"
    assert kinds(out["events"], "funding_entry_blocked")


@xfail
def test_a_shortened_interval_raises_the_staleness_alert_once_per_instrument(tmp_path, monkeypatch, binance):
    """Two strategies on BTC/USDT, one journal, both running when the venue settles at 12:00: one funding_stale warning
    in all, naming the instrument, from 12:00 on (the same alert as a missing rate, 18:39)."""
    store = journal()
    # Orders a minute apart, so the two replays' order ids differ in the shared journal.
    for name, side, start, opens, closes in (("w1", 1, LATE, f"{DAY} 11:52", f"{DAY} 12:45"),
                                             ("w2", -1, "2025-10-03 11:49", f"{DAY} 11:53", f"{DAY} 12:44")):
        paper(tmp_path, monkeypatch, binance, win((opens, closes, side)), start, 60, rates=EIGHT_THEN_FOUR,
              name=name, store=store)
    stale = kinds(list(reversed(store.events(None, limit=5000))), "funding_stale")
    assert len(stale) == 1, [(e["ts"], e["sleeve"], e["message"]) for e in stale]
    assert stale[0]["level"] == "warning" and PAIR in stale[0]["message"]
    assert utc(stale[0]["ts"]) >= utc(f"{DAY} 12:00")


def test_a_lengthened_interval_is_not_blocked(tmp_path, monkeypatch, binance):
    """Plain (Advisor 19:11: a lengthened interval is never blocked): the venue settled every 4 hours up to 08:00, then
    every 8 (no 12:00; 16:00 comes). The 16:25 entry fills and nothing is blocked. 18:35's interim guard refuses only a
    step SHORTER than the charged schedule (19:11), so this start is allowed."""
    rates = {"2025-10-02 20:00": 0.0001, f"{DAY} 00:00": 0.0001, f"{DAY} 04:00": 0.0001, f"{DAY} 08:00": 0.0001,
             f"{DAY} 16:00": 0.0001}
    out = paper(tmp_path, monkeypatch, binance,
                win((f"{DAY} 15:52", f"{DAY} 16:20", 1), (f"{DAY} 16:25", f"{DAY} 16:35", 1)),
                "2025-10-03 15:50", 50, rates=rates, step=10)
    assert [(t, s) for t, s, _ in out["fills"]][:3] == [
        (utc(f"{DAY} 15:52"), "BUY"), (utc(f"{DAY} 16:20"), "SELL"), (utc(f"{DAY} 16:25"), "BUY")]
    assert not kinds(out["events"], "funding_entry_blocked")


def test_a_lengthened_interval_is_not_read_as_a_missed_settlement_or_charged_a_false_baseline(tmp_path, monkeypatch,
                                                                                               binance):
    """Advisor 19:11 / ~19:20: the instrument is charged every 4 hours; after 08:00 the venue lengthens to 8 hours (it
    publishes the new interval, and the stored rates show 08:00 then 16:00). The expected next settlement comes from
    the stored or published interval, so 12:00 is not missing: no 12:00 charge (today a false 0.01% baseline at the
    12:15 check), no staleness alert, no block, and the start (whose latest step equals the 4-hour schedule) is not
    refused. The published interval is offered as VenueProfile.funding_interval(pair), set here without raising."""
    from o17_harness import O17Win

    monkeypatch.setattr(binance, "funding_hours", (0, 4, 8, 12, 16, 20))

    def published_interval(pair):
        now = O17Win.last.clock.utc_now() if O17Win.last is not None else utc(f"{DAY} 07:50")
        return pd.Timedelta(hours=8 if utc(now) > utc(f"{DAY} 08:00") else 4)

    monkeypatch.setattr(binance, "funding_interval", published_interval, raising=False)
    rates = {"2025-10-02 20:00": 0.0001, f"{DAY} 00:00": 0.0001, f"{DAY} 04:00": 0.0001, f"{DAY} 08:00": 0.0001,
             f"{DAY} 16:00": 0.0003}
    out = paper(tmp_path, monkeypatch, binance,
                win((f"{DAY} 07:52", f"{DAY} 16:30", 1), (f"{DAY} 16:33", f"{DAY} 16:38", 1)),
                START, 530, rates=rates, step=10)
    assert out["fills"] and out["fills"][0][:2] == (utc(f"{DAY} 07:52"), "BUY"), "the start was refused"
    assert not kinds(out["events"], "strategy_refused")
    charged = [r["ts"] for r in out["funding"]]
    assert utc(f"{DAY} 12:00") not in charged, f"12:00 read as a missed settlement and charged: {out['funding']}"
    assert charged == [utc(f"{DAY} 08:00"), utc(f"{DAY} 16:00")]
    assert all(r.get("kind", "settled") != "baseline" for r in out["funding"])
    assert not kinds(out["events"], "funding_stale") and not kinds(out["events"], "funding_entry_blocked")
    assert (utc(f"{DAY} 16:33"), "BUY") in [(t, sd) for t, sd, _ in out["fills"]]


def test_a_perp_add_is_refused_today_in_paper(tmp_path, monkeypatch, binance):
    """Plain (HoE + DA): until P2-1, a perp cannot add. A weight-sized model is refused before it can start on a perp
    (check_perp_sizing, as paper/node.py and the supervisor call it), and a paper strategy on the perp that holds half
    its weight and then asks for all of it does not increase the position, with every rate on time (no funding
    block). This goes red if anything opens perp adds before P2-1; then the O17b add xfail above must pass with it."""
    from sleeve_fund.strategies import check_perp_sizing

    params = win((f"{DAY} 11:52", f"{DAY} 12:35", 0.5), (f"{DAY} 12:35", "2026-01-01", 1.0))
    with pytest.raises(ValueError, match="not available on perpetuals"):
        check_perp_sizing("o17weight", params)
    out = paper(tmp_path, monkeypatch, binance, params, LATE, 60, strategy="o17weight", rates=EIGHT_ONLY)
    buys = [f for f in out["fills"] if f[1] == "BUY"]
    assert len(out["fills"]) == 1 and len(buys) == 1, f"the position was increased or changed: {out['fills']}"
    assert buys[0][0] == utc(f"{DAY} 11:52")


# ---------------------------------------------------------------------------------------------------------------
# #163 x O17b (Head of QA, 7 Oct, accepting the DA's per-settlement interface; Advisor ~02:12 O17a-11): entries are
# blocked while ANY staleness episode on the instrument is open; a settlement the venue never published (still missing
# 24 h after due with a later one stored, marked funding_never_published) does not block, so no episode blocks
# entries forever.
# ---------------------------------------------------------------------------------------------------------------

def _np_hub_pass(m, b, store, at, rates: dict) -> None:
    """One pass of the hub's funding refresh (history._refresh_funding) at `at` on the journal `store`; the store
    file and the venue's history hold `rates` (settlement -> rate), the events it writes are stamped `at`."""
    import sleeve_fund.store as store_mod
    from o17_harness import _REAL_RATES, ms, write_rates
    from sleeve_fund import funding, history

    at = utc(at)
    m.setattr(funding, "rates", _REAL_RATES)
    write_rates({utc(k): v for k, v in rates.items()})
    rows = [(ms(t), r) for t, r in sorted((utc(k), v) for k, v in rates.items())]
    real_stale = funding.stale

    class Stamped:
        def __getattr__(self, name):
            return getattr(store, name)

        def event(self, sleeve, level, kind, message, ts=None):
            return store.event(sleeve, level, kind, message, ts=ts or at.to_pydatetime())

    with m.context() as c:
        c.setattr(funding, "stale", lambda v, p, root=None, now=None: real_stale(v, p, root, now=at))
        c.setattr(history, "_now", lambda: at, raising=False)
        c.setattr(b, "funding_loader", lambda pair, start: [x for x in rows if x[0] >= start])
        c.setattr(b, "stats_loaders", {})
        inbox = Stamped()
        c.setattr(store_mod, "Store", lambda *a, **k: inbox)
        history._warned.clear()
        history._refresh_funding(b, PAIR, funding.DEFAULT_ROOT, None)
        history._warned.clear()
    funding._cache.clear()


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON)
def test_a_never_published_settlement_does_not_block_entries_but_an_open_episode_does(tmp_path, monkeypatch, binance):
    """Paper "w" marks 08:00 missing at 08:15 (never published; every later settlement is). Control: a flat strategy
    started at 16:35, whose entry is due at 16:40, is held back (08:00's episode is open, under 24 h). After the hub's
    passes at 4 Oct 00:30 and 08:30, 08:00 is never published and its episode closed: a flat strategy started at
    08:35 enters at 08:40, with no funding_entry_blocked."""
    from sleeve_fund import funding, history

    assert callable(getattr(funding, "stale_open", None)), "not built: the per-instrument staleness episode"
    monkeypatch.setattr(history, "_stale", set(), raising=False)
    store = journal()
    made, create = set(), store.create_sleeve
    store.create_sleeve = lambda **kw: None if kw["name"] in made else (made.add(kw["name"]), create(**kw))[1]
    later = {"2025-10-02 16:00": 0.0001, "2025-10-03 00:00": 0.0001, "2025-10-03 16:00": 0.0002,
             "2025-10-04 00:00": 0.0001, "2025-10-04 08:00": 0.0001}
    run = dict(rates=later, store=store, published={"2025-10-03 08:00": NEVER})
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-05 00:00", 1)), "2025-10-03 07:50", 30,
          name="w", **run)
    ctl = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 16:40", "2025-10-05 00:00", 1)), "2025-10-03 16:35",
                25, name="f1", **run)
    assert not _entries_after(ctl, "2025-10-03 16:39"), f"control: entered while 08:00's episode is open {ctl['fills']}"
    for at in ("2025-10-04 00:30", "2025-10-04 08:30"):
        _np_hub_pass(monkeypatch, binance, store, at, {k: v for k, v in later.items() if utc(k) < utc(at)})
    never = [e for e in store.events(None, limit=5000) if e["kind"] == "funding_never_published"]
    assert never, "setup: 08:00 not marked never published by the 4 Oct 08:30 pass"
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-04 08:40", "2025-10-05 00:00", 1)), "2025-10-04 08:35",
                25, name="f2", **run)
    assert _entries_after(out, "2025-10-04 08:39"), "a never-published settlement blocked the entry"
    assert not [e for e in out["events"] if e["kind"] == "funding_entry_blocked" and utc(e["ts"]) > utc("2025-10-04 08:00")]
