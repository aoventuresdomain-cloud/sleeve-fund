"""O17a, missing funding rate: no credit; invalid rates; the "N of M at baseline" count and the held-time run; the G1
bands on the out-of-sample windows and the holdout; simulated perps; the staleness alert. Owner: Data Architect (board
split, HoE 17:52). Strict xfails written by QA BEFORE the build on main c7a73f1; the engineer makes each pass and removes
its mark before hand-over. Plain tests pin what already holds and must keep holding.

Rulings pinned (advisor-rulings.md; the latest wins where they differ):
- line 76 (Advisor 6 Oct 17:52, to DA): "baseline is NEVER a credit (both sides pay 0.01% when missing). Backtest: tear
  sheet "N of M at baseline" over held settlements; 1-5% warn; >5% or any baseline run >7 days -> G1 not judged until
  backfill. Paper: ... staleness alert."
- line 80 (O17a review, ~18:25, to DA): holdout counted too (own row); baseline scaled 0.01% x interval/8h per
  instrument; stale alert also logs recovery. (Its point 3, the stretch in clock time, is superseded by line 81.)
- lines 81-82 (O17 QA points ~18:30, and O17 open points 18:18, to QA; same content): run = unbroken stretch of
  missing rates, broken only by a stored real rate; flat time neither breaks nor adds; length = HELD time; >7 days ->
  G1 not judged. Thresholds on the OOS test windows G1 stands on, and separately on the holdout ("holdout not judged");
  full-period figure an info line only; benchmark charges never count. 1% and 5% inside the warn band; <1% shown, no
  warning; held settlements only. Simulated perps: baseline, labelled; G1 not judged unless a modelled rate series is
  in the trials register BEFORE the run. Staleness alert per instrument, once per episode, replaces funding_fallback.
  NaN, +-inf and |rate| above the venue's funding cap are invalid: alerted, never stored as real, charged as missing.
- line 111 (O17a-4 funding cap, Advisor 6 Oct 20:44): the validity cap is each instrument's own published cap,
  POINT-IN-TIME (the cap that applied at that settlement, stored with effective-from in contract data). Missing cap:
  the WIDEST cap the venue publishes for that contract type, plus an alert (lean to accepting). Rejected rates are
  never dropped: the raw value is quarantined with the alert for DA review (journalled true-up if real), and charged
  the baseline until then. Simulated perps keep the flat fallback.
- v5, 7 Oct (sections 8-10): Advisor ~02:12 O17a-11 FUNDING EPISODES (per-settlement states, episodes, backfill,
  never published at 24 h; P1-O17a-10/-11/-12), the DA's event interface as accepted by the Head of QA, Advisor ~02:40
  FUNDING GAP READING full text (rules (a)-(d) and the default), and the G1-NP follow-up (before strategy testing).
- 7 Oct 06:30 EPISODE AFTER RECLASSIFICATION and 06:33 EPISODE MEMBERSHIP + ALERT TEXT (Advisor; section 9c, and the
  section 8 membership re-pin): the episode is the outage; a miss while it is open joins it whatever lies between; it
  closes once every settlement is terminal (stored, reclassified, never published), with one closing notice giving
  each outcome. Not in #163: follow-up O17a-EPISODE (DA, MINOR, before strategy testing; HoE 06:30), REASON_EPISODE.

Today (c7a73f1, strategies/base.py _apply_funding / _funding_rate ~1859-1923): a missing rate is charged the terms'
fixed 0.01% as amount = -qty x price x rate, so a long pays and a SHORT IS CREDITED; one "funding_fallback" warning
per run; nothing is counted; a NaN or inf kept in the store is charged as it is; a simulated perp charges 0.01% the
same way (shorts credited) and its study is judged.

ASSUMED INTERFACES (adapt the names, never the assertions):
- Funding row `kind` = "settled" | "baseline" | "true_up" (O17b), on Store.funding rows (a new column on `funding`),
  the backtest journal's `funding_` rows and BacktestResult.funding rows. A simulated perp's rows are "baseline".
- BacktestResult.funding_held (M), .funding_at_baseline (N), .funding_baseline_longest (pd.Timedelta: the longest
  HELD time inside one unbroken stretch of missing rates; pd.Timedelta(0) when none).
- funding.baseline_check(at_baseline, held, longest_held) -> (verdict, words): "PASS" | "WARN" | tearsheet.NOT_JUDGED;
  words carry "N of M"; constants BASELINE_WARN = 0.01, BASELINE_LIMIT = 0.05, BASELINE_RUN_LIMIT = 7 days.
- VenueProfile.funding_cap(pair) -> float: O17a's own validity cap on |rate| per settlement (built on O17a: a sanity
  default of 5%). The invalid-rate tests set it to 0.3%; the cap pins in section 7 never touch it.
- VenueProfile.published_funding_caps: pair -> cap, the caps the venue publishes now (the DA's interim static table on
  O17a, 0.3% for BTC); a rate beyond it raises "funding_over_cap" and is still kept and charged until DA-11.
- DA-11: funding.keep_cap(venue, pair, cap, effective_from) keeps a cap in the contract data with its effective-from;
  funding.cap_at(venue, pair, ts) -> the cap that applied at ts. With no cap for the instrument, the widest the venue
  publishes for that contract type applies, with a "funding_cap_missing" alert.
- funding.quarantined(venue, pair) -> pd.Series: each rejected settlement's raw value (NaN and inf included), held
  until the DA reviews it; the settlement is charged the baseline meanwhile.
- Tear sheet: tearsheet.FUNDING_CHECK, the G1 row judged on the OOS test windows, its line carrying "N of M",
  "baseline" and "out-of-sample"; an info line with the full-period "N of M"; a holdout line reading "holdout not
  judged" when the holdout breaks a band; StudyResult.not_judged names funding when the row is NOT JUDGED;
  tearsheet.g1_verdict treats "WARN" as not failing.
- Events: "funding_stale" (warning, per instrument, once per episode; no "funding_fallback" any more),
  "funding_stale_cleared" (recovery), "funding_invalid" (warning, in the alerts inbox, from the collector's refresh,
  carrying the raw value), "funding_cap_missing" (warning, naming the instrument whose cap is missing),
  "funding_over_cap" (the DA's interim guard, naming the instrument).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from o17_harness import (  # noqa: F401  (fixtures are used by name)
    BASELINE,
    NEVER,
    PAIR,
    _no_swallowed_strategy_errors,
    _o17win,
    backtest,
    binance,
    hourly_bars,
    isnan,
    journal,
    kinds,
    ms,
    paper,
    settlements,
    utc,
    win,
    write_rates,
)
from sleeve_fund import funding

# Funding mechanics with the harness's stopless 1x perp: #182's interim 5% open-risk limit would refuse the entries
# (QA 7 Oct 08:15, DA). The open-risk limit has its own pins in the gate 5 masters.
pytestmark = pytest.mark.no_open_risk_limit(reason="funding mechanics with a stopless perp; open-risk limit pinned in gate 5")

REASON = "QA O17a: not built yet (Advisor 17:52)"
REASON_EPISODE = "O17a-EPISODE follow-up (Advisor 06:28/06:30 episode rulings; HoE 06:30): not in #163"
xfail = pytest.mark.xfail(strict=True, reason=REASON)

LONG = pytest.param(1, id="long")
SHORT_X = pytest.param(-1, id="short")


def _paid_baseline(rows):
    for r in rows:
        assert not isnan(r["amount"]), f"NaN charged at {r['ts']}"
        assert r["amount"] == pytest.approx(-abs(r["qty"]) * r["price"] * BASELINE, rel=1e-6), r


# ---------------------------------------------------------------------------------------------------------------
# 1. Baseline is never a credit: both sides pay 0.01% when the rate is missing (backtest and paper, perp only)
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", [LONG, SHORT_X])
def test_a_missing_rate_is_paid_by_both_sides_in_the_backtest(binance, side):
    """No rate kept at all: every held settlement is charged the baseline, and both sides pay it."""
    bars = hourly_bars("2025-10-03 00:00", "2025-10-05 00:00")
    r = backtest(binance, bars, win(("2025-10-03 01:00", "2026-01-01", side)))
    rows = r.journal.funding_
    assert [utc(x["ts"]) for x in rows] == settlements("2025-10-03 01:00", bars.index[-1])
    assert all(np.sign(x["qty"]) == side for x in rows)
    _paid_baseline(rows)
    assert all(f["amount"] < 0 for f in r.funding)  # the result the equity curve is built from agrees


@pytest.mark.parametrize("side", [LONG, SHORT_X])
def test_a_missing_rate_is_paid_by_both_sides_on_paper(tmp_path, monkeypatch, binance, side):
    """The venue never publishes 08:00: after the 15-minute check paper charges the baseline, adverse to either side."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 08:25", side)),
                "2025-10-03 07:50", 35, published={"2025-10-03 08:00": NEVER})
    (row,) = out["funding"]
    assert row["ts"] == utc("2025-10-03 08:00") and np.sign(row["qty"]) == side
    _paid_baseline([row])


# ---------------------------------------------------------------------------------------------------------------
# 2. A real rate, positive, negative or zero, is charged exactly as settled (plain: holds today, must keep holding)
# ---------------------------------------------------------------------------------------------------------------

REAL = {"2025-10-03 08:00": 0.0003, "2025-10-03 16:00": -0.0002, "2025-10-04 00:00": 0.0}


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_settled_rate_is_charged_as_settled_in_the_backtest(binance, side):
    write_rates({utc(k): v for k, v in REAL.items()})
    bars = hourly_bars("2025-10-03 00:00", "2025-10-04 04:00")
    r = backtest(binance, bars, win(("2025-10-03 01:00", "2026-01-01", side)))
    rows = r.journal.funding_
    assert [utc(x["ts"]) for x in rows] == [utc(k) for k in REAL]
    for x in rows:
        rate = REAL[utc(x["ts"]).strftime("%Y-%m-%d %H:%M")]
        assert x["rate"] == pytest.approx(rate)
        assert x["amount"] == pytest.approx(-x["qty"] * x["price"] * rate, rel=1e-6, abs=1e-9)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("rate", [0.0003, -0.0002, 0.0], ids=["positive", "negative", "zero"])
def test_a_rate_published_within_15_minutes_is_charged_as_settled_on_paper(tmp_path, monkeypatch, binance, side, rate):
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 08:25", side)),
                "2025-10-03 07:50", 35, rates={"2025-10-03 08:00": rate},
                published={"2025-10-03 08:00": "2025-10-03 08:10"})
    (row,) = out["funding"]
    assert row["rate"] == pytest.approx(rate)
    assert row["amount"] == pytest.approx(-row["qty"] * row["price"] * rate, abs=1e-6)
    assert not kinds(out["events"], "funding_stale")  # the venue answered inside the check: nothing is stale


def test_spot_pays_no_funding():
    """Perp only: a spot holding across every settlement is charged nothing."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD")
    r = run_backtest("buy_and_hold", hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"), inst, {},
                     starting_capital=10_000, risk_profile="balanced", bar_minutes=60, half_spread=0)
    assert len(r.fills) >= 1 and r.funding == [] and r.journal.funding_ == []


# ---------------------------------------------------------------------------------------------------------------
# 1b. Simulated perps (no venue rates): every settlement is the baseline, adverse, labelled. Interval scaling.
# ---------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("side", [LONG, SHORT_X])
def test_a_simulated_perp_charges_the_baseline_to_both_sides(side):
    """Kraken has no perpetual of its own: a perp there is simulated at the fixed 0.01%, which is the baseline (18:18)."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD")
    r = run_backtest("o17win", hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"), inst,
                     win(("2025-10-03 01:00", "2026-01-01", side)), starting_capital=10_000,
                     risk_profile="balanced", bar_minutes=60, half_spread=0)
    assert r.journal.funding_
    _paid_baseline(r.journal.funding_)


def test_a_simulated_perps_charges_are_labelled_baseline(monkeypatch):
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD")
    r = run_backtest("o17win", hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"), inst,
                     win(("2025-10-03 01:00", "2026-01-01", 1)), starting_capital=10_000, bar_minutes=60, half_spread=0)
    assert r.funding and {f.get("kind") for f in r.funding} == {"baseline"}
    assert (r.funding_at_baseline, r.funding_held) == (len(r.funding), len(r.funding))


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_the_baseline_is_scaled_to_the_instruments_interval(binance, monkeypatch, side):
    """O17a review (~18:25): baseline = 0.01% x interval / 8 h. An instrument settling every 4 hours pays 0.005% for a
    missing settlement, either side."""
    monkeypatch.setattr(binance, "funding_hours", (0, 4, 8, 12, 16, 20))
    every = settlements("2025-10-03 00:00", "2025-10-05 00:00")
    missing = set(every[3:6])
    write_rates({t: 0.0002 for t in every if t not in missing})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"),
                 win(("2025-10-03 01:00", "2026-01-01", side)))
    rows = {utc(x["ts"]): x for x in r.journal.funding_}
    assert missing <= set(rows), "setup: the 4-hourly settlements are charged"
    for t in missing:
        x = rows[t]
        assert x["amount"] == pytest.approx(-abs(x["qty"]) * x["price"] * BASELINE / 2, rel=1e-6), x


# ---------------------------------------------------------------------------------------------------------------
# 2. Baseline charges are marked, and counted as N of M over HELD settlements; invalid rates are missing
# ---------------------------------------------------------------------------------------------------------------

# Long 3 Oct 01:00 -> 5 Oct 12:00, flat, short 6 Oct 04:00 -> end (7 Oct 23:00).
SPANS = (("2025-10-03 01:00", "2025-10-05 12:00", 1), ("2025-10-06 04:00", "2026-01-01", -1))
HELD = ["2025-10-03 08:00", "2025-10-03 16:00", "2025-10-04 00:00", "2025-10-04 08:00", "2025-10-04 16:00",
        "2025-10-05 00:00", "2025-10-05 08:00", "2025-10-06 08:00", "2025-10-06 16:00", "2025-10-07 00:00",
        "2025-10-07 08:00", "2025-10-07 16:00"]
FLAT = ["2025-10-03 00:00", "2025-10-05 16:00", "2025-10-06 00:00"]  # all three missing: never counted
MISSING_HELD = ["2025-10-03 16:00", "2025-10-04 08:00", "2025-10-05 00:00", "2025-10-06 08:00", "2025-10-07 00:00"]
KEPT = {"2025-10-03 08:00": 0.0003, "2025-10-04 00:00": -0.0002, "2025-10-04 16:00": 0.0, "2025-10-05 08:00": 0.0001,
        "2025-10-06 16:00": 0.0002, "2025-10-07 08:00": -0.0001, "2025-10-07 16:00": 0.0003}


def _mixed(binance, risk_profile):
    write_rates({utc(k): v for k, v in KEPT.items()})
    return backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-07 23:00"), win(*SPANS), risk_profile)


@pytest.mark.parametrize("risk_profile", [None, "balanced"], ids=["research", "risk-profile"])
def test_n_of_m_counts_baseline_settlements_over_held_settlements_only(binance, risk_profile):
    """12 held settlements (7 with a kept rate, one of them exactly 0), 5 of them missing; 3 more missing while flat.
    N = 5, M = 12, whether or not the backtest runs with a risk profile (research studies run without one)."""
    r = _mixed(binance, risk_profile)
    assert sorted(utc(f["ts"]) for f in r.funding) == [utc(t) for t in HELD]  # setup: charged only while held
    assert (r.funding_at_baseline, r.funding_held) == (5, 12)


def test_each_charge_says_whether_it_was_the_venue_rate_or_the_baseline(binance):
    r = _mixed(binance, "balanced")
    want = {utc(t): ("baseline" if t in MISSING_HELD else "settled") for t in HELD}
    assert {utc(x["ts"]): x.get("kind") for x in r.journal.funding_} == want
    assert {utc(x["ts"]): x.get("kind") for x in r.funding} == want
    for x in r.journal.funding_:
        if x["kind"] == "baseline":
            assert x["amount"] == pytest.approx(-abs(x["qty"]) * x["price"] * BASELINE, rel=1e-6)


INVALID = [("nan", float("nan")), ("inf", float("inf")), ("minus-inf", float("-inf")),
           ("over-the-cap", 0.05), ("under-minus-the-cap", -0.05)]


@pytest.mark.strategy_errors
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("bad", [b for _, b in INVALID], ids=[n for n, _ in INVALID])
def test_an_invalid_kept_rate_is_missing_charged_the_baseline_to_both_sides_and_counted(binance, monkeypatch, side, bad):
    """NaN, +-inf, or |rate| over the venue's cap (0.3% here) kept in the store (a hand-written file or a bad backfill)
    is a missing rate: the baseline, never the bad number, and counted at baseline (18:18)."""
    if np.isfinite(bad):
        monkeypatch.setattr(binance, "funding_cap", lambda pair: 0.003, raising=False)  # O17a's own validity cap
    write_rates({utc("2025-10-03 08:00"): 0.0003, utc("2025-10-03 16:00"): bad, utc("2025-10-04 00:00"): 0.0001})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-04 04:00"),
                 win(("2025-10-03 01:00", "2026-01-01", side)))
    bad_row = next(x for x in r.journal.funding_ if utc(x["ts"]) == utc("2025-10-03 16:00"))
    _paid_baseline([bad_row])
    assert r.handler_error_count == 0, r.handler_errors[:1]  # today a NaN or inf charge also breaks on_order_filled
    assert np.isfinite([f["amount"] for f in r.funding]).all() and not r.equity.isna().any()
    assert (r.funding_at_baseline, r.funding_held) == (1, 3)


def test_the_collector_never_stores_an_invalid_rate_and_alerts_it(tmp_path, binance, monkeypatch):
    """funding.refresh through the hub's pass (history._refresh_funding): NaN, +-inf and rates over the cap are not
    kept as real rates, and one funding_invalid warning reaches the alerts inbox. (NaN and inf are dropped today, but
    only printed; a rate over the cap is kept.)"""
    from types import SimpleNamespace

    import sleeve_fund.store as store_mod
    from sleeve_fund import history

    monkeypatch.setattr(binance, "funding_cap", lambda pair: 0.003, raising=False)  # O17a's own validity cap
    times = settlements("2025-10-02 23:00", "2025-10-05 00:00")  # 7 settlements
    bad = {times[1]: float("nan"), times[2]: float("inf"), times[3]: float("-inf"), times[4]: 0.05, times[5]: -0.004}
    rows = [(int(t.timestamp() * 1000), bad.get(t, 0.0001)) for t in times]
    monkeypatch.setattr(binance, "funding_loader", lambda pair, start: [r for r in rows if r[0] >= start])
    sent = []

    class Inbox:
        def __init__(self, *a, **k):
            pass

        def event(self, sleeve, level, kind, message, ts=None):
            sent.append((level, kind, message))

    monkeypatch.setattr(store_mod, "Store", Inbox)
    history._warned.clear()
    history._refresh_funding(SimpleNamespace(name="binance", funding_loader=True, stats_loaders={}), PAIR,
                             funding.DEFAULT_ROOT, None)
    history._warned.clear()
    kept = funding.rates("BINANCE", PAIR)
    assert list(kept.index) == [t for t in times if t not in bad], f"kept {kept.to_dict()}"
    invalid = [m for lv, k, m in sent if k == "funding_invalid"]
    assert invalid and all(lv == "warning" for lv, k, _ in sent if k == "funding_invalid")


def test_a_rate_of_exactly_zero_is_a_real_rate_not_missing(binance):
    """Zero is a settled rate: charged 0 (no baseline) and not counted at baseline."""
    write_rates({t: 0.0 for t in settlements("2025-10-03 00:00", "2025-10-05 00:00")})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"),
                 win(("2025-10-03 01:00", "2026-01-01", -1)))
    assert r.funding and all(f["amount"] == 0 for f in r.funding)  # holds today
    assert (r.funding_at_baseline, r.funding_held) == (0, len(r.funding))


# ---------------------------------------------------------------------------------------------------------------
# 3. The run: an unbroken stretch of missing rates, broken only by a stored real rate; its length is HELD time
# ---------------------------------------------------------------------------------------------------------------

D = pd.Timedelta(days=1)
S0, S1 = utc("2025-09-01 00:00"), utc("2025-09-25 00:00")


def _run(binance, spans, missing_from, missing_to, keep_inside=()):
    """Held on `spans`; every settlement kept at 0.01% except those in [missing_from, missing_to), apart from
    `keep_inside` (a real rate stored inside the stretch)."""
    every = settlements(S0, S1)
    gap = {t for t in every if utc(missing_from) <= t < utc(missing_to)} - {utc(t) for t in keep_inside}
    write_rates({t: 0.0001 for t in every if t not in gap})
    return backtest(binance, hourly_bars(S0, S1), win(*spans), None), gap


@pytest.mark.parametrize("run_days, lo, hi", [(10, 9, 11), (3, 2, 4)], ids=["held-10-days", "held-3-days"])
def test_the_longest_run_held_throughout_is_its_length(binance, run_days, lo, hi):
    start = utc("2025-09-05 00:00")
    r, gap = _run(binance, [("2025-09-01 01:00", "2026-01-01", 1)], start, start + run_days * D)
    assert r.funding_at_baseline == len(gap)
    assert lo * D <= r.funding_baseline_longest <= hi * D


def test_flat_time_inside_a_stretch_neither_breaks_it_nor_counts(binance):
    """Missing 5 Sep -> 16 Sep (11 days of clock). Held 5-9 Sep and 12-16 Sep, flat 9-12 Sep: one run of about 8 days
    held, so over 7 (clock time would read 11; two runs broken by the flat spell would read 4 each)."""
    r, _ = _run(binance, [("2025-09-01 01:00", "2025-09-09 00:00", 1), ("2025-09-12 00:00", "2026-01-01", 1)],
                "2025-09-05 00:00", "2025-09-16 00:00")
    assert pd.Timedelta(days=7, hours=8) <= r.funding_baseline_longest <= 9 * D


def test_a_long_flat_spell_does_not_make_a_short_held_run_long(binance):
    """Missing 5 Sep -> 17 Sep (12 days of clock). Held 5-8 Sep and 14-17 Sep, flat between: about 6 days held, so at
    most 7 and judged (clock time would read 12)."""
    r, _ = _run(binance, [("2025-09-01 01:00", "2025-09-08 00:00", 1), ("2025-09-14 00:00", "2026-01-01", 1)],
                "2025-09-05 00:00", "2025-09-17 00:00")
    assert 5 * D <= r.funding_baseline_longest <= pd.Timedelta(days=6, hours=16)


def test_a_real_rate_stored_while_flat_breaks_the_run(binance):
    """Missing 5 Sep -> 17 Sep except one real rate stored on 11 Sep 00:00, while flat (9-13 Sep). Held 5-9 and 13-17:
    two runs of about 4 days, not one of 8."""
    r, _ = _run(binance, [("2025-09-01 01:00", "2025-09-09 00:00", 1), ("2025-09-13 00:00", "2026-01-01", 1)],
                "2025-09-05 00:00", "2025-09-17 00:00", keep_inside=["2025-09-11 00:00"])
    assert 3 * D <= r.funding_baseline_longest <= 5 * D


# ---------------------------------------------------------------------------------------------------------------
# 4. The bands: 0%, under 1%, exactly 1%, 1-5% warn, exactly 5%, over 5%; held run over 7 days not judged
# ---------------------------------------------------------------------------------------------------------------

H8, D7 = pd.Timedelta(hours=8), pd.Timedelta(days=7)
BANDS = [
    ("nothing-held", 0, 0, pd.Timedelta(0), "PASS"),
    ("0pct", 0, 1000, pd.Timedelta(0), "PASS"),
    ("0.9pct", 9, 1000, H8, "PASS"),
    ("exactly-1pct", 1, 100, H8, "WARN"),
    ("exactly-1pct-of-3000", 30, 3000, H8, "WARN"),
    ("3pct", 3, 100, H8, "WARN"),
    ("exactly-5pct", 5, 100, H8, "WARN"),
    ("exactly-5pct-of-1000", 50, 1000, H8, "WARN"),
    ("5.1pct", 51, 1000, H8, "NOT JUDGED"),
    ("6pct", 6, 100, H8, "NOT JUDGED"),
    ("held-run-exactly-7d-0.7pct", 21, 3000, D7, "PASS"),
    ("held-run-exactly-7d-2pct", 60, 3000, D7, "WARN"),
    ("held-run-7d-1min-0.73pct", 22, 3000, D7 + pd.Timedelta(minutes=1), "NOT JUDGED"),
    ("held-run-7d8h-0.73pct", 22, 3000, D7 + H8, "NOT JUDGED"),
]


@pytest.mark.parametrize("n, m, run, verdict", [b[1:] for b in BANDS], ids=[b[0] for b in BANDS])
def test_the_baseline_bands(n, m, run, verdict):
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    assert NOT_JUDGED == "NOT JUDGED"
    assert (funding.BASELINE_WARN, funding.BASELINE_LIMIT, funding.BASELINE_RUN_LIMIT) == (0.01, 0.05, D7)
    got, words = funding.baseline_check(n, m, run)
    assert got == verdict, words
    if m:
        assert f"{n} of {m}" in words
    if verdict == NOT_JUDGED:
        assert "backfill" in words.lower()


def test_a_funding_warn_does_not_fail_g1():
    """WARN is shown, not a fail: today g1_verdict counts any verdict other than PASS, INFO or N/A as a fail."""
    from sleeve_fund.research.tearsheet import g1_verdict

    assert g1_verdict([("Runs complete enough to judge", "PASS", ""), ("Funding rates", "WARN", "3 of 100")]) == \
        ("PASS", [])


# ---------------------------------------------------------------------------------------------------------------
# 5. Through a real study: OOS windows judged, holdout on its own row, full period info only, simulated perps
# ---------------------------------------------------------------------------------------------------------------

def _every(start, end):
    return settlements(start, end)


# buy_and_hold, 400 synthetic days from 2 Jan 2018: research to 6 Jan 2019 (about 1,107 held settlements), four test
# windows (1 May 2018 -> 27 Dec 2018, about 720 settlements), holdout 6 Jan -> 5 Feb 2019.
STUDY_GAPS = {
    # name: (missing settlements, verdict of the OOS funding row, holdout not judged?)
    "none": (lambda: [], "PASS", False),
    "oos-6-day-run-plus-10-scattered": (lambda: _every("2018-05-31 23:00", "2018-06-06 16:00")
                                        + _every("2018-09-01", "2018-11-30")[::7][:10], "WARN", False),
    "oos-8-day-run-only": (lambda: _every("2018-05-31 23:00", "2018-06-08 16:00"), "NOT JUDGED", False),
    "oos-every-10th": (lambda: _every("2018-05-10", "2018-12-20")[::10], "NOT JUDGED", False),
    "training-only-every-3rd": (lambda: _every("2018-01-07", "2018-04-21")[::3], "PASS", False),
    "holdout-every-5th": (lambda: _every("2019-01-08", "2019-02-04")[::5], "PASS", True),
}


def _lines(sheet, *words):
    return [ln for ln in sheet.splitlines() if all(w in ln.lower() for w in words)]


def _counts(lines):
    return [(int(a), int(b)) for ln in lines for a, b in re.findall(r"(\d+) of (\d+)", ln)]


@pytest.mark.parametrize("case", list(STUDY_GAPS))
def test_the_tear_sheet_judges_funding_on_the_oos_windows_and_the_holdout_apart(tmp_path, binance, case):
    """N and M are worked out here from the full-period run's own charges, which buy and hold shares with every fold
    run and the holdout run (held throughout). The OOS M is allowed one settlement per fold either way (window edges)."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict, render
    from sleeve_fund.strategies.buy_and_hold import SPEC

    make, verdict, holdout_unjudged = STUDY_GAPS[case]
    missing = set(make())
    prices = synthetic_ohlcv(days=400, seed=3, start_price=60_000)
    write_rates({t: 0.00012 for t in settlements(prices.index[0] - D, prices.index[-1]) if t not in missing})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, binance.instrument("BTC", "USDT"), dataset="syn-binance-1440m", ledger=ledger,
                  synthetic=True, holdout_days=30, train_days=120, test_days=60, use_holdout=True)
    full = [utc(f["ts"]) for f in r.full_period.funding]
    in_oos = [t for t in full if any(f.train_end < t <= f.test_end for f in r.folds)]
    nf, mf = sum(t in missing for t in full), len(full)
    n_oos, m_oos = sum(t in missing for t in in_oos), len(in_oos)
    assert mf > 1000 and m_oos > 600 and (r.holdout is not None or not holdout_unjudged)  # setup
    sheet = render(r, ledger)
    oos = _counts(_lines(sheet, "baseline", "out-of-sample"))
    assert any(n == n_oos and abs(m - m_oos) <= len(r.folds) for n, m in oos), \
        f"no '{n_oos} of ~{m_oos} ... baseline ... out-of-sample' line on the tear sheet: {oos}"
    assert (nf, mf) in _counts(_lines(sheet, "baseline", "full")), f"no full-period '{nf} of {mf}' info line"
    from sleeve_fund.research.tearsheet import FUNDING_CHECK

    checks = {name: (v, ev) for name, v, ev in g1_checks(r, ledger)}
    assert checks[FUNDING_CHECK][0] == verdict, checks[FUNDING_CHECK]
    g1, failed = g1_verdict(list((k, *v) for k, v in checks.items()))
    if verdict == NOT_JUDGED:
        assert g1 == NOT_JUDGED and "funding" in r.not_judged.lower()
        assert "**G1: NOT JUDGED**" in sheet
    else:
        assert "funding" not in r.not_judged.lower() and FUNDING_CHECK not in failed
    assert bool(_lines(sheet, "holdout not judged")) == holdout_unjudged


def test_a_simulated_perp_study_is_not_judged_without_a_registered_rate_model(tmp_path):
    """Kraken BTC/USD as a perp: no venue rates, so every settlement is the labelled baseline, and with no modelled
    rate series in the trials register before the run, G1 is not judged (18:18)."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict, render
    from sleeve_fund.strategies.buy_and_hold import SPEC
    from sleeve_fund.venues import venue

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, synthetic_ohlcv(days=400, seed=3, start_price=60_000), venue("KRAKEN").instrument("BTC", "USD"),
                  dataset="syn-kraken-perp", ledger=ledger, synthetic=True, holdout_days=30, train_days=120,
                  test_days=60, default_params={"market": "perp"})
    assert len(r.full_period.funding) > 1000  # setup: the simulated perp is charged at every held settlement
    assert g1_verdict(g1_checks(r, ledger))[0] == NOT_JUDGED
    assert "funding" in r.not_judged.lower()
    assert _lines(render(r, ledger), "simulated", "baseline")


# ---------------------------------------------------------------------------------------------------------------
# 6. Paper: the staleness alert, per instrument, once per episode, replacing funding_fallback; recovery logged
# ---------------------------------------------------------------------------------------------------------------

def test_paper_raises_one_staleness_alert_per_episode_instead_of_funding_fallback(tmp_path, monkeypatch, binance):
    """Long from 07:52 to 16:25; neither 08:00 nor 16:00 is ever published. One funding_stale warning, raised when the
    15-minute check on 08:00 fails (not before), naming the instrument; the second missing settlement in the same
    episode does not raise another; no funding_fallback."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 16:25", 1)),
                "2025-10-03 07:50", 520, published={"2025-10-03 08:00": NEVER, "2025-10-03 16:00": NEVER})
    assert [r["ts"] for r in out["funding"]] == [utc("2025-10-03 08:00"), utc("2025-10-03 16:00")]  # setup
    stale = kinds(out["events"], "funding_stale")
    assert len(stale) == 1, [(e["ts"], e["message"]) for e in stale]
    (alert,) = stale
    assert alert["level"] == "warning" and PAIR in alert["message"]
    assert utc("2025-10-03 08:15") <= alert["ts"] < utc("2025-10-03 08:16")
    assert not kinds(out["events"], "funding_fallback")


def test_two_strategies_on_the_instrument_raise_one_staleness_alert(tmp_path, monkeypatch, binance):
    """Per instrument: two paper strategies on BTC/USDT, both holding over the same missing 08:00, share one journal.
    One funding_stale alert in all, not one each."""
    store = journal()
    # Started a minute apart, so the two replays' order ids differ in the shared journal.
    for name, side, start, opens, closes in (("w1", 1, "2025-10-03 07:50", "2025-10-03 07:52", "2025-10-03 08:25"),
                                             ("w2", -1, "2025-10-03 07:49", "2025-10-03 07:53", "2025-10-03 08:24")):
        out = paper(tmp_path, monkeypatch, binance, win((opens, closes, side)),
                    start, 36, published={"2025-10-03 08:00": NEVER}, name=name, store=store)
        assert len(out["funding"]) == 1  # setup: each was charged
    assert len(kinds(out["events"], "funding_stale")) == 1
    assert not kinds(out["events"], "funding_fallback")


def test_the_staleness_alert_logs_its_recovery(tmp_path, monkeypatch, binance):
    """08:00 published at 08:30: stale from 08:15, then one funding_stale_cleared at or after 08:30."""
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 08:45", 1)),
                "2025-10-03 07:50", 50, rates={"2025-10-03 08:00": 0.0002},
                published={"2025-10-03 08:00": "2025-10-03 08:30"})
    assert len(kinds(out["events"], "funding_stale")) == 1
    (cleared,) = kinds(out["events"], "funding_stale_cleared")
    assert cleared["ts"] >= utc("2025-10-03 08:30") and PAIR in cleared["message"]


def test_paper_marks_the_baseline_row_in_the_journal(tmp_path, monkeypatch, binance):
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 08:25", -1)),
                "2025-10-03 07:50", 35, published={"2025-10-03 08:00": NEVER})
    (row,) = out["funding"]
    assert row.get("kind") == "baseline"



def test_a_holdout_not_judged_for_funding_blocks_promotion_but_not_g1(tmp_path, binance):
    """Advisor 19:11 / ~19:20, point 2: G1 can still pass on the OOS windows, but a strategy whose holdout is "not
    judged" for missing rates cannot go past the holdout step: no promotion to paper evaluation until the rates are
    backfilled and the holdout judged.
    The real study (20% of the holdout's settlements missing, OOS clean) gives the facts the pipeline reads. Assumed:
    sheet_facts(path)["holdout"] == "NOT JUDGED", G1 not made NOT JUDGED by it, and
    sleeve_fund.dashboard.pipeline.promotable(facts) -> (bool, reason). Written to fail on an assertion, never on a
    missing name."""
    from sleeve_fund.dashboard import pipeline
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.strategies.buy_and_hold import SPEC

    missing = set(_every("2019-01-08", "2019-02-04")[::5])
    prices = synthetic_ohlcv(days=400, seed=3, start_price=60_000)
    write_rates({t: 0.00012 for t in settlements(prices.index[0] - D, prices.index[-1]) if t not in missing})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, binance.instrument("BTC", "USDT"), dataset="syn-binance-1440m", ledger=ledger,
                  synthetic=True, holdout_days=30, train_days=120, test_days=60, use_holdout=True)
    sheet = tmp_path / "buy_and_hold_syn-binance-1440m.md"
    sheet.write_text(render(r, ledger), encoding="utf-8")
    facts = pipeline.sheet_facts(sheet)
    assert facts.get("holdout") == "NOT JUDGED", f"the pipeline does not see the holdout as not judged: {facts}"
    assert facts["g1"] != "NOT JUDGED", "the holdout's funding made G1 itself not judged; G1 stands on the OOS windows"
    promotable = getattr(pipeline, "promotable", None)
    assert callable(promotable), "no promotion gate past the holdout step (pipeline.promotable)"
    ok, why = promotable(dict(facts, g1="PASS"))
    assert ok is False and "backfill" in why.lower(), why
    assert promotable(dict(facts, g1="PASS", holdout="PASS"))[0] is True  # control: the same pass, holdout judged


# ---------------------------------------------------------------------------------------------------------------
# 7. The validity cap (O17a-4, Advisor 20:44; the pins close with DA-11): each instrument's own published cap, point-
#    in-time; a missing cap uses the venue's widest for that contract type, with an alert; a rejected rate is
#    quarantined raw, alerted, charged the baseline until reviewed; simulated perps keep the flat fallback. Plus the
#    DA's interim guard (O17a): a rate beyond the static published cap is alerted, and still kept and charged.
#    These tests NEVER patch the production validity cap (VenueProfile.funding_cap): every cap they need comes from
#    the venue's published caps, fed through the test-only sources below.
# ---------------------------------------------------------------------------------------------------------------

REASON_CAP = "DA-11"  # the cap pins close with DA-11, not O17a (HoE, pending the Advisor's confirmation)
CAP_TIMES = settlements("2025-10-02 23:00", "2025-10-04 09:00")  # 3 Oct 00/08/16, 4 Oct 00/08
ETH = "ETH/USDT"


def _published_now(binance, monkeypatch, caps: dict) -> None:
    """The caps the venue publishes now, pair -> cap on |rate|: the DA's interim static table
    (VenueProfile.published_funding_caps). A pair left out has no published cap. Set with raising=False, so it is
    inert where that table is not built (c7a73f1)."""
    monkeypatch.setattr(binance, "published_funding_caps", dict(caps), raising=False)


def _cap_table(rows) -> None:
    """DA-11's point-in-time cap table: (pair, effective-from, cap) rows kept in the contract data. Not built at
    c7a73f1 or on O17a: the test fails here, on an assertion, as "not built". Rename here if the DA names it
    differently, never the assertions."""
    keep = getattr(funding, "keep_cap", None)
    assert callable(keep), "not built (DA-11): funding.keep_cap(venue, pair, cap, effective_from), the point-in-time " \
                           "cap table in the contract data"
    for pair, t, cap in rows:
        keep("BINANCE", pair, cap, utc(t))


def _collect(binance, monkeypatch, rates: dict, pair: str = PAIR) -> list:
    """One collector pass (history._refresh_funding, as the hub runs it) over the venue's settled `rates` for `pair`;
    returns what reached the alerts inbox as (level, kind, message)."""
    import sleeve_fund.store as store_mod
    from sleeve_fund import history

    rows = [(ms(t), r) for t, r in sorted((utc(k), v) for k, v in rates.items())]
    monkeypatch.setattr(binance, "funding_loader", lambda p, start: [x for x in rows if x[0] >= start])
    monkeypatch.setattr(binance, "stats_loaders", {})
    sent = []

    class Inbox:
        def __init__(self, *a, **k):
            pass

        def event(self, sleeve, level, kind, message, ts=None):
            sent.append((level, kind, message))

    with monkeypatch.context() as m:
        m.setattr(store_mod, "Store", Inbox)
        history._warned.clear()
        history._refresh_funding(binance, pair, funding.DEFAULT_ROOT, None)
        history._warned.clear()
    funding._cache.clear()
    return sent


def _held_backtest(binance):
    """Long from 3 Oct 01:00: holds the 3 Oct 08:00 and 16:00 and 4 Oct 00:00 settlements."""
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-04 04:00"),
                 win(("2025-10-03 01:00", "2026-01-01", 1)))
    return r, {utc(x["ts"]): x for x in r.journal.funding_}


def _charged_as_settled(row, rate):
    assert row["rate"] == pytest.approx(rate)
    assert row["amount"] == pytest.approx(-row["qty"] * row["price"] * rate, rel=1e-6)


def _names(message: str, pair: str) -> bool:
    return pair in message or pair.replace("/", "") in message


def _shows(text: str, raw: float) -> bool:
    """`text` carries the raw value as the venue sent it (as a number or a percentage)."""
    if isnan(raw):
        return "nan" in text.lower()
    pct = raw * 100
    forms = {repr(raw), f"{raw:g}", f"{pct:g}%", f"{pct:.1f}%", f"{pct:.2f}%", f"{pct:.3f}%", f"{pct:.4f}%"}
    return any(f in text for f in forms)


def test_a_rate_within_the_instruments_own_published_cap_is_kept_and_charged_as_settled(binance, monkeypatch):
    """Plain (holds at c7a73f1 and on O17a; must keep holding under DA-11). BTC/USDT publishes a 0.75% cap: 0.5% at
    3 Oct 16:00 is kept as real, charged as settled, and raises no funding_over_cap alert."""
    _published_now(binance, monkeypatch, {PAIR: 0.0075})
    at = utc("2025-10-03 16:00")
    sent = _collect(binance, monkeypatch, {t: (0.005 if t == at else 0.0001) for t in CAP_TIMES})
    assert not [m for _, k, m in sent if k == "funding_over_cap"], sent
    kept = funding.rates("BINANCE", PAIR)
    assert kept.get(at) == pytest.approx(0.005), f"not kept: {kept.to_dict()}"
    _, rows = _held_backtest(binance)
    _charged_as_settled(rows[at], 0.005)


def test_the_interim_guard_alerts_a_btc_rate_over_the_published_cap_but_keeps_and_charges_it(binance, monkeypatch):
    """Plain, for the DA's interim guard on O17a (not in c7a73f1, so it fails there): with the static published cap as
    shipped (VenueProfile.published_funding_caps: 0.3% for BTC; not patched here), 0.5% at 3 Oct 16:00 raises one
    funding_over_cap alert naming BTC/USDT, and is still kept as real and charged as settled. Until DA-11 nothing
    over a published cap is refused on that cap alone."""
    at = utc("2025-10-03 16:00")
    sent = _collect(binance, monkeypatch, {t: (0.005 if t == at else 0.0001) for t in CAP_TIMES})
    over = [m for _, k, m in sent if k == "funding_over_cap"]
    assert len(over) == 1 and _names(over[0], PAIR), f"one funding_over_cap alert naming {PAIR}; sent {sent}"
    kept = funding.rates("BINANCE", PAIR)
    assert kept.get(at) == pytest.approx(0.005), f"not kept: {kept.to_dict()}"
    _, rows = _held_backtest(binance)
    _charged_as_settled(rows[at], 0.005)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_CAP)
def test_a_rate_above_the_instruments_own_cap_is_rejected_where_a_flat_cap_would_keep_it(binance, monkeypatch):
    """Two instruments, one rate. BTC/USDT's own cap is 0.75%; ETH/USDT's is 3% (the venue's widest here). 2% at
    3 Oct 16:00 is rejected on BTC and kept on ETH: no flat cap does both (one at or above 2% keeps both, one below
    rejects both). And 2% kept on BTC by hand is charged the baseline, 1 of 3. Today 2% is kept and charged on both."""
    caps = {PAIR: 0.0075, ETH: 0.03}
    _published_now(binance, monkeypatch, caps)
    _cap_table([(p, "2025-09-01", c) for p, c in caps.items()])
    at, ok = utc("2025-10-03 16:00"), utc("2025-10-03 08:00")
    btc = {t: {at: 0.02, ok: 0.005}.get(t, 0.0001) for t in CAP_TIMES}
    _collect(binance, monkeypatch, btc, PAIR)
    _collect(binance, monkeypatch, {t: (0.02 if t == at else 0.0001) for t in CAP_TIMES}, ETH)
    kept_btc, kept_eth = funding.rates("BINANCE", PAIR), funding.rates("BINANCE", ETH)
    assert kept_eth.get(at) == pytest.approx(0.02), "control: 2% within ETH's own 3% cap is kept"
    assert kept_btc.get(ok) == pytest.approx(0.005), "control: 0.5% within BTC's own cap is kept"
    assert at not in kept_btc.index, f"2% kept on BTC over its own 0.75% cap: {kept_btc.get(at)}"
    write_rates(btc)  # a hand-written file or a bad backfill
    r, rows = _held_backtest(binance)
    _paid_baseline([rows[at]])
    assert (r.funding_at_baseline, r.funding_held) == (1, 3)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_CAP)
def test_the_cap_is_point_in_time(binance, monkeypatch):
    """BTC/USDT's cap was 3% until 4 Oct 00:00 and is 0.75% from then (published now: 0.75%). 2% at 3 Oct 16:00 is
    within the cap that applied then: kept and charged as settled. 2% at 4 Oct 08:00 is rejected. funding.cap_at
    gives 3% then and 0.75% now. Today both are kept."""
    _published_now(binance, monkeypatch, {PAIR: 0.0075, ETH: 0.03})
    _cap_table([(PAIR, "2025-09-01", 0.03), (PAIR, "2025-10-04 00:00", 0.0075), (ETH, "2025-09-01", 0.03)])
    then, now = utc("2025-10-03 16:00"), utc("2025-10-04 08:00")
    _collect(binance, monkeypatch, {t: 0.02 if t in (then, now) else 0.0001 for t in CAP_TIMES})
    kept = funding.rates("BINANCE", PAIR)
    assert now not in kept.index, "2% kept at 4 Oct 08:00, over the 0.75% cap in force from 4 Oct 00:00"
    assert kept.get(then) == pytest.approx(0.02), "2% at 3 Oct 16:00 refused, though within the 3% that applied then"
    cap_at = getattr(funding, "cap_at", None)
    assert callable(cap_at), "not built (DA-11): funding.cap_at(venue, pair, ts) -> the cap that applied at ts"
    assert cap_at("BINANCE", PAIR, then) == pytest.approx(0.03)
    assert cap_at("BINANCE", PAIR, now) == pytest.approx(0.0075)
    _, rows = _held_backtest(binance)
    _charged_as_settled(rows[then], 0.02)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_CAP)
def test_a_missing_cap_uses_the_venues_widest_and_alerts(binance, monkeypatch):
    """The venue publishes caps for ETH/USDT (3%) and SOL/USDT (2%) but none for BTC/USDT, and none is kept for it:
    the WIDEST cap the venue publishes for a linear perpetual applies (3%), and one funding_cap_missing alert names
    BTC/USDT. 2% is kept and charged as settled (lean to accepting); 4% is rejected. Never a TypeError: the cap
    lookup for an instrument with no cap is the thing being built."""
    _published_now(binance, monkeypatch, {ETH: 0.03, "SOL/USDT": 0.02})
    _cap_table([(ETH, "2025-09-01", 0.03), ("SOL/USDT", "2025-09-01", 0.02)])
    within, over = utc("2025-10-03 08:00"), utc("2025-10-03 16:00")
    sent = _collect(binance, monkeypatch, {t: {within: 0.02, over: 0.04}.get(t, 0.0001) for t in CAP_TIMES})
    missing = [(lv, m) for lv, k, m in sent if k == "funding_cap_missing"]
    assert missing, f"no funding_cap_missing alert for {PAIR} (sent {sent})"
    assert all(lv == "warning" and "cap" in m.lower() and _names(m, PAIR) for lv, m in missing), missing
    kept = funding.rates("BINANCE", PAIR)
    assert kept.get(within) == pytest.approx(0.02), "lean to accepting: 2% is within the venue's widest cap"
    assert over not in kept.index, "4% kept, over the venue's widest cap (3%)"
    _, rows = _held_backtest(binance)
    _charged_as_settled(rows[within], 0.02)


@pytest.mark.strategy_errors
@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_CAP)
def test_a_rejected_rate_is_quarantined_raw_alerted_and_charged_the_baseline(binance, monkeypatch):
    """Never silently dropped: the raw value (2% over BTC's own 0.75% cap, NaN, -2%) is quarantined for the DA's
    review with a funding_invalid alert carrying it, and until reviewed the settlement is charged the baseline and
    counted at baseline. Today NaN is dropped with a printed line only, and 2% and -2% are kept and charged as real."""
    _published_now(binance, monkeypatch, {PAIR: 0.0075, ETH: 0.03})
    bad = {utc("2025-10-03 08:00"): 0.02, utc("2025-10-03 16:00"): float("nan"), utc("2025-10-04 00:00"): -0.02}
    quarantined = getattr(funding, "quarantined", None)
    assert callable(quarantined), ("not built (DA-11): funding.quarantined(venue, pair) -> the rejected settlements "
                                   "and their raw values, held for review")
    _cap_table([(PAIR, "2025-09-01", 0.0075), (ETH, "2025-09-01", 0.03)])
    sent = _collect(binance, monkeypatch, {t: bad.get(t, 0.0001) for t in CAP_TIMES})
    q = quarantined("BINANCE", PAIR)
    for t, raw in bad.items():
        assert t in q.index, f"{t:%d %b %H:%M} ({raw}) dropped, not quarantined"
        assert isnan(raw) and isnan(float(q[t])) or float(q[t]) == pytest.approx(raw)
    alerts = " | ".join(m for lv, k, m in sent if k == "funding_invalid" and lv == "warning")
    assert alerts, f"no funding_invalid alert (sent {sent})"
    for raw in bad.values():
        assert _shows(alerts, raw), f"the alert does not carry the raw value {raw}: {alerts}"
    kept = funding.rates("BINANCE", PAIR)
    assert not set(bad) & set(kept.index), "a rejected rate was also kept as real"
    r, rows = _held_backtest(binance)
    assert r.handler_error_count == 0, r.handler_errors[:1]
    _paid_baseline([rows[t] for t in bad])
    assert {rows[t].get("kind") for t in bad} == {"baseline"}
    assert (r.funding_at_baseline, r.funding_held) == (3, 3)


def test_a_simulated_perp_keeps_the_flat_fallback():
    """Plain. Kraken has no perpetual of its own and no venue rates to check against any cap: every settlement stays
    the flat 0.01% fallback. The short side is pinned by test_a_simulated_perp_charges_the_baseline_to_both_sides."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.venues import venue

    k = venue("KRAKEN")
    r = run_backtest("o17win", hourly_bars("2025-10-03 00:00", "2025-10-05 00:00"), k.instrument("BTC", "USD"),
                     win(("2025-10-03 01:00", "2026-01-01", 1)), starting_capital=10_000,
                     risk_profile="balanced", bar_minutes=60, half_spread=0)
    rows = r.journal.funding_
    assert len(rows) == 6 and all(x["rate"] == pytest.approx(BASELINE) for x in rows)
    _paid_baseline(rows)


# ---------------------------------------------------------------------------------------------------------------
# 8. #163 staleness episodes, per settlement (QA P1-O17a-10, -11, -12; Advisor 7 Oct ~02:12 O17a-11 FUNDING EPISODES;
#    the DA's event interface as accepted by the Head of QA, 7 Oct). Each settlement is stored, missing or never
#    published. A settlement t is missing once it is due, the 15-minute wait has passed and there is no rate. An episode
#    is a contiguous run of missing settlements: t joins the open episode o when o <= t and every settlement in [o, t)
#    is marked missing, otherwise t opens a new one with its own funding_stale (several can be open at once). After a
#    later settlement is stored, the hub's refresh first refetches from the earliest missing settlement (the backfill);
#    still missing 24 h after due, with a later settlement stored, t is never published (marked once, by the hub or
#    paper, never both). Episode o closes when every settlement from o to the latest due is stored or never published.
#    On start, paper rebuilds its watched settlements from the baseline rows of open episodes (never-published ones out).
#    Legacy funding_stale events without "from" read as an episode opened at the event's time.
#    MEMBERSHIP SUPERSEDED (Advisor 06:30 / 06:33): a settlement missed while an episode is open (any of its settlements
#    still provisionally missing or missing) joins it whatever lies between; the contiguity rule above is re-pinned by
#    the two cells that replace test_a_missing_settlement_not_contiguous_with_an_open_episode_opens_its_own (O17a-EPISODE).
#
#    Events (global, sleeve None; every text starts with the instrument tag "[BTC/USDT]" and names no venue, O17a-9):
#    funding_missing (info, once per settlement): "{tag} {t:%Y-%m-%d %H:%M} UTC settlement missing", plus
#      " (inferred time)" when t came from gap inference or was foreseen past the newest record;
#    funding_stale (warning, opens an episode): as before, plus "from {o:%Y-%m-%d %H:%M} UTC" (o its first missing
#      settlement); in paper its ts is the raise time, not the settlement;
#    funding_never_published (warning, once per settlement): "{tag} {t} UTC rate never published; baseline kept,
#      true-up impossible" (pinned loosely: "never published", "baseline", "true-up");
#    funding_stale_cleared (info, closes episode o): as before, plus "from {o} UTC".
#
#    The collector's pass runs at a simulated time `at`: its clock is history._now() (funding.stale's `now` too), and
#    the events it writes are stamped `at`. Each pass is a fresh hub process as far as the daily alert cap (_warned)
#    goes, so "once" must come from the journal, not from memory. Paper runs use 5-second trades unless the cell is
#    about trades more than 15 s apart (#146's unseen-trade deferral, UNSEEN_GAP_NS).
# ---------------------------------------------------------------------------------------------------------------

TAG = f"[{PAIR}]"
EPISODE = ("funding_stale", "funding_stale_cleared")
FUNDING_EVENTS = ("funding_missing", "funding_stale", "funding_never_published", "funding_stale_cleared")
# 2 Oct 16:00 and 3 Oct 00:00 are kept before the run (the instrument has a history); 08:00 and 16:00 per cell.
HISTORY = {"2025-10-02 16:00": 0.0001, "2025-10-03 00:00": 0.0001}
RATES8 = {**HISTORY, "2025-10-03 08:00": 0.0002, "2025-10-03 16:00": 0.0002, "2025-10-04 00:00": 0.0001,
          "2025-10-04 08:00": 0.0001}


def _at(t) -> str:
    return f"{utc(t):%Y-%m-%d %H:%M} UTC"


def _from(t) -> str:
    return f"from {_at(t)}"


def _hub_built() -> None:
    """Checked before any paper run, so a missing collector check fails here as "not built", never inside an engine
    callback (where the engine would swallow it)."""
    from sleeve_fund import history

    built = all(callable(getattr(funding, n, None)) for n in ("stale", "stale_open", "stale_tag"))
    assert built and callable(getattr(history, "_refresh_funding", None)), \
        "not built: the collector's per-instrument staleness episode (funding.stale / stale_open / stale_tag)"


def _restartable_journal():
    """A journal a paper strategy can be restarted on: the harness's replay creates the sleeve, a restart finds it."""
    store = journal()
    made, create = set(), store.create_sleeve

    def create_once(**kw):
        if kw["name"] not in made:
            made.add(kw["name"])
            return create(**kw)

    store.create_sleeve = create_once
    return store


class _Stamped:
    """The journal as the collector's pass at `at` sees it: every event it writes is stamped `at`."""

    def __init__(self, store, at):
        self._store, self._at = store, utc(at).to_pydatetime()

    def __getattr__(self, name):
        return getattr(self._store, name)

    def event(self, sleeve, level, kind, message, ts=None):
        return self._store.event(sleeve, level, kind, message, ts=ts or self._at)


def _hub_pass(m, b, store, at, venue: dict | None = None) -> None:
    """One pass of the hub's funding refresh (history._refresh_funding) at `at`, on the journal `store`. venue: the
    settled rates the venue's history holds at `at` (settlement -> rate; default none). The store is the file as
    kept (funding.rates), or the paper harness's view of it during a paper run."""
    import sleeve_fund.store as store_mod
    from sleeve_fund import history

    at = utc(at)
    rows = [(ms(t), r) for t, r in sorted((utc(k), v) for k, v in (venue or {}).items())]
    real_stale = funding.stale
    with m.context() as c:
        c.setattr(funding, "stale", lambda v, p, root=None, now=None: real_stale(v, p, root, now=at))
        c.setattr(history, "_now", lambda: at, raising=False)
        c.setattr(b, "funding_loader", lambda pair, start: [x for x in rows if x[0] >= start])
        c.setattr(b, "stats_loaders", {})
        inbox = _Stamped(store, at)
        c.setattr(store_mod, "Store", lambda *a, **k: inbox)
        history._warned.clear()
        history._refresh_funding(b, PAIR, funding.DEFAULT_ROOT, None)
        history._warned.clear()
    funding._cache.clear()


def _hub_during(monkeypatch, b, store, at_times) -> list:
    """Collector passes at `at_times` while a paper replay runs (at the strategy's first timer tick at or after each):
    the hub and the paper node side by side, as deployed. Returns the passes still to run (empty once all ran)."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    pending = sorted(utc(t) for t in at_times)
    orig = LongFlatStrategy._on_tick

    def tick(self, *a, **k):
        now = utc(self.clock.utc_now())
        while pending and now >= pending[0]:
            _hub_pass(monkeypatch, b, store, pending.pop(0))
        return orig(self, *a, **k)

    monkeypatch.setattr(LongFlatStrategy, "_on_tick", tick)
    return pending


def _fresh_hub(monkeypatch) -> None:
    from sleeve_fund import history

    monkeypatch.setattr(history, "_stale", set(), raising=False)


def _kept(monkeypatch, rates: dict) -> None:
    """The store file as the collector keeps it, outside a paper run (the store reader itself again)."""
    from o17_harness import _REAL_RATES, write_rates

    monkeypatch.setattr(funding, "rates", _REAL_RATES)
    write_rates({utc(k): v for k, v in rates.items()})


def _events(store, *which) -> list:
    """The instrument's funding events, oldest first."""
    which = which or FUNDING_EVENTS
    evs = [dict(e, ts=utc(e["ts"])) for e in reversed(store.events(None, limit=5000)) if e["kind"] in which]
    return sorted(evs, key=lambda e: e["ts"])


def _show(evs) -> list:
    return [(f"{e['ts']:%d %H:%M}", e["kind"], e["message"][:110]) for e in evs]


def _texts_ok(store) -> None:
    """Every funding event starts with the instrument tag and names no venue (O17a-9)."""
    for e in _events(store):
        assert e["message"].startswith(TAG), f"{e['kind']} does not start with {TAG}: {e['message']}"
        assert not re.search(r"binance", e["message"], re.I), f"{e['kind']} names the venue: {e['message']}"


def _missing_of(store, t) -> list:
    return [e for e in _events(store, "funding_missing") if f"{_at(t)} settlement missing" in e["message"]]


def _stales(store) -> list:
    return _events(store, "funding_stale")


def _cleared(store) -> list:
    return _events(store, "funding_stale_cleared")


def _never(store, t=None) -> list:
    return [e for e in _events(store, "funding_never_published") if t is None or _at(t) in e["message"]]


def _built_missing(store, t) -> None:
    assert _events(store, "funding_missing"), \
        f"not built: no funding_missing marks (the per-settlement model); events {_show(_events(store))}"
    assert len(_missing_of(store, t)) == 1, f"{_at(t)} not marked missing exactly once: {_show(_events(store))}"


def test_each_missing_settlement_is_marked_once_after_the_15_minute_wait_and_one_outage_alerts_once(
        tmp_path, monkeypatch, binance):
    """Held 07:52 -> 16:25; neither 08:00 nor 16:00 is ever published (the store keeps 2 Oct 16:00 and 3 Oct 00:00).
    Each is marked funding_missing once, between 15 and 16 minutes after it settled, flagged "(inferred time)"
    (foreseen past the newest record). 16:00 joins the episode 08:00 opened: one funding_stale "from 2025-10-03
    08:00 UTC", raised at 08:15 (the raise time), and nothing clears it."""
    store = journal()
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 16:25", 1)), "2025-10-03 07:50", 520,
          rates=HISTORY, store=store)
    for t in ("2025-10-03 08:00", "2025-10-03 16:00"):
        _built_missing(store, t)
        (e,) = _missing_of(store, t)
        assert utc(t) + pd.Timedelta(minutes=15) <= e["ts"] < utc(t) + pd.Timedelta(minutes=16), _show([e])
        assert e["level"] == "info" and "(inferred time)" in e["message"], _show([e])
    stale = _stales(store)
    assert len(stale) == 1 and _from("2025-10-03 08:00") in stale[0]["message"], _show(_events(store))
    assert stale[0]["level"] == "warning" and utc("2025-10-03 08:15") <= stale[0]["ts"] < utc("2025-10-03 08:16")
    assert not _cleared(store), _show(_events(store))
    _texts_ok(store)


def test_a_rate_published_inside_the_15_minute_wait_is_never_missing(tmp_path, monkeypatch, binance):
    """The 15-minute boundary: 08:00 is published at 08:14 (inside the wait: settled, never missing); 16:00 at 16:16
    (one minute past it: missing at 16:15, an episode from 16:00, closed once the rate is in, from 16:16)."""
    store = journal()
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-03 16:40", 1)), "2025-10-03 07:50",
                530, rates=RATES8, store=store,
                published={"2025-10-03 08:00": "2025-10-03 08:14", "2025-10-03 16:00": "2025-10-03 16:16"})
    _built_missing(store, "2025-10-03 16:00")
    assert not _missing_of(store, "2025-10-03 08:00"), _show(_events(store))
    (m16,) = _missing_of(store, "2025-10-03 16:00")
    assert utc("2025-10-03 16:15") <= m16["ts"] < utc("2025-10-03 16:16"), _show([m16])
    (stale,) = _stales(store)
    assert _from("2025-10-03 16:00") in stale["message"], _show([stale])
    (cleared,) = _cleared(store)
    assert cleared["ts"] >= utc("2025-10-03 16:16") and _from("2025-10-03 16:00") in cleared["message"], _show([cleared])
    kinds_ = {r["ts"]: r.get("kind") for r in out["funding"]}
    assert kinds_.get(utc("2025-10-03 08:00")) == "settled", kinds_
    _texts_ok(store)


def _repin_run(tmp_path, monkeypatch, binance, store, published=None):
    """4 h records (00:00, 04:00, 08:00), then 16:00 kept; 12:00 and 20:00 missing at their 15-minute wait (12:00 unless
    `published` brings it in first). Held throughout; paper 07:50 -> 20:30."""
    rates = {**HISTORY, "2025-10-03 04:00": 0.0001, "2025-10-03 08:00": 0.0002, "2025-10-03 16:00": 0.0002}
    rates.update({"2025-10-03 12:00": 0.0002} if published else {})
    return paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-06 00:00", 1)), "2025-10-03 07:50",
                 12 * 60 + 40, rates=rates, store=store, published=published)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_EPISODE)
def test_a_miss_while_the_open_episode_is_unresolved_joins_it_whatever_lies_between(tmp_path, monkeypatch, binance):
    """Re-pinned per Advisor 06:30 (episode membership; confirmed 06:33, which supersedes the contiguity reading this
    section's header gives). Replaces test_a_missing_settlement_not_contiguous_with_an_open_episode_opens_its_own.
    A kept 16:00 lies between a missing 12:00 and a missing 20:00. At 20:15 12:00 is still unresolved (provisionally
    missing: s_next unknown), so the episode opened at 12:15 is open and 20:00 joins it: one funding_stale, from
    12:00, raised at 12:15, nothing cleared, and 20:00 marked missing once (its state, not a new alert)."""
    store = journal()
    _repin_run(tmp_path, monkeypatch, binance, store)
    for t in ("2025-10-03 12:00", "2025-10-03 20:00"):
        _built_missing(store, t)
    stale = _stales(store)
    assert len(stale) == 1, \
        f"20:00 missed while the episode from 12:00 is unresolved must join it, not raise its own: {_show(stale)}"
    assert _from("2025-10-03 12:00") in stale[0]["message"], _show(stale)
    assert utc("2025-10-03 12:15") <= stale[0]["ts"] < utc("2025-10-03 12:16"), _show(stale)
    assert not _cleared(store), f"the episode was closed while 12:00 is unresolved: {_show(_events(store, *EPISODE))}"
    _texts_ok(store)


def test_a_miss_after_the_episode_resolved_opens_a_new_one(tmp_path, monkeypatch, binance):
    """Re-pinned per Advisor 06:30 (episode membership; confirmed 06:33). The other half: 12:00 is published at 13:00,
    so its episode (opened 12:15) closes between 13:00 and 13:16, before 20:15. 20:00, missed after that, opens a new
    episode: two funding_stale (from 12:00, then from 20:00 at 20:15), the first closed, the second open at 20:30."""
    store = journal()
    _repin_run(tmp_path, monkeypatch, binance, store, published={"2025-10-03 12:00": "2025-10-03 13:00"})
    for t in ("2025-10-03 12:00", "2025-10-03 20:00"):
        _built_missing(store, t)
    stale = _stales(store)
    assert len(stale) == 2, f"two outages, two episodes: {_show(_events(store, *EPISODE))}"
    assert _from("2025-10-03 12:00") in stale[0]["message"] and _from("2025-10-03 20:00") in stale[1]["message"], \
        _show(stale)
    assert utc("2025-10-03 20:15") <= stale[1]["ts"] < utc("2025-10-03 20:16"), _show(stale)
    closed = _cleared(store)
    assert len(closed) == 1 and _from("2025-10-03 12:00") in closed[0]["message"], \
        f"only the episode from 12:00 closes: {_show(_events(store, *EPISODE))}"
    assert utc("2025-10-03 13:00") <= closed[0]["ts"] < utc("2025-10-03 13:16"), _show(closed)
    _texts_ok(store)


def test_two_missing_the_first_published_late_never_reads_clear_and_alerts_once(tmp_path, monkeypatch, binance):
    """P1-O17a-10. 08:00 is published at 17:00 and 16:00 never; held throughout; the hub passes at 08:30, 12:00,
    16:30, 17:30 and 18:30. 16:00 is still charged the baseline, so the hub must not write "kept up again" at 17:30
    (on 89d4f6e it does, and the strategy then opens a second funding_stale for the same outage). One funding_stale
    from 08:00, no funding_stale_cleared."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    left = _hub_during(monkeypatch, binance, store, ["2025-10-03 08:30", "2025-10-03 12:00", "2025-10-03 16:30",
                                                     "2025-10-03 17:30", "2025-10-03 18:30"])
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-05 00:00", 1)), "2025-10-03 07:50",
                11 * 60, rates=RATES8, store=store,
                published={"2025-10-03 08:00": "2025-10-03 17:00", "2025-10-03 16:00": NEVER})
    assert not left, f"setup: hub passes not run {left}"
    rows = [(r["ts"], r.get("kind")) for r in out["funding"]]
    assert (utc("2025-10-03 16:00"), "baseline") in rows, f"setup: 16:00 charged the baseline {rows}"
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale"], \
        f"one outage, one funding_stale and no clear while 16:00 is missing: {_show(got)}"
    assert _from("2025-10-03 08:00") in got[0]["message"], _show(got)
    _texts_ok(store)


def test_a_paper_restart_mid_episode_keeps_it_open_and_alerts_once(tmp_path, monkeypatch, binance):
    """P1-O17a-10, with the paper node restarted at 16:45 (a deploy) after 08:00 and 16:00 were both charged the
    baseline. 08:00 arrives at 17:00, 16:00 never. The restarted strategy rebuilds its watched settlements from the
    journal's baseline rows; hub passes at 17:30, 18:30, 20:00, 22:00 and 23:45. The inbox is not clear from 17:30
    to 00:00: one funding_stale from 08:00, no clear, each settlement marked missing once across the restart."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    params = win(("2025-10-03 07:52", "2025-10-05 00:00", 1))
    run = dict(rates=RATES8, store=store, name="w",
               published={"2025-10-03 08:00": "2025-10-03 17:00", "2025-10-03 16:00": NEVER})
    _hub_during(monkeypatch, binance, store, ["2025-10-03 08:30", "2025-10-03 16:30"])
    paper(tmp_path, monkeypatch, binance, params, "2025-10-03 07:50", 8 * 60 + 55, **run)  # stops at 16:45
    left = _hub_during(monkeypatch, binance, store, ["2025-10-03 17:30", "2025-10-03 18:30", "2025-10-03 20:00",
                                                     "2025-10-03 22:00", "2025-10-03 23:45"])
    paper(tmp_path, monkeypatch, binance, params, "2025-10-03 17:10", 6 * 60 + 50, **run)  # restarted, to 00:00
    assert not left, f"setup: hub passes not run {left}"
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale"], f"read clear after the restart: {_show(got)}"
    assert _from("2025-10-03 08:00") in got[0]["message"], _show(got)
    for t in ("2025-10-03 08:00", "2025-10-03 16:00"):
        _built_missing(store, t)
    _texts_ok(store)


def test_a_restarted_paper_strategy_rebuilds_its_watched_settlements_and_closes_the_episode_itself(
        tmp_path, monkeypatch, binance):
    """The rebuild itself (P1-O17a-10's fix): 08:00 and 16:00 both missing and charged the baseline; paper restarted
    at 16:45; 08:00 arrives at 17:00 and 16:00 at 18:00. No hub pass after the restart, so only the restarted
    strategy can see the rates arrive: it closes the episode (from 08:00) between 18:00 and 18:15. On 89d4f6e the
    restarted strategy watches nothing and the episode stays open until the hub's next pass."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    params = win(("2025-10-03 07:52", "2025-10-05 00:00", 1))
    run = dict(rates=RATES8, store=store, name="w",
               published={"2025-10-03 08:00": "2025-10-03 17:00", "2025-10-03 16:00": "2025-10-03 18:00"})
    _hub_during(monkeypatch, binance, store, ["2025-10-03 08:30", "2025-10-03 16:30"])
    paper(tmp_path, monkeypatch, binance, params, "2025-10-03 07:50", 8 * 60 + 55, **run)  # stops at 16:45
    _hub_during(monkeypatch, binance, store, [])
    paper(tmp_path, monkeypatch, binance, params, "2025-10-03 17:10", 90, **run)  # restarted, to 18:40
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got[:1]] == ["funding_stale"], f"setup: the episode opened {_show(got)}"
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], \
        f"the restarted strategy did not close the episode once both rates were in: {_show(got)}"
    assert utc("2025-10-03 18:00") <= got[1]["ts"] < utc("2025-10-03 18:15"), _show(got)
    assert _from("2025-10-03 08:00") in got[1]["message"], _show(got)
    _texts_ok(store)


def _sparse_after(monkeypatch, after, every: int = 30) -> None:
    """From `after` on, the harness's trades and quotes come `every` seconds apart (more than #146's 15 s
    UNSEEN_GAP_NS), instead of every 5 s."""
    from sleeve_fund.paper.recorder import Recorder

    cut = utc(after).value // 1_000_000_000
    for name in ("trade", "quote"):
        orig = getattr(Recorder, name)

        def keep(self, tick, _orig=orig):
            s = tick.ts_event // 1_000_000_000
            if s < cut or (s - cut) % every == 0:
                return _orig(self, tick)

        monkeypatch.setattr(Recorder, name, keep)


def test_the_strategy_closes_an_episode_itself_with_trades_more_than_15_s_apart(tmp_path, monkeypatch, binance):
    """P1-O17a-12. Trades every 5 s until 09:00 (paper opens the episode for 08:00 at 08:15), then every 30 s. 08:00
    is published at 12:00; the hub passes at 08:30, 10:00 and 12:30. The strategy's recovery watch runs before #146's
    unseen-trade deferral, so it closes the episode itself between 12:00 and 12:15, not the hub at 12:30."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    _sparse_after(monkeypatch, "2025-10-03 09:00")
    left = _hub_during(monkeypatch, binance, store, ["2025-10-03 08:30", "2025-10-03 10:00", "2025-10-03 12:30"])
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-05 00:00", 1)), "2025-10-03 07:50",
          5 * 60, rates=RATES8, store=store, published={"2025-10-03 08:00": "2025-10-03 12:00"})
    assert not left, f"setup: hub passes not run {left}"
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got[:1]] == ["funding_stale"], f"setup: the episode opened at 08:15 {_show(got)}"
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], _show(got)
    assert got[1]["ts"] < utc("2025-10-03 12:15"), \
        f"closed at {got[1]['ts']:%H:%M}, by the hub's pass, not by the strategy when 08:00 arrived: {_show(got)}"
    assert got[1]["ts"] >= utc("2025-10-03 12:00") and _from("2025-10-03 08:00") in got[1]["message"], _show(got)
    _texts_ok(store)


def _open_08(tmp_path, monkeypatch, binance, store) -> None:
    """Paper holds 07:50 -> 08:20 with 08:00 unpublished: it marks 08:00 missing and opens the episode (08:15)."""
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-05 00:00", 1)), "2025-10-03 07:50", 30,
          rates=HISTORY, store=store, name="w")


def test_the_collector_backfills_a_missing_settlement_from_the_venue_history_first(tmp_path, monkeypatch, binance):
    """O17a-11 ruling: once a later settlement is stored, the hub's refresh first refetches from the earliest missing
    one. Paper opened the episode for 08:00; the store then kept 16:00 (the venue published 08:00 late, after the
    collector had moved past it). At 17:30 the venue's history has 08:00: the pass stores it, the episode closes
    (from 08:00), and nothing is called never published. On 89d4f6e the refresh tops up from the newest kept rate
    only, so 08:00 is never fetched."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    _open_08(tmp_path, monkeypatch, binance, store)
    venue = {k: v for k, v in RATES8.items() if utc(k) <= utc("2025-10-03 16:00")}
    _kept(monkeypatch, {k: v for k, v in venue.items() if k != "2025-10-03 08:00"})
    _hub_pass(monkeypatch, binance, store, "2025-10-03 16:30", {k: v for k, v in venue.items() if k != "2025-10-03 08:00"})
    _hub_pass(monkeypatch, binance, store, "2025-10-03 17:30", venue)
    kept = funding.rates("BINANCE", PAIR)
    assert kept.get(utc("2025-10-03 08:00")) == pytest.approx(0.0002), \
        f"08:00 not backfilled from the venue's history: kept {list(kept.index.strftime('%d %H:%M'))}"
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], _show(got)
    assert _from("2025-10-03 08:00") in got[1]["message"] and not _never(store), _show(_events(store))
    _texts_ok(store)


def _pass_days(monkeypatch, binance, store, rates: dict, at_times) -> None:
    """Hub passes at `at_times`, the store and the venue's history holding `rates` settled by then (published on
    time)."""
    for at in at_times:
        upto = {k: v for k, v in rates.items() if utc(k) + pd.Timedelta(minutes=5) <= utc(at)}
        _kept(monkeypatch, upto)
        _hub_pass(monkeypatch, binance, store, at, upto)


LATER = {k: v for k, v in RATES8.items() if k != "2025-10-03 08:00"}  # 08:00 never published; the rest on time


def test_a_settlement_still_missing_24_h_after_due_is_never_published_once_with_its_own_alert(tmp_path, monkeypatch,
                                                                                              binance):
    """O17a-11 ruling. 08:00 is never published; every later settlement is. Hub passes at 16:30, 00:30 and 4 Oct
    07:50 (23 h 50 min after due: still missing); then paper runs again from 07:55 to 08:45 (rebuilding its watched
    settlements) beside hub passes at 08:30 and 08:40; one more pass at 09:30. 08:00 becomes never published once:
    one funding_never_published (a warning, at or after 4 Oct 08:00, "never published", "baseline", "true-up"),
    whether the hub or paper marks it; the episode then closes (from 08:00), with no second funding_stale."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    _open_08(tmp_path, monkeypatch, binance, store)
    _pass_days(monkeypatch, binance, store, LATER, ["2025-10-03 16:30", "2025-10-04 00:30", "2025-10-04 07:50"])
    assert not _never(store), f"never published before 24 h: {_show(_events(store))}"
    left = _hub_during(monkeypatch, binance, store, ["2025-10-04 08:30", "2025-10-04 08:40"])
    paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-05 00:00", 1)), "2025-10-04 07:55", 50,
          rates=LATER, store=store, name="w", published={"2025-10-03 08:00": NEVER})
    assert not left, f"setup: hub passes not run {left}"
    _pass_days(monkeypatch, binance, store, LATER, ["2025-10-04 09:30"])
    never = _never(store, "2025-10-03 08:00")
    assert len(never) == 1, f"08:00 not marked never published exactly once: {_show(_events(store))}"
    (np_,) = never
    text = np_["message"].lower()
    assert np_["level"] == "warning" and np_["ts"] >= utc("2025-10-04 08:00"), _show(never)
    assert "never published" in text and "baseline" in text and "true-up" in text, np_["message"]
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], _show(got)
    assert got[1]["ts"] >= utc("2025-10-04 08:00") and _from("2025-10-03 08:00") in got[1]["message"], _show(got)
    _texts_ok(store)


def test_a_never_published_settlement_does_not_silence_a_later_outage(tmp_path, monkeypatch, binance):
    """P1-O17a-11. 08:00 is never published; the feed keeps up from 16:00 to 4 Oct 08:00 and then stops. 08:00 is
    never published by the 4 Oct 08:30 pass and its episode closes; the new outage (4 Oct 16:00 on) raises its own
    funding_stale, from 4 Oct 16:00, by the 5 Oct 02:00 pass. On 89d4f6e the episode held open by 08:00 silences it."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    _open_08(tmp_path, monkeypatch, binance, store)
    _pass_days(monkeypatch, binance, store, LATER, ["2025-10-03 16:30", "2025-10-04 00:30", "2025-10-04 08:30",
                                                    "2025-10-04 16:30", "2025-10-05 02:00"])
    stale = _stales(store)
    assert len(stale) == 2, f"the later outage raised no funding_stale of its own: {_show(_events(store))}"
    assert _from("2025-10-04 16:00") in stale[1]["message"], _show(stale)
    assert utc("2025-10-04 16:15") <= stale[1]["ts"] <= utc("2025-10-05 02:00"), _show(stale)
    first_closed = [e for e in _cleared(store) if _from("2025-10-03 08:00") in e["message"]]
    assert first_closed and first_closed[0]["ts"] < stale[1]["ts"], _show(_events(store))
    _texts_ok(store)


def test_a_missing_settlement_is_not_never_published_while_nothing_later_is_stored(tmp_path, monkeypatch, binance):
    """Never published needs a later settlement stored: the feed stops after 3 Oct 00:00, so at 4 Oct 08:30 and 09:00
    (past 24 h) 08:00 is still missing, not never published, and its episode stays open (an outage, not a hole)."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    _open_08(tmp_path, monkeypatch, binance, store)
    _pass_days(monkeypatch, binance, store, HISTORY, ["2025-10-04 08:30", "2025-10-04 09:00"])
    _built_missing(store, "2025-10-03 08:00")
    assert not _never(store), _show(_events(store))
    got = _events(store, *EPISODE)
    assert got and got[-1]["kind"] == "funding_stale", _show(got)
    _texts_ok(store)


def test_a_legacy_staleness_event_without_from_is_read_as_opened_at_its_time(tmp_path, monkeypatch, binance):
    """A funding_stale written before this interface (no "from"), at 3 Oct 08:15, is an episode opened at 08:15. The
    store keeps every settlement to 16:00: the 16:30 pass closes it once, "from 2025-10-03 08:15 UTC"."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    store.event(None, "warning", "funding_stale", f"{TAG} No settled funding rate from the venue for {PAIR} at "
                "03 Oct 2025 08:00 UTC, 15 minutes after it settled; charging the baseline, whichever side is held, "
                "until it arrives", ts=utc("2025-10-03 08:15").to_pydatetime())
    rates = {k: v for k, v in RATES8.items() if utc(k) <= utc("2025-10-03 16:00")}
    _pass_days(monkeypatch, binance, store, rates, ["2025-10-03 16:30"])
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], _show(got)
    assert _from("2025-10-03 08:15") in got[1]["message"], _show(got)
    _texts_ok(store)


# ---------------------------------------------------------------------------------------------------------------
# 9. Reading a gap between stored records (markets.settlement_times / _missing_between; Advisor 7 Oct ~02:40 "FUNDING
#    GAP READING, full text", which supersedes the 02:13 line's "(a)-(d)"). P = the profile's interval (8 h here),
#    g = the gap between two stored records, s_prev / s_next = the steps before / after it; missing settlements are
#    placed every f = min(P, s_prev) from the gap's start, strictly before its end.
#    (a) g > P: missing, filled at f.  (b) g <= P, g > s_prev and s_next == s_prev: a lost record after a move to
#    shorter settlements, filled at f.  (c), as amended (Advisor 04:31, 04:33): g <= P is NOT missing only once
#    s_next >= g (the venue moving back towards its schedule); while s_next is unknown and g > s_prev it is
#    PROVISIONALLY missing (baseline and alert), reversed by its own journaled "reversal" row and its alert closed when
#    the next record shows s_next >= g, or missing for good under (b) when s_next == s_prev (no new row). At most one
#    charge and one reversal per settlement. s_prev and s_next come from STORED records only; provisional settlements
#    feed foresight only: past the newest record, the published interval, else min(latest step counting provisional
#    settlements, P).  (d) a lengthening beyond P is honoured only where the venue publishes its
#    interval (VenueProfile.funding_interval), and only on the newest gap; a sustained wider step is not a lengthening.
#    Default: a gap matching none of (a)-(c) is missing (adverse), with an alert. 02:13 pins: every inferred missing
#    settlement enters the per-settlement model flagged "inferred time"; the venue-history backfill replaces it; the
#    8 h single-missing case is filled at 8 h and charged once, long and short.
#    Backtest cells check every charge from 3 Oct 00:00 to the last bar: settled at each stored record, the baseline
#    (kind "baseline") at each inferred one, each exactly once.
# ---------------------------------------------------------------------------------------------------------------

def _grid(start, end, hours) -> list:
    return list(pd.date_range(utc(start), utc(end), freq=f"{hours}h"))


B8 = _grid("2025-10-01 00:00", "2025-10-03 00:00", 8)  # 8-hourly records up to 3 Oct 00:00


def _t(*times) -> list:
    return [utc(f"2025-{t}") for t in times]


# name: (records after B8, last bar, the charges expected from 3 Oct 00:00: settlement -> kind)
S_, B_ = "settled", "baseline"
GAPS = {
    # (a) 3 Oct 00:00 -> 16:00, g = 16 h > P: 08:00 filled at f = 8 h (02:13: the 8 h single-missing case)
    "a-8h-one-missing": (_t("10-03 16:00", "10-04 00:00", "10-04 08:00"), "2025-10-04 04:00",
                         {"10-03 00:00": S_, "10-03 08:00": B_, "10-03 16:00": S_, "10-04 00:00": S_}),
    # (a) at its edge, g = P + 1 min (a record at 08:01): see section 9b, the record snap (Advisor 03:12)
    # (b) moved to 4 h at 12:00 (s_prev = 4 h), 16:00 lost: g = 8 h <= P, > s_prev, s_next == s_prev: 16:00 at f = 4 h
    "b-lost-right-after-a-shortening": (_t("10-03 08:00", "10-03 12:00", "10-03 20:00", "10-04 00:00", "10-04 04:00",
                                           "10-04 08:00"), "2025-10-04 06:00",
                                        {"10-03 00:00": S_, "10-03 08:00": S_, "10-03 12:00": S_, "10-03 16:00": B_,
                                         "10-03 20:00": S_, "10-04 00:00": S_, "10-04 04:00": S_}),
    # (c) 4 h steps, then back to 8 h: 08:00 -> 16:00, g = P, > s_prev, s_next = 8 h >= g: no 12:00
    "c-gap-equal-to-P-step-after-as-wide": (_t("10-03 04:00", "10-03 08:00", "10-03 16:00", "10-04 00:00",
                                               "10-04 08:00"), "2025-10-04 04:00",
                                            {"10-03 00:00": S_, "10-03 04:00": S_, "10-03 08:00": S_,
                                             "10-03 16:00": S_, "10-04 00:00": S_}),
    # amended (c) (Advisor 04:31, 04:33): the same gap ending the records (s_next unknown, g > s_prev) is PROVISIONALLY
    # missing: 12:00 at the baseline; foresight past 16:00 is min(4 h counting the provisional 12:00, P), so 20:00 is
    # due and at the baseline too. s_prev / s_next come from stored records only.
    "c-gap-ending-the-records-is-provisionally-missing": (_t("10-03 04:00", "10-03 08:00", "10-03 16:00"),
                                                          "2025-10-03 23:00",
                                                          {"10-03 00:00": S_, "10-03 04:00": S_, "10-03 08:00": S_,
                                                           "10-03 12:00": B_, "10-03 16:00": S_, "10-03 20:00": B_}),
    # default: 2 h step (00:00 -> 02:00), then g = 6 h (02:00 -> 08:00) <= P, > s_prev, s_prev < s_next (4 h) < g:
    # missing, filled at f = 2 h (04:00, 06:00)
    "default-none-of-a-to-c": (_t("10-03 02:00", "10-03 08:00", "10-03 12:00", "10-03 16:00", "10-03 20:00"),
                               "2025-10-03 23:00",
                               {"10-03 00:00": S_, "10-03 02:00": S_, "10-03 04:00": B_, "10-03 06:00": B_,
                                "10-03 08:00": S_, "10-03 12:00": S_, "10-03 16:00": S_, "10-03 20:00": S_}),
    # (d) a sustained 16 h step and no published interval: every 16 h gap is missing at 8 h, the adverse side
    "d-sustained-wider-step-not-published": (_t("10-03 16:00", "10-04 08:00", "10-05 00:00"), "2025-10-05 04:00",
                                             {"10-03 00:00": S_, "10-03 08:00": B_, "10-03 16:00": S_,
                                              "10-04 00:00": B_, "10-04 08:00": S_, "10-04 16:00": B_,
                                              "10-05 00:00": S_}),
    # (d) the venue publishes 16 h: honoured on the newest gap only (4 Oct 08:00 -> 5 Oct 00:00); the older ones stay
    "d-published-16h-newest-gap-only": (_t("10-03 16:00", "10-04 08:00", "10-05 00:00"), "2025-10-05 04:00",
                                        {"10-03 00:00": S_, "10-03 08:00": B_, "10-03 16:00": S_, "10-04 00:00": B_,
                                         "10-04 08:00": S_, "10-05 00:00": S_}),
}


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("case", list(GAPS))
def test_a_gap_between_stored_records_is_read_by_the_rules(binance, monkeypatch, case, side):
    records, last_bar, want = GAPS[case]
    if case.startswith("d-published"):
        monkeypatch.setattr(binance, "funding_interval", lambda pair: pd.Timedelta(hours=16), raising=False)
    write_rates({t: 0.0001 for t in B8 + records})
    r = backtest(binance, hourly_bars("2025-10-02 00:00", last_bar), win(("2025-10-02 01:00", "2026-01-01", side)))
    got = sorted((utc(x["ts"]), x.get("kind")) for x in r.journal.funding_ if utc(x["ts"]) >= utc("2025-10-03 00:00"))
    expect = sorted((utc(f"2025-{t}"), k) for t, k in want.items())
    assert got == expect, (f"{case}: charged {[(f'{t:%d %H:%M}', k) for t, k in got]}, the rules give "
                           f"{[(f'{t:%d %H:%M}', k) for t, k in expect]}")
    for x in r.journal.funding_:
        if x.get("kind") == B_:
            assert x["amount"] < 0, x  # the baseline is paid whichever side is held


R_ = "reversal"  # the journaled row that reverses a provisional baseline (amended (c), Advisor 04:31)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_c_step_after_as_wide_reverses_both_in_the_backtest(binance, side):
    """Sibling of c-gap-ending-the-records-is-provisionally-missing, with 4 Oct 00:00 stored (run to 02:00). Gap
    08:00 -> 16:00: s_next = 8 h >= g, so 12:00 is not missing. Gap 16:00 -> 00:00: from STORED records s_prev = 8 h
    (08:00 -> 16:00), so g <= s_prev is no gap and 20:00 is not missing either (counting the provisional 12:00 would
    make s_prev 4 h and wrongly keep 20:00 missing). The backtest may book this net (no rows) or as a baseline and its
    own reversal row each: 12:00 and 20:00 each net to zero, with at most one charge and one reversal, no orphan
    reversal, and every record charged once as settled."""
    write_rates({t: 0.0001 for t in B8 + _t("10-03 04:00", "10-03 08:00", "10-03 16:00", "10-04 00:00")})
    r = backtest(binance, hourly_bars("2025-10-02 00:00", "2025-10-04 02:00"),
                 win(("2025-10-02 01:00", "2026-01-01", side)))
    rows = [(utc(x["ts"]), x.get("kind"), x["amount"]) for x in r.journal.funding_
            if utc(x["ts"]) >= utc("2025-10-03 00:00")]
    shown = [(f"{t:%d %H:%M}", k, round(a, 6)) for t, k, a in sorted(rows, key=lambda x: (x[0], str(x[1])))]
    for t in _t("10-03 12:00", "10-03 20:00"):
        at = [(k, a) for ts, k, a in rows if ts == t]
        kinds_at = [k for k, _ in at]
        assert kinds_at.count(B_) <= 1 and kinds_at.count(R_) <= 1 and set(kinds_at) <= {B_, R_}, shown
        assert (R_ in kinds_at) == (B_ in kinds_at), f"{t:%H:%M}: a charge without its reversal, or an orphan: {shown}"
        assert sum(a for _, a in at) == pytest.approx(0, abs=1e-9), f"{t:%H:%M} does not net to zero: {shown}"
    others = sorted((ts, k) for ts, k, _ in rows if ts not in _t("10-03 12:00", "10-03 20:00"))
    assert others == [(t, S_) for t in _t("10-03 00:00", "10-03 04:00", "10-03 08:00", "10-03 16:00", "10-04 00:00")], \
        shown


def _gap_paper(tmp_path, monkeypatch, binance, store, rates, start, minutes, side=1, opens=None, published=None):
    opens = opens or utc(start) + pd.Timedelta(minutes=2)
    return paper(tmp_path, monkeypatch, binance, win((opens, "2025-10-05 00:00", side)), start, minutes,
                 rates=rates, store=store, published=published)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_paper_the_8h_single_missing_settlement_is_inferred_marked_and_charged_once(tmp_path, monkeypatch, binance,
                                                                                    side):
    """02:13 pin, paper: 08:00 is never published, 16:00 is on time. 08:00 is foreseen past the newest record, marked
    missing between 08:15 and 08:16 "(inferred time)", charged the baseline once (and not again when 16:00 arrives and
    the gap 00:00 -> 16:00 is read by (a)); 16:00 is charged as settled."""
    store = journal()
    out = _gap_paper(tmp_path, monkeypatch, binance, store, {**HISTORY, "2025-10-03 16:00": 0.0002},
                     "2025-10-03 07:50", 530, side)
    _built_missing(store, "2025-10-03 08:00")
    (m,) = _missing_of(store, "2025-10-03 08:00")
    assert "(inferred time)" in m["message"] and utc("2025-10-03 08:15") <= m["ts"] < utc("2025-10-03 08:16"), _show([m])
    rows = sorted((r["ts"], r.get("kind")) for r in out["funding"])
    assert rows == [(utc("2025-10-03 08:00"), B_), (utc("2025-10-03 16:00"), S_)], rows
    _paid_baseline([r for r in out["funding"] if r.get("kind") == B_])
    _texts_ok(store)


def test_paper_a_record_lost_right_after_a_shortening_is_missing_at_the_15_minute_wait(tmp_path, monkeypatch, binance):
    """(b) in paper: the venue moved to 4 h at 12:00 (08:00, 12:00 stored); 16:00 is never published, 20:00 is. 16:00
    (12:00 + 4 h, foreseen) is marked missing "(inferred time)" between 16:15 and 16:16 (the 15-minute wait, not a
    longer one), opens an episode from 16:00 and is charged the baseline once; 12:00 and 20:00 are settled. At 20:30
    s_next is unknown, so 16:00 is still (provisionally) missing: no reversal row (amended (c), Advisor 04:31)."""
    store = journal()
    rates = {**{f"{t:%Y-%m-%d %H:%M}": 0.0001 for t in B8}, "2025-10-03 08:00": 0.0001, "2025-10-03 12:00": 0.0001,
             "2025-10-03 20:00": 0.0001}
    out = _gap_paper(tmp_path, monkeypatch, binance, store, rates, "2025-10-03 11:50", 520)
    _built_missing(store, "2025-10-03 16:00")
    (m,) = _missing_of(store, "2025-10-03 16:00")
    assert "(inferred time)" in m["message"] and utc("2025-10-03 16:15") <= m["ts"] < utc("2025-10-03 16:16"), _show([m])
    assert [e for e in _stales(store) if _from("2025-10-03 16:00") in e["message"]], _show(_events(store))
    rows = sorted((r["ts"], r.get("kind")) for r in out["funding"])
    assert rows == [(utc("2025-10-03 12:00"), S_), (utc("2025-10-03 16:00"), B_), (utc("2025-10-03 20:00"), S_)], rows
    _texts_ok(store)


def _rows(out) -> list:
    return sorted((r["ts"], r.get("kind"), r["amount"]) for r in out["funding"])


def _rshow(rows) -> list:
    return [(f"{t:%d %H:%M}", k, round(a, 6)) for t, k, a in rows]


def _net_zero(rows, *times) -> None:
    for t in times:
        assert sum(a for ts, _, a in rows if ts == utc(t)) == pytest.approx(0, abs=1e-6), \
            f"{t} does not net to zero: {_rshow(rows)}"


def _closed_from(store, t, not_before) -> None:
    """The episode from settlement t was opened and then closed (its alert), not before `not_before`."""
    opened = [e for e in _stales(store) if _from(t) in e["message"]]
    closed = [e for e in _cleared(store) if _from(t) in e["message"]]
    assert len(opened) == 1 and len(closed) == 1, f"episode from {t}: {_show(_events(store, *EPISODE))}"
    assert closed[0]["ts"] >= utc(not_before), f"closed before {not_before}: {_show(_events(store, *EPISODE))}"


def test_paper_c_step_after_as_wide_reverses_both(tmp_path, monkeypatch, binance):
    """Amended (c) in paper (Advisor 04:31 / 04:33; replaces the 02:40 reading "back on schedule, not missing"). 4 h
    records (00:00, 04:00, 08:00), then 16:00 and 4 Oct 00:00. 12:00 (foreseen at 4 h) is charged the baseline at
    12:15 and alerted; once 16:00 lands, s_next is unknown and g (8 h) > s_prev (4 h): still provisionally missing.
    Foresight past 16:00 is min(4 h counting the provisional 12:00, P), so 20:00 is charged at 20:15 and alerted (a
    new episode: 16:00 between is stored). When 00:00 lands, 08:00 -> 16:00 has s_next = 8 h >= g (12:00 reversed)
    and 16:00 -> 00:00 is g <= s_prev = 8 h from stored records (20:00 reversed): each by its own "reversal" row,
    net zero, both alerts closed then, not before."""
    store = journal()
    rates = {**HISTORY, "2025-10-03 04:00": 0.0001, "2025-10-03 08:00": 0.0002, "2025-10-03 16:00": 0.0002,
             "2025-10-04 00:00": 0.0001}
    out = _gap_paper(tmp_path, monkeypatch, binance, store, rates, "2025-10-03 07:50", 1010)
    for t in ("2025-10-03 12:00", "2025-10-03 20:00"):
        _built_missing(store, t)
        (m,) = _missing_of(store, t)
        assert "(inferred time)" in m["message"], _show([m])
        assert utc(t) + pd.Timedelta(minutes=15) <= m["ts"] < utc(t) + pd.Timedelta(minutes=16), _show([m])
    rows = _rows(out)
    want = [("2025-10-03 08:00", S_), ("2025-10-03 12:00", B_), ("2025-10-03 12:00", R_), ("2025-10-03 16:00", S_),
            ("2025-10-03 20:00", B_), ("2025-10-03 20:00", R_), ("2025-10-04 00:00", S_)]
    assert [(t, k) for t, k, _ in rows] == [(utc(t), k) for t, k in want], _rshow(rows)
    _net_zero(rows, "2025-10-03 12:00", "2025-10-03 20:00")
    for t in ("2025-10-03 12:00", "2025-10-03 20:00"):
        _closed_from(store, t, "2025-10-04 00:00")
    _texts_ok(store)


B8_RATES = {f"{t:%Y-%m-%d %H:%M}": 0.0001 for t in B8}
SHORTENED = {**B8_RATES, "2025-10-03 08:00": 0.0001, "2025-10-03 12:00": 0.0001, "2025-10-03 20:00": 0.0001}


def test_paper_a_record_lost_after_a_shortening_stays_missing_when_the_step_after_matches(tmp_path, monkeypatch,
                                                                                         binance):
    """(b) settled by the next record: as the 20:30 cell, but 4 Oct 00:00 lands (4 h after 20:00), run to 00:40.
    s_next (4 h) == s_prev (4 h): 16:00 is missing for good (backfill, then never published at 24 h). It stays charged
    once, with no reversal and no new row; its episode stays open; 00:00 is settled."""
    store = journal()
    out = _gap_paper(tmp_path, monkeypatch, binance, store, {**SHORTENED, "2025-10-04 00:00": 0.0001},
                     "2025-10-03 11:50", 770)
    _built_missing(store, "2025-10-03 16:00")
    rows = _rows(out)
    want = [("2025-10-03 12:00", S_), ("2025-10-03 16:00", B_), ("2025-10-03 20:00", S_), ("2025-10-04 00:00", S_)]
    assert [(t, k) for t, k, _ in rows] == [(utc(t), k) for t, k in want], _rshow(rows)
    assert not [e for e in _cleared(store) if _from("2025-10-03 16:00") in e["message"]], _show(_events(store))
    assert [e for e in _stales(store) if _from("2025-10-03 16:00") in e["message"]], _show(_events(store))
    _texts_ok(store)


def test_paper_the_04_00_branch_reverses_16_00_and_00_00_when_04_00_lands(tmp_path, monkeypatch, binance):
    """The Advisor's 04:00 branch (04:31). Records 08:00, 12:00, (16:00 lost), 20:00, no 4 Oct 00:00, then 04:00
    lands; run to 04:40. 16:00 is charged at 16:15 and stays provisional at 20:00 (s_next unknown). Foresight past
    20:00 is 4 h (counting the provisional 16:00), so 00:00 is charged at 00:15 (its own episode: 20:00 between is
    stored). When 04:00 lands: 12:00 -> 20:00 has s_next = 8 h >= g = 8 h, so 16:00 is reversed; 20:00 -> 04:00 is
    8 h <= s_prev = 8 h from stored records (12:00 -> 20:00), no gap, so 00:00 is reversed. Each by its own "reversal"
    row, net zero, both alerts closed then, not before."""
    store = journal()
    out = _gap_paper(tmp_path, monkeypatch, binance, store, {**SHORTENED, "2025-10-04 04:00": 0.0001},
                     "2025-10-03 11:50", 1010)
    for t in ("2025-10-03 16:00", "2025-10-04 00:00"):
        _built_missing(store, t)
        (m,) = _missing_of(store, t)
        assert "(inferred time)" in m["message"], _show([m])
        assert utc(t) + pd.Timedelta(minutes=15) <= m["ts"] < utc(t) + pd.Timedelta(minutes=16), _show([m])
    rows = _rows(out)
    want = [("2025-10-03 12:00", S_), ("2025-10-03 16:00", B_), ("2025-10-03 16:00", R_), ("2025-10-03 20:00", S_),
            ("2025-10-04 00:00", B_), ("2025-10-04 00:00", R_), ("2025-10-04 04:00", S_)]
    assert [(t, k) for t, k, _ in rows] == [(utc(t), k) for t, k in want], _rshow(rows)
    _net_zero(rows, "2025-10-03 16:00", "2025-10-04 00:00")
    for t in ("2025-10-03 16:00", "2025-10-04 00:00"):
        _closed_from(store, t, "2025-10-04 04:00")
    _texts_ok(store)


def test_a_restart_after_a_reversal_does_not_refund_the_settlement_twice(tmp_path, monkeypatch, binance):
    """P1-O17a-15 (Code Reviewer, #163 at 32099ca): a paper restart while an episode is still open, after a provisional
    baseline in it was already reversed. 4 h records (00:00, 04:00, 08:00), then 16:00 and 4 Oct 04:00 (4 Oct 00:00
    never published). 12:00 (foreseen at 4 h) is charged at 12:15; foresight past 16:00 is 4 h (counting the
    provisional 12:00), so 20:00 is charged at 20:15 and opens its own episode (16:00 between is stored), and 00:00 is
    charged at 00:15 and joins it. When 04:00 lands: 08:00 -> 16:00 has s_next = 12 h >= g, so 12:00 is reversed;
    16:00 -> 04:00 is g = 12 h > P, read by (a) with f = s_prev = 8 h from stored records, so 00:00 stays missing and
    20:00 (not on that grid) is reversed. Paper stops at 04:45 with the episode from 20:00 still open on 00:00, and is
    restarted at 04:55 to 05:20. On start it rebuilds its watched settlements from the journal's baseline rows; the
    reversed 12:00 and 20:00 must not come back as owed: each keeps exactly one baseline and one reversal (net zero,
    amended (c): at most one charge and one reversal per settlement), 00:00 keeps its one baseline and no reversal,
    and the restart raises no new funding_stale and clears nothing while 00:00 is missing."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    params = win(("2025-10-03 07:52", "2025-10-05 00:00", 1))
    rates = {**HISTORY, "2025-10-03 04:00": 0.0001, "2025-10-03 08:00": 0.0002, "2025-10-03 16:00": 0.0002,
             "2025-10-04 04:00": 0.0001}
    run = dict(rates=rates, store=store, name="w")
    first = _rows(paper(tmp_path, monkeypatch, binance, params, "2025-10-03 07:50", 20 * 60 + 55, **run))  # to 04:45
    want = [("2025-10-03 08:00", S_), ("2025-10-03 12:00", B_), ("2025-10-03 12:00", R_), ("2025-10-03 16:00", S_),
            ("2025-10-03 20:00", B_), ("2025-10-03 20:00", R_), ("2025-10-04 00:00", B_), ("2025-10-04 04:00", S_)]
    assert [(t, k) for t, k, _ in first] == [(utc(t), k) for t, k in want], \
        f"setup: before the restart 12:00 and 20:00 are reversed and 00:00 is still missing {_rshow(first)}"
    _built_missing(store, "2025-10-04 00:00")
    restart = utc("2025-10-04 04:55")
    rows = _rows(paper(tmp_path, monkeypatch, binance, params, "2025-10-04 04:55", 25, **run))  # restarted, to 05:20
    for t in ("2025-10-03 12:00", "2025-10-03 20:00"):
        at = [(k, a) for ts, k, a in rows if ts == utc(t)]
        assert [k for k, _ in at].count(R_) == 1, \
            (f"{t} refunded {[k for k, _ in at].count(R_)} times for one baseline charge: the restart re-watched a "
             f"settlement already reversed and reversed it again {_rshow(rows)}")
        assert sorted(k for k, _ in at) == [B_, R_], f"{t}: one baseline and one reversal, nothing else {_rshow(rows)}"
    _net_zero(rows, "2025-10-03 12:00", "2025-10-03 20:00")
    at00 = [(k, a) for ts, k, a in rows if ts == utc("2025-10-04 00:00")]
    assert [k for k, _ in at00] == [B_], f"00:00 (still missing) keeps its one baseline, no reversal {_rshow(rows)}"
    assert [(t, k) for t, k, _ in rows] == [(utc(t), k) for t, k in want], \
        f"the restart changed the money rows: before {_rshow(first)}, after {_rshow(rows)}"
    got = _events(store, *EPISODE)
    assert not [e for e in got if e["ts"] >= restart], f"the restart raised or cleared an episode: {_show(got)}"
    assert len([e for e in _stales(store) if _from("2025-10-03 20:00") in e["message"]]) == 1, _show(got)
    assert not [e for e in _cleared(store) if _from("2025-10-03 20:00") in e["message"]], \
        f"the episode from 20:00 was cleared while 00:00 in it is still missing: {_show(got)}"
    _built_missing(store, "2025-10-04 00:00")
    _texts_ok(store)


@pytest.mark.parametrize("case", ["a-8h-one-missing", "default-none-of-a-to-c"])
def test_the_collector_marks_an_inferred_settlement_and_the_backfill_replaces_it(tmp_path, monkeypatch, binance, case):
    """02:13 pins through the hub. The store holds records with a gap, as the collector kept them (a: 08:00 lost on
    8 h; default: 04:00 and 06:00 lost on 2 h). The pass at 20:30 marks each inferred settlement funding_missing
    "(inferred time)" and opens one episode from the first (the alert). The venue then publishes them late: the pass
    at 21:30 backfills them from its history and the episode closes (from the first), nothing never published."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    records, _, want = GAPS[case]
    lost = [utc(f"2025-{t}") for t, k in want.items() if k == B_]
    kept = {t: 0.0001 for t in B8 + records if t <= utc("2025-10-03 20:00")}
    _kept(monkeypatch, kept)
    _hub_pass(monkeypatch, binance, store, "2025-10-03 20:30", kept)
    assert _events(store, "funding_missing"), f"not built: no funding_missing marks: {_show(_events(store))}"
    for t in lost:
        marks = _missing_of(store, t)
        assert len(marks) == 1 and "(inferred time)" in marks[0]["message"], (t, _show(_events(store)))
    stale = _stales(store)
    assert len(stale) == 1 and _from(lost[0]) in stale[0]["message"], _show(_events(store))
    _hub_pass(monkeypatch, binance, store, "2025-10-03 21:30", {**kept, **{t: 0.0002 for t in lost}})
    stored = funding.rates("BINANCE", PAIR)
    assert all(stored.get(t) == pytest.approx(0.0002) for t in lost), f"not backfilled: {list(stored.index)}"
    got = _events(store, *EPISODE)
    assert [e["kind"] for e in got] == ["funding_stale", "funding_stale_cleared"], _show(got)
    assert _from(lost[0]) in got[1]["message"] and not _never(store), _show(_events(store))
    _texts_ok(store)


# ---------------------------------------------------------------------------------------------------------------
# 9b. The record snap (Advisor 7 Oct ~03:12 FUNDING RECORD SNAP, option 1, on the a-gap-P-plus-1-min pin). One shared
#     constant N = 1 minute, either side, for the gap reader and the engine's match window. Snapped BEFORE the gaps are
#     read: a record within +/-N of a due settlement IS that settlement's rate, so 00:00 -> 08:01 reads as g = P (no
#     missing settlement, no baseline, no alert) and 08:00 is charged once, at the 08:01 record's rate. One-to-one: a
#     record pays at most one settlement and a settlement takes at most one record (no rate is charged twice). The
#     record keeps its own stamp for audit, flagged "snapped (+1 min)" ("snapped (-1 min)" when early). Outside the
#     window the 02:40 rules apply unchanged, and a record is charged once at most. The charge is stamped at the
#     settlement. Rates differ per record here, so each charge shows which record paid it.
# ---------------------------------------------------------------------------------------------------------------

SNAP_RATE = 0.0003  # the drifting record's rate; every other record is 0.0001


def _snap_rows(binance, records: dict, last_bar, side):
    """A backtest held from 2 Oct 01:00 over the B8 records plus `records` (time -> rate); its charges from 3 Oct
    00:00 as (time, kind, rate), oldest first, and the result."""
    write_rates({**{t: 0.0001 for t in B8}, **{utc(t): r for t, r in records.items()}})
    r = backtest(binance, hourly_bars("2025-10-02 00:00", last_bar), win(("2025-10-02 01:00", "2026-01-01", side)))
    rows = sorted((utc(x["ts"]), x.get("kind"), x["rate"]) for x in r.journal.funding_
                  if utc(x["ts"]) >= utc("2025-10-03 00:00"))
    return rows, r


def _shown(rows) -> list:
    return [(f"{t:%d %H:%M:%S}", k, f"{rate:.4%}") for t, k, rate in rows]


ON_TIME = {"2025-10-03 16:00": 0.0001, "2025-10-04 00:00": 0.0001}
SNAPS = {
    # name: (records after B8, last bar, the charges the ruling gives as (settlement, kind, rate))
    "snap-plus-1-min": ({"2025-10-03 08:01": SNAP_RATE, **ON_TIME}, "2025-10-04 04:00",
                        [("10-03 00:00", S_, 0.0001), ("10-03 08:00", S_, SNAP_RATE), ("10-03 16:00", S_, 0.0001),
                         ("10-04 00:00", S_, 0.0001)]),
    "snap-minus-1-min": ({"2025-10-03 07:59": SNAP_RATE, **ON_TIME}, "2025-10-04 04:00",
                         [("10-03 00:00", S_, 0.0001), ("10-03 08:00", S_, SNAP_RATE), ("10-03 16:00", S_, 0.0001),
                          ("10-04 00:00", S_, 0.0001)]),
    # 08:02 is outside +/-N: the 02:40 rules, g = 8 h 2 min > P, so 08:00 is missing (baseline, strictly before
    # 08:02) and the 08:02 record is charged once, at its own time and rate
    "outside-the-window-plus-2-min": ({"2025-10-03 08:02": SNAP_RATE, **ON_TIME}, "2025-10-04 04:00",
                                      [("10-03 00:00", S_, 0.0001), ("10-03 08:00", B_, BASELINE),
                                       ("10-03 08:02", S_, SNAP_RATE), ("10-03 16:00", S_, 0.0001),
                                       ("10-04 00:00", S_, 0.0001)]),
}


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("case", list(SNAPS))
def test_a_record_within_a_minute_of_its_settlement_is_that_settlements_rate(binance, case, side):
    """The rewritten a-gap-P-plus-1-min pin and its edges: 08:01 (+1 min) and 07:59 (-1 min) snap to 08:00, charged
    once, settled, at that record's rate, with no baseline; 08:02 (outside the window) follows the 02:40 rules. In
    every case no record's rate is charged twice."""
    records, last_bar, want = SNAPS[case]
    rows, _ = _snap_rows(binance, records, last_bar, side)
    paid = [t for t, _, rate in rows if rate == pytest.approx(SNAP_RATE)]
    assert len(paid) <= 1, f"{case}: the {SNAP_RATE:.2%} record charged {len(paid)} times: {_shown(rows)}"
    expect = [(utc(f"2025-{t}"), k, rate) for t, k, rate in want]
    assert [(t, k) for t, k, _ in rows] == [(t, k) for t, k, _ in expect], \
        f"{case}: charged {_shown(rows)}, the ruling gives {_shown(expect)}"
    for (t, _, rate), (_, _, want_rate) in zip(rows, expect):
        assert rate == pytest.approx(want_rate), f"{case}: {t:%d %H:%M} charged at {rate:.4%}: {_shown(rows)}"


# Drifting stamps over a day, each record its own rate: (stamp, rate) -> the settlement it is
DRIFT = [("2025-10-03 07:59:00", 0.00011, "2025-10-03 08:00"),  # -60 s
         ("2025-10-03 16:00:30", 0.00012, "2025-10-03 16:00"),  # +30 s
         ("2025-10-04 00:01:00", 0.00013, "2025-10-04 00:00"),  # +60 s
         ("2025-10-04 07:59:30", 0.00014, "2025-10-04 08:00")]  # -30 s


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_records_and_settlements_match_one_to_one_with_drifting_stamps(binance, side):
    """A day of stamps drifting by -60 s, +30 s, +60 s and -30 s: each settlement (3 Oct 08:00 to 4 Oct 08:00) is
    charged once, at the settlement, as settled, with its own record's rate; no record pays two settlements, no
    settlement takes two records, nothing is charged the baseline."""
    rows, _ = _snap_rows(binance, {s: r for s, r, _ in DRIFT}, "2025-10-04 10:00", side)
    for _, rate, _ in DRIFT:
        n = sum(x == pytest.approx(rate) for _, _, x in rows)
        assert n == 1, f"the {rate:.4%} record paid {n} settlements: {_shown(rows)}"
    expect = [(utc("2025-10-03 00:00"), S_, 0.0001)] + [(utc(t), S_, rate) for _, rate, t in DRIFT]
    assert [(t, k) for t, k, _ in rows] == [(t, k) for t, k, _ in expect], \
        f"charged {_shown(rows)}, one-to-one gives {_shown(expect)}"
    assert all(rate == pytest.approx(w) for (_, _, rate), (_, _, w) in zip(rows, expect)), _shown(rows)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("stamp, signed", [("2025-10-03 08:01", "+1 min"), ("2025-10-03 07:59", "-1 min")],
                         ids=["late", "early"])
def test_a_snapped_record_keeps_its_own_stamp_flagged_for_audit(binance, stamp, signed, side):
    """The 08:00 charge paid by a record stamped 08:01 (or 07:59) carries the record's own stamp and a flag containing
    "snapped" and the signed minute ("snapped (+1 min)" / "snapped (-1 min)"), on the journal's funding row or on
    BacktestResult.funding (pinned loosely: anywhere in the row)."""
    _, r = _snap_rows(binance, {stamp: SNAP_RATE, **ON_TIME}, "2025-10-04 04:00", side)
    rows = [x for x in list(r.journal.funding_) + list(r.funding) if utc(x["ts"]) == utc("2025-10-03 08:00")]
    assert rows, f"08:00 not charged: {[(str(x['ts']), x.get('kind')) for x in r.journal.funding_]}"
    text = " | ".join(" ".join(str(v) for v in x.values()) for x in rows)
    assert "snapped" in text.lower() and signed in text, f"no 'snapped ({signed})' flag on the 08:00 charge: {text}"
    assert f"{utc(stamp):%H:%M}" in text, f"the record's own stamp {utc(stamp):%H:%M} is not kept: {text}"


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_paper_a_record_stamped_a_minute_late_is_charged_once_at_the_settlement(tmp_path, monkeypatch, binance, side):
    """Paper reads the venue's records too: 08:00's record is stamped 08:01 (published 08:05, inside the wait).
    08:00 is charged once, settled, at that record's rate; nothing at 08:01; no funding_missing and no funding_stale."""
    store = journal()
    out = _gap_paper(tmp_path, monkeypatch, binance, store, {**HISTORY, "2025-10-03 08:01": SNAP_RATE},
                     "2025-10-03 07:50", 50, side, published={"2025-10-03 08:01": "2025-10-03 08:05"})
    rows = sorted((r["ts"], r.get("kind"), r["rate"]) for r in out["funding"])
    assert [(t, k) for t, k, _ in rows] == [(utc("2025-10-03 08:00"), S_)], _shown(rows)
    assert rows[0][2] == pytest.approx(SNAP_RATE), _shown(rows)
    assert not _events(store, "funding_missing", "funding_stale"), _show(_events(store))


def test_the_collector_reads_a_snapped_record_as_no_gap_but_still_marks_a_real_one(tmp_path, monkeypatch, binance):
    """The store keeps 08:01 (08:00's record, late) and 4 Oct 00:00, and lost 16:00. The 4 Oct 00:30 pass marks 16:00
    missing "(inferred time)" (the control), and nothing for 08:00 or 08:01: no funding_missing, no episode from them."""
    _hub_built()
    _fresh_hub(monkeypatch)
    store = _restartable_journal()
    kept = {**{t: 0.0001 for t in B8}, utc("2025-10-03 08:01"): SNAP_RATE, utc("2025-10-04 00:00"): 0.0001}
    _kept(monkeypatch, kept)
    _hub_pass(monkeypatch, binance, store, "2025-10-04 00:30", kept)
    _built_missing(store, "2025-10-03 16:00")
    early = [e for e in _events(store) if re.search(r"2025-10-03 08:0[01]", e["message"])]
    assert not early, f"a snapped record read as a gap: {_show(early)}"
    (stale,) = _stales(store)
    assert _from("2025-10-03 16:00") in stale["message"], _show(_events(store))
    _texts_ok(store)


# ---------------------------------------------------------------------------------------------------------------
# 9c. The episode after reclassification (Advisor 7 Oct 06:30 EPISODE AFTER RECLASSIFICATION, and 06:33 EPISODE
#     MEMBERSHIP + ALERT TEXT). The episode is the outage, not its first settlement: it is open while any of its
#     settlements is provisionally missing or missing, and closes once every one is terminal (stored or backfilled,
#     reclassified, never published); a settlement missed while it is open joins it, whatever lies between. Its alert
#     text is written once, at open; the current missing set is read live from the episode's state (missing and never
#     published marks, the store) and its funding_reclassified events, with no new alert when the set changes. ONE
#     closing notice (funding_stale_cleared, "episode from ...") gives each settlement's outcome: backfilled /
#     reclassified / never published. G1 counts stay per settlement: a reclassified settlement is never counted as
#     never published. If every settlement of an episode is reclassified, it closes as "reclassified, no missing
#     settlement" and is kept (its events and rows stay).
#     Scenario (P1-O17a-15's, without the restart): 4 h records 3 Oct 00:00, 04:00, 08:00, then 16:00, then 4 Oct 04:00
#     and every 4 h after it; 4 Oct 00:00 is never published unless a cell publishes it late. 12:00 is charged at
#     12:15 (the episode opens); 20:00 at 20:15 and 00:00 at 00:15 join it; when 04:00 lands, 12:00 and 20:00 are
#     reversed (reclassified) and 00:00 stays missing.
#     Assumed (DA, 7 Oct): funding_reclassified (info, once per settlement, global; starts with the instrument tag,
#     names no venue): names the settlement, "reclassified, not missing", the reversal row's journal id (funding.id)
#     and the episode ("episode from {o} UTC"). Settlements are named "YYYY-MM-DD HH:MM" (or "DD Mon YYYY HH:MM").
#     O17b's entry block while 4 Oct 00:00 is missing belongs to test_o17b_xfails.py (not pinned here).
# ---------------------------------------------------------------------------------------------------------------

RECLASS = "funding_reclassified"
EP_OPEN = "2025-10-03 12:00"
EP_RATES = {**HISTORY, "2025-10-03 04:00": 0.0001, "2025-10-03 08:00": 0.0002, "2025-10-03 16:00": 0.0002,
            **{f"{t:%Y-%m-%d %H:%M}": 0.0001 for t in _grid("2025-10-04 04:00", "2025-10-05 04:00", 4)}}
EP_REVERSED = ("2025-10-03 12:00", "2025-10-03 20:00")
EP_MISSING = "2025-10-04 00:00"
_FROM_ANY = re.compile(r"from \d{4}-\d\d-\d\d \d\d:\d\d UTC")


def _episode_run(tmp_path, monkeypatch, binance, until, published=None):
    """The 9c scenario as paper, held from 07:52, run from 3 Oct 07:50 to `until`; `published` gives 4 Oct 00:00 a
    late publication time (its rate is then the venue's). Checks the money rows the episode rulings stand on (built on
    #163: 12:00 and 20:00 each one baseline and one reversal, 00:00 its baseline, not reversed)."""
    store = journal()
    minutes = int((utc(until) - utc("2025-10-03 07:50")).total_seconds() // 60)
    rates = {**EP_RATES, **({EP_MISSING: 0.0001} if published else {})}
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-06 00:00", 1)), "2025-10-03 07:50",
                minutes, rates=rates, store=store, published=published)
    rows = _rows(out)
    for t in EP_REVERSED:
        assert sorted(k for ts, k, _ in rows if ts == utc(t)) == [B_, R_], \
            f"setup: {t} one baseline and one reversal {_rshow(rows)}"
    at00 = [k for ts, k, _ in rows if ts == utc(EP_MISSING)]
    assert B_ in at00 and R_ not in at00, f"setup: 00:00 charged the baseline, not reversed {_rshow(rows)}"
    _built_missing(store, EP_MISSING)
    return out, store


def _stamp_at(text: str, t) -> int:
    """Where settlement t is named in `text` (episode names removed), or -1."""
    t = utc(t)
    found = [text.find(s) for s in (f"{t:%Y-%m-%d %H:%M}", f"{t:%d %b %Y %H:%M}") if s in text]
    return min(found) if found else -1


def _says(message: str, t) -> bool:
    return _stamp_at(_FROM_ANY.sub("", message), t) >= 0


def _reclassified(store) -> list:
    return _events(store, RECLASS)


def _live_missing(store) -> list:
    """The episode's current missing set, as the ruling reads it: settlements marked missing, less those reclassified
    (funding_reclassified), never published, or stored since."""
    state = funding.journal_state(store, TAG)
    gone = {t for t in state["missing"] for e in _reclassified(store) if _says(e["message"], t)}
    stored = set(funding.rates("BINANCE", PAIR).index)
    return sorted(t for t in state["missing"] - gone - state["never"] if t not in stored)


def _late_alerts(store, after) -> list:
    return [e for e in _stales(store) if e["ts"] >= utc(after)]


def _closing(store) -> list:
    return [e for e in _cleared(store) if _from(EP_OPEN) in e["message"]]


def _outcomes_listed(notice: str, outcomes: dict) -> None:
    """The closing notice names every settlement of the episode, each followed by its outcome word."""
    text = _FROM_ANY.sub("", notice)
    where = {t: _stamp_at(text, t) for t in outcomes}
    assert all(p >= 0 for p in where.values()), \
        f"the closing notice does not name {[t for t, p in where.items() if p < 0]}: {notice}"
    order = sorted(where, key=where.get)
    for i, t in enumerate(order):
        seg = text[where[t]:where[order[i + 1]] if i + 1 < len(order) else len(text)].lower()
        assert re.search(outcomes[t], seg), f"{t}: outcome {outcomes[t]!r} not given after it in: {notice}"


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_EPISODE)
def test_after_the_reversals_one_episode_stays_open_naming_only_the_missing_settlement(tmp_path, monkeypatch, binance):
    """Advisor 06:30 pin for 4 Oct 05:00: exactly ONE funding_stale is open, the episode opened 3 Oct 12:15 (from
    12:00), its opened_at unchanged; it was never closed and reopened and no second alert was raised (20:00 and 00:00
    joined it; the 04:00 reversals changed its set, not its alert). Its live missing set is 4 Oct 00:00 only."""
    _, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-04 05:00")
    stale = _stales(store)
    assert len(stale) == 1, f"one outage, one funding_stale: {_show(_events(store, *EPISODE))}"
    assert _from(EP_OPEN) in stale[0]["message"], f"not the episode from 12:00: {_show(stale)}"
    assert utc("2025-10-03 12:15") <= stale[0]["ts"] < utc("2025-10-03 12:16"), f"opened_at moved: {_show(stale)}"
    assert not _cleared(store), f"closed (and reopened?) while 00:00 is missing: {_show(_events(store, *EPISODE))}"
    opened = funding.journal_state(store, TAG)["open"]
    assert list(opened) == [utc(EP_OPEN)], f"open episodes {sorted(opened)}, want only the one from 12:00"
    assert _live_missing(store) == [utc(EP_MISSING)], \
        (f"the episode's missing set is {[f'{t:%d %H:%M}' for t in _live_missing(store)]}, want 4 Oct 00:00 only "
         f"(12:00 and 20:00 reclassified): {_show(_events(store, *FUNDING_EVENTS, RECLASS))}")
    _texts_ok(store)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_EPISODE)
def test_the_reversed_settlements_are_shown_on_the_episode_as_reclassified_with_their_reversal_ids(
        tmp_path, monkeypatch, binance):
    """Advisor 06:30: on the episode, 12:00 and 20:00 are shown as "reclassified, not missing", each with its reversal
    journal id. One funding_reclassified each (info), once 04:00 has landed, naming the settlement, the episode (from
    12:00) and the id of that settlement's own reversal row; none for 00:00, which is still missing."""
    out, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-04 05:00")
    rec = _reclassified(store)
    assert rec, f"not built: no {RECLASS} events (the episode's reclassified settlements): {_show(_events(store))}"
    ids = {}
    for t in EP_REVERSED:
        (rid,) = [r["id"] for r in out["funding"] if r["ts"] == utc(t) and r.get("kind") == R_]
        ids[t] = rid
        mine = [e for e in rec if _says(e["message"], t)]
        assert len(mine) == 1, f"{t} not shown as reclassified exactly once: {_show(rec)}"
        (e,) = mine
        text = e["message"]
        assert e["level"] == "info" and e["ts"] >= utc("2025-10-04 04:00"), _show([e])
        assert re.search(r"reclassified,?\s+not missing", text, re.I), f"{t}: no 'reclassified, not missing': {text}"
        assert _from(EP_OPEN) in text, f"{t}: does not name the episode from 12:00: {text}"
        bare = re.sub(r"\d{4}-\d\d-\d\d \d\d:\d\d|\d\d \w{3} \d{4} \d\d:\d\d", "", text)
        assert re.search(rf"(?<![\w.:-]){rid}(?![\w.:])", bare), f"{t}: its reversal journal id {rid} not shown: {text}"
        assert text.startswith(TAG) and not re.search(r"binance", text, re.I), text
    assert ids[EP_REVERSED[0]] != ids[EP_REVERSED[1]], ids
    assert not [e for e in rec if _says(e["message"], EP_MISSING)], f"00:00 is missing, not reclassified: {_show(rec)}"


def test_the_episode_closes_when_the_missing_settlement_is_backfilled(tmp_path, monkeypatch, binance):
    """Advisor 06:30: when 4 Oct 00:00 is backfilled (published at 06:00), the episode closes as normal: the one from
    12:00 closes once, between 06:00 and 06:16, nothing is left open, nothing is never published, and no alert was
    raised after 00:15 (no close and reopen)."""
    _, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-04 06:40",
                            published={EP_MISSING: "2025-10-04 06:00"})
    closed = _closing(store)
    assert len(closed) == 1, f"the episode from 12:00 did not close once: {_show(_events(store, *EPISODE))}"
    assert utc("2025-10-04 06:00") <= closed[0]["ts"] < utc("2025-10-04 06:16"), _show(closed)
    assert not funding.journal_state(store, TAG)["open"], f"left open: {_show(_events(store, *EPISODE))}"
    assert not _never(store), _show(_events(store))
    assert not _late_alerts(store, "2025-10-04 00:16"), _show(_events(store, *EPISODE))
    _texts_ok(store)


def test_the_episode_closes_once_the_missing_settlement_is_never_published_and_it_counts_for_g1(
        tmp_path, monkeypatch, binance):
    """Advisor 06:30: 4 Oct 00:00 still absent at 5 Oct 00:00 (due + 24 h; later settlements stored): it is never
    published, marked once (a warning, "never published", "baseline", "true-up"), and the episode closes then, not
    before. It counts towards the G1 1% rule (in the journal's never-published set); the reclassified 12:00 and 20:00
    do not. Its baseline row stays, with no reversal."""
    out, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-05 00:40")
    never = _never(store, EP_MISSING)
    assert len(never) == 1, f"00:00 not marked never published exactly once: {_show(_events(store))}"
    text = never[0]["message"].lower()
    assert never[0]["level"] == "warning" and never[0]["ts"] >= utc("2025-10-05 00:00"), _show(never)
    assert "never published" in text and "baseline" in text and "true-up" in text, never[0]["message"]
    state = funding.journal_state(store, TAG)
    assert utc(EP_MISSING) in state["never"], f"00:00 not in the never-published set G1 counts: {state['never']}"
    assert not {utc(t) for t in EP_REVERSED} & state["never"], f"a reclassified settlement counted for G1: {state}"
    closed = _closing(store)
    assert len(closed) == 1 and closed[0]["ts"] >= utc("2025-10-05 00:00"), \
        f"the episode from 12:00 must close once 00:00 is never published, not before: {_show(_events(store, *EPISODE))}"
    assert not state["open"], f"left open: {_show(_events(store, *EPISODE))}"
    assert not _late_alerts(store, "2025-10-04 00:16"), _show(_events(store, *EPISODE))
    assert [k for t, k, _ in _rows(out) if t == utc(EP_MISSING)] == [B_], _rshow(_rows(out))
    _texts_ok(store)


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_EPISODE)
@pytest.mark.parametrize("case", ["backfilled", "never-published"])
def test_one_closing_notice_gives_each_settlements_outcome(tmp_path, monkeypatch, binance, case):
    """Advisor 06:33 (2): ONE closing notice for the episode from 12:00, stating each settlement's outcome: 12:00 and
    20:00 reclassified; 4 Oct 00:00 backfilled (published at 06:00) or never published (still absent at 5 Oct 00:00)."""
    if case == "backfilled":
        _, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-04 06:40",
                                published={EP_MISSING: "2025-10-04 06:00"})
        last = "backfill"
    else:
        _, store = _episode_run(tmp_path, monkeypatch, binance, "2025-10-05 00:40")
        last = "never published"
    closed = _closing(store)
    assert len(_cleared(store)) == 1 and len(closed) == 1, \
        f"one episode, one closing notice: {_show(_events(store, *EPISODE))}"
    _outcomes_listed(closed[0]["message"], {EP_REVERSED[0]: "reclassif", EP_REVERSED[1]: "reclassif",
                                            EP_MISSING: last})


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_EPISODE)
def test_an_episode_whose_settlements_are_all_reclassified_closes_as_such_and_is_kept(tmp_path, monkeypatch, binance):
    """Advisor 06:30 variant: every settlement of the episode is reclassified. As the 9c scenario but 4 Oct 00:00 is
    published on time (the amended (c) sibling, run to 00:40): 12:00 opens the episode at 12:15, 20:00 joins it at
    20:15; when 00:00 lands both are reversed. The episode closes once, at or after 00:00, as "reclassified, no missing
    settlement", its notice giving 12:00 and 20:00 as reclassified, and it is kept: its funding_stale and missing
    marks stay in the journal, each settlement keeps its baseline and reversal rows, nothing is never published."""
    store = journal()
    out = paper(tmp_path, monkeypatch, binance, win(("2025-10-03 07:52", "2025-10-06 00:00", 1)), "2025-10-03 07:50",
                1010, rates={**EP_RATES, EP_MISSING: 0.0001}, store=store)
    rows = _rows(out)
    for t in EP_REVERSED:
        assert sorted(k for ts, k, _ in rows if ts == utc(t)) == [B_, R_], f"setup: {t} {_rshow(rows)}"
        _built_missing(store, t)
    stale = _stales(store)
    assert len(stale) == 1 and _from(EP_OPEN) in stale[0]["message"], \
        f"one episode, from 12:00 (20:00 joins it): {_show(_events(store, *EPISODE))}"
    closed = _cleared(store)
    assert len(closed) == 1 and _from(EP_OPEN) in closed[0]["message"], _show(_events(store, *EPISODE))
    assert closed[0]["ts"] >= utc(EP_MISSING), _show(closed)
    assert re.search(r"reclassified,?\s+no missing settlement", closed[0]["message"], re.I), \
        f"not closed as 'reclassified, no missing settlement': {closed[0]['message']}"
    _outcomes_listed(closed[0]["message"], {t: "reclassif" for t in EP_REVERSED})
    assert _stales(store) == stale and all(_missing_of(store, t) for t in EP_REVERSED), \
        f"the episode was not kept: {_show(_events(store))}"
    assert not _never(store), _show(_events(store))
    _texts_ok(store)


# ---------------------------------------------------------------------------------------------------------------
# 10. G1-NP follow-up (before strategy testing). NOT #163: a follow-up item before strategy testing (Head of QA,
#     7 Oct). Advisor 7 Oct ~02:12: never-published settlements in out-of-sample windows are counted and shown; above
#     1% of those windows' settlements, or if a re-run at the trailing 30-day max rate flips the verdict, G1 is not
#     judged until resolved; below both, shown but not blocking.
#     Assumed: funding.never_published_check(never, held, verdict, rerun_verdict) -> (verdict, words), where `verdict`
#     is G1's and `rerun_verdict` G1's on the re-run with each never-published settlement charged the trailing 30-day
#     max |rate|; NOT JUDGED or a verdict that does not block; words carry "N of M" and "never published". A study
#     reads the never-published settlements the collector has marked.
# ---------------------------------------------------------------------------------------------------------------

REASON_NP = "G1-NP follow-up (before strategy testing)"

NP_RULE = [
    # name, never published, OOS settlements, G1, G1 re-run at the 30-day max, not judged?
    ("none", 0, 700, "PASS", "PASS", False),
    ("exactly-1pct", 7, 700, "PASS", "PASS", False),
    ("just-over-1pct", 8, 700, "PASS", "PASS", True),
    ("one-and-the-re-run-flips-pass-to-fail", 1, 700, "PASS", "FAIL", True),
    ("one-a-fail-stays-a-fail", 1, 700, "FAIL", "FAIL", False),
    ("one-no-flip", 1, 700, "PASS", "PASS", False),
]


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_NP)
@pytest.mark.parametrize("n, m, g1, rerun, unjudged", [c[1:] for c in NP_RULE], ids=[c[0] for c in NP_RULE])
def test_g1_np_never_published_settlements_in_the_oos_windows_rule(n, m, g1, rerun, unjudged):
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    check = getattr(funding, "never_published_check", None)
    assert callable(check), "not built (G1-NP): funding.never_published_check(never, held, verdict, rerun_verdict)"
    got, words = check(n, m, g1, rerun)
    assert (got == NOT_JUDGED) == unjudged, (got, words)
    assert f"{n} of {m}" in words and "never published" in words.lower(), words


NP_STUDY = {
    # name: (never-published settlements, G1 not judged for funding?)
    "oos-just-over-1pct": (lambda: _every("2018-05-10", "2018-12-20")[::60], True),
    "oos-under-1pct-no-flip": (lambda: _every("2018-05-10", "2018-12-20")[::150], False),
}


@pytest.mark.xfail(strict=True, raises=AssertionError, reason=REASON_NP)
@pytest.mark.parametrize("case", list(NP_STUDY))
def test_g1_np_a_study_counts_and_shows_never_published_settlements(tmp_path, monkeypatch, binance, case):
    """The study of section 5 (buy and hold, 400 days, four OOS windows), with settlements the venue never published
    in the OOS windows; the collector's passes have marked them never published (more than 24 h after due, later
    settlements stored, the backfill finding nothing). The tear sheet shows how many are in the OOS windows ("never
    published", with that count); over 1% G1 is NOT JUDGED naming funding; under 1% with a flat 0.012% everywhere
    (the re-run at the 30-day max cannot flip it) funding does not make it not judged."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import NOT_JUDGED, g1_checks, g1_verdict, render
    from sleeve_fund.strategies.buy_and_hold import SPEC

    _hub_built()
    make, unjudged = NP_STUDY[case]
    holes = set(make())
    prices = synthetic_ohlcv(days=400, seed=3, start_price=60_000)
    venue = {t: 0.00012 for t in settlements(prices.index[0] - D, prices.index[-1]) if t not in holes}
    _fresh_hub(monkeypatch)
    store = journal()
    _kept(monkeypatch, venue)
    now = pd.Timestamp.now(tz="UTC").floor("min")
    for at in (now, now + pd.Timedelta(minutes=1)):
        _hub_pass(monkeypatch, binance, store, at, venue)
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, binance.instrument("BTC", "USDT"), dataset="syn-binance-1440m", ledger=ledger,
                  synthetic=True, holdout_days=30, train_days=120, test_days=60, use_holdout=True)
    full = [utc(f["ts"]) for f in r.full_period.funding]
    n_oos = sum(t in holes and any(f.train_end < t <= f.test_end for f in r.folds) for t in full)
    assert n_oos >= 4, f"setup: {n_oos} never-published settlements held in the OOS windows"
    sheet = render(r, ledger)
    shown = [ln for ln in sheet.splitlines() if re.search(r"never[- ]published", ln, re.I)]
    assert any(re.search(rf"\b{n_oos}\b", ln) for ln in shown), \
        f"the {n_oos} never-published OOS settlements are not counted on the tear sheet: {shown}"
    g1 = g1_verdict(g1_checks(r, ledger))[0]
    if unjudged:
        assert g1 == NOT_JUDGED and "funding" in r.not_judged.lower(), (g1, r.not_judged)
    else:
        assert "funding" not in r.not_judged.lower(), r.not_judged
