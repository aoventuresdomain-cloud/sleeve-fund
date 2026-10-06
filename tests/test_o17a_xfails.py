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

REASON = "QA O17a: not built yet (Advisor 17:52)"
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
