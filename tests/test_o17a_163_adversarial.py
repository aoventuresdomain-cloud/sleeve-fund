"""QA adversarial checks on #163 part 2, O17a (commit 8165352), unannounced to the DA. Copy beside tests/o17_harness.py
(the harness from quant-review/o17-xfails) and run from the checkout:

    TEST_DATABASE_URL=... BACKTEST_ISOLATE=0 PYTHONPATH=. python -m pytest -q -p no:cacheprovider tests/<this file>

Findings are strict xfails; behaviour that belongs to part 3 (O17b) is a strict xfail with an "O17b" reason."""
from __future__ import annotations

import math
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from o17_harness import (  # noqa: F401 - fixtures
    BASELINE,
    NEVER,
    PAIR,
    _no_swallowed_strategy_errors,
    _o17win,
    backtest,
    binance,
    hourly_bars,
    kinds,
    paper,
    settlements,
    utc,
    win,
    write_rates,
)
from sleeve_fund import funding

DAY = "2025-10-03"
D = pd.Timedelta(days=1)


def _baseline(row, rate=BASELINE):
    assert math.isfinite(row["amount"]), row
    assert row["amount"] == pytest.approx(-abs(row["qty"]) * row["price"] * rate, rel=1e-6), row


# --- 1. A NaN, inf or over-cap rate arriving from the venue mid-run (paper) -----------------------------------

BAD = [("nan", float("nan")), ("inf", float("inf")), ("over-cap", 0.004)]


@pytest.mark.strategy_errors
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
@pytest.mark.parametrize("bad", [b for _, b in BAD], ids=[n for n, _ in BAD])
def test_paper_a_bad_rate_published_on_time_is_missing_both_sides_pay_the_baseline(tmp_path, monkeypatch, binance,
                                                                                 side, bad):
    """The venue publishes 08:00 at 08:05, but the number is NaN, inf or over the cap (0.3% here). Paper must charge
    the baseline after the 15-minute check, never the bad number, keep running, and say so."""
    monkeypatch.setattr(binance, "funding_cap", lambda pair: 0.003)
    out = paper(tmp_path, monkeypatch, binance, win((f"{DAY} 07:52", f"{DAY} 16:25", side)), f"{DAY} 07:50", 520,
                rates={f"{DAY} 08:00": bad, f"{DAY} 16:00": 0.0002},
                published={f"{DAY} 08:00": f"{DAY} 08:05", f"{DAY} 16:00": f"{DAY} 16:02"}, step=10)
    rows = {r["ts"]: r for r in out["funding"]}
    assert set(rows) == {utc(f"{DAY} 08:00"), utc(f"{DAY} 16:00")}
    _baseline(rows[utc(f"{DAY} 08:00")])
    assert rows[utc(f"{DAY} 08:00")]["kind"] == "baseline" and rows[utc(f"{DAY} 16:00")]["kind"] == "settled"
    assert rows[utc(f"{DAY} 16:00")]["rate"] == pytest.approx(0.0002)  # the run went on and charged the next one
    warned = [e for e in out["events"] if e["level"] in ("warning", "error") and e["kind"].startswith("funding")]
    print(f"\n{bad} {side}: alerts {[(e['kind'], e['message'][:90]) for e in warned]}")
    assert warned, "no alert for the bad rate"


# --- 2. A rate arriving late: baseline charged, then the true-up (O17b, part 3) -------------------------------

@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_paper_a_late_rate_is_charged_the_baseline_adversely_and_the_episode_clears(tmp_path, monkeypatch, binance, side):
    out = paper(tmp_path, monkeypatch, binance, win((f"{DAY} 07:52", f"{DAY} 08:50", side)), f"{DAY} 07:50", 60,
                rates={f"{DAY} 08:00": 0.0003}, published={f"{DAY} 08:00": f"{DAY} 08:30"}, step=10)
    base = [r for r in out["funding"] if r["kind"] == "baseline"]
    assert len(base) == 1 and base[0]["ts"] == utc(f"{DAY} 08:00") and np.sign(base[0]["qty"]) == side
    _baseline(base[0])
    assert len(kinds(out["events"], "funding_stale")) == 1 and len(kinds(out["events"], "funding_stale_cleared")) == 1


@pytest.mark.xfail(strict=True, reason="O17b (part 3): no journalled true-up yet")
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_paper_a_late_rate_is_trued_up_with_the_right_sign(tmp_path, monkeypatch, binance, side):
    """True-up = actual amount - baseline amount. r = +0.03%, b = 0.01%: a long paid b, owes r, so -n(r - b); a short
    paid b and receives r, so +n(r + b) (18:18)."""
    out = paper(tmp_path, monkeypatch, binance, win((f"{DAY} 07:52", f"{DAY} 08:50", side)), f"{DAY} 07:50", 60,
                rates={f"{DAY} 08:00": 0.0003}, published={f"{DAY} 08:00": f"{DAY} 08:30"}, step=10)
    (base,) = [r for r in out["funding"] if r["kind"] == "baseline"]
    (true_up,) = [r for r in out["funding"] if r["kind"] == "true_up"]
    n = abs(base["qty"]) * base["price"]
    want = -n * (0.0003 - BASELINE) if side > 0 else n * (0.0003 + BASELINE)
    assert true_up["amount"] == pytest.approx(want, rel=1e-6)


# --- 3. A lengthened interval (4h -> 8h) charges no false baseline --------------------------------------------

def _four_then_eight(change="2025-10-04 08:00"):
    four = pd.date_range(utc("2025-10-02 20:00"), utc(change), freq="4h")
    eight = pd.date_range(utc(change) + pd.Timedelta(hours=8), utc("2025-10-07 00:00"), freq="8h")
    return four.append(eight)


def test_backtest_a_run_after_a_lengthening_to_the_charged_8h_schedule_is_judged_and_charges_no_baseline(binance):
    """As deployed (Binance charged on 0/8/16): the instrument settled every 4 h, then every 8 h from 4 Oct 08:00. A run
    wholly after the change sees only 8 h steps: no mismatch, every held settlement charged at the venue rate."""
    write_rates({t: 0.0001 for t in _four_then_eight()})
    r = backtest(binance, hourly_bars("2025-10-04 09:00", "2025-10-06 23:00"),
                 win(("2025-10-04 10:00", "2026-01-01", 1)))
    assert r.funding_schedule == "" and r.funding_at_baseline == 0 and r.funding_held >= 7
    assert all(f["kind"] == "settled" for f in r.funding)


def test_backtest_a_run_across_the_lengthening_is_not_judged_only_for_the_shorter_steps(binance):
    """The same store, a run that starts while the instrument was still on 4 h: the 18:35 guard (steps shorter than
    the charged 8 h) makes it not judged; the lengthening itself charges no baseline."""
    write_rates({t: 0.0001 for t in _four_then_eight()})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-06 23:00"),
                 win(("2025-10-03 01:00", "2026-01-01", 1)))
    assert "mismatch" in r.funding_schedule and "4 hours apart" in r.funding_schedule
    assert r.funding_at_baseline == 0


def test_backtest_charged_on_4h_a_lengthening_to_8h_reads_as_missing_settlements(binance, monkeypatch):
    """Evidence only (DA-11 territory): with the engine charging every 4 h, a venue that lengthens to 8 h leaves 12:00
    and 20:00 'missing', so they are charged the baseline and counted. Not reachable as deployed (Binance is charged on
    0/8/16); 19:11's pin for this is paper (O17b file)."""
    monkeypatch.setattr(binance, "funding_hours", (0, 4, 8, 12, 16, 20))
    write_rates({t: 0.0001 for t in _four_then_eight()})
    r = backtest(binance, hourly_bars("2025-10-04 09:00", "2025-10-06 23:00"),
                 win(("2025-10-04 10:00", "2026-01-01", 1)))
    print(f"\ncharged on 4 h after a lengthening: {r.funding_at_baseline} of {r.funding_held} at baseline; "
          f"schedule {r.funding_schedule!r}")
    assert r.funding_at_baseline > 0


# --- 4. The staleness episode: once per episode, and its recovery, past 24 hours ------------------------------

class _Rt:
    def __init__(self):
        self.events, self.t = [], utc(f"{DAY} 08:15").to_pydatetime()
        self.store = SimpleNamespace(event=self._event, events_of=self._events_of)

    def _event(self, sleeve, level, kind, message, ts=None):
        self.events.insert(0, {"kind": kind, "message": message, "ts": ts or self.t, "level": level})

    def _events_of(self, kinds_, limit=500):
        return [e for e in self.events if e["kind"] in kinds_][:limit]

    def now(self):
        return self.t


def test_a_staleness_episode_longer_than_a_day_alerts_once():
    from sleeve_fund.strategies.base import LongFlatStrategy

    rt = _Rt()
    s = SimpleNamespace(runtime=rt)
    LongFlatStrategy._funding_episode(s, PAIR, True, "08:00 missing")
    for h in (8, 16, 24, 32, 40):  # the venue's funding feed stays down; each new missing settlement is noticed
        rt.t += timedelta(hours=8)
        LongFlatStrategy._funding_episode(s, PAIR, True, f"+{h} h missing")
    rt.t += timedelta(hours=2)
    LongFlatStrategy._funding_episode(s, PAIR, False, "arrived")
    got = [e["kind"] for e in reversed(rt.events)]
    print(f"\nepisode of 42 h, feed down: {got}")
    assert got == ["funding_stale", "funding_stale_cleared"]


def test_a_single_rate_arriving_after_a_day_logs_its_recovery():
    from sleeve_fund.strategies.base import LongFlatStrategy

    rt = _Rt()
    s = SimpleNamespace(runtime=rt)
    LongFlatStrategy._funding_episode(s, PAIR, True, "08:00 missing")
    rt.t += timedelta(hours=30)
    LongFlatStrategy._funding_episode(s, PAIR, False, "08:00 arrived")
    got = [e["kind"] for e in reversed(rt.events)]
    print(f"\none rate 30 h late: {got}")
    assert got == ["funding_stale", "funding_stale_cleared"]


# --- 5. N of M: OOS windows vs holdout vs full period, and a >7-day stretch in a G1 study ------------------------

def _study(tmp_path, binance, missing, holdout=True):
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.research.tearsheet import render
    from sleeve_fund.strategies.buy_and_hold import SPEC

    prices = synthetic_ohlcv(days=400, seed=3, start_price=60_000)
    write_rates({t: 0.00012 for t in settlements(prices.index[0] - D, prices.index[-1]) if t not in missing})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, prices, binance.instrument("BTC", "USDT"), dataset="syn-binance-1440m", ledger=ledger,
                  synthetic=True, holdout_days=30, train_days=120, test_days=60, use_holdout=holdout)
    return r, render(r, ledger)


# Folds of this study (no gaps): test windows end 2018-06-29 23:00 (approx), 08-28, 10-27, 12-26 (read from the run).
_BOUNDARY = {}


def _first_boundary(tmp_path, binance):
    if "b" not in _BOUNDARY:
        r, _ = _study(tmp_path / "b", binance, set(), holdout=False)
        _BOUNDARY["b"] = pd.Timestamp(r.folds[0].test_end)
    return _BOUNDARY["b"]


def test_an_eight_day_stretch_across_two_oos_windows_is_not_judged(tmp_path, binance):
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    b = _first_boundary(tmp_path, binance)
    missing = set(settlements(b - pd.Timedelta(days=4), b + pd.Timedelta(days=4)))
    r, sheet = _study(tmp_path, binance, missing, holdout=False)
    print(f"\nboundary {b}; folds' longest {[str(f.funding_gap) for f in r.folds]}; check {r.funding_check}")
    assert r.funding_check[0] == NOT_JUDGED


def test_an_eight_day_stretch_inside_one_oos_window_is_not_judged(tmp_path, binance):
    from sleeve_fund.research.tearsheet import NOT_JUDGED

    b = _first_boundary(tmp_path, binance)
    missing = set(settlements(b - pd.Timedelta(days=20), b - pd.Timedelta(days=12)))
    r, sheet = _study(tmp_path, binance, missing, holdout=False)
    assert r.funding_check[0] == NOT_JUDGED and "**G1: NOT JUDGED**" in sheet


def test_a_seven_day_stretch_exactly_is_judged(tmp_path, binance):
    """21 missing 8-hour settlements = 7 days held: not more than 7, so WARN (2.9% of ~720), not NOT JUDGED."""
    b = _first_boundary(tmp_path, binance)
    every = settlements(b - pd.Timedelta(days=30), b - pd.Timedelta(days=10))
    missing = set(every[:21])
    r, _ = _study(tmp_path, binance, missing, holdout=False)
    assert r.funding_baseline[2] == pd.Timedelta(days=7) and r.funding_check[0] == "WARN", r.funding_check


def test_full_period_over_five_percent_with_clean_oos_and_holdout_is_info_only(tmp_path, binance):
    every_train = settlements("2018-01-07", "2018-04-25")
    r, sheet = _study(tmp_path, binance, set(every_train[::2]))  # ~50% of training, none OOS or in the holdout
    full = r.full_period
    assert full.funding_at_baseline / full.funding_held > 0.05  # setup
    assert r.funding_check[0] == "PASS" and not r.holdout_not_judged and "**G1: NOT JUDGED**" not in sheet


def test_holdout_bad_oos_clean_g1_judged_holdout_not(tmp_path, binance):
    missing = set(settlements("2019-01-10", "2019-01-20"))  # ten days, all inside the holdout
    r, sheet = _study(tmp_path, binance, missing)
    assert r.funding_check[0] == "PASS" and r.holdout_not_judged and "holdout not judged" in sheet.lower()
