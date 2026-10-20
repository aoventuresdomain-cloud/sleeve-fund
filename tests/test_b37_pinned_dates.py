"""B-37 (Advisor: PIN IT): a study list pins its own dates (research.pinned). A study run from it reads those days only,
whatever the day it runs: the same list on two different days gives identical results, never a bar after the pinned
end is read, and a list with no pinned dates is refused in words.

Fold dates are worked by hand from the windows (train 730 days, test 365), never read back from the study."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import pinned as pinning
from sleeve_fund.research import rescore
from sleeve_fund.research import study as study_mod
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.research.tearsheet import render
from sleeve_fund.strategies.trend_filter import SPEC
from sleeve_fund.venues import venue

PRICES = synthetic_ohlcv(days=2100, seed=3)  # daily bars; each closes at 00:00 UTC the day after the one it covers
FIRST = (PRICES.index[0] - pd.Timedelta(days=1)).date()
KW = dict(dataset="syn", synthetic=True, train_days=730, test_days=365)
INSTRUMENT = venue("KRAKEN").instrument("BTC", "USD")  # conftest's


def _pins(start: date = FIRST, folds: int = 2, holdout: int = 200) -> pinning.Pinned:
    """Research from `start`: `folds` folds of 730 days' training then 365 testing, stepping 365; then the holdout."""
    fs = tuple(pinning.Fold(start + timedelta(days=365 * k), start + timedelta(days=365 * k + 730),
                            start + timedelta(days=365 * k + 730 + 364)) for k in range(folds))
    end = fs[-1].test_end
    return pinning.Pinned(start, end, end + timedelta(days=1), end + timedelta(days=holdout), fs)


def _block(p: pinning.Pinned) -> str:
    folds = "\n".join(f"| {k} | {f.train_start} | {f.test_start} | {f.test_end} |" for k, f in enumerate(p.folds, 1))
    return (f"## Pinned dates\n\n- **Data start:** {p.data_start}\n- **Research end:** {p.research_end}\n"
            f"- **Holdout:** {p.holdout_start} to {p.holdout_end}\n\n| Fold | Train start | Test start | Test end |\n"
            f"| --- | --- | --- | --- |\n{folds}\n")


def _upto(day: date) -> pd.DataFrame:
    """The history as it stood on `day`: every bar closed by then."""
    return PRICES[PRICES.index <= pd.Timestamp(day, tz="UTC")]


def _study(tmp_path, history, pins, **kw):
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, history, INSTRUMENT, ledger=ledger, holdout_days=pins.holdout_days if pins else 200,
                  pinned=pins, **KW, **kw)
    return r, render(r, ledger)


def test_b37a_the_list_block_parses_to_its_dates():
    p = _pins()
    assert pinning.parse("# list\n\n" + _block(p) + "\n## Studies\n") == p
    assert p.holdout_days == 200 and p.end == p.holdout_end


@pytest.mark.parametrize("text, words", [
    ("# a list with no dates\n", "pins no dates"),
    ("- **Data start:** 2019-01-01\n", "exactly one data start, one research end and one holdout"),
    ("- **Data start:** 2019-01-01\n- **Research end:** 2020-01-01\n- **Holdout:** 2020-01-03 to 2020-12-31\n"
     "| 1 | 2019-01-01 | 2019-06-01 | 2020-01-01 |\n", "must start the day after the research end"),
    ("- **Data start:** 2019-01-01\n- **Research end:** 2020-01-01\n- **Holdout:** 2020-01-02 to 2020-12-31\n",
     "no walk-forward folds"),
    ("- **Data start:** 2019-01-01\n- **Research end:** 2020-01-01\n- **Holdout:** 2020-01-02 to 2020-12-31\n"
     "| 2 | 2019-01-01 | 2019-06-01 | 2020-01-01 |\n", "numbered 1, 2, 3"),
])
def test_b37b_a_list_that_pins_no_dates_or_bad_ones_is_refused_in_words(text, words):
    with pytest.raises(ValueError, match=words):
        pinning.parse(text)


def test_b37c_the_same_list_on_two_different_days_gives_identical_results(tmp_path):
    """Run on the day the holdout ends and again 60 days later, with 60 more days of history: identical. Without the
    pin the later day's holdout and walk-forward slide forward 60 days, so its results move (what the pin fixes)."""
    p = _pins()
    today, later = p.holdout_end + timedelta(days=1), p.holdout_end + timedelta(days=61)
    a, sheet_a = _study(tmp_path / "a", _upto(today), p)
    b, sheet_b = _study(tmp_path / "b", _upto(later), p)
    assert sheet_a == sheet_b
    assert a.oos_returns.equals(b.oos_returns) and a.research_end == b.research_end
    assert [(f.train_start, f.train_end, f.test_end) for f in a.folds] == [(f.train_start, f.train_end, f.test_end)
                                                                          for f in b.folds]
    assert a.research_end == pd.Timestamp(p.research_end + timedelta(days=1), tz="UTC")  # its last bar's close
    loose_a, _ = _study(tmp_path / "la", _upto(today)[PRICES.index[0]:], None)
    loose_b, _ = _study(tmp_path / "lb", _upto(later)[PRICES.index[0]:], None)
    assert loose_a.research_end != loose_b.research_end  # unpinned, the day it runs moves the study


@pytest.mark.parametrize("use_holdout", [False, True])
def test_b37d_no_bar_after_the_pinned_end_is_ever_read(tmp_path, monkeypatch, use_holdout):
    p = _pins()
    seen, real = [], study_mod.run_backtest

    def spy(name, df, inst, params, **k):
        seen.append(df.index.max())
        for x in (k.get("exec_prices"),):
            if x is not None and len(x):
                seen.append(x.index.max())
        return real(name, df, inst, params, **k)

    draws, real_bench = [], study_mod._benchmarks

    def bench(prices, folds, *a, **k):  # the random draws: the bars they are drawn and priced on, and their windows
        draws.append((prices.index.max(), max((f.test_end for f in folds), default=None)))
        return real_bench(prices, folds, *a, **k)

    monkeypatch.setattr(study_mod, "run_backtest", spy)
    monkeypatch.setattr(study_mod, "_benchmarks", bench)
    history = _upto(p.holdout_end + timedelta(days=120))
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(SPEC, history, INSTRUMENT, ledger=ledger, holdout_days=p.holdout_days, pinned=p,
                  use_holdout=use_holdout, **KW)
    last, research_last = (pd.Timestamp(d + timedelta(days=1), tz="UTC") for d in (p.holdout_end, p.research_end))
    assert seen and max(seen) <= last and draws
    # Advisor (B-37 check): no fold, warm-up or random draw reads past the research end without the holdout asked for
    assert all(c <= research_last and (w is None or w <= research_last) for c, w in draws), draws
    if use_holdout:  # the holdout run alone reads past it, to its pinned end and no further, and is recorded as used
        assert r.holdout is not None and max(seen) == last
        assert ledger.holdout_used(SPEC.name, "syn")
    else:
        assert max(seen) <= research_last and r.holdout is None
        assert not ledger.holdout_used(SPEC.name, "syn")


def test_b37e_a_history_that_does_not_reach_the_pinned_end_is_refused(tmp_path):
    p = _pins()
    with pytest.raises(ValueError, match="history covers .* not the pinned"):
        _study(tmp_path, _upto(p.holdout_end), p)  # one day short


def test_b37f_a_walk_forward_other_than_the_pinned_folds_is_refused(tmp_path):
    p = _pins()
    moved = pinning.Pinned(p.data_start, p.research_end, p.holdout_start, p.holdout_end,
                           (p.folds[0], pinning.Fold(p.folds[1].train_start, p.folds[1].test_start + timedelta(days=1),
                                                     p.folds[1].test_end)))
    with pytest.raises(ValueError, match="walk-forward fold 2 would be"):
        _study(tmp_path, _upto(p.holdout_end + timedelta(days=1)), moved)
    with pytest.raises(ValueError, match="holdout is 200 days, not 365"):
        run_study(SPEC, _upto(p.holdout_end + timedelta(days=1)), INSTRUMENT,
                  ledger=IdeaLedger(tmp_path / "x.jsonl"), holdout_days=365, pinned=p, **KW)


def test_b37g_the_runner_refuses_a_list_with_no_pinned_dates_before_anything_runs(tmp_path, capfd):
    lst = tmp_path / "list.md"
    lst.write_text("# no dates\n\n| # | Definition | Idea family | Decision candle |\n|---|---|---|---|\n"
                   "| 1 | qr01_trend_ema_long | QR-B1 | 4h |\n")
    ran = []
    assert rescore.main([str(lst)], run=lambda *a, **k: ran.append(a), journal=lambda: None) == 2
    assert not ran and "refused: the study list pins no dates" in capfd.readouterr().out


def test_b37h_each_row_carries_the_lists_pinned_dates(tmp_path):
    p = _pins(holdout=365)
    lst = tmp_path / "list.md"
    lst.write_text(_block(p) + "\n| # | Definition | Idea family | Decision candle |\n|---|---|---|---|\n"
                   "| 1 | qr01_trend_ema_long | QR-B1 | 4h |\n| 2 | qr02_dip_long | QR-B4 | 4h |\n")
    seen = []
    assert rescore.main([str(lst)], run=lambda req, store=None: seen.append(req), journal=lambda: None) == 0
    assert [r.pinned for r in seen] == [p, p]


def test_b37i_the_qr_rescore_list_pins_its_dates_and_the_walk_forward_is_its_folds():
    """The Researcher's dates (20 Oct): research 2020-10-21 to 2025-10-19 (1,825 days, 2024-02-29 in it), the holdout
    the 365 days after, three folds of 730/365 that end on the research end."""
    p = pinning.parse(rescore.ROOT.joinpath("configs", "studies", "qr-rescore-list.md").read_text())
    assert (p.data_start, p.research_end, p.holdout_start, p.holdout_end, p.holdout_days) == (
        date(2020, 10, 21), date(2025, 10, 19), date(2025, 10, 20), date(2026, 10, 19), rescore.HOLDOUT_DAYS)
    assert [(f.train_start, f.test_start, f.test_end) for f in p.folds] == [
        (date(2020, 10, 21), date(2022, 10, 21), date(2023, 10, 20)),
        (date(2021, 10, 21), date(2023, 10, 21), date(2024, 10, 19)),
        (date(2022, 10, 21), date(2024, 10, 20), date(2025, 10, 19))]
    days = pd.date_range("2020-10-22", "2026-10-20", freq="1D", tz="UTC")  # each day's bar, closing the next 00:00
    bars = pd.DataFrame({"close": 1.0}, index=days)
    research = pinning.cut(bars, p).iloc[:-p.holdout_days]
    pinning.check_folds(p, research, 730, 365, pd.Timedelta(days=1))  # the study's own walk-forward: these folds
    assert len(research) == 1825


class _History:
    """The history store as the server's study job reads it (run.run_store_study): the daily bars as they stood on
    `day`. Shorter bars are not served: the study's execution bars are cut by b37k, not here."""

    def __init__(self, day: date):
        self.bars = _upto(day)

    def read(self, venue, pair, minutes, start=None, end=None):
        assert minutes == 1440, f"only daily bars are stored here, not {minutes}-minute ones"
        return self.bars

    def coverage(self, venue, pair):
        return None


def test_b37j_the_servers_study_job_reads_the_lists_pinned_dates_whatever_day_it_runs(tmp_path, monkeypatch):
    """CR MUST (B-37): through run_store_study, the job the re-score runner starts. The same pinned request on the day
    the holdout ends and 60 days later, with 60 more days of history, writes the same tear sheet. RED with run.py's
    pinned=req.pinned dropped: the later day's study then reads its 60 extra days."""
    from sleeve_fund.research import run as run_mod

    monkeypatch.setattr(run_mod, "exec_step", lambda span, minutes: None)  # daily bars only (see _History)
    monkeypatch.setattr(run_mod, "OOS_EXEC_BUDGET", 0)
    p = _pins()
    req = run_mod.StudyRequest(strategy=SPEC.name, pair="BTC/USD", venue="kraken", minutes=1440, train_days=730,
                               test_days=365, holdout_days=p.holdout_days, pinned=p)
    sheets = []
    for k, day in enumerate((p.holdout_end + timedelta(days=1), p.holdout_end + timedelta(days=61))):
        out = run_mod.run_store_study(req, ledger_path=tmp_path / f"{k}" / "l.jsonl", out_dir=tmp_path / f"ts{k}",
                                      history=_History(day))
        sheets.append(out.read_text())
    assert sheets[0] == sheets[1] and "Tear sheet" in sheets[0]


def test_b37k_no_execution_bar_after_the_pinned_end_reaches_a_run(tmp_path, monkeypatch):
    """CR follow-up: the shorter bars a study executes on (exec_prices, and oos_exec_prices for the out-of-sample
    windows and the holdout) run 120 days past the pinned end; no run is handed one after it, with or without the
    holdout. The bars are 6-hourly stamps of the daily closes; the spy records what each run is handed and runs it on
    its daily bars alone, so only the cut is under test."""
    p = _pins()
    last = pd.Timestamp(p.holdout_end + timedelta(days=1), tz="UTC")
    history = _upto(p.holdout_end + timedelta(days=120))
    six = pd.date_range(history.index[0] - pd.Timedelta(hours=18), history.index[-1], freq="6h")
    fine = history.reindex(six, method="bfill").dropna()
    seen, real = [], study_mod.run_backtest

    def spy(name, df, inst, params, **k):
        x = k.get("exec_prices")
        if x is not None and len(x):
            seen.append(x.index.max())
        return real(name, df, inst, params, **{**k, "exec_prices": None, "exec_minutes": 1})

    monkeypatch.setattr(study_mod, "run_backtest", spy)
    for use_holdout in (False, True):
        run_study(SPEC, history, INSTRUMENT, ledger=IdeaLedger(tmp_path / f"l{use_holdout}.jsonl"),
                  holdout_days=p.holdout_days, pinned=p, use_holdout=use_holdout, exec_prices=fine,
                  oos_exec_prices=fine, **KW)
    assert seen and max(seen) <= last, (max(seen), last)
