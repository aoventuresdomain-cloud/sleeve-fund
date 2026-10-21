"""R-11, R-12 and R-14 (HoE 21 Oct; PM chose "Five years" 10:13 UK; Advisor SCORE-WINDOW addendum in
research/quant-researcher/trend-variants-preregistration.md): a pinned-setting study scored continuously after a
warm-up of at least 10x its longest indicator, to the research end, the holdout locked; this round, QR-B1/B2/B3 also
show the old walk-forward's verdict and pass only if both do; random entries over the continuous window, per year too."""

import numpy as np
import pandas as pd
import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import rescore
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import (CONTINUOUS, MIN_SCORED_DAYS, WARMUP_MULTIPLE, block_warmup,
                                        continuous_warmup, run_study)
from sleeve_fund.research.tearsheet import OLD_WINDOW_CHECK, SCREENING_DISCLOSURE, render
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC as RULES
from sleeve_fund.venues import venue

INSTRUMENT = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
EMA = {"version": 1, "reason": "A test case, written before it is run.",
       "blocks": {"fast": {"kind": "ema", "period": 21}, "slow": {"kind": "ema", "period": 55}},
       "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
                "exit": {"left": "fast", "op": "<", "right": "slow"}}}
HOLDOUT = 120


def _four_hourly(days=900, seed=5):
    idx = pd.date_range("2022-01-01 04:00", periods=days * 6, freq="4h", tz="UTC")
    rng = np.random.default_rng(seed)
    c = np.round(30_000 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, len(idx)) + 0.004 * np.sin(np.arange(len(idx))
                                                                                                 / 90))), 1)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999, "close": c,
                         "volume": 1e9}, index=idx)


PRICES = _four_hourly(days=860)  # its scored window ends flat: every trip in it is counted


def _study(tmp_path, window=CONTINUOUS, prices=PRICES, **kw):
    ledger = IdeaLedger(tmp_path / f"{window}.jsonl")
    r = run_study(RULES, prices, INSTRUMENT, dataset="syn", ledger=ledger, synthetic=True, holdout_days=HOLDOUT,
                  train_days=365, test_days=180, default_params=to_params(EMA), score_window=window, **kw)
    return r, ledger


# ---- R-11 --------------------------------------------------------------------------------------------------------------

def test_the_warm_up_is_the_most_any_block_needs_by_the_advisors_rule():
    """Recursive blocks (EMA, Wilder's ATR and RSI) need 10N bars; exact rolling ones (SMA, Donchian) N."""
    assert continuous_warmup("rules", to_params(EMA), 240) == WARMUP_MULTIPLE * 55 == 550  # about 92 days on 4h
    assert block_warmup("atr", {"period": 14}) == 140 and block_warmup("rsi", {"period": 14}) == 140
    assert block_warmup("sma", {"period": 200}) == 200 and block_warmup("donchian", {"period": 55}) == 55
    daily = {**EMA, "blocks": {**EMA["blocks"], "slow": {"kind": "ema", "period": 55, "timeframe": "1d"}}}
    assert continuous_warmup("rules", to_params(daily), 240) == (550 + 1) * 6  # a daily block in 4h candles
    with pytest.raises(ValueError, match="rule-builder definition"):
        continuous_warmup("trend_filter", {}, 1440)


QR21 = {"version": 1, "reason": "The 200-day trend, as the Researcher's qr21 file has it.",
        "blocks": {"sma": {"kind": "sma", "period": 200}, "atr": {"kind": "atr", "period": 14}},
        "long": {"entry": {"left": "close", "op": ">", "right": "sma"},
                 "exit": {"left": "close", "op": "<", "right": "sma"}},
        "exits": {"stop": {"level": {"sub": ["close", {"mul": [3.0, "atr"]}]}}}}


def test_qr21s_200_day_average_warms_up_200_days_not_its_atrs_library_settle():
    """HoE cell: the 200-day SMA decides (its ATR(14) needs 140); the library's own settle reads 280 for the ATR."""
    assert continuous_warmup("rules", to_params(QR21), 1440) == 200


def test_a_mixed_ema_and_sma_definition_takes_whichever_needs_more():
    mixed = {**QR21, "blocks": {"sma": {"kind": "sma", "period": 200}, "fast": {"kind": "ema", "period": 21}},
             "exits": {}}
    assert continuous_warmup("rules", to_params(mixed), 1440) == 210  # EMA 21: 210 beats SMA 200
    mixed["blocks"]["fast"]["period"] = 15
    assert continuous_warmup("rules", to_params(mixed), 1440) == 200  # EMA 15: 150, so the SMA's 200


def test_a_55_bar_strategy_on_4h_gets_at_least_550_bars_and_nothing_is_scored_inside_them(tmp_path):
    """Red first: before R-11, a pinned study's scoring started after a 365-day training fold, not the warm-up."""
    r, _ = _study(tmp_path)
    research = PRICES.iloc[:-HOLDOUT * 6]
    w = r.warmup_bars
    assert w == 550 and r.score_window == CONTINUOUS
    [fold] = r.folds
    first_scored = research.index[w]
    assert fold.train_end == research.index[w - 1] and fold.test_end == research.index[-1]
    assert all(pd.Timestamp(o) >= first_scored for o, _, _ in fold.trips) and fold.trips
    assert r.oos_returns.index[0] >= first_scored.normalize() - pd.Timedelta(days=1)
    assert f"continuous: a warm-up of {w:,} bars" in r.settings
    assert f"to {(research.index[-1] - pd.Timedelta(hours=4)):%d %b %Y}" in r.settings  # the last day scored


def test_no_scored_bar_falls_in_the_holdout(tmp_path):
    r, ledger = _study(tmp_path)
    holdout_first = PRICES.index[-HOLDOUT * 6]
    assert r.research_end < holdout_first
    assert r.oos_returns.index[-1] < holdout_first and r.folds[0].test_end < holdout_first
    assert not ledger.holdout_used(RULES.name, "syn")


def test_a_study_that_tunes_settings_keeps_the_walk_forward_and_is_refused_a_continuous_window(tmp_path):
    from sleeve_fund.strategies.trend_filter import SPEC as TUNED

    prices = synthetic_ohlcv(days=1500, seed=3)
    kw = dict(dataset="syn", synthetic=True, holdout_days=200, train_days=730, test_days=365)
    with pytest.raises(ValueError, match="pinned settings only: this study tunes 11 settings"):
        run_study(TUNED, prices, venue("KRAKEN").instrument("BTC", "USD"), ledger=IdeaLedger(tmp_path / "x"),
                  score_window=CONTINUOUS, **kw)
    sheets = []
    for window in ({}, {"score_window": "walk_forward"}):
        ledger = IdeaLedger(tmp_path / str(len(sheets)) / "l.jsonl")
        r = run_study(TUNED, prices, venue("KRAKEN").instrument("BTC", "USD"), ledger=ledger, **kw, **window)
        sheets.append(render(r, ledger))
        assert r.warmup_bars is None and len(r.folds) == 1 and not r.random_entry_by_year
    assert sheets[0] == sheets[1]
    assert "walk-forward 730 days training, 365 days testing" in sheets[0] and "Scored continuously" not in sheets[0]


def test_a_window_too_short_to_score_after_its_warm_up_is_refused(tmp_path):
    with pytest.raises(ValueError, match=f"leaves under {MIN_SCORED_DAYS} days to score"):
        run_study(RULES, PRICES.iloc[: (HOLDOUT + 400) * 6], INSTRUMENT, dataset="syn", ledger=IdeaLedger(tmp_path / "l"),
                  synthetic=True, holdout_days=HOLDOUT, train_days=60, test_days=60, default_params=to_params(EMA),
                  score_window=CONTINUOUS)


# ---- R-14 --------------------------------------------------------------------------------------------------------------

def test_random_entry_over_the_continuous_window_matches_trades_and_time_in_market_and_reports_each_year(tmp_path):
    from sleeve_fund.research.random_entry import DRAWS

    r, ledger = _study(tmp_path)
    [fold] = r.folds
    re_ = r.random_entry
    assert DRAWS >= 1000 and re_.draws == DRAWS and re_.trades == len(fold.trips)
    held = sum(int((pd.Timestamp(c) - pd.Timestamp(o)) / pd.Timedelta(hours=4)) for o, c, _ in fold.trips)
    assert fold.carried_out == 0 and abs(held - fold.in_market_bars) <= 1  # the draws hold these: within one bar
    years = r.random_entry_by_year
    assert sorted(years) == sorted({ts.year for ts in r.oos_returns.index})[: len(years)] and len(years) >= 2
    assert sum(y.trades for y in years.values()) == re_.trades  # every trip in exactly one year, whole
    sheet = render(r, ledger)
    assert "> Random entry by test year" in sheet and f"{min(years)}: " in sheet
    again, _ = _study(tmp_path / "again")
    assert {y: v.return_percentile for y, v in again.random_entry_by_year.items()} == {
        y: v.return_percentile for y, v in years.items()}  # a fixed seed


# ---- R-12 --------------------------------------------------------------------------------------------------------------

def test_the_old_windows_verdict_sits_beside_the_new_and_the_sheet_passes_only_if_both_do(tmp_path):
    r, ledger = _study(tmp_path)
    old, _ = _study(tmp_path / "old", window="walk_forward")
    sheet = render(r, ledger, old=old)
    row = next(line for line in sheet.splitlines() if line.startswith(f"| {OLD_WINDOW_CHECK} |"))
    assert "walk-forward 365 days training, 180 days testing" in row
    assert SCREENING_DISCLOSURE in sheet
    plain = render(r, ledger)
    assert OLD_WINDOW_CHECK not in plain and SCREENING_DISCLOSURE not in plain
    from sleeve_fund.research.tearsheet import g1_checks, g1_verdict

    new = g1_checks(r, ledger)
    assert g1_verdict(new + [(OLD_WINDOW_CHECK, "FAIL", "")])[0] == "FAIL"  # the old window failing fails the sheet
    assert g1_verdict(new + [(OLD_WINDOW_CHECK, "PASS", "")])[0] == g1_verdict(new)[0]  # passing, the new decides


def test_the_list_line_turns_continuous_on_and_only_the_qr_b_rows_also_show_the_old_window(tmp_path):
    pinned_text = (rescore.ROOT / "configs" / "studies" / "qr-rescore-list.md").read_text()
    on = pinned_text.replace("## Common settings\n", "## Common settings\n\n- **Scoring window:** continuous: a test.\n")
    assert rescore.CONTINUOUS_LINE.search(on) and not rescore.CONTINUOUS_LINE.search(
        pinned_text.replace("Scoring window:** continuous", "Scoring window:** walk-forward"))
    from sleeve_fund.research.pinned import parse

    pins = parse(on)
    b1 = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT", pins, continuous=True)
    other = rescore.request("qr02_dip_long", 240, "binance", "BTC/USDT", pins, continuous=True)
    off = rescore.request("qr01_trend_ema_long", 240, "binance", "BTC/USDT", pins)
    assert (b1.score_window, b1.old_window_too) == (CONTINUOUS, True)
    assert (other.score_window, other.old_window_too) == (CONTINUOUS, False)
    assert (off.score_window, off.old_window_too) == ("walk_forward", False)
    assert len(rescore.OLD_WINDOW_TOO) == 6 and all((rescore.DEFINITIONS / f"{n}.toml").is_file()
                                                     for n in rescore.OLD_WINDOW_TOO)
    # the runner itself reads the line: with it every row goes continuous, without it every row keeps the walk-forward
    walk = pinned_text.replace("Scoring window:** continuous", "Scoring window:** walk-forward")
    for text, want in ((on, CONTINUOUS), (walk, "walk_forward")):
        (tmp_path / "list.md").write_text(text)
        ran = []
        assert rescore.main([str(tmp_path / "list.md")], run=lambda req, store: ran.append(req),
                            journal=lambda: None) == 0
        assert ran and {r.score_window for r in ran} == {want}
        assert {r.old_window_too for r in ran} == ({True, False} if want == CONTINUOUS else {False})


class _History:
    """The history store as the server's study job reads it: these daily bars only."""

    def __init__(self, bars):
        self.bars = bars

    def read(self, venue, pair, minutes, start=None, end=None):
        assert minutes == 1440
        return self.bars

    def coverage(self, venue, pair):
        return None


def test_the_study_job_runs_the_old_window_beside_the_new_and_counts_only_the_new(tmp_path, monkeypatch):
    """R-12 through run_store_study: the same settings on the old walk-forward, never logged as a try (no new tries,
    Advisor SCORE-WINDOW): its runs go to a scratch ledger and no register; the sheet carries its verdict."""
    from sleeve_fund.research import run as run_mod
    from sleeve_fund.research import study as study_mod

    monkeypatch.setattr(run_mod, "exec_step", lambda span, minutes: None)
    monkeypatch.setattr(run_mod, "OOS_EXEC_BUDGET", 0)
    calls, real = [], study_mod.run_study

    def spy(*a, **k):
        calls.append((k["score_window"], k["ledger"].path, k["register"]))
        return real(*a, **k)

    monkeypatch.setattr(study_mod, "run_study", spy)
    req = run_mod.StudyRequest(strategy="rules", pair="BTC/USD", venue="kraken", minutes=1440, risk_profile=None,
                               train_days=365, test_days=365, holdout_days=200, params=to_params(EMA), label="ema",
                               score_window=CONTINUOUS, old_window_too=True)
    ledger = tmp_path / "l.jsonl"
    sheet = run_mod.run_store_study(req, ledger_path=ledger, out_dir=tmp_path / "ts",
                                    history=_History(synthetic_ohlcv(days=2100, seed=3))).read_text()
    assert [(w, p == ledger) for w, p, _ in calls] == [(CONTINUOUS, True), ("walk_forward", False)]
    assert calls[1][2] is None
    assert f"| {OLD_WINDOW_CHECK} |" in sheet and SCREENING_DISCLOSURE in sheet
    with pytest.raises(ValueError, match="beside a continuous window's only"):
        run_mod.StudyRequest(strategy="rules", pair="BTC/USD", old_window_too=True).validate()


def test_a_trip_across_new_year_stays_whole_in_the_year_it_opened():
    """Advisor R-14 ruling: a trip belongs to the year it opened, never split; that year's window runs on to its close
    and the next year starts after it."""
    from types import SimpleNamespace

    from sleeve_fund.research.study import Fold, _years

    idx = pd.date_range("2023-12-01", "2024-03-01", freq="1D", tz="UTC")
    trips = [(pd.Timestamp("2023-12-10", tz="UTC"), pd.Timestamp("2023-12-20", tz="UTC"), 1),
             (pd.Timestamp("2023-12-28", tz="UTC"), pd.Timestamp("2024-01-09", tz="UTC"), 1),  # across new year
             (pd.Timestamp("2024-01-20", tz="UTC"), pd.Timestamp("2024-02-02", tz="UTC"), 1)]
    fold = Fold(train_start=idx[0], train_end=idx[0], test_end=idx[-1], chosen={}, train_sharpe=float("nan"), test={},
                benchmark_test=None, trips=trips, window_bars=len(idx))
    run = SimpleNamespace(exposure=pd.Series(0.0, index=idx))
    years = _years(fold, run, idx)
    assert [(y, len(f.trips)) for y, f, _ in years] == [(2023, 2), (2024, 1)]
    (_, y23, n23), (_, y24, n24) = years
    assert y23.test_end == pd.Timestamp("2024-01-09", tz="UTC")  # on to the close of the trip it opened
    assert n23 + n24 == len(idx) and n23 == idx.get_loc(y23.test_end) + 1  # no bar in two years, none left out


TIMED = {"prior_period": ({"kind": "prior_period", "period": "week"}, "high"),
         "session_levels": ({"kind": "session_levels", "session": "asia"}, "high"),
         "vwap": ({"kind": "vwap", "anchor": "day"}, "value"),
         "funding_decile": ({"kind": "funding_decile", "window_days": 90}, "value")}


def _with(kind: str) -> dict:
    block, output = TIMED[kind]
    return {**EMA, "blocks": {**EMA["blocks"], "t": block},
            "long": {"entry": {"all": [EMA["long"]["entry"], {"left": "close", "op": ">", "right": f"t.{output}"}]},
                     "exit": EMA["long"]["exit"]}}


def test_a_block_that_warms_up_over_time_is_counted_in_candles_of_its_size():
    """Advisor 21 Oct, defect 1: a calendar period, a session and a day VWAP warm up over time; as the builder counts
    them (definitions.py), ceil(minutes / candle) + 1 candles of the block's size, never 1-minute bars."""
    assert block_warmup("prior_period", {"period": "week"}, 60) == 2 * 7 * 24 + 1 == 337  # two weeks, not 20,161
    assert block_warmup("prior_period", {"period": "day"}, 60) == 2 * 24 + 1 == 49
    assert block_warmup("session_levels", {"session": "asia"}, 60) == 49
    assert block_warmup("vwap", {"anchor": "day"}, 60) == 24 + 1
    assert continuous_warmup("rules", to_params(_with("prior_period")), 60) == 550  # the EMA's 10 x 55 decides
    assert continuous_warmup("rules", to_params(_with("prior_period")), 15) == 2 * 7 * 96 + 1 == 1345
    four_hourly = {**_with("prior_period")}
    four_hourly["blocks"] = {**four_hourly["blocks"], "t": {**TIMED["prior_period"][0], "timeframe": "4h"}}
    assert continuous_warmup("rules", to_params(four_hourly), 15) == (2 * 7 * 6 + 1 + 1) * 16  # in 4h candles


def test_the_funding_deciles_trailing_days_are_counted_in_the_warm_up():
    """Advisor 21 Oct, defect 2: the decile reads the settlements of its trailing window_days, so a continuous window
    warms up that many days of candles before it scores, not 0."""
    assert block_warmup("funding_decile", {"window_days": 90}, 60) == 90 * 24 + 1 == 2161
    assert continuous_warmup("rules", to_params(_with("funding_decile")), 60) == 2161
    assert continuous_warmup("rules", to_params(_with("funding_decile")), 1440) == 550


def test_a_block_fed_another_block_warms_up_after_its_input_has():
    """A block read from another's output settles only once its input has: the builder adds the two warm-ups, and
    the continuous window does too (an EMA of an RSI: 10 x 30 after the RSI's 10 x 30)."""
    chained = {**EMA, "blocks": {**EMA["blocks"], "rsi": {"kind": "rsi", "period": 30},
                                 "smooth": {"kind": "ema", "period": 30, "input": "rsi"}},
               "long": {"entry": {"all": [EMA["long"]["entry"], {"left": "smooth", "op": ">", "right": 50}]},
                        "exit": EMA["long"]["exit"]}}
    assert continuous_warmup("rules", to_params(chained), 240) == 300 + 300
