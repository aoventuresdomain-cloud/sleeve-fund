"""QA P1-O17 (Advisor, 6 Oct 2026): out-of-sample funding charged the baseline for a missing rate is counted on
the tear sheet, warned above 1% of the settlements a position was held through, and above 5%, or any stretch
longer than 7 days, G1 doesn't judge the study."""

from types import SimpleNamespace

import pandas as pd

from sleeve_fund.research import tearsheet
from sleeve_fund.research.study import Fold, StudyResult, _window_funding

T0 = pd.Timestamp("2026-01-01", tz="UTC")
H8 = pd.Timedelta(hours=8)


def _marks(n: int, missing: set[int], flat: set[int] = frozenset()) -> list:
    return [(T0 + i * H8, i in missing, i not in flat) for i in range(n)]


def test_a_windows_marks_count_held_baseline_and_the_longest_held_stretch():
    """The window is (after, end], as its returns are; a stretch is the held time inside an unbroken run of missing
    rates, which only a real rate breaks (Advisor, 18:18)."""
    out = _window_funding(_marks(30, {3, 4, 5, 20}), T0, T0 + 25 * H8)
    assert out["funding_held"] == 25 and out["funding_baseline"] == 4 and out["funding_gap"] == 3 * H8
    # Flat through most of a stretch, held at its end: only the held time counts, and the flat time doesn't break it.
    out = _window_funding(_marks(40, set(range(5, 30)), flat=set(range(0, 29))), T0 - H8, T0 + 39 * H8)
    assert out["funding_held"] == 11 and out["funding_baseline"] == 1 and out["funding_gap"] == H8
    assert _window_funding(_marks(10, {2, 3, 4}, flat={3}), T0 - H8, T0 + 9 * H8)["funding_gap"] == 2 * H8
    assert _window_funding(_marks(10, {2, 4}, flat={3}), T0 - H8, T0 + 9 * H8)["funding_gap"] == H8  # a real rate
    # A stretch only ever flat doesn't count against the study.
    assert _window_funding(_marks(40, set(range(5, 30)), flat=set(range(0, 31))), T0, T0 + 39 * H8)["funding_gap"] \
        == pd.Timedelta(0)


def test_the_baseline_is_scaled_to_the_settlement_interval():
    import dataclasses

    from sleeve_fund import markets

    eight = markets.LOW_FEE_PERP
    assert markets.baseline_rate(eight) == 0.0001
    hourly = dataclasses.replace(eight, funding_hours=tuple(range(24)))
    assert abs(markets.baseline_rate(hourly) - 0.0001 / 8) < 1e-15


def _study(*folds: dict, holdout: dict | None = None) -> SimpleNamespace:
    made = [Fold(train_start=T0, train_end=T0, test_end=T0, chosen={}, train_sharpe=1.0, test={}, benchmark_test={},
                 test_trades=5, **f) for f in folds]
    r = SimpleNamespace(folds=made, error_count=0, errors=[], oos_trades=10, holdout={"sharpe": 1.0} if holdout else None,
                        holdout_funding=holdout or {},
                        full_period=SimpleNamespace(funding_held=0, funding_at_baseline=0, funding_simulated=False, funding_schedule="",
                                                    funding_baseline_longest=pd.Timedelta(0)))
    for name in ("funding_baseline", "funding_check", "not_judged", "holdout_funding_baseline", "holdout_not_judged"):
        setattr(r, name, getattr(StudyResult, name).fget(r))
    return r


def test_g1_judges_under_1_percent_warns_to_5_and_not_judged_above_5_or_a_long_stretch():
    clean = _study(dict(funding_held=100, funding_baseline=0))
    assert clean.not_judged == "" and tearsheet._funding_rows(clean)[0][1:] == (
        "PASS", "0 of 100 out-of-sample funding settlements held through charged the baseline for a missing rate (0.0%)")
    warned = _study(dict(funding_held=100, funding_baseline=3, funding_gap=H8))
    assert warned.not_judged == "" and tearsheet._funding_rows(warned)[0][1] == "WARN"
    edge = _study(dict(funding_held=100, funding_baseline=5, funding_gap=H8))  # 5% is inside the warn band
    assert tearsheet._funding_rows(edge)[0][1] == "WARN"
    many = _study(dict(funding_held=50, funding_baseline=2), dict(funding_held=50, funding_baseline=4))
    assert "6 of 100" in many.not_judged and "backfill" in many.not_judged
    assert tearsheet._funding_rows(many)[0][1] == tearsheet.NOT_JUDGED
    long = _study(dict(funding_held=1000, funding_baseline=22, funding_gap=pd.Timedelta(days=7, hours=8)))
    assert "7.3 days" in long.not_judged
    assert tearsheet._funding_rows(_study(dict())) == []  # spot, or never held a perpetual through a settlement


def test_the_holdout_is_counted_and_judged_on_its_own_row():
    """The holdout is the last check before G1 means anything, so it doesn't quietly pass on baseline funding."""
    gap = pd.Timedelta(days=8)
    r = _study(dict(funding_held=100), holdout=dict(funding_held=50, funding_baseline=1, funding_gap=gap))
    assert r.not_judged == "" and "1 of 50 holdout funding settlements" in r.holdout_not_judged
    rows = tearsheet._funding_rows(r)
    assert [name.split(" ")[0] for name, _, _ in rows] == ["Funding", "Holdout"]
    assert rows[1][2].startswith("holdout not judged: 1 of 50 holdout") and "8.0 days" in rows[1][2]
    fine = _study(dict(funding_held=100), holdout=dict(funding_held=50, funding_baseline=0))
    assert fine.holdout_not_judged == ""


def test_a_funding_feed_that_stops_is_raised_once_per_episode(tmp_path, monkeypatch):
    """gaps() only sees holes between kept rates, so a feed that simply stops is raised on its own (QA P1-O17)."""
    import json

    from sleeve_fund import funding, history
    from sleeve_fund.venues import venue

    path = funding._path("BINANCE", "BTC/USDT", tmp_path)
    path.parent.mkdir(parents=True)
    kept = [T0 + i * H8 for i in range(3)]
    path.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0001] for t in kept]}))
    last = kept[-1]
    assert funding.stale("BINANCE", "BTC/USDT", tmp_path, now=last + 2 * H8) is None  # one late, one due: not yet
    problem = funding.stale("BINANCE", "BTC/USDT", tmp_path, now=last + 5 * H8)
    assert "funding last kept 2026-01-01 16:00 UTC, about 5 settlements ago" in problem
    assert funding.stale("BINANCE", "ETH/USDT", tmp_path) is None  # nothing collected yet

    profile = venue("BINANCE")
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [])  # the venue returns nothing new
    monkeypatch.setattr(profile, "stats_loaders", {})
    sent = []

    class Inbox:
        def event(self, sleeve, level, kind, message, ts=None):
            sent.append((level, kind, message))
    monkeypatch.setattr("sleeve_fund.store.Store", Inbox)
    history._warned.clear()
    monkeypatch.setattr(history, "_stale", set())
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    history._warned.clear()  # a day later, the same episode: still one alert
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(level, kind) for level, kind, _ in sent] == [("warning", "funding_stale")]
    # The rates catch up: the gap gets its end, once (Advisor).
    fresh = [int(t.timestamp() * 1000) for t in pd.date_range(last + H8, pd.Timestamp.now(tz="UTC"), freq="8h")]
    monkeypatch.setattr(profile, "funding_loader", lambda pair, start: [(t, 0.0001) for t in fresh if t >= start])
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    history._refresh_funding(profile, "BTC/USDT", tmp_path, None)
    assert [(level, kind) for level, kind, _ in sent][1:] == [("info", "funding_stale_cleared")]


def _switching_rates(tmp_path):
    """BTC/USDT settling every 8 hours, then every hour from 3 Oct 2025 16:00 (the venue's move at its funding cap)."""
    import json

    from sleeve_fund import funding

    eight = pd.date_range("2025-10-02 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
    hourly = pd.date_range("2025-10-03 17:00", "2025-10-04 23:00", freq="1h", tz="UTC")
    path = funding._path("BINANCE", "BTC/USDT", tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0003] for t in eight.append(hourly)]}))
    funding._cache.clear()
    return eight, hourly


def test_a_settlement_schedule_shorter_than_the_one_charged_is_a_mismatch(tmp_path):
    """Interim guard (Advisor, 6 Oct 2026) until funding is charged at each stored rate's own time (DA-11)."""
    from sleeve_fund import funding

    _switching_rates(tmp_path)
    why = funding.schedule_mismatch("BINANCE", "BTC/USDT", (0, 8, 16), tmp_path)
    assert why.startswith("funding schedule mismatch: BTC/USDT settled 1 hours apart at 2025-10-03 17:00 UTC")
    assert funding.schedule_mismatch("BINANCE", "BTC/USDT", (0, 8, 16), tmp_path, end=pd.Timestamp("2025-10-03 16:00",
                                                                                                    tz="UTC")) is None
    assert funding.schedule_mismatch("BINANCE", "BTC/USDT", tuple(range(24)), tmp_path) is None
    assert funding.schedule_mismatch("BINANCE", "BTC/USDT", (0, 8, 16), tmp_path, latest=True)


def test_a_study_with_a_schedule_mismatch_is_not_judged():
    r = _study(dict(funding_held=100, funding_baseline=0))
    r.full_period.funding_schedule = "funding schedule mismatch: BTC/USDT settled 1 hours apart"
    verdict, words = StudyResult.funding_check.fget(r)
    assert verdict == tearsheet.NOT_JUDGED and words.startswith("funding schedule mismatch")


def test_paper_refuses_to_start_a_perp_whose_latest_settlement_step_is_shorter(tmp_path, monkeypatch):
    import pytest

    from sleeve_fund import funding
    from sleeve_fund.supervisor import check_funding_schedule

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    _switching_rates(tmp_path)
    s = SimpleNamespace(strategy="buy_and_hold", instrument="BTC/USDT", venue="BINANCE",
                        params={"market": "perp", "venue": "BINANCE"})
    monkeypatch.setattr("sleeve_fund.paper.config.from_store", lambda s: SimpleNamespace(venue="BINANCE"))
    with pytest.raises(ValueError, match="funding schedule mismatch"):
        check_funding_schedule(s)
    monkeypatch.setattr("sleeve_fund.paper.config.from_store", lambda s: SimpleNamespace(venue="KRAKEN"))
    check_funding_schedule(SimpleNamespace(**{**vars(s), "params": {"market": "perp"}}))  # simulated: no venue rates
