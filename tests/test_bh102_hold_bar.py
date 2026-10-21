"""BH-102 (Advisor, 21 Oct), the study's side. Rule 3: a long-only perp study must beat the HIGHER of the 1x perp
hold and the spot hold, by Sharpe over the out-of-sample days both cover, chosen once per sheet, and every reader of
the benchmark takes that one. On a perp-only venue the spot hold is the price itself held, paying one entry and one
exit at the base-tier spot taker fee plus half the spread. A liquidated perp hold is said on the sheet. Expected
values are worked out here from the fixtures, not read from the code under test."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.research import study as study_mod
from sleeve_fund.research.study import PRICE_HOLD, liquidated_at, price_hold_equity
from test_binance import offline_binance  # noqa: F401 - the fixture

HALF_SPREAD = 0.0005
COST = 0.001 + HALF_SPREAD  # the base-tier spot taker fee both venues publish, plus half the spread


def test_the_price_only_hold_pays_one_entry_and_one_exit():
    closes = pd.Series([100.0, 120.0, 90.0, 110.0], index=pd.date_range("2026-01-01", periods=4, freq="1D", tz="UTC"))
    eq = price_hold_equity(closes, 0.002, 1000.0)
    assert eq.tolist() == pytest.approx([1000.0, 1197.6, 898.2, 1097.8 * 0.998])  # the capital before the entry first


def test_a_liquidation_is_read_from_the_order_the_engine_decided_as_one():
    ts = pd.date_range("2026-01-01", periods=2, freq="1D", tz="UTC")
    fills = pd.DataFrame({"side": ["OrderSide.BUY", "OrderSide.SELL"], "filled_qty": [1.0, 1.0],
                          "avg_px": [100.0, 40.0], "commissions": [0.1, 0.1], "ts_last": ts},
                         index=["O-1", "O-2"])
    run = SimpleNamespace(fills=fills, decisions={"O-1": {"intent": "entry"}, "O-2": {"intent": "liquidation"}})
    assert liquidated_at(run) == [ts[1]]
    run.decisions["O-2"] = {"intent": "stop_loss"}
    assert liquidated_at(run) == []


def _study(tmp_path, binance, rate, **kw):
    """A trend study on the venue's perpetual, 800 daily bars, the venue's funding stored at `rate` every 8 hours."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC

    prices = synthetic_ohlcv(days=800, seed=4)
    times = pd.date_range(prices.index[0] - pd.Timedelta(days=2), prices.index[-1] + pd.Timedelta(days=1), freq="8h")
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), rate] for t in times]}))
    r = run_study(SPEC, prices, binance.instrument("BTC", "USDT"), dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"),
                  synthetic=True, train_days=300, test_days=200, half_spread=HALF_SPREAD,
                  default_params={"fast": 20, "slow": 50}, **{"holdout_days": 0, **kw})
    return r, prices


@pytest.fixture(scope="module")
def drained(tmp_path_factory):
    """Funding at 0.3% every 8 hours: the perp hold pays almost 1% a day, so the price held beats it."""
    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("bh102")
    try:
        from sleeve_fund.venues import binance_contract, venue
        from test_binance import INFO

        b = venue("binance")
        mp.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
        mp.setattr(funding, "DEFAULT_ROOT", d / "funding")
        yield _study(d, b, 0.003)
    finally:
        mp.undo()


def test_when_the_price_held_beats_the_drained_perp_hold_it_is_the_bar_on_every_line(drained):
    r, prices = drained
    assert r.bar_hold == PRICE_HOLD
    assert r.bar_note.startswith(f"bar: the {PRICE_HOLD}, the higher Sharpe over the ")
    # the full-period benchmark is the price held, its entry and exit paid
    closes = prices["close"].astype(float)
    want = 10_000.0 * (1 - COST) * closes / closes.iloc[0]
    want.iloc[0] = 10_000.0
    want.iloc[-1] *= 1 - COST
    pd.testing.assert_series_equal(r.full_period_benchmark.equity, want, check_names=False, check_freq=False)
    # each test window's benchmark is the price held over that window: it compounds to (1 - cost)^2 x the price's move
    # from the close before the window to its last close, and the fold's own benchmark row is that same series
    assert len(r.folds) == 2
    for f in r.folds:
        days = r.oos_benchmark_returns[(r.oos_benchmark_returns.index > f.test_end - pd.Timedelta(days=200))
                                       & (r.oos_benchmark_returns.index <= f.test_end)]
        first = days.index[0] - pd.Timedelta(days=1)
        assert math.prod(1 + v for v in days) == pytest.approx((1 - COST) ** 2 * closes[f.test_end] / closes[first])
        assert f.benchmark_test["sharpe"] == pytest.approx(study_mod.summary(days)["sharpe"])
    # the strategy is compared on the days the bar covers, and nothing says the perp hold judged
    assert r.compare_returns.index.equals(r.oos_benchmark_returns.index)
    assert not r.hold_insufficient and r.spot_hold_returns is None  # no spot market here
    assert r.perp_hold_returns is not None and len(r.perp_hold_returns) > 300  # shown beside the bar
    assert "set the bar, so it is the benchmark on every line" in r.hold_note


def test_the_sheet_names_the_bar_with_both_sharpes(drained, tmp_path):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.tearsheet import render

    r, _ = drained
    sheet = render(r, IdeaLedger(tmp_path / "l.jsonl"))
    line = next(x for x in sheet.splitlines() if x.startswith("G1 bar: "))
    assert "(perp hold " in line and f", {PRICE_HOLD} " in line


def test_when_the_perp_hold_is_paid_funding_it_stays_the_bar(tmp_path, offline_binance):  # noqa: F811
    r, _ = _study(tmp_path, offline_binance, -0.003)  # longs are paid nearly 1% a day
    assert r.bar_hold == "perp hold" and r.bar_note.startswith("bar: the perp hold, ")
    assert "1x long perpetual paying its fees" in r.hold_note and "set the bar" not in r.hold_note


def test_a_liquidated_perp_hold_is_said_on_the_sheet(tmp_path, offline_binance, monkeypatch):  # noqa: F811
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.tearsheet import render

    at = pd.Timestamp("2019-05-16", tz="UTC")
    monkeypatch.setattr(study_mod, "liquidated_at", lambda run: [at] if run.fills is not None and len(run.fills) == 1
                        else [])  # the hold's one fill: the engine's liquidation itself is PE4's side, pinned there
    r, _ = _study(tmp_path, offline_binance, 0.0001)
    assert "16 May 2019 (the research period)" in r.hold_liquidated
    sheet = render(r, IdeaLedger(tmp_path / "l.jsonl"))
    assert "The perp hold was liquidated on 16 May 2019 (the research period)" in sheet


def test_the_holdout_is_judged_against_the_same_bar(tmp_path, monkeypatch):
    """Rule 3: every reader takes the one hold, the holdout too. Here the price held over the last 100 days, worked out
    from the closes: its entry at the close before the holdout's first day, its exit on the last."""
    from sleeve_fund.venues import binance_contract, venue
    from test_binance import INFO

    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    # a perpetual's holdout needs 1-minute bars (its liquidation rests); that rule is pinned elsewhere, and this cell
    # is only about which hold the holdout is compared with
    monkeypatch.setattr(study_mod.StudyResult, "not_judged", property(lambda self: ""))
    r, prices = _study(tmp_path, b, 0.003, holdout_days=100, use_holdout=True)
    assert r.bar_hold == PRICE_HOLD and r.holdout is not None, (r.holdout_withheld, r.not_judged)
    closes = prices["close"].astype(float).iloc[-101:]
    eq = closes / closes.iloc[0] * (1 - COST)
    eq.iloc[0] = 1.0  # the capital before the entry, so the first holdout day carries the entry fee
    eq.iloc[-1] *= 1 - COST
    want = eq.pct_change().iloc[1:]
    assert r.holdout_benchmark["sharpe"] == pytest.approx(study_mod.summary(want)["sharpe"])
