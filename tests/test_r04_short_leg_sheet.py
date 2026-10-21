"""R-04 (HoE, 21 Oct 2026): a short-leg study's tear sheet counts its short trips. The full-period table read every
trip long (Closed trades 0) and only positive exposure as in the market (Time in market 0%) while the cost ladder
counted the round trips; the perpetual caveat said "long only" and the asset class "spot" on every sheet. Real
end-to-end study on the perpetual venue (contract and funding offline, as test_binance)."""
from __future__ import annotations

import dataclasses
import json
import re

import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.research import tearsheet
from sleeve_fund.research.ledger import IdeaLedger
from sleeve_fund.research.study import run_study
from sleeve_fund.strategies.definitions import to_params
from sleeve_fund.strategies.rules import SPEC
from sleeve_fund.venues import binance_contract, venue
from test_binance import INFO

BLOCKS = {"fast": {"kind": "ema", "period": 10}, "slow": {"kind": "ema", "period": 30}}
SHORT = {"version": 1, "reason": "R-04 probe: short while the fast average is under the slow.", "blocks": BLOCKS,
         "short": {"entry": {"left": "fast", "op": "crosses_below", "right": "slow"},
                   "exit": {"left": "fast", "op": ">", "right": "slow"}}}
LONG = {"version": 1, "reason": "R-04 probe: long while the fast average is over the slow.", "blocks": BLOCKS,
        "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
                 "exit": {"left": "fast", "op": "<", "right": "slow"}}}


@pytest.fixture
def perp_venue(tmp_path, monkeypatch):
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    bars = synthetic_ohlcv(days=1300, seed=7, vol=0.03, start_price=30_000)
    times = pd.date_range(bars.index[0].floor("D") - pd.Timedelta("1D"), bars.index[-1] + pd.Timedelta("1D"), freq="8h")
    path = funding._path("BINANCE", "BTC/USDT")
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.0001] for t in times]}))
    return b.instrument("BTC", "USDT"), bars


def _study(tmp_path, perp_venue, defn, **kw):
    inst, bars = perp_venue
    params = {**to_params(defn), "market": "perp", "allow_short": "short" in defn}
    spec = dataclasses.replace(SPEC, param_grid={k: [v] for k, v in params.items()})
    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(spec, bars, inst, dataset="r04", ledger=ledger, default_params=params, synthetic=True,
                  holdout_days=0, train_days=730, test_days=365, **kw)
    return r, ledger, tearsheet.render(r, ledger)


def _line(sheet: str, start: str) -> str:
    return next(line for line in sheet.splitlines() if line.startswith(start))


def test_a_short_leg_sheet_counts_its_short_trips_and_time_in_the_market_as_the_cost_ladder_does(tmp_path, perp_venue):
    r, _, sheet = _study(tmp_path, perp_venue, SHORT)
    ladder = r.cost_ladder[0].round_trips
    assert ladder > 0, "set-up: the short leg trades"
    assert r.trade_stats["trades"] == ladder
    row = sheet.splitlines()[sheet.splitlines().index(_line(sheet, "| Closed trades |")) + 2]
    assert row.startswith(f"| {ladder} (")
    held = float((r.full_period.exposure.abs() > 0.001).mean())
    assert held > 0
    assert _line(sheet, "- Time in the market:") == f"- Time in the market: {held:.0%} of bars hold a position"


def test_the_caveat_and_asset_class_are_the_studys_own_market_and_sides(tmp_path, perp_venue):
    _, _, sheet = _study(tmp_path, perp_venue, SHORT)
    assert "**Asset class:** perpetual" in sheet
    assert "traded the perpetual, short only," in sheet and "long only" not in sheet
    _, _, long_sheet = _study(tmp_path, perp_venue, LONG)
    assert "**Asset class:** perpetual" in long_sheet
    assert "traded the perpetual, long only," in long_sheet


def test_a_short_legs_cost_ladder_benchmark_is_the_oos_benchmark_taken_short(tmp_path, perp_venue, monkeypatch):
    """R-08 (Advisor): each rung's timing and random-entry columns are the out-of-sample benchmark at the rung's
    cost, on the same short trips. A rung charged the study's own cost (the rung's fee plus the ladder's slippage
    equal to its taker) must read exactly the OOS benchmark, which the random-side test confirms is the short one."""
    from sleeve_fund.research import study

    taker = 0.00055  # the perpetual venue's taker in the fixture
    monkeypatch.setattr(study, "COST_LADDER", (0.0, taker - study.ladder_slippage("BTC/USDT")))
    r, _, _ = _study(tmp_path, perp_venue, SHORT)
    assert float(r.fee_basis.taker) == taker, "set-up: the rung is charged the study's own cost"
    oos, rung = r.random_entry, r.cost_ladder[1]
    assert oos.trades > 0 and r.random_side is not None
    assert r.random_side.strategy_return == pytest.approx(oos.strategy_return), "the OOS benchmark is the short reading"
    assert rung.oos_timing_return == pytest.approx(oos.strategy_return)
    assert rung.random_return == pytest.approx(oos.median_random_return)


@pytest.mark.parametrize("defn", [LONG, SHORT], ids=["long", "short"])
def test_the_trading_tables_net_pnl_is_the_funded_equitys_to_the_cent(tmp_path, perp_venue, defn):
    """R-16: the trading table's net P&L books each trip's funding, as the equity does: the closed trips' P&L is
    the equity's gain from the start to the bar the last of them closed, to the cent."""
    from sleeve_fund.research.metrics import fills_to_rows, trades

    r, _, sheet = _study(tmp_path, perp_venue, defn)
    run = r.full_period
    assert run.funding, "set-up: the perpetual settles funding"
    last = trades(fills_to_rows(run.fills), run.shorts)[-1]["closed"]
    eq = run.equity
    gain = float(eq[eq.index >= pd.Timestamp(last)].iloc[0] - eq.iloc[0])
    assert round(r.trade_stats["pnl"], 2) == round(gain, 2)
    assert _line(sheet, "## Trading and costs") == (f"## Trading and costs (full research period, "
                                                    f"{r.research_start:%d %b %Y} to {r.research_end:%d %b %Y}, default params)")


def test_a_spot_study_keeps_its_asset_class(tmp_path, instrument):
    from sleeve_fund.strategies.trend_filter import SPEC as TREND

    ledger = IdeaLedger(tmp_path / "l.jsonl")
    r = run_study(TREND, synthetic_ohlcv(days=1300, seed=3), instrument, dataset="syn", ledger=ledger, synthetic=True,
                  holdout_days=0, train_days=730, test_days=365)
    sheet = tearsheet.render(r, ledger)
    assert re.search(r"\*\*Asset class:\*\* spot ", sheet) and "traded the perpetual" not in sheet


@pytest.mark.parametrize("sides, allow_short, words", [
    ({1}, False, "long only"), ({-1}, True, "short only"), ({1, -1}, True, "long and short"),
    (set(), True, "long and short"), (set(), False, "long only")])
def test_the_caveat_names_the_sides_the_run_traded_or_may_take(monkeypatch, sides, allow_short, words):
    """CR F2: a long and short idea's caveat, and a run with no trades, which names the sides it may take."""
    import types

    from sleeve_fund.research import study

    monkeypatch.setattr(study, "trades", lambda rows, shorts: [{"side": s} for s in sorted(sides)])
    monkeypatch.setattr(study, "fills_to_rows", lambda fills: [])
    run = types.SimpleNamespace(fills=None, shorts=True)
    assert study._sides_words(run, {"allow_short": allow_short}) == words


def test_a_cost_ladder_rung_draws_no_random_side_test(monkeypatch):
    """CR F1: the rung reads random entry only, so its random-side test, which it would throw away, is never drawn."""
    from sleeve_fund.research import study

    calls = []
    monkeypatch.setattr(study, "random_side", lambda *a, **k: calls.append(1))
    monkeypatch.setattr(study, "random_entry", lambda *a, **k: "entry")
    import types

    days = pd.date_range("2021-01-01", periods=8, freq="D", tz="UTC")
    prices = pd.DataFrame({"close": [1.0 + i for i in range(8)]}, index=days)
    # A fold with a short trip in it: its side (-1) must not stand in for the rung's flag (CR M1).
    fold = types.SimpleNamespace(test_end=days[-1], trips=[(days[1], days[4], -1)], chosen={}, window_bars=8,
                                 in_market_bars=3)
    assert study._benchmarks(prices, [fold], 8, 0.001, 0.0, True, random_side_test=False) == ("entry", None)
    assert calls == []
    assert study._benchmarks(prices, [fold], 8, 0.001, 0.0, True)[1] is None and calls == [1]
