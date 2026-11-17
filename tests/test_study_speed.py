"""STUDY-SPEED (PM 17 Nov: our own slowness, no bigger server; Advisor's ruling 17 Nov): basket_study(fast=True),
which every store basket study runs. Done when:

1. a counted basket that loses before fees, or makes fewer than 40 test-window trades, is screened out after its base
   run and its 0-fee rung: a FAIL naming why, its try counted as before, its sheet carrying the trade count, the
   0-fee and real-fee lines, the gross edge per trade and the 4-rung ladder line; its full study fails too;
2. a survivor's cost ladder is four rungs, 0, 0.02%, the venue's fee (the base run itself) and the stress rung at
   twice it (the 2x fees run itself when the venue charges one fee on every fill), its break-even fee interpolated
   between the two rungs that bracket it, or "> " the stress rung above it;
3. on the same basket, the fast and the full study agree on every pass condition, the verdict and every rung both
   ran, and the fast sheet differs from the full one only in its break-even lines."""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
from test_b38_basket_study import (
    CAPITAL,
    PARAMS,
    RESEARCH_END,
    WINDOWS,
    _asset,
    _basket,
    _register,
)
from test_portfolio_run import LOOSE

from sleeve_fund.instruments import FeeSchedule, spot_pair
from sleeve_fund.research import basket as bk
from sleeve_fund.research import portfolio_run, trials

LOSER = (_asset("BTC", 7), _asset("ETH", 8), _asset("SOL", 10))  # loses before fees on PARAMS
CROSSES = {"rsi_period": 4, "long_entry": 60.0, "long_exit": 65.0}  # breaks even below the venue's 0.2% taker


def _studied(tmp_path, basket, fast, min_trades=0):
    """The study and the fees per side of every run it made (None: the venue's own). min_trades: the screen's trade
    floor (the synthetic baskets make under 40 test-window trades, so a survivor's tests lower it)."""
    fees, real = [], portfolio_run.run_portfolio

    def spy(specs, profile=None, **k):
        f = specs[0].fees
        fees.append(None if f is None else (float(f.maker), float(f.taker)))
        return real(specs, profile, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", spy)
    mp.setattr(bk, "MIN_TRADES", min_trades)
    try:
        register = _register(tmp_path / ("fast" if fast else "full"))
        study = bk.basket_study(basket, WINDOWS, RESEARCH_END, register=register, dataset="syn", capital=CAPITAL,
                                profile=LOOSE, warm_days=40, fast=fast)
        text = bk.render(study)
    finally:
        mp.undo()
    return study, fees, register, text


def _dirs(tmp_path):
    (tmp_path / "fast").mkdir(), (tmp_path / "full").mkdir()


# ---- 1. the screen ----------------------------------------------------------------------------------------------

def test_a_basket_losing_before_fees_is_screened_out_after_two_runs_and_its_full_study_fails_too(tmp_path):
    _dirs(tmp_path)
    b = _basket(LOSER)
    fast, fast_runs, fast_reg, text = _studied(tmp_path, b, True)
    full, _, full_reg, _ = _studied(tmp_path, b, False)
    assert fast_runs == [None, (0.0, 0.0)]  # the base run at the venue's fees, then the 0-fee rung: nothing else
    assert fast.screened == "loses before fees" and bk.verdict(fast, None) == "FAIL"
    assert fast.ladder == [(0.0, fast.ladder[0][1])] and fast.ladder[0][1] <= 0
    assert fast.double_fees is None and fast.left_out is None and fast.random is None
    # the full study of the same basket: the same base run and 0-fee rung, and it fails
    assert full.base.net == fast.base.net and dict(full.ladder)[0.0] == fast.ladder[0][1]
    assert bk.verdict(full, None) == "FAIL" and not full.double_fees.passed
    # its try is counted as the full study's is
    idea = trials.legacy_idea_hash("rsi_cross")
    keys = [{(t["definition_hash"], t["dataset"], t["sharpe"]) for t in reg._counted(idea)}
            for reg in (fast_reg, full_reg)]
    assert keys[0] == keys[1] and len(keys[0]) == 1
    assert (fast.tries, fast.configuration) == (full.tries, full.configuration) == (1, 1)
    # the sheet: the verdict and why, the trade count, the 0-fee and real-fee lines, the gross edge, the ladder line
    zero = fast.ladder[0][1]
    assert "**Basket verdict:** FAIL (loses before fees)" in text
    assert f"| Net above 0 before fees (0% a side) | FAIL, net {bk._pct(zero)} |" in text
    assert f"| Net at the venue's fees | net {bk._pct(fast.base.net)} |" in text
    assert fast.trades > 0 and f"| At least 0 test-window trades | PASS, {fast.trades} |" in text
    assert f"{fast.trades} test-window trades; net {bk._pct(zero)} before fees" in text
    assert not math.isnan(fast.gross_edge) and f"gross edge {bk._pct_fine(fast.gross_edge)} per trade" in text
    assert "cost ladder: 4 rungs (17 Nov)" in text and "were not run" in text
    assert "**Tries:** 1 (one per fixed basket and settings)" in text


def test_the_gross_edge_is_the_mean_return_per_test_window_trade_of_the_run_it_is_given(monkeypatch):
    from types import SimpleNamespace as Row

    monkeypatch.setattr(bk, "test_trades", lambda m, bars, windows: [Row(ret=0.01), Row(ret=-0.004), Row(ret=0.0)])
    assert bk.gross_edge(None, {}, WINDOWS) == pytest.approx(0.002)
    monkeypatch.setattr(bk, "test_trades", lambda m, bars, windows: [])
    assert math.isnan(bk.gross_edge(None, {}, WINDOWS)) and bk._pct_fine(math.nan) == "n/a"


def test_a_basket_with_fewer_than_40_test_window_trades_is_screened_out_even_when_it_makes_money(tmp_path):
    _dirs(tmp_path)
    fast, runs, _, text = _studied(tmp_path, _basket(), True, min_trades=40)
    assert runs == [None, (0.0, 0.0)] and fast.ladder[0][1] > 0 and fast.trades < 40
    assert fast.screened == "fewer than 40 test-window trades" and bk.verdict(fast, None) == "FAIL"
    assert "**Basket verdict:** FAIL (fewer than 40 test-window trades)" in text
    assert f"| At least 40 test-window trades | FAIL, {fast.trades} |" in text
    assert f"| Net above 0 before fees (0% a side) | PASS, net {bk._pct(fast.ladder[0][1])} |" in text
    assert f"made {fast.trades} trades in the test windows, under the 40 a verdict needs" in text


def test_a_confirming_run_is_never_screened(tmp_path):
    """Not a try: its fitness lines are its point, so it runs whole whatever its 0-fee rung says."""
    _dirs(tmp_path)
    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "MIN_TRADES", 40)
    try:
        study = bk.basket_study(_basket(LOSER), WINDOWS, RESEARCH_END, register=_register(tmp_path / "fast"),
                                dataset="syn", capital=CAPITAL, profile=LOOSE, warm_days=40, fast=True, counted=False)
    finally:
        mp.undo()
    assert not study.screened and study.double_fees is not None and study.left_out is not None


# ---- 2. the four rungs ------------------------------------------------------------------------------------------

@pytest.mark.parametrize("params", [PARAMS, CROSSES], ids=["above-the-stress-rung", "between-rungs"])
def test_a_survivors_ladder_is_four_rungs_and_its_break_even_is_interpolated_or_beyond_the_stress_rung(tmp_path,
                                                                                                       params):
    _dirs(tmp_path)
    fast, runs, _, text = _studied(tmp_path, _basket(params=params), True)
    real = 0.002  # the fixture venue's taker fee per side
    assert [f for f, _ in fast.ladder] == [0.0, 0.0002, real, 2 * real]
    assert dict(fast.ladder)[real] == fast.base.net  # the base run is the venue's-fee rung, not run again
    assert runs.count((real, real)) == 0 and (0.0002, 0.0002) in runs
    assert "cost ladder: 4 rungs (17 Nov)" in text
    value, words = fast.breakeven
    nets = dict(fast.ladder)
    if nets[2 * real] > 0:
        assert (value, words) == (None, "> 0.40%") and "**Pooled break-even fee:** > 0.40% (" in text
    else:
        lo = max(f for f, n in fast.ladder if n > 0)
        hi = min(f for f, n in fast.ladder if n <= 0)
        assert lo < value < hi and words == f"{value:.3%} per side"
        assert value == pytest.approx(bk._between(lo, nets[lo], hi, nets[hi]), abs=1e-15)
    assert "interpolated between real runs on the cost ladder: 4 rungs (17 Nov)" in text


def test_the_stress_rung_is_the_2x_fees_run_when_the_venue_charges_one_fee_on_every_fill(tmp_path):
    from nautilus_trader.model import Venue

    flat = FeeSchedule(bk._dec(0.002), bk._dec(0.002))
    assets = tuple(bk.BasketAsset(spot_pair(a.instrument.id.symbol.value.split("/")[0], "USD", flat,
                                            Venue("KRAKEN")), a.prices, a.exec_prices)
                   for a in _basket().assets)
    _dirs(tmp_path)
    fast, runs, _, _ = _studied(tmp_path, _basket(assets, CROSSES), True)
    full, _, _, _ = _studied(tmp_path, _basket(assets, CROSSES), False)
    assert runs.count((0.002, 0.002)) == 0 and runs.count((0.004, 0.004)) == 1  # the 2x run, once, as the stress rung
    assert dict(fast.ladder)[0.004] == fast.double_fees.net == dict(full.ladder)[0.004]
    assert dict(fast.ladder)[0.002] == fast.base.net == dict(full.ladder)[0.002]
    assert 0.004 in dict(fast.asset_ladder["BTC"])  # each asset's line at that fee is the 2x run's too


# ---- 3. parity with the full study ------------------------------------------------------------------------------

@pytest.mark.parametrize("params", [PARAMS, CROSSES], ids=["above-the-stress-rung", "between-rungs"])
def test_a_survivor_is_judged_as_the_full_study_judges_it_and_its_sheet_differs_only_in_break_even(tmp_path,
                                                                                                    params):
    _dirs(tmp_path)
    b = _basket(params=params)
    fast, fast_runs, _, fast_text = _studied(tmp_path, b, True)
    full, full_runs, _, full_text = _studied(tmp_path, b, False)
    assert not fast.screened and fast.ladder[0][1] > 0
    for got, want in ((fast.base, full.base), (fast.double_fees, full.double_fees), (fast.left_out.run,
                                                                                     full.left_out.run)):
        assert (got.net, got.passed) == (want.net, want.passed) and got.returns.equals(want.returns)
    assert fast.left_out.asset == full.left_out.asset and repr(fast.random) == repr(full.random)
    assert repr(fast.left_out.random) == repr(full.left_out.random)
    assert bk.verdict(fast, None) == bk.verdict(full, None)
    assert fast.gate == full.gate and fast.intents == full.intents
    full_nets = dict(full.ladder)
    for f, net in fast.ladder:  # every rung both ran agrees to the cent (the venue's fee: the base run's own net)
        if f in full_nets:
            assert net == full_nets[f]
    assert len(fast_runs) < len(full_runs)
    # the sheets: the same but for the pooled break-even line and the per-asset table's break-even cells
    a, b_ = fast_text.splitlines(), full_text.splitlines()
    assert len(a) == len(b_)
    for x, y in zip(a, b_):
        if x == y:
            continue
        if x.startswith("**Pooled break-even fee:**"):
            assert y.startswith("**Pooled break-even fee:**") and "cost ladder: 4 rungs (17 Nov)" in x
            continue
        assert x.startswith("| ") and x.rsplit("|", 2)[0] == y.rsplit("|", 2)[0], (x, y)  # only the last cell


# ---- 4. a store study runs fast ---------------------------------------------------------------------------------

def test_a_store_basket_study_is_screened_and_writes_its_fail_sheet(tmp_path):
    """The study lists' path (research.basket_run, as rescore runs it): a 4-hour list with too few trades on 1-minute
    fills is screened out after its two runs, its sheet and its trade files written, its try counted."""
    from test_basket_4h import _History, _req

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.store import Store

    runs, real = [], portfolio_run.run_portfolio
    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", lambda specs, *a, **k: runs.append(specs) or real(specs, *a, **k))
    try:
        store = Store(f"sqlite:///{tmp_path}/t.db")
        out = run_mod.run_store_study(_req(), store=store, out_dir=tmp_path / "sheets", history=_History())
    finally:
        mp.undo()
    assert len(runs) == 2 and runs[0][0].fees is None and float(runs[1][0].fees.taker) == 0.0
    text = out.read_text(encoding="utf-8")
    assert "**Basket verdict:** FAIL (fewer than 40 test-window trades)" in text
    assert out.with_name(f"{out.stem}-basket-trades.csv").exists() and out.with_name(f"{out.stem}-pooled.csv").exists()
    rows = [t for t in trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules"))
            if str(t["definition_name"]).startswith("basket:")]
    assert len(rows) == 1


# ---- 5. CR on #365: M1, F1, F2 ----------------------------------------------------------------------------------

def test_a_screened_store_study_still_refuses_trips_that_disagree_with_the_book(tmp_path):
    """CR M1: an own-sized run (guards off, as the edge verdict's) refuses, with no sheet and no try, when the pooled
    trips and the book disagree; a study screened out before its draws checks the same thing first."""
    from test_basket_4h import _History, _req

    from sleeve_fund.research import run as run_mod
    from sleeve_fund.store import Store

    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "pooled_legs", lambda *a, **k: ([], (0.05, 0.0), ""))
    try:
        store = Store(f"sqlite:///{tmp_path}/t.db")
        with pytest.raises(ValueError, match="the trades and the book disagree, so no sheet and no try"):
            run_mod.run_store_study(replace(_req(), fund_guards=False), store=store, out_dir=tmp_path / "sheets",
                                    history=_History())
    finally:
        mp.undo()
    assert trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules")) == []
    assert not (tmp_path / "sheets").exists() or not list((tmp_path / "sheets").glob("*.md"))


def test_a_basket_that_never_trades_reads_as_too_few_trades_and_a_zero_net_is_a_loss_before_fees(tmp_path, monkeypatch):
    """CR F2: the trade count is judged first; at exactly 0 before fees, nothing can pass after them."""
    _dirs(tmp_path)
    monkeypatch.setattr(bk, "test_trades", lambda m, bars, windows: [])
    none, _, _, _ = _studied(tmp_path, _basket(), True, min_trades=40)
    assert none.screened == "fewer than 40 test-window trades"
    (tmp_path / "zero").mkdir()
    monkeypatch.setattr(bk, "test_trades", lambda m, bars, windows: [object()] * 40)
    monkeypatch.setattr(bk, "gross_edge", lambda zero, bars, windows: 0.0)
    real_run, real_measure, fees = bk._run, bk._measure, []

    def run(basket, assets, first, last, f, *a, **k):
        fees.append(f)
        return real_run(basket, assets, first, last, f, *a, **k)

    def measure(*a, **k):  # the 0-fee rung's net exactly 0: a flat book before fees
        m = real_measure(*a, **k)
        if fees[-1] is not None and float(fees[-1].taker) == 0.0:
            m.net = 0.0
        return m
    monkeypatch.setattr(bk, "_run", run)
    monkeypatch.setattr(bk, "_measure", measure)
    study = bk.basket_study(_basket(), WINDOWS, RESEARCH_END, register=_register(tmp_path / "zero"), dataset="syn",
                            capital=CAPITAL, profile=LOOSE, warm_days=40, fast=True)
    assert study.ladder == [(0.0, 0.0)] and study.screened == "loses before fees"


def test_the_real_fee_rung_is_its_own_flat_run_when_the_base_run_made_a_maker_fill(tmp_path, monkeypatch):
    """CR F1: on a venue whose maker fee differs, a base run with any passive fill is not one flat fee."""
    _dirs(tmp_path)
    monkeypatch.setattr(bk, "_maker_filled", lambda m: True)
    fast, runs, _, _ = _studied(tmp_path, _basket(params=CROSSES), True)
    assert runs.count((0.002, 0.002)) == 1 and runs[0] is None
    monkeypatch.setattr(bk, "_maker_filled", lambda m: False)
    (tmp_path / "again" / "fast").mkdir(parents=True)
    again, runs2, _, _ = _studied(tmp_path / "again", _basket(params=CROSSES), True)
    assert runs2.count((0.002, 0.002)) == 0 and dict(again.ladder)[0.002] == again.base.net
