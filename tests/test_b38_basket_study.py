"""B-38 (Advisor spec, 20 Oct 2026): a multi-asset basket's study report. One fund run of the whole basket, pooled
daily net P&L over the out-of-sample days; PASS needs it to beat random at the 95th percentile, net above 0 at twice
the fees, hold with its best asset left out, and the locked holdout (a separate final step) net above 0. Per-asset
lines are diagnostic only.

Expected values are worked by hand or from the runs the study made, captured as they ran; never read back from its
own figures."""

from __future__ import annotations

import math
from datetime import date, timedelta
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest
from nautilus_trader.model import Venue

from sleeve_fund.instruments import FeeSchedule, spot_pair
from sleeve_fund.research import basket as bk
from sleeve_fund.research import portfolio_run
from sleeve_fund.research.holdout import HoldoutLocks
from sleeve_fund.research import trials
from sleeve_fund.research.candidates import configurations
from sleeve_fund.research.study import COST_LADDER, _between
from sleeve_fund.risk import PortfolioProfile
from sleeve_fund.store import Store
from test_portfolio_run import FEES, LOOSE, _daily, _hourly

DAYS = 300  # 2018-01-01 to 2018-10-27: warm-up, two test windows, the holdout
FIRST = date(2018, 1, 1)
WINDOWS = ((date(2018, 3, 2), date(2018, 5, 30)), (date(2018, 5, 31), date(2018, 8, 28)))
RESEARCH_END = WINDOWS[-1][1]
HOLDOUT = (date(2018, 8, 29), date(2018, 10, 20))
CAPITAL = 30_000.0


def _asset(base: str, seed: int, vol: float = 0.007) -> bk.BasketAsset:
    h = _hourly(DAYS, seed, vol=vol)
    return bk.BasketAsset(spot_pair(base, "USD", FEES, Venue("KRAKEN")), _daily(h), h)


ASSETS = (_asset("BTC", 2), _asset("ETH", 5), _asset("SOL", 9))


PARAMS = {"rsi_period": 4, "long_entry": 40.0, "long_exit": 55.0}  # trades often enough to draw from


def _basket(assets=ASSETS, params=None, name="trend-x") -> bk.Basket:
    return bk.Basket(name, "rsi_cross", dict(params or PARAMS), tuple(assets), "2026-10-20")


def _register(tmp_path) -> trials.TrialsRegister:
    return trials.TrialsRegister(Store(f"sqlite:///{tmp_path}/t.db"))


def _study(tmp_path, basket=None, profile=LOOSE, register=None, **kw):
    return bk.basket_study(basket or _basket(), WINDOWS, RESEARCH_END, register=register or _register(tmp_path),
                           dataset="syn", capital=CAPITAL, profile=profile, warm_days=40, **kw)


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    """One study with every fund run it made captured: its legs' specs (fees, bars) and its result."""
    runs, real = [], portfolio_run.run_portfolio

    def spy(specs, profile=None, **k):
        res = real(specs, profile, **k)
        runs.append({"specs": specs, "res": res})
        return res

    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", spy)
    register = _register(tmp_path_factory.mktemp("b38"))
    try:
        r = _study(None, register=register)
    finally:
        mp.undo()
    return r, runs, register


def _day(ts: pd.Timestamp) -> date:
    return (ts - pd.Timedelta(days=1)).date()


def test_b38a_pooled_daily_return_is_the_summed_net_pnl_over_the_funds_equity_by_hand():
    """Two assets, capital 1,000, test days 2, 3 and 5 (day 4 between windows): day 2 +10 on 1,000, day 3 +10 on
    1,010, day 5 +10 on 1,070, the book's close on day 4 after it made 50 there (CR B-38 b: the gap's P&L counts in
    the denominator, as the random check's)."""
    idx = pd.date_range("2024-01-02", periods=4, tz="UTC")
    book = pd.Series([1_010.0, 1_020.0, 1_070.0, 1_080.0], index=idx)  # the fund book's daily closes, days 2 to 5
    pnl = pd.DataFrame({"A": [10.0, -5.0, 20.0], "B": [0.0, 15.0, -10.0]}, index=idx[[0, 1, 3]])
    got = bk.pooled_returns(pnl, book, 1_000.0)
    assert list(got) == pytest.approx([10 / 1_000, 10 / 1_010, 10 / 1_070], abs=1e-15)


def test_b38b_the_pooled_series_is_the_out_of_sample_days_only_from_one_fund_run(ran):
    r, runs, _ = ran
    base = runs[0]
    assert len(base["specs"]) == 3 and r.base.returns.index.equals(r.base.pnl.index)
    days = [_day(t) for t in r.base.returns.index]
    assert days[0] == WINDOWS[0][0] and days[-1] == RESEARCH_END
    assert len(days) == (RESEARCH_END - WINDOWS[0][0]).days + 1
    # The legs' own equity, day by day, is the per-asset P&L (worked here from the run, not the study's figures)
    for spec, col in zip(base["specs"], r.base.pnl.columns):
        eq = base["res"].legs[spec.name].equity
        closes = eq.resample("1D", closed="right", label="right").last().dropna()
        closes = closes[closes.index <= pd.Timestamp(RESEARCH_END + timedelta(days=1), tz="UTC")]
        want = closes.diff().fillna(closes.iloc[0] - spec.capital)
        assert r.base.pnl[col].to_numpy() == pytest.approx(want.reindex(r.base.pnl.index).to_numpy(), abs=1e-9)
    assert r.base.net == pytest.approx(float((1 + r.base.returns).prod() - 1), abs=1e-12)


def test_b38c_no_run_of_the_study_reads_a_bar_after_the_research_end(ran):
    _, runs, _ = ran
    end = pd.Timestamp(RESEARCH_END + timedelta(days=1), tz="UTC")
    for run in runs:
        for s in run["specs"]:
            assert s.prices.index.max() <= end and s.exec_prices.index.max() <= end
            assert s.warmup_prices.index.max() <= s.prices.index.min() - pd.Timedelta(days=1)


def test_b38d_double_fees_is_a_real_re_run_at_exactly_twice_every_legs_fees(ran):
    r, runs, _ = ran
    twice = [x for x in runs if all(s.fees is not None and s.fees.maker == 2 * FEES.maker
                                    and s.fees.taker == 2 * FEES.taker for s in x["specs"])]
    assert len(twice) == 1 and len(twice[0]["specs"]) == 3
    base = runs[0]
    differ = [not base["res"].legs[s.name].fills["filled_qty"].reset_index(drop=True).equals(
        twice[0]["res"].legs[s.name].fills["filled_qty"].reset_index(drop=True)) for s in base["specs"]]
    assert any(differ)  # the trades themselves moved: never the fees subtracted
    assert r.double_fees.net < r.base.net and r.double_fees.passed is (r.double_fees.net > 0)
    assert r.double_fees.returns.index.equals(r.base.returns.index)


def test_b38e_best_asset_left_out_is_the_top_contributor_re_run_without_it(ran):
    r, runs, _ = ran
    totals = r.base.pnl.sum()
    top = totals.idxmax()
    assert r.left_out.asset == top
    out = [x for x in runs if len(x["specs"]) == 2]
    assert len(out) == 1
    assert top not in [bk.underlying(s.instrument) for s in out[0]["specs"]]
    assert all(s.fees is None for s in out[0]["specs"])  # base fees, same settings
    # A real re-run: the assets left in trade differently (the fund sizes them off a different book)
    base_fills = {bk.underlying(s.instrument): runs[0]["res"].legs[s.name].fills for s in runs[0]["specs"]}
    kept = {bk.underlying(s.instrument): out[0]["res"].legs[s.name].fills for s in out[0]["specs"]}
    assert any(len(base_fills[a]) != len(kept[a]) or not np.array_equal(base_fills[a]["filled_qty"].to_numpy(),
                                                                        kept[a]["filled_qty"].to_numpy())
               for a in kept)
    assert r.left_out.passed is (r.left_out.net > 0)
    assert r.left_out.share == pytest.approx(totals[top] / totals.sum())
    assert r.left_out.concentrated is (r.left_out.share > 0.5)


@pytest.mark.parametrize("pnl, top, share, flag", [({"A": 60.0, "B": 30.0, "C": 10.0}, "A", 0.6, True),
                                                   ({"A": 50.0, "B": 30.0, "C": 20.0}, "A", 0.5, False)])
def test_b38f_concentrated_when_the_top_asset_is_over_half_the_pooled_net(pnl, top, share, flag):
    assert bk.top_contributor(pd.Series(pnl)) == (top, pytest.approx(share), flag)


def test_b38g_break_even_interpolates_log_growth_between_bracketing_rungs_and_never_extrapolates():
    rungs = [(0.0, 0.10), (0.0005, 0.04), (0.001, -0.02), (0.002, -0.09)]
    value, words = bk.ladder_breakeven(rungs)
    assert value == pytest.approx(_between(0.0005, 0.04, 0.001, -0.02))
    assert value == pytest.approx(0.0005 + 0.0005 * math.log1p(0.04) / (math.log1p(0.04) - math.log1p(-0.02)))
    assert bk.ladder_breakeven([(0.0, -0.01), (0.001, -0.05)]) == (None, "<0")
    assert bk.ladder_breakeven([(0.0, 0.2), (0.008, 0.01)]) == (None, "> top rung")


def test_b38h_the_ladder_is_real_runs_one_fee_per_side_on_every_leg(ran):
    r, runs, _ = ran
    flat = [x for x in runs if all(s.fees is not None and s.fees.maker == s.fees.taker for s in x["specs"])]
    assert sorted({float(x["specs"][0].fees.maker) for x in flat}) == sorted(COST_LADDER)
    assert all(len(x["specs"]) == 3 for x in flat)
    assert [f for f, _ in r.ladder] == list(COST_LADDER)
    value, words = r.breakeven
    assert (value, words) == bk.ladder_breakeven(r.ladder)
    assert set(r.asset_breakeven) == {"BTC", "ETH", "SOL"}


def test_b38i_random_draws_keep_each_assets_trade_count_sides_and_holds_and_move_only_entry_times():
    rng = np.random.default_rng(0)
    trades = [bk.TradeAt(entry=5, exit=9, side=1, notional=1_000.0), bk.TradeAt(entry=20, exit=22, side=-1,
                                                                                   notional=500.0)]
    drawn = bk.draw_trades(rng, trades, [(0, 40)])
    assert len(drawn) == 2
    assert sorted((t.exit - t.entry, t.side, t.notional) for t in drawn) == [(2, -1, 500.0), (4, 1, 1_000.0)]
    assert all(0 <= t.entry and t.exit <= 40 for t in drawn)
    entries = {tuple(sorted(t.entry for t in bk.draw_trades(rng, trades, [(0, 40)]))) for _ in range(50)}
    assert len(entries) > 25  # the entry times move, draw to draw


def test_b38j_trade_pnl_by_hand():
    """Long 1,000 of notional at 100 (10 units), held two days to 110 then 105, 0.1% per side: day 1 -1.0 (entry
    cost), day 2 +100, day 3 -50 - 1.05 (exit cost on 10 x 105)."""
    closes = np.array([100.0, 110.0, 105.0])
    got = bk.trade_pnl(closes, [bk.TradeAt(0, 2, 1, 1_000.0)], 0.001)
    assert list(got) == pytest.approx([-1.0, 100.0, -51.05])


def test_b38k_random_benchmark_is_judged_only_when_the_gate_touched_under_5_percent(ran):
    r, _, _ = ran
    assert r.random.draws == 1_000 and r.gate_touched < 0.05 and r.random.judged and r.random.seed == 0
    assert r.random.beats is (r.random.sharpe > r.random.bar)  # strictly above the draws' 95th percentile
    cut, asked = r.gate
    assert asked > 0 and r.gate_touched == cut / asked
    words = bk.random_words(bk.RandomBeat(judged=False, percentile=float("nan"), beats=False, draws=1_000,
                                          gate_touched=0.07))
    assert words == ("not judged: the compliance tests trimmed 7.0% of entry notional (5% or more), so the random "
                     "benchmark needs gated re-runs")


def test_b38l_a_tight_gate_over_5_percent_is_not_judged_and_the_basket_cannot_pass(tmp_path):
    tight = PortfolioProfile(gross=0.4, net_instrument=0.2, margin=100, open_risk=100, drawdown=0.99, daily_loss=0.99)
    r = _study(tmp_path, profile=tight, ladder=False)
    assert r.gate_touched >= 0.05
    assert not r.random.judged and r.verdict in ("NOT JUDGED", "FAIL")
    assert "not judged: the compliance tests trimmed" in bk.render(r)


def test_b38m_a_basket_edit_is_a_new_try_counted_in_its_models_family_budget(tmp_path):
    register = _register(tmp_path)
    idea = trials.legacy_idea_hash("rsi_cross")
    # the model's own single-asset setting, tried before: configuration 1 of its family
    register.record(definition_hash=trials.legacy_definition_hash("rsi_cross", PARAMS), idea_hash=idea,
                    name="rsi_cross", family="mean_reversion", settings=PARAMS, dataset="syn", stage="in_sample",
                    source="study", sharpe=0.5)
    when = dict(data_start=pd.Timestamp("2018-01-21", tz="UTC"), data_end=pd.Timestamp("2018-08-29", tz="UTC"))
    b = _basket()
    assert bk.record_try(register, b, "syn", **when) == (1, 2)  # a basket try: the family's second configuration
    assert len(configurations(register, idea)) == 2
    assert bk.record_try(register, b, "syn", **when) == (1, 2)  # the same basket and settings again: the same try
    assert bk.record_try(register, _basket(ASSETS[:2]), "syn", **when) == (2, 3)  # an asset dropped: a new basket
    assert bk.record_try(register, _basket(ASSETS[::-1]), "syn", **when) == (3, 4)  # reordered: legs fill differently
    assert bk.record_try(register, _basket(params={**PARAMS, "rsi_period": 10}), "syn", **when) == (4, 5)
    row = next(t for t in register._counted(idea) if t["definition_hash"] == bk.basket_key(b))
    import json

    assert json.loads(row["settings"])["parent"] == trials.legacy_definition_hash("rsi_cross", PARAMS)


def test_b38n_the_study_counts_one_try_and_its_left_out_run_none(ran):
    r, _, register = ran
    assert (r.tries, r.configuration) == (1, 1)
    assert configurations(register, trials.legacy_idea_hash("rsi_cross")) == {bk.basket_key(_basket())}


def test_b38o_the_holdout_stays_sealed_until_its_own_step_which_marks_every_asset_used(ran):
    r, _, register = ran
    locks = HoldoutLocks(register.store)  # the study's own store: its trial row read nothing in the holdout
    idea = bk.idea_hash(_basket())
    assert all(locks.lock(idea, a) is None for a in ("BTC", "ETH", "SOL"))  # the study opened nothing
    seen, real = [], portfolio_run.run_portfolio

    def spy(specs, profile=None, **k):
        seen.append(specs)
        return real(specs, profile, **k)

    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", spy)
    try:
        with pytest.raises(ValueError, match="differ from the ones the study judged"):
            bk.open_holdout(r, HOLDOUT, locks, basket=_basket(ASSETS[:2]), capital=CAPITAL, profile=LOOSE,
                            warm_days=40)
        with pytest.raises(ValueError, match="differ from the ones the study judged"):
            bk.open_holdout(r, HOLDOUT, locks, basket=_basket(params={**PARAMS, "long_exit": 60.0}),
                            capital=CAPITAL, profile=LOOSE, warm_days=40)
        assert not seen and all(locks.lock(idea, a) is None for a in ("BTC", "ETH", "SOL"))
        h = bk.open_holdout(r, HOLDOUT, locks, capital=CAPITAL, profile=LOOSE, warm_days=40)
    finally:
        mp.undo()
    assert len(seen) == 1 and all(s.fees is None for s in seen[0])  # one run, at the venue's fees
    assert _day(seen[0][0].prices.index.min()) == HOLDOUT[0] and _day(seen[0][0].prices.index.max()) == HOLDOUT[1]
    assert all(locks.lock(idea, a)["status"] == "opened" for a in ("BTC", "ETH", "SOL"))
    assert h.passed is (h.net > 0)
    with pytest.raises(ValueError, match="holdout on BTC was opened"):
        bk.open_holdout(r, HOLDOUT, locks, capital=CAPITAL, profile=LOOSE, warm_days=40)
    final = bk.verdict(r, h)
    assert final == ("PASS" if r.random.beats and r.double_fees.passed and r.left_out.passed and h.passed else "FAIL")


def test_b38p_a_holdout_look_that_crashes_spends_every_assets_holdout(tmp_path, monkeypatch, ran):
    r, _, _ = ran
    store = Store(f"sqlite:///{tmp_path}/t.db")
    locks = HoldoutLocks(store)

    def boom(*a, **k):
        raise RuntimeError("engine fell over")

    monkeypatch.setattr(bk, "run_portfolio", boom)
    with pytest.raises(RuntimeError):
        bk.open_holdout(r, HOLDOUT, locks, capital=CAPITAL, profile=LOOSE, warm_days=40)
    idea = bk.idea_hash(_basket())
    assert all(locks.lock(idea, a)["status"] == "crashed" for a in ("BTC", "ETH", "SOL"))


def test_b38q_the_report_says_the_verdict_tries_and_next_step_with_per_asset_lines_diagnostic_only(ran, monkeypatch):
    r, _, _ = ran
    text = bk.render(r)
    want = r.random.beats and r.double_fees.passed and r.left_out.passed  # the study's three, by hand
    assert r.random.judged and r.verdict == ("NOT YET" if want else "FAIL")
    assert f"**Basket verdict:** {'NOT YET: the locked holdout is sealed' if want else 'FAIL'}" in text
    monkeypatch.setattr(bk, "verdict", lambda study, holdout: "NOT YET")
    assert "**Basket verdict:** NOT YET: the locked holdout is sealed until its own final step" in bk.render(r)
    monkeypatch.undo()
    assert (f"**Tries:** {r.tries} (one per fixed basket and settings); this basket is try {r.configuration} of 20 "
            "for rsi_cross's family") in text
    cut, asked = r.gate
    assert f"trimmed or refused {cut:,.2f} of {asked:,.2f} intended entry notional on the test days" in text
    assert "next: 4-6 weeks paper" in text
    assert "Per asset (diagnostic only, never a verdict)" in text
    for a in ("BTC", "ETH", "SOL"):
        assert f"| {a} |" in text
    assert f"{r.gate_touched:.1%} of entry notional" in text
    assert "Fixed list chosen on 2026-10-20" in text


def test_b38r_base_fees_are_the_venues_for_each_leg():
    assert bk.base_fees(ASSETS[0].instrument, {}) == FeeSchedule(Decimal("0.001"), Decimal("0.002"))


GAP = ((date(2018, 3, 2), date(2018, 4, 5)), (date(2018, 5, 1), RESEARCH_END))  # BTC and SOL hold across 5 April


@pytest.fixture(scope="module")
def gapped(tmp_path_factory):
    """Test windows with a gap (6 to 30 April): one run spans it (settings fixed), its runs captured."""
    runs, real = [], portfolio_run.run_portfolio

    def spy(specs, profile=None, **k):
        res = real(specs, profile, **k)
        runs.append({"specs": specs, "res": res})
        return res

    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", spy)
    try:
        r = bk.basket_study(_basket(), GAP, RESEARCH_END, register=_register(tmp_path_factory.mktemp("gap")),
                            dataset="syn", capital=CAPITAL, profile=LOOSE, warm_days=40, ladder=False)
    finally:
        mp.undo()
    return r, runs


def _in(d: date, windows) -> bool:
    return any(a <= d <= b for a, b in windows)


def test_b38s_days_between_test_windows_never_count(gapped):
    """Only the windows' days count, and a trade across the gap counts by its daily marks inside the windows."""
    r, _ = gapped
    days = {_day(t) for t in r.base.returns.index}
    want = {GAP[0][0] + timedelta(days=k) for k in range(35)} | {GAP[1][0] + timedelta(days=k) for k in range(120)}
    assert days == want


def test_b38x_the_trim_share_counts_the_test_days_decisions_only(gapped):
    """Numerator and denominator worked here from the run's own gate decisions: requested at the expected fill, and
    what was trimmed or refused of it, on the windows' days; a decision in the gap is left out."""
    r, runs = gapped
    decisions = runs[0]["res"].decisions
    day = [(pd.Timestamp(d["ts"]) - pd.Timedelta(1, "ns")).date() for d in decisions]
    assert any(not _in(x, GAP) for x in day)  # the run decided in the gap too
    asked = sum(float(d["requested"]) * float(d["price"]) for d, x in zip(decisions, day) if _in(x, GAP))
    cut = sum((float(d["requested"]) - float(d["approved"])) * float(d["price"])
              for d, x in zip(decisions, day) if _in(x, GAP))
    assert r.gate == pytest.approx((cut, asked), abs=1e-9)


def _sharpe(x: np.ndarray) -> float:
    return float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(365))


def _basket_side(r, run) -> float:
    """The random check's basket side, worked here: each closed trip at its gate request (requested x expected
    fill, the leg's n-th decision for its n-th trip), cut to the windows, P&L on the closes at the venue's taker fee
    plus half spread, over the fund book's close the day before; Sharpe on the windows' days."""
    specs, res = run["specs"], run["res"]
    clock = specs[0].prices.index
    keep = np.array([_in(_day(t), r.windows) for t in clock])
    pos = [i for i, k in enumerate(keep) if k]
    ranges, start = [], pos[0]
    for a, b in zip(pos, pos[1:] + [None]):
        if b != a + 1:
            ranges.append((start, a))
            start = b
    book = res.book.resample("1D", closed="right", label="right").last().dropna()
    equity = book.shift(1).reindex(clock).fillna(CAPITAL).to_numpy()
    legs, crossing = [], []
    for s in specs:
        trips = bk._trips(res, s.name, s, PARAMS)
        crossing.extend(t for t in trips if any(t.entry <= b < t.exit for _, b in ranges[:-1]))
        asked = [float(d["requested"]) * float(d["price"]) for d in res.decisions
                 if d["strategy"] == s.name and float(d["approved"]) > 0]
        assert [t.notional for t in trips] == pytest.approx(asked[:len(trips)])  # the intended notional
        cps = float(FEES.taker) + bk._half_spread(s.instrument, PARAMS)
        legs.append((s.prices["close"].to_numpy(), bk.clip(trips, ranges), cps))
    return _sharpe(bk.pooled_trade_returns(legs, equity)[keep]), crossing


def test_b38y_the_random_checks_basket_side_is_its_intended_trades_over_the_funds_equity(ran, gapped):
    for (r, runs) in ((ran[0], ran[1]), gapped):
        sharpe, crossing = _basket_side(r, runs[0])
        assert r.random.sharpe == pytest.approx(sharpe, rel=1e-12)
    assert crossing  # with the gap, trades held across a window's end: cut to it, the same on both sides


@pytest.mark.parametrize("judged, beats, twice, out, hold, want", [
    (True, True, True, True, None, "NOT YET"), (True, True, True, True, True, "PASS"),
    (True, True, True, True, False, "FAIL"), (True, False, True, True, True, "FAIL"),
    (False, False, True, True, None, "NOT JUDGED"), (False, False, True, True, True, "NOT JUDGED"),
    (False, False, False, True, True, "FAIL"), (True, True, True, False, None, "FAIL")])
def test_b38z_verdict_pass_needs_all_four_and_not_judged_is_never_pass_or_fail(judged, beats, twice, out, hold, want):
    from types import SimpleNamespace as NS

    study = NS(random=NS(judged=judged, beats=beats), double_fees=NS(passed=twice), left_out=NS(passed=out))
    assert bk.verdict(study, None if hold is None else NS(passed=hold)) == want


def test_b38t_below_5_percent_the_basket_side_is_its_trades_pooled_over_the_funds_equity_by_hand():
    """Two assets, three days. A: long 1,000 at 100, out at 105 on day 2, 0.1% a side; B: short 500 at 50, out at 45
    on day 1, 0.2% a side. Fund equity before each day 10,000, 10,050, 10,100.
    Day 0: -1.0 (A in) - 1.0 (B in) = -2.0; day 1: A +50 (10 x 5), B +50 (10 x 5) - 0.9 (B out on 450) = +99.1;
    day 2: A 0 (flat close) - 1.05 (out on 1,050) = -1.05."""
    a = (np.array([100.0, 105.0, 105.0]), [bk.TradeAt(0, 2, 1, 1_000.0)], 0.001)
    b = (np.array([50.0, 45.0, 44.0]), [bk.TradeAt(0, 1, -1, 500.0)], 0.002)
    got = bk.pooled_trade_returns([a, b], np.array([10_000.0, 10_050.0, 10_100.0]))
    assert list(got) == pytest.approx([-2.0 / 10_000, 99.1 / 10_050, -1.05 / 10_100], abs=1e-15)


def test_b38u_the_same_seed_gives_the_same_percentile_and_another_seed_draws_again(ran):
    r, runs, _ = ran
    base = runs[0]
    names = [s.name for s in base["specs"]]
    again = bk.random_beat(r.base, names, base["specs"], PARAMS, WINDOWS, CAPITAL, r.gate, seed=0)
    assert (again.percentile, again.bar, again.sharpe) == (r.random.percentile, r.random.bar, r.random.sharpe)
    other = bk.random_beat(r.base, names, base["specs"], PARAMS, WINDOWS, CAPITAL, r.gate, seed=1)
    assert other.seed == 1 and other.bar != r.random.bar and other.sharpe == r.random.sharpe


@pytest.mark.parametrize("cut, judged", [(4.99, True), (5.0, False), (5.01, False)])
def test_b38v_exactly_5_percent_trimmed_is_not_judged(ran, cut, judged):
    r, runs, _ = ran
    base = runs[0]
    got = bk.random_beat(r.base, [s.name for s in base["specs"]], base["specs"], PARAMS, WINDOWS, CAPITAL,
                         (cut, 100.0))
    assert got.judged is judged and got.gate_touched == pytest.approx(cut / 100)


def test_b38w_trades_across_a_fold_edge_count_only_inside_the_windows():
    """Windows (0, 9) and (11, 20): a trade from bar 3 to bar 12 is cut to bars 3-9 with no exit paid at the cut; one
    entered on bar 9 keeps its entry and no marks; one exiting inside its window is whole; one entered on bar 10,
    between the windows, is dropped (the draws' entries are inside the windows only)."""
    cut = bk.clip([bk.TradeAt(3, 12, 1, 100.0), bk.TradeAt(9, 11, 1, 50.0), bk.TradeAt(10, 13, 1, 70.0),
                   bk.TradeAt(12, 15, -1, 30.0)], [(0, 9), (11, 20)])
    assert cut == [bk.TradeAt(3, 9, 1, 100.0, exit_cost=False), bk.TradeAt(9, 9, 1, 50.0, exit_cost=False),
                   bk.TradeAt(12, 15, -1, 30.0)]


def test_b38zb_a_refused_entry_counts_in_the_trim_share_so_4_trimmed_and_2_refused_is_not_judged():
    """Advisor (B-38 review 20 Oct): a refused entry never traded, so it is out of both P&L sides, but its requested
    notional at the expected fill is in the trim share's numerator and denominator. 4 of 100 trimmed and 2 refused
    is 6%: not judged (without the refusal it would read 4 / 98 = 4.1%, judged)."""
    from types import SimpleNamespace

    ts = pd.Timestamp("2018-03-10", tz="UTC")
    run = SimpleNamespace(decisions=[
        {"ts": ts, "price": 100.0, "requested": 0.98, "approved": 0.94, "strategy": "a"},  # trimmed 4
        {"ts": ts, "price": 100.0, "requested": 0.02, "approved": 0.0, "strategy": "b"},  # refused 2
    ])
    touched = bk.gate_touched(run, [(date(2018, 3, 2), date(2018, 4, 5))])
    assert touched == pytest.approx((6.0, 100.0))
    r = bk.random_beat(None, [], [], PARAMS, WINDOWS, CAPITAL, touched)
    assert not r.judged and r.gate_touched == pytest.approx(0.06)
    assert "not judged: the compliance tests trimmed 6.0% of entry notional" in bk.random_words(r)


def test_b38zc_a_trade_held_past_a_window_books_its_marks_to_the_last_window_day_with_no_fee_at_the_cut():
    """Advisor (B-38 review 20 Oct): bought 1,000 at bar 1's close (101) and held to bar 5, the window ending at bar 3.
    Its marks count to bar 3; its entry pays 0.1% (1.00); the cut pays nothing. The same on the draws' side: a draw
    of it pays its entry and no exit."""
    closes = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
    cut = bk.clip([bk.TradeAt(1, 5, 1, 1_000.0)], [(0, 3)])
    assert cut == [bk.TradeAt(1, 3, 1, 1_000.0, exit_cost=False)]
    q = 1_000.0 / 101.0
    assert bk.trade_pnl(closes, cut, 0.001) == pytest.approx([0.0, -1.0, q, q, 0.0, 0.0])
    drawn = bk.draw_trades(np.random.default_rng(3), cut, [(0, 3)])
    assert len(drawn) == 1 and drawn[0].exit - drawn[0].entry == 2 and not drawn[0].exit_cost
    fee = bk.trade_pnl(closes, drawn, 0.0) - bk.trade_pnl(closes, drawn, 0.001)
    assert fee.sum() == pytest.approx(0.001 * 1_000.0)  # the entry only
    whole = bk.clip([bk.TradeAt(1, 3, 1, 1_000.0)], [(0, 3)])  # one that exits inside its window pays both sides
    assert bk.trade_pnl(closes, whole, 0.001)[3] == pytest.approx(q - 0.001 * q * 103.0)


def test_b38za_retitling_a_rules_definition_is_the_same_basket_try():
    """A rule-builder definition's title renames it on screen only (B-16): the same basket and settings."""
    named = {"definition": {"title": "Dip buy", "rules": [1, 2]}}
    renamed = {"definition": {"title": "Dip buy, BTC and ETH", "rules": [1, 2]}}
    assert bk.basket_key(_basket(params=named)) == bk.basket_key(_basket(params=renamed))
    assert bk.basket_key(_basket(params=named)) != bk.basket_key(_basket(params={"definition": {"rules": [1, 3]}}))


def test_b38zd_a_holdout_starting_inside_the_research_period_is_refused_before_any_lock_or_run(ran, tmp_path):
    """CR MUST (B-38): a holdout period starting on the research end, or before it, is refused in words; no asset's
    lock is claimed and nothing runs."""
    r, _, _ = ran
    locks = HoldoutLocks(Store(f"sqlite:///{tmp_path}/t.db"))
    seen = []
    mp = pytest.MonkeyPatch()
    mp.setattr(bk, "run_portfolio", lambda *a, **k: seen.append(a))
    try:
        for first in (r.research_end, r.research_end - timedelta(days=30)):
            with pytest.raises(ValueError, match=f"the holdout starts {first}, inside the research period"):
                bk.open_holdout(r, (first, HOLDOUT[1]), locks, capital=CAPITAL, profile=LOOSE, warm_days=40)
    finally:
        mp.undo()
    idea = bk.idea_hash(_basket())
    assert not seen and all(locks.lock(idea, a) is None for a in ("BTC", "ETH", "SOL"))


def test_b38ze_the_random_check_refuses_to_judge_on_fewer_than_1000_draws(ran):
    r, runs, _ = ran
    base = runs[0]
    with pytest.raises(ValueError, match="draws at least 1,000 times, not 999"):
        bk.random_beat(r.base, [s.name for s in base["specs"]], base["specs"], PARAMS, WINDOWS, CAPITAL, (0.0, 1.0),
                       draws=999)


def test_b38zf_the_edges_level_with_the_95th_is_no_beat_and_a_bottom_rung_at_0_is_below_0():
    assert not bk.beats_bar(1.25, 1.25) and bk.beats_bar(1.2500001, 1.25)
    assert bk.ladder_breakeven([(0.001, 0.0), (0.002, -0.05)]) == (None, "<0")


def test_b38zg_each_trip_takes_its_own_opening_entrys_request_whatever_adds_or_open_trips_come_before():
    """CR B-38 (a): leg "x" opens 1 @ 100 (request 1.0, sent), adds 0.5 inside that trip (an add: not an opening),
    has an entry refused, then opens 2 @ 110 (request 2.5, trimmed to 2). Trip 1 is 1.0 x 100, trip 2 2.5 x 110: the
    add and the refusal never shift it."""
    from types import SimpleNamespace

    idx = pd.date_range("2024-01-02", periods=10, tz="UTC")
    t = lambda k: idx[k] + pd.Timedelta(minutes=1)  # noqa: E731 - filled just after bar k's close
    fills = pd.DataFrame({"side": ["BUY", "BUY", "SELL", "BUY", "SELL"], "filled_qty": [1.0, 0.5, 1.5, 2.0, 2.0],
                          "avg_px": [100.0, 101.0, 104.0, 110.0, 115.0], "commissions": [0.0] * 5,
                          "ts_last": [t(0), t(1), t(3), t(5), t(7)]}, index=[f"o{k}" for k in range(5)])
    dec = lambda k, kind, req, ok, px: {"strategy": "x", "bar_ts": idx[k].to_pydatetime(), "ts": t(k),  # noqa: E731
                                        "kind": kind, "requested": req, "approved": ok, "price": px}
    run = SimpleNamespace(legs={"x": SimpleNamespace(fills=fills)}, decisions=[
        dec(0, "entry", 1.0, 1.0, 100.0), dec(1, "add", 0.5, 0.5, 101.0), dec(4, "entry", 3.0, 0.0, 108.0),
        dec(5, "entry", 2.5, 2.0, 110.0)])
    spec = SimpleNamespace(prices=pd.DataFrame({"close": 100.0}, index=idx))
    got = bk._trips(run, "x", spec, PARAMS)
    assert [(g.entry, g.exit, g.notional) for g in got] == [(0, 3, 100.0), (5, 7, 275.0)]
