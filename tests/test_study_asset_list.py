"""STUDY-ASSET-LIST (PM 21 Oct; Deploy 3): a study request's asset list runs one definition as one B-38 basket.

StudyRequest(pairs=[...]) takes 2-4 assets of the research pool (BTC, ETH, SOL, XRP) on one quote, daily bars (or
4-hour on a pinned list: tests/test_basket_4h.py), its pair the list's first; a request without pairs runs exactly as
before. The basket's days are the single study's (the walk-forward's test windows, or one continuous window after the
warm-up), its sheet shows each asset and the pooled book, and it counts as one try."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from sleeve_fund.research import basket_run, run as run_mod
from sleeve_fund.research import trials
from sleeve_fund.store import Store
from test_portfolio_run import _daily, _hourly

PAIRS = ("BTC/USD", "ETH/USD", "SOL/USD")
PARAMS = {"rsi_period": 4, "long_entry": 40.0, "long_exit": 55.0}  # trades often enough to draw from


def _req(**kw):
    base = dict(strategy="rsi_cross", pair=PAIRS[0], venue="kraken", minutes=1440, risk_profile=None, train_days=60,
                test_days=90, holdout_days=50, params=PARAMS, label="trend-x", pairs=list(PAIRS))
    return run_mod.StudyRequest(**{**base, **kw})


def test_a_request_takes_an_asset_list_and_keeps_it_as_given():
    r = _req()
    r.validate()
    assert r.pairs == PAIRS  # a list becomes a tuple: the request stays hashable and frozen


def test_a_single_pair_request_is_unchanged():
    r = run_mod.StudyRequest(strategy="rsi_cross", pair="BTC/USD")
    r.validate()
    assert r.pairs is None


@pytest.mark.parametrize("kw, words", [
    ({"pairs": ["BTC/USD"]}, "2 to 4 assets, not 1"),
    ({"pairs": ["BTC/USD", "BTC/USD"]}, "listed twice: BTC, BTC"),
    ({"pairs": ["BTC/USD", "SUI/USD"]}, "SUI is not in the research pool (BTC, ETH, SOL, XRP)"),
    ({"pairs": ["BTC/USD", "ETH/USDT"]}, "different currencies"),
    ({"pairs": ["BTC/USD", "ETHUSD"]}, "should look like BTC/USDT"),
    ({"pair": "ETH/USD"}, "is not its list's first asset BTC/USD"),
    ({"minutes": 15}, "daily, 4-hour or 1-hour bars only"),
    ({"minutes": 60}, "a 1-hour asset list runs on a pinned list's dates only"),
    ({"minutes": 240}, "a 4-hour asset list runs on a pinned list's dates only"),
    ({"old_window_too": True, "score_window": "continuous"}, "without the placebo or the old window"),
])
def test_an_asset_list_is_refused_in_words(kw, words):
    with pytest.raises(ValueError) as e:
        _req(**kw).validate()
    assert words in str(e.value)


def test_only_a_request_with_a_list_goes_to_the_basket(monkeypatch, tmp_path):
    monkeypatch.setattr(basket_run, "run_store_basket", lambda req, store, **k: ("basket", req.pairs))
    assert run_mod.run_store_study(_req(), store=object()) == ("basket", PAIRS)

    class _Empty:
        def read(self, *a, **k):
            raise KeyError("none")

    with pytest.raises(ValueError, match="no stored history for BTC/USD"):  # the single study's own path
        run_mod.run_store_study(_req(pairs=None), ledger_path=tmp_path / "l.jsonl", history=_Empty())


def test_the_walk_forward_windows_are_the_single_studys_test_windows():
    clock = pd.date_range("2018-01-02", periods=400, freq="1D", tz="UTC")  # day k's bar closes at k + 1, 00:00
    tests, warm, end, holdout = basket_run.windows(clock, 50, 100, 120)
    assert warm == 100 and end == date(2018, 12, 16) and holdout == (date(2018, 12, 17), date(2019, 2, 4))
    # research: 350 days; folds at 0 and 120 (240 + 120 > 350 stops a third): tests from day 100 and day 220
    assert tests == ((date(2018, 4, 11), date(2018, 8, 8)), (date(2018, 8, 9), date(2018, 12, 6)))


def test_a_continuous_window_runs_from_the_warm_up_to_the_research_end():
    clock = pd.date_range("2018-01-02", periods=400, freq="1D", tz="UTC")
    tests, warm, end, _ = basket_run.windows(clock, 50, 100, 120, warmup_bars=30)
    assert warm == 30 and tests == ((date(2018, 1, 31), date(2018, 12, 16)),) and end == date(2018, 12, 16)


def test_the_listed_assets_run_on_the_days_they_all_have():
    a = pd.DataFrame({"close": 1.0}, index=pd.date_range("2018-01-02", periods=10, freq="1D", tz="UTC"))
    b = a.iloc[3:]
    c = a.iloc[:8]
    got = basket_run.common_clock([a, b, c], ["BTC", "ETH", "SOL"])
    assert all(g.index[0] == a.index[3] and g.index[-1] == a.index[7] for g in got)


class _History:
    """Each asset's synthetic hourly bars and their daily bars, as the history store serves them."""

    def __init__(self):
        self.hourly = {p: _hourly(300, seed) for p, seed in zip(PAIRS, (2, 5, 9))}
        self.calls = []

    def read(self, venue, pair, minutes, start=None, end=None):
        self.calls.append((pair, minutes))
        h = self.hourly[pair]
        bars = _daily(h) if minutes == 1440 else h
        if start is not None:
            bars = bars[bars.index > start]
        return bars if end is None else bars[bars.index <= end]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sal")
    store = Store(f"sqlite:///{tmp}/t.db")
    history = _History()
    out = run_mod.run_store_study(_req(), store=store, out_dir=tmp / "sheets", history=history)
    return out, store, history, tmp


def test_the_sheet_shows_each_asset_and_the_pooled_book(ran):
    out, _, history, _ = ran
    text = out.read_text()
    assert text.startswith("# Basket study: trend-x (rsi_cross on BTC, ETH, SOL)")
    assert "**Basket verdict:**" in text and "one fund run of the whole basket" in text
    per_asset = text[text.index("## Per asset (diagnostic only, never a verdict)"):]
    for a in ("BTC", "ETH", "SOL"):
        assert f"\n| {a} | " in per_asset
    assert {(p, m) for p, m in history.calls} == {(p, m) for p in PAIRS for m in (1440, 60)}
    assert out.name.startswith("trend-x_kraken-basket-btc-eth-sol-usd-store_")


def test_the_basket_counts_as_one_try_and_a_rerun_adds_none(ran):
    out, store, history, tmp = ran
    reg = trials.TrialsRegister(store)
    rows = [t for t in reg._counted(trials.legacy_idea_hash("rsi_cross"))
            if str(t["definition_name"]).startswith("basket:")]
    assert len(rows) == 1 and rows[0]["definition_name"] == "basket:trend-x"
    run_mod.run_store_study(_req(), store=store, out_dir=tmp / "again", history=history)
    again = [t for t in reg._counted(trials.legacy_idea_hash("rsi_cross"))
             if str(t["definition_name"]).startswith("basket:")]
    assert len(again) == 1


def test_a_basket_sheet_is_never_read_as_a_single_assets_g1_pass(ran):
    from sleeve_fund.dashboard import pipeline

    facts = pipeline.sheet_facts(ran[0])
    assert facts["strategy"] is None and facts["g1"] is None  # no "# Tear sheet:" head, no G1 row: never a pass


def test_a_basket_study_needs_the_journal():
    with pytest.raises(ValueError, match="needs the journal"):
        run_mod.run_store_study(_req(), store=None, history=_History())


# ---- the server's study list (rescore, study.yml) ------------------------------------------------------------------

def _list():
    from sleeve_fund.research import rescore

    return rescore, rescore.ROOT / "configs" / "studies" / "qr-rescore-list.md"


def test_a_list_rows_pair_list_is_one_basket_request_on_its_first_asset():
    rescore, path = _list()
    daily = [(n, name, m) for n, name, m in rescore.rows_of(path.read_text()) if m == 1440]
    assert daily
    for _, name, m in daily:
        req = rescore.request(name, m, "binance", "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT", continuous=True)
        assert req.pair == "BTC/USDT" and req.pairs == ("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT")
        assert req.old_window_too is False and req.use_holdout is False
        if req.placebo_parent is None:
            req.validate()
    one = rescore.request(daily[0][1], 1440, "binance", "BTC/USDT")
    assert one.pairs is None and one.pair == "BTC/USDT"


@pytest.mark.parametrize("pair", ["BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,SUI/USDT", "BTC/USDT,", "BTC/USDT;ETH/USDT"])
def test_a_bad_pair_list_is_refused_before_anything_runs(pair, capfd):
    rescore, path = _list()
    called = []
    assert rescore.main([str(path), "--pair", pair], run=lambda *a, **k: called.append(1), journal=lambda: None) == 2
    assert called == [] and capfd.readouterr().out.startswith("refused:")


def test_the_workflow_takes_the_same_pair_list():
    import re

    rescore, _ = _list()
    text = (rescore.ROOT / ".github" / "workflows" / "study.yml").read_text()
    pattern = re.search(r'\[\[ "\$PAIR" =~ \^(.+)\$ \]\]', text)[1]
    assert pattern == rescore.PAIRS


# ---- the pooled verdict (Advisor ruling 21 Oct) ----------------------------------------------------------------------

import math  # noqa: E402

import numpy as np  # noqa: E402

from sleeve_fund.research import basket as bk  # noqa: E402
from sleeve_fund.research.fills import DrawFills  # noqa: E402
from sleeve_fund.research.random_entry import Trade  # noqa: E402

DEFINITION = {
    "version": 1, "name": "sal-trend", "title": "Trend test", "reason": "test",
    "blocks": {"fast": {"kind": "ema", "period": 2}, "slow": {"kind": "ema", "period": 4},
               "atr": {"kind": "atr", "period": 2}},  # a 40-bar warm-up: the fixture's
    "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
             "exit": {"left": "fast", "op": "<", "right": "slow"}},
    "exits": {"stop": {"level": {"sub": ["close", {"mul": [3.0, "atr"]}]}}},
}


class _Spy:
    """A pricer that books each draw trip at +1% (or holds `held` of each call back) and records what it was given."""

    def __init__(self, held: int = 0):
        self.calls, self.held = [], held

    def draws(self, entries, holds, sides, levs=None):
        self.calls.append(np.array(entries))
        return DrawFills(np.full(len(entries), 0.01), min(self.held, len(entries)), 0)


def _leg(name, pricer, n=3, start=0, end=60):
    trades = [Trade(start + 10 * k, start + 10 * k + 3, 1) for k in range(n)]
    return bk.PooledLeg(name, trades, np.full(n, 0.02), np.full(n, 0.1), [(start, end)], pricer, np.zeros(n),
                        np.zeros(n, dtype=bool), 0.15)


def test_a_pooled_draw_moves_both_assets_entries_in_one_draw():
    a, b = _Spy(), _Spy()
    r = bk.random_entry_pooled([_leg("BTC", a), _leg("ETH", b)], (0.012, 0.012), draws=5, seed=3)
    assert len(a.calls) == len(b.calls) == 5  # each draw prices every asset once
    assert any(not np.array_equal(x, y) for x, y in zip(a.calls, a.calls[1:]))  # and moves its entries
    assert any(not np.array_equal(x, y) for x, y in zip(b.calls, b.calls[1:]))
    # the statistic: every asset's trips at their notional over the fund's capital, summed: 6 x 2% x 0.1
    assert r.strategy_return == pytest.approx(6 * 0.02 * 0.1)
    assert r.median_random_return == pytest.approx(6 * 0.01 * 0.1)  # each draw: 6 trips at 1% x 0.1
    assert r.trades == 6 and r.verdict == "PASS"


def test_held_back_above_2_percent_pooled_is_not_judged():
    # 1 of 3 held back on one asset, none on the other: 1 / 6 of every draw's trips, 16.7%, above 2%
    r = bk.random_entry_pooled([_leg("BTC", _Spy(held=1)), _leg("ETH", _Spy())], (0.012, 0.012), draws=4)
    assert r.held_back_share == pytest.approx(1 / 6) and r.verdict == "NOT JUDGED"
    assert "not judged: too many draws held back" in bk.pooled_words(r)
    ok = bk.random_entry_pooled([_leg("BTC", _Spy(), n=3), _leg("ETH", _Spy(), n=3)], (0.012, 0.012), draws=4)
    assert ok.held_back_share == 0 and ok.verdict == "PASS"


def test_the_pooled_trips_must_reconcile_to_the_book():
    r = bk.random_entry_pooled([_leg("BTC", _Spy())], (0.006, 0.0125), draws=2)
    assert r.verdict == "NOT JUDGED" and "more than 0.5 points apart: not judged" in bk.pooled_words(r)


def _measured(returns: list[float], start="2018-03-02") -> bk.Measured:
    idx = pd.date_range(pd.Timestamp(start, tz="UTC") + pd.Timedelta(days=1), periods=len(returns), freq="1D")
    return bk.Measured(None, [], pd.DataFrame(), pd.Series(returns, index=idx), 0.0)


def _spec(prices: list[float], start="2018-03-02"):
    from types import SimpleNamespace as NS

    from nautilus_trader.model import Venue

    from sleeve_fund.instruments import spot_pair
    from test_portfolio_run import FEES

    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(prices), freq="1D")
    bars = pd.DataFrame({"close": prices}, index=idx)
    return NS(prices=bars.iloc[1:], warmup_prices=bars.iloc[:1], bar_minutes=1440,
              instrument=spot_pair("BTC", "USD", FEES, Venue("KRAKEN")))


def test_a_long_short_basket_prints_no_hold_bar():
    m = _measured([0.01, -0.01, 0.02])
    params = {"definition": {**DEFINITION, "short": DEFINITION["long"]}}
    assert bk.hold_bar(m, [_spec([100, 101, 102, 103])], params, ((date(2018, 3, 2), date(2018, 3, 4)),), 1e4) is None
    assert bk.hold_bar(m, [_spec([100, 101, 102, 103])], {"allow_short": True}, ((date(2018, 3, 2), date(2018, 3, 4)),),
                       1e4) is None
    assert bk.hold_words(None) == "no hold bar: the basket has a short side (Advisor: long-only baskets only)"


def test_the_hold_bar_takes_the_price_only_hold_when_its_sharpe_is_higher_and_names_it():
    windows = ((date(2018, 3, 2), date(2018, 3, 6)),)
    m = _measured([0.001, 0.002, -0.001, 0.0, 0.001])
    rising = [100, 101, 102.5, 103, 104.5, 105]  # the price-only hold: steady gains
    perp = {"market": "perp"}

    def funded(bars, instrument, params, capital):  # the funded hold: the same days, choppier
        eq = pd.Series([1.0, 1.02, 0.99, 1.03, 0.98, 1.04], index=bars.index) * capital
        return eq

    h = bk.hold_bar(m, [_spec(rising), _spec(rising)], perp, windows, 1e4, hold=funded)
    assert h.chosen == bk.PRICE_ONLY and h.price_sharpe > h.perp_sharpe
    assert h.bar_sharpe == pytest.approx(h.price_sharpe) and not h.beats
    words = bk.hold_words(h)
    assert "against the price-only hold (no spot market at this venue)'s" in words and bk.UNPROVEN in words
    assert "PASS" not in words and "FAIL" not in words  # unproven until BH-102 is live: no verdict words

    def smooth(bars, instrument, params, capital):  # now the funded hold is the steadier one
        return pd.Series([1.0, 1.01, 1.02, 1.03, 1.04, 1.05], index=bars.index) * capital

    h2 = bk.hold_bar(m, [_spec(rising), _spec(rising)], perp, windows, 1e4, hold=smooth)
    assert h2.chosen == bk.FUNDED and h2.bar_sharpe == pytest.approx(h2.perp_sharpe)


def test_an_unproven_hold_bar_never_lets_a_basket_pass():
    from types import SimpleNamespace as NS

    study = NS(random=NS(judged=True, beats=True), double_fees=NS(passed=True), left_out=NS(passed=True),
               pooled=NS(verdict="PASS"), hold=NS(beats=True))
    assert bk.verdict(study, NS(passed=True)) == "NOT JUDGED"  # BH102_LIVE is False
    study.hold = None  # a long/short basket: random entry and the rest decide
    assert bk.verdict(study, NS(passed=True)) == "PASS"
    study.pooled = NS(verdict="FAIL")
    assert bk.verdict(study, NS(passed=True)) == "FAIL"
    study.pooled = NS(verdict="NOT JUDGED")
    assert bk.verdict(study, None) == "NOT JUDGED"


@pytest.fixture(scope="module")
def filled(tmp_path_factory):
    """A two-asset rule-builder basket, untrimmed, its draws filled as the engine fills the definition."""
    from sleeve_fund.strategies.definitions import to_params
    from test_b38_basket_study import LOOSE, RESEARCH_END, WINDOWS, _asset, _register

    params = to_params(DEFINITION, "KRAKEN")
    basket = bk.Basket("sal-trend", "rules", params, (_asset("BTC", 2), _asset("ETH", 5)), "2026-10-21")
    reg = _register(tmp_path_factory.mktemp("filled"))
    study = bk.basket_study(basket, WINDOWS, RESEARCH_END, register=reg, dataset="syn", capital=30_000.0,
                            profile=LOOSE, warm_days=40, ladder=False, filled=True)
    return study, reg


def test_a_rule_basket_is_judged_on_draws_filled_on_the_pooled_book(filled):
    study, _ = filled
    p = study.pooled
    assert p.why_not == "" and p.trades > 0 and p.draws == 1000 and p.held_back_share <= 0.02
    assert not p.unreconciled, p.reconciled  # its trips add to what the book did over them
    assert study.gate_touched < bk.GATE_TOUCH_CAP
    text = bk.render(study)
    assert "| Beats random entries, filled on the pooled book (95th percentile) | " in text
    assert "| Random entries, close-priced | " in text and "no verdict" in text
    assert "| Hold bar (long-only baskets) | pooled Sharpe" in text and study.hold is not None
    assert "Without SOL" not in text  # no SOL in this basket


def test_the_holdout_row_and_close_priced_row_read_what_is_true(filled):
    """Advisor, B0's ruling 31 Oct: a sealed run never reads "already viewed" unless an earlier study holds a lock, and
    the close-priced row says how it prices and gives the book's filled Sharpe beside it."""
    study, _ = filled
    holdout_row = lambda text: next(x for x in text.splitlines() if x.startswith("| Locked holdout"))
    sealed = bk.render(study)
    assert "already viewed" not in sealed and "sealed: opened once, as its own final step |" in holdout_row(sealed)
    assert "already viewed on BTC by an earlier study, so paper is the forward test" in holdout_row(
        bk.render(study, viewed=("BTC",)))
    row = next(x for x in sealed.splitlines() if x.startswith("| Random entries, close-priced | "))
    assert (f"trips at decision closes, no stop fills, against the book's filled Sharpe "
            f"{bk._sharpe(study.base.returns.to_numpy()):.2f} |") in row


def test_a_confirming_run_never_gives_a_rounds_try_line(filled):
    """A run that records no try (counted False: a guarded run at a size scale) keeps "not a new try" even when its
    notes carry the round's try line (CR F2 on #341)."""
    from dataclasses import replace

    study, _ = filled
    line = "Round next-1 (study list SHA-256 0123456789ab): registered try 3 of 21, family A try 3"
    tries = lambda text: next(x for x in text.splitlines() if x.startswith("**Tries:**"))
    assert tries(bk.render(study, round_try=line)) == f"**Tries:** {line}"
    assert "not a new try" in tries(bk.render(replace(study, counted=False), round_try=line))


@pytest.mark.parametrize("name, coin", [("SOLUSDT-PERP", "SOL"), ("SOL", "SOL"), ("ETHUSDC-PERP", "ETH"),
                                        ("XBTUSD", "XBT"), ("USDT", "USDT")])
def test_a_perps_symbol_is_its_coin(name, coin):
    assert bk.base_asset(name) == coin  # so a perp basket with SOL gets its Without SOL line (Researcher, B0)


def test_a_two_asset_basket_registers_exactly_one_try(filled):
    study, reg = filled
    rows = [t for t in reg._counted(bk.idea_hash(study.basket)) if str(t["definition_name"]).startswith("basket:")]
    assert len(rows) == 1 and (study.tries, study.configuration) == (1, 1)


def test_a_basket_with_sol_shows_itself_without_it(ran):
    text = ran[0].read_text()
    line = next(x for x in text.splitlines() if x.startswith("**Without SOL** (BTC, ETH;"))
    assert "diagnostic only, never a condition or a try" in line and "net " in line and "Sharpe " in line


def test_a_one_pair_request_runs_the_single_study_untouched(tmp_path, monkeypatch):
    """The single path never reaches the basket code, and writes the single tear sheet (the B-37 job's fixture). Its
    bytes were checked equal to the sheet before STUDY-ASSET-LIST (c0969acd) by hand; not pinned here, as every later
    sheet change would have to move a pinned hash."""
    import test_b37_pinned_dates as b37
    from datetime import timedelta

    monkeypatch.setattr(run_mod, "exec_step", lambda span, minutes: None)
    monkeypatch.setattr(run_mod, "OOS_EXEC_BUDGET", 0)

    def refuse(*a, **k):
        raise AssertionError("a one-pair request reached the basket")

    monkeypatch.setattr(basket_run, "run_store_basket", refuse)
    monkeypatch.setattr(bk, "basket_study", refuse)
    p = b37._pins()
    req = run_mod.StudyRequest(strategy=b37.SPEC.name, pair="BTC/USD", venue="kraken", minutes=1440, train_days=730,
                               test_days=365, holdout_days=p.holdout_days, pinned=p)
    out = run_mod.run_store_study(req, ledger_path=tmp_path / "l.jsonl", out_dir=tmp_path / "ts",
                                  history=b37._History(p.holdout_end + timedelta(days=1)))
    text = out.read_text()
    assert text.startswith("# Tear sheet:") and "Basket" not in text and "pooled" not in text


def test_the_basket_passes_the_studys_market_to_every_leg(monkeypatch):
    """RESEARCH-SPOT-FEES (DA wires the market's meaning): whatever study_market returns goes into every leg's params."""
    seen = []
    monkeypatch.setattr(run_mod, "study_market", lambda req: {"market": "perp", "probe": req.market})

    def spy(basket, *a, **k):
        seen.append(basket.params)
        raise RuntimeError("stop here")

    monkeypatch.setattr(bk, "basket_study", spy)
    with pytest.raises(RuntimeError, match="stop here"):
        run_mod.run_store_study(_req(market="perp"), store=Store("sqlite://"), history=_History())
    assert seen and seen[0]["market"] == "perp" and seen[0]["probe"] == "perp" and seen[0]["rsi_period"] == 4


def test_a_basket_takes_the_study_market_and_refuses_an_unknown_one_in_words():
    # RESEARCH-SPOT-FEES wired the market QD's stub refused (HoE, Deploy 3 merge)
    assert run_mod.study_market(_req(market="spot-lowfee")) == {"market": "spot-lowfee"}
    with pytest.raises(ValueError, match="perp or spot-lowfee"):
        run_mod.study_market(_req(market="spot"))
    assert run_mod.study_market(_req()) == {}
