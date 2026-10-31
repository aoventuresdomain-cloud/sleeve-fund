"""4-hour asset lists (PM 27 Oct, research only; Advisor's joint random-entry check for V5 long-only pooled): a basket
of 4-hour decisions on a pinned list, filled on 1-minute bars as its single sheets are. Done when:

1. a basket study takes a 4-hour definition on an asset list: its days, test windows and holdout are the daily
   basket's on the same dates, and its four checks run as a daily basket's do;
2. B-38's joint random draws move entries on 4-hour bars inside the test windows, each draw's P&L summed into days
   as the book's are; a one-asset basket draws the same entry times as the single-asset filled check, same seed;
3. the basket fills every asset on 1-minute bars, as the single-asset sheets do: a one-asset basket's fills are the
   single-asset backtest's on the same bars;
4. a daily basket is as it was: its key, its dataset name, its hourly fills."""

from __future__ import annotations

import copy
from datetime import date

import numpy as np
import pandas as pd
import pytest
from qa155_lib import resample, synth_1m

from sleeve_fund.research import basket as bk
from sleeve_fund.research import basket_run, trials
from sleeve_fund.research import pinned as pinning
from sleeve_fund.research import run as run_mod
from sleeve_fund.research.fills import DrawFills
from sleeve_fund.research.random_entry import Trade, random_entry_filled
from sleeve_fund.store import Store

H4 = pd.Timedelta(hours=4)
PAIRS = ("BTC/USD", "ETH/USD")
# A short-period trend rule, long only, with a stop: trades often on 40 days of 4-hour bars.
DEFINITION = {
    "version": 1, "name": "b4h-trend", "title": "Trend test", "reason": "test",
    "blocks": {"fast": {"kind": "ema", "period": 3}, "slow": {"kind": "ema", "period": 8},
               "atr": {"kind": "atr", "period": 3}},
    "long": {"entry": {"left": "fast", "op": "crosses_above", "right": "slow"},
             "exit": {"left": "fast", "op": "<", "right": "slow"}},
    "exits": {"stop": {"level": {"sub": ["close", {"mul": [3.0, "atr"]}]}}},
}


# ---- 1. the days ------------------------------------------------------------------------------------------------

def test_a_4h_clock_has_the_daily_clocks_windows_research_end_and_holdout():
    daily = pd.date_range("2018-01-02", periods=400, freq="1D", tz="UTC")  # day k's bar closes at k + 1, 00:00
    four = pd.date_range(daily[0] - pd.Timedelta(days=1) + H4, daily[-1], freq=H4)
    assert len(four) == 6 * len(daily)
    want = basket_run.windows(daily, 50, 100, 120)
    got = basket_run.windows(four, 50, 100, 120, minutes=240)
    assert got[0] == want[0] and got[2:] == want[2:]
    assert got[1] == 6 * want[1]  # the warm-up is counted in decision bars: 100 days of 4-hour bars
    cont = basket_run.windows(four, 50, 100, 120, warmup_bars=30, minutes=240)
    assert cont[0][0][0] == date(2018, 1, 6) and cont[1] == 30  # 30 bars: five days, scored from the sixth


def test_every_4h_bar_closing_on_a_test_day_is_out_of_sample_midnight_included():
    idx = pd.date_range("2024-01-01 04:00", "2024-01-05 00:00", freq=H4, tz="UTC")
    win = ((date(2024, 1, 2), date(2024, 1, 3)),)
    oos = bk._oos(idx, win, H4)
    days = [(t - H4).date() for t in idx]
    assert list(oos) == [date(2024, 1, 2) <= d <= date(2024, 1, 3) for d in days]
    assert oos[idx.get_loc(pd.Timestamp("2024-01-04 00:00", tz="UTC"))]  # the 3rd's last bar closes at midnight
    assert not oos[idx.get_loc(pd.Timestamp("2024-01-02 00:00", tz="UTC"))]  # the 1st's last bar
    assert bk._ranges(idx, win, H4) == [(6, 17)]


def test_a_4h_asset_list_runs_on_pinned_dates_only_and_on_daily_4h_or_1h_bars():
    req = run_mod.StudyRequest(strategy="rules", pair=PAIRS[0], pairs=PAIRS, venue="kraken", minutes=240,
                               train_days=30, test_days=30, holdout_days=8, params={"definition": DEFINITION},
                               pinned=_pins())
    req.validate()
    with pytest.raises(ValueError, match="pinned list's dates only"):
        run_mod.StudyRequest(**{**req.__dict__, "pinned": None}).validate()
    run_mod.StudyRequest(**{**req.__dict__, "minutes": 60}).validate()  # the floor lifted to 1-hour (PM 30 Oct)
    with pytest.raises(ValueError, match="daily, 4-hour or 1-hour bars only"):
        run_mod.StudyRequest(**{**req.__dict__, "minutes": 15}).validate()


# ---- 2. the joint draws -----------------------------------------------------------------------------------------

class _Spy:
    """A pricer that books each draw trip at +1% and records the entries it was given."""

    def __init__(self):
        self.calls = []

    def draws(self, entries, holds, sides, levs=None):
        self.calls.append(np.array(entries))
        return DrawFills(np.full(len(entries), 0.01), 0, 0)


def test_a_one_asset_basket_draws_the_single_asset_checks_entry_times():
    """The pooled draws and the single-asset filled check consume the same seeded stream in the same order, so one
    asset's draws land on the same 4-hour bars."""
    trades = [Trade(6 + 25 * k, 6 + 25 * k + 4, 1) for k in range(5)]
    windows = [(5, 60), (61, 120)]
    pooled, single = _Spy(), _Spy()
    leg = bk.PooledLeg("BTC", trades, np.full(5, 0.02), np.full(5, 0.1), windows, pooled, np.zeros(5),
                       np.zeros(5, dtype=bool), 0.3)
    bk.random_entry_pooled([leg], (0.01, 0.01), draws=1000, seed=7)
    random_entry_filled(trades, np.full(5, 0.02), windows, single, (0.01, 0.01), draws=1000, seed=7)
    assert len(pooled.calls) == len(single.calls) == 2000  # 1,000 draws, two windows each
    assert all(np.array_equal(a, b) for a, b in zip(pooled.calls, single.calls))


def test_the_close_priced_draws_sum_each_4h_bars_pnl_into_its_day():
    """B-38's close-priced check on 4-hour bars: each day's return is that day's six bars' P&L over the equity at
    the day before's close, the same daily returns a daily basket's are judged on."""
    idx = pd.date_range("2024-01-01 04:00", periods=6 * 20, freq=H4, tz="UTC")
    closes = 100 * np.exp(np.cumsum(np.random.default_rng(3).normal(0, 0.01, len(idx))))
    trips = [bk.TradeAt(8, 20, 1, 1_000.0), bk.TradeAt(40, 41, 1, 1_000.0), bk.TradeAt(70, 95, 1, 1_000.0)]
    pnl = bk.trade_pnl(closes, trips, 0.001)
    days = idx.ceil("D")
    by_day = pd.Series(pnl, index=days).groupby(level=0).sum()
    assert len(by_day) == 20 and by_day.index[0] == pd.Timestamp("2024-01-02", tz="UTC")
    assert by_day.sum() == pytest.approx(pnl.sum())
    # every bar of a day (04:00 to the next midnight) is in one day: the 2nd's six bars
    second = pd.Timestamp("2024-01-03", tz="UTC")
    assert by_day[second] == pytest.approx(pnl[(idx > pd.Timestamp("2024-01-02", tz="UTC")) & (idx <= second)].sum())


# ---- 3. 1-minute fills, end to end ------------------------------------------------------------------------------

STORE_START = "2025-01-01"


def _pins():
    """60 research days from 2 Jan (30 to train, 30 to test), then an 8-day holdout, inside the 70-day store."""
    folds = (pinning.Fold(date(2025, 1, 2), date(2025, 2, 1), date(2025, 3, 2)),)
    return pinning.Pinned(date(2025, 1, 2), date(2025, 3, 2), date(2025, 3, 3), date(2025, 3, 10), folds)


class _History:
    """Each asset's synthetic 1-minute bars and every coarser bar from them, as the history store serves them."""

    def __init__(self, pairs=PAIRS):
        self.minute = {p: synth_1m(days=70, seed=s, start=STORE_START, vol_day=0.04)
                       for p, s in zip(pairs, (11, 23, 37))}
        self.calls = []

    def read(self, venue, pair, minutes, start=None, end=None):
        self.calls.append((pair, minutes))
        bars = resample(self.minute[pair], minutes)
        if start is not None:
            bars = bars[bars.index > start]
        return bars if end is None else bars[bars.index <= end]


def _req(pairs=PAIRS):
    return run_mod.StudyRequest(strategy="rules", pair=pairs[0], pairs=pairs, venue="kraken", minutes=240,
                                risk_profile=None, train_days=30, test_days=30, holdout_days=8,
                                params={"definition": DEFINITION}, label="b4h", pinned=_pins())


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b4h")
    store = Store(f"sqlite:///{tmp}/t.db")
    history, specs, studies = _History(), [], []
    mp = pytest.MonkeyPatch()
    real_run, real_render = bk.run_portfolio, bk.render

    def spy_run(legs, *a, **k):
        specs.append(list(legs))
        return real_run(legs, *a, **k)

    def spy_render(study, opened=None):
        studies.append(study)
        return real_render(study, opened)

    mp.setattr(bk, "run_portfolio", spy_run)
    mp.setattr(bk, "render", spy_render)
    try:
        out = run_mod.run_store_study(_req(), store=store, out_dir=tmp / "sheets", history=history)
    finally:
        mp.undo()
    return out, store, history, specs, studies[0]


def test_every_run_decides_on_4h_bars_and_fills_on_1_minute_bars(ran):
    _, _, history, specs, _ = ran
    assert specs and all(len(run) in (1, 2) for run in specs)
    for run in specs:
        for s in run:
            assert (s.bar_minutes, s.exec_minutes) == (240, 1)
            assert (s.prices.index.to_series().diff().dropna() == H4).all()
            assert (s.exec_prices.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()
            assert s.exec_prices.index[-1] <= s.prices.index[-1]  # never a minute after the run's last decision
    assert {(p, 1) for p in PAIRS} <= set(history.calls) and not {(p, 60) for p in PAIRS} & set(history.calls)


def test_the_sheet_names_its_bars_and_its_test_days_are_the_pinned_ones(ran):
    out, _, _, _, study = ran
    assert out.name.startswith("b4h_kraken-basket-btc-eth-usd-store-240m_")
    assert study.windows == ((date(2025, 2, 1), date(2025, 3, 2)),)
    assert study.research_end == date(2025, 3, 2)
    assert study.base.returns.index.equals(study.base.returns.index.normalize())  # judged on daily returns
    text = out.read_text()
    assert text.startswith("# Basket study: b4h (rules on BTC, ETH)") and "**Basket verdict:**" in text


def test_its_joint_draws_are_filled_on_the_assets_own_1_minute_bars(ran):
    """Every asset's draws get a pricer on its own 1-minute bars and 4-hour candles, and 1,000 joint draws price;
    the study's own pooled line is judged or not by the compliance tests' trims alone (Advisor fork 3), never
    because its draws can't be filled."""
    from sleeve_fund.venues import venue

    *_, study = ran
    assert study.pooled.trades > 0 and study.pooled.draws == 1000
    assert study.pooled.why_not == "" or "compliance tests trimmed" in study.pooled.why_not
    assert study.random.draws == 1000

    h = _History()
    assets = tuple(bk.BasketAsset(venue("KRAKEN").instrument(p.split("/")[0], "USD", price_precision=1),
                                  resample(h.minute[p], 240), h.minute[p], bar_minutes=240, exec_minutes=1)
                   for p in PAIRS)
    basket = bk.Basket("two", "rules", study.basket.params, assets, "2025-01-02")
    win = study.windows
    run, names, specs = bk._run(basket, list(assets), win[0][0], win[-1][1], None, 10_000.0, bk.PORTFOLIO, 180)
    m = bk._measure(run, names, specs, ["BTC", "ETH"], win, 10_000.0)
    legs, reconciled, why = bk.pooled_legs(m, names, specs, basket.params, win, 10_000.0)
    assert why == "" and all(leg.pricer is not None and len(leg.trades) for leg in legs)
    r = bk.random_entry_pooled(legs, reconciled, draws=1000, seed=0)
    assert r.trades == sum(len(leg.trades) for leg in legs) and r.draws == 1000
    # judged: the trips reconcile to the book, no draw is held back, and the draws priced to real returns
    assert not r.unreconciled and r.held_back == 0 and r.verdict in ("PASS", "FAIL")
    assert np.isfinite(r.median_random_return) and np.isfinite(r.return_percentile)


def test_it_counts_one_try_of_its_own_beside_any_daily_basket(ran):
    _, store, *_ = ran
    rows = [t for t in trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules"))
            if str(t["definition_name"]).startswith("basket:")]
    assert len(rows) == 1 and rows[0]["dataset"] == "kraken-basket-btc-eth-usd-store-240m"


def test_a_one_asset_baskets_fills_are_the_single_asset_backtests():
    """The same 4-hour decisions on the same 1-minute bars fill at the same times and prices in a one-asset fund run
    as in the single-asset backtest a single sheet runs, sizes apart (each run sizes off its own capital). Compared
    where both trade: the single run waits out its own daily-ATR warm-up, and the fund run's guards (the daily pause,
    the drawdown halt) can stop it before the single run stops."""
    from sleeve_fund.research.runner import run_backtest
    from sleeve_fund.strategies.definitions import to_params
    from sleeve_fund.venues import venue

    m1 = _History().minute["BTC/USD"]
    four = resample(m1, 240)
    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    params = {**to_params(copy.deepcopy(DEFINITION)), "market": "spot"}
    a = bk.BasketAsset(inst, four, m1, bar_minutes=240, exec_minutes=1)
    basket = bk.Basket("one", "rules", params, (a,), "2025-01-02")
    run, names, specs = bk._run(basket, [a], date(2025, 1, 17), date(2025, 2, 15), None, 10_000.0, bk.PORTFOLIO, 90)
    s = specs[0]
    single = run_backtest("rules", pd.concat([s.warmup_prices, s.prices]), inst, params, bar_minutes=240,
                          exec_prices=s.exec_prices, exec_minutes=1, half_spread=bk._half_spread(inst, params))

    def rows(f):
        return [(side, pd.Timestamp(ts).floor("s"), float(px)) for side, ts, px in zip(f["side"], f["ts_last"],
                                                                                         f["avg_px"])]

    mine, theirs = rows(run.legs[names[0]].fills), rows(single.fills)
    lo, hi = max(mine[0][1], theirs[0][1]), min(mine[-1][1], theirs[-1][1])
    mine, theirs = ([r for r in x if lo <= r[1] <= hi] for x in (mine, theirs))
    assert len(mine) >= 4 and [r[:2] for r in mine] == [r[:2] for r in theirs]  # same sides at the same minutes
    assert np.allclose([r[2] for r in mine], [r[2] for r in theirs], rtol=1e-6)  # at the same prices


# ---- 4. a daily basket is unchanged -----------------------------------------------------------------------------

def test_a_daily_baskets_key_and_dataset_are_as_they_were_and_a_4h_ones_differ():
    from sleeve_fund.venues import venue

    inst = venue("KRAKEN").instrument("BTC", "USD", price_precision=1)
    eth = venue("KRAKEN").instrument("ETH", "USD", price_precision=1)
    empty = pd.DataFrame()
    params = {"definition": DEFINITION}
    daily = bk.Basket("d", "rules", params, (bk.BasketAsset(inst, empty, empty), bk.BasketAsset(eth, empty, empty)),
                      "x")
    four = bk.Basket("d", "rules", params, tuple(bk.BasketAsset(i, empty, empty, bar_minutes=240, exec_minutes=1)
                                                 for i in (inst, eth)), "x")
    assert bk.basket_key(daily) == trials.content_hash({"strategy": "rules", "params": trials._untitled(params),
                                                         "assets": ["BTC", "ETH"]})
    assert bk.basket_key(four) != bk.basket_key(daily)
    assert basket_run.dataset_name("kraken", PAIRS) == "kraken-basket-btc-eth-usd-store"
    assert basket_run.dataset_name("kraken", PAIRS, 240) == "kraken-basket-btc-eth-usd-store-240m"


# ---- memory (27 Oct: three assets' whole pinned windows of minutes went past the study container's 1.5 GB) ------

def test_a_4h_list_reads_only_the_minutes_its_runs_fill_on_and_refuses_a_gap():
    h = _History()
    got = basket_run._minutes(h, "kraken", "BTC/USD", date(2025, 2, 1), date(2025, 3, 2))
    assert got.index[0] == pd.Timestamp("2025-02-01 00:01", tz="UTC")
    assert got.index[-1] == pd.Timestamp("2025-03-03", tz="UTC")
    with pytest.raises(ValueError, match="BTC/USD: the stored 1-minute bars don't cover"):
        basket_run._minutes(h, "kraken", "BTC/USD", date(2025, 2, 1), date(2025, 6, 1))  # past the store's end


def test_a_runs_cut_of_its_minutes_is_a_view_never_a_copy():
    from sleeve_fund.venues import venue

    h = _History()
    m = h.minute["BTC/USD"]
    a = bk.BasketAsset(venue("KRAKEN").instrument("BTC", "USD", price_precision=1), resample(m, 240), m,
                       bar_minutes=240, exec_minutes=1)
    cut = bk._cut(a, date(2025, 2, 1), date(2025, 3, 2), 30)
    assert cut.exec_prices.index[0] == pd.Timestamp("2025-02-01 00:01", tz="UTC")
    assert cut.exec_prices.index[-1] == pd.Timestamp("2025-03-03", tz="UTC")
    assert np.shares_memory(cut.exec_prices["close"].to_numpy(), m["close"].to_numpy())
