"""STUDY-ASSET-LIST (PM 21 Oct; Deploy 3): a study request's asset list runs one definition as one B-38 basket.

StudyRequest(pairs=[...]) takes 2-4 assets of the research pool (BTC, ETH, SOL, XRP) on one quote, daily bars, its
pair the list's first; a request without pairs runs exactly as before. The basket's days are the single study's (the
walk-forward's test windows, or one continuous window after the warm-up), its sheet shows each asset and the pooled
book, and it counts as one try."""

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
    ({"minutes": 240}, "daily bars only"),
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
