"""The basket interval floor lifted to 1-hour (PM 30 Oct, "Yes, lift it"): a BTC/ETH asset list deciding on 1-hour
bars runs as the 4-hour one does (tests/test_basket_4h.py), on a pinned list's dates only, filled on each asset's
1-minute bars on the days it trades, judged on daily returns, and counted as one try of its own."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from test_basket_4h import DEFINITION, PAIRS, _History, _pins

from sleeve_fund.research import basket as bk
from sleeve_fund.research import basket_run, trials
from sleeve_fund.research import run as run_mod
from sleeve_fund.store import Store

H1 = pd.Timedelta(hours=1)


def _req():
    return run_mod.StudyRequest(strategy="rules", pair=PAIRS[0], pairs=PAIRS, venue="kraken", minutes=60,
                                risk_profile=None, train_days=30, test_days=30, holdout_days=8,
                                params={"definition": DEFINITION}, label="b1h", pinned=_pins())


def test_a_1h_clock_has_the_daily_clocks_windows_research_end_and_holdout():
    daily = pd.date_range("2024-01-02", periods=70, freq="D", tz="UTC")
    hourly = pd.date_range("2024-01-01 01:00", periods=70 * 24, freq=H1, tz="UTC")
    assert basket_run.windows(hourly, 8, 30, 30, minutes=60)[0] == basket_run.windows(daily, 8, 30, 30)[0]
    assert basket_run.windows(hourly, 8, 30, 30, minutes=60)[2:] == basket_run.windows(daily, 8, 30, 30)[2:]


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("b1h")
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


def test_every_run_decides_on_1h_bars_and_fills_on_1_minute_bars(ran):
    _, _, history, specs, _ = ran
    assert specs
    for run in specs:
        for s in run:
            assert (s.bar_minutes, s.exec_minutes) == (60, 1)
            assert (s.prices.index.to_series().diff().dropna() == H1).all()
            assert (s.exec_prices.index.to_series().diff().dropna() == pd.Timedelta(minutes=1)).all()
            assert s.exec_prices.index[-1] <= s.prices.index[-1]  # never a minute after the run's last decision
    assert {(p, 1) for p in PAIRS} <= set(history.calls)


def test_its_sheet_test_days_draws_and_try_are_the_4h_baskets_on_1h_bars(ran):
    out, store, _, _, study = ran
    assert out.name.startswith("b1h_kraken-basket-btc-eth-usd-store-60m_")
    assert study.windows == ((date(2025, 2, 1), date(2025, 3, 2)),) and study.research_end == date(2025, 3, 2)
    assert study.base.returns.index.equals(study.base.returns.index.normalize())  # judged on daily returns
    assert study.pooled.trades > 0 and study.pooled.draws == 1000 and study.random.draws == 1000
    rows = [t for t in trials.TrialsRegister(store)._counted(trials.legacy_idea_hash("rules"))
            if str(t["definition_name"]).startswith("basket:")]
    assert len(rows) == 1 and rows[0]["dataset"] == "kraken-basket-btc-eth-usd-store-60m"
    assert "**Basket verdict:**" in out.read_text()


def test_a_1h_asset_list_without_pinned_dates_is_refused():
    with pytest.raises(ValueError, match="a 1-hour asset list runs on a pinned list's dates only"):
        run_mod.StudyRequest(**{**_req().__dict__, "pinned": None}).validate()
