"""REGIME-BH (21 Oct): the regime table's "Buy and hold over those days (price only, no fees)" column is the price
path itself, compounded close to close over each regime's days, never the funded perp hold's equity (which a heavy
funding drain takes below zero, so its "returns" read -1526% and +898%). Expected values are worked out here from the
fixture's closes, not read from the code under test."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.research import regime_split as rs


@pytest.fixture(scope="module")
def drained(tmp_path_factory):
    """A trend study on the venue's perpetual, 800 daily bars, funding at 0.3% every 8 hours: the 1x perp hold pays
    almost 1% a day and is drained to nothing well inside the window."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import binance_contract, venue
    from test_binance import INFO

    mp = pytest.MonkeyPatch()
    d = tmp_path_factory.mktemp("regime_bh")
    try:
        b = venue("binance")
        mp.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
        mp.setattr(funding, "DEFAULT_ROOT", d / "funding")
        prices = synthetic_ohlcv(days=800, seed=4)
        times = pd.date_range(prices.index[0] - pd.Timedelta(days=2), prices.index[-1] + pd.Timedelta(days=1),
                              freq="8h")
        p = funding._path("BINANCE", "BTC/USDT")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"rates": [[int(t.timestamp() * 1000), 0.003] for t in times]}))
        r = run_study(SPEC, prices, b.instrument("BTC", "USDT"), dataset="syn", ledger=IdeaLedger(d / "l.jsonl"),
                      synthetic=True, train_days=300, test_days=200, half_spread=0.0005,
                      default_params={"fast": 20, "slow": 50}, holdout_days=0)
        yield r, prices
    finally:
        mp.undo()


def test_the_hold_column_is_the_price_compounded_over_each_regime_s_days(drained):
    r, prices = drained
    close = prices["close"].astype(float)
    move = close / close.shift(1) - 1  # each day's close-to-close move, the first day has none
    lab = rs.labels(rs.daily(prices))
    checked = 0
    for s in r.by_regime:
        prior = lab[s.dimension].shift(1, fill_value=rs.UNLABELLED)  # a day is the regime its previous close named
        for b in s.buckets:
            days = move[(prior == b.name).to_numpy()].dropna()
            want = float((1 + days).prod() - 1)
            assert b.hold_return == pytest.approx(want, rel=1e-9, abs=1e-12), (s.dimension, b.name)
            assert b.hold_return >= -1.0
            checked += 1
    assert checked >= 7  # bull, bear, high, low, trend, neutral, range at least


def test_the_sheet_shows_no_hold_figure_below_minus_100_percent(drained):
    r, _ = drained
    text = "\n".join(rs.render(r.by_regime))
    cells = [line.split("|")[-2].strip() for line in text.splitlines()
             if line.startswith("| ") and not line.startswith("| Regime")]
    shown = [float(c.rstrip("%")) for c in cells if c.endswith("%")]
    assert shown and min(shown) >= -100.0
