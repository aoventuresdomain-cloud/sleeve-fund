"""SOL-FUND (Advisor, 21 Oct; TEC-SOLFUND-GUARD): the venue moved SOL from 8-hour to 4-hour settlements and back
(2022-11-09, 101 steps). The engine charges each stored settlement at its own time and rate (markets.settlement_times),
so research no longer refuses to judge such a run: the sheet says so in an informational line instead. Paper Start
still refuses (funding.schedule_mismatch, latest step only) until it reads the venue's published interval.
Expected values are worked out here from the stored rates and the bars, not read from the code under test."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from o17_harness import _no_swallowed_strategy_errors, _o17win, backtest, binance, hourly_bars, utc, win, write_rates  # noqa: F401
from sleeve_fund import funding


def _eight_four_eight() -> dict:
    """8-hourly to 16:00 on the 3rd, 4-hourly from 20:00 to 16:00 on the 4th, 8-hourly again from 00:00 on the 5th;
    every rate different, so each settlement shows it was charged at its own."""
    times = (list(pd.date_range("2025-10-03 00:00", "2025-10-03 16:00", freq="8h", tz="UTC"))
             + list(pd.date_range("2025-10-03 20:00", "2025-10-04 16:00", freq="4h", tz="UTC"))
             + list(pd.date_range("2025-10-05 00:00", "2025-10-06 00:00", freq="8h", tz="UTC")))
    return {t: 0.0001 * (1 + k) for k, t in enumerate(times)}


def test_a_held_long_across_8_4_8_pays_every_settlement_at_its_own_rate_and_the_run_says_so(binance):
    rates = _eight_four_eight()
    write_rates(rates)
    bars = hourly_bars("2025-10-03 00:00", "2025-10-06 00:00")
    r = backtest(binance, bars, win(("2025-10-03 01:00", "2026-01-01", 1)))
    qty = float(r.fills["filled_qty"].iloc[0])
    entered = pd.Timestamp(r.fills["ts_last"].iloc[0])
    owed = {t: -qty * bars["close"][t] * rate for t, rate in rates.items() if entered < t <= bars.index[-1]}
    charged = {pd.Timestamp(f["ts"]): f["amount"] for f in r.funding}
    assert sorted(charged) == sorted(owed) and len(owed) == 12  # 2 + 6 + 4 settlements after the entry
    for t, amount in owed.items():
        assert charged[t] == pytest.approx(amount, rel=1e-6), t
    # informational, not a refusal: the steps from 16:00 on the 3rd to 16:00 on the 4th are 4 hours (the 16:00 to
    # 20:00 step is the first)
    assert r.funding_schedule == ("BTC/USDT settled every 4 hours for 6 steps from 2025-10-03 20:00 UTC; each "
                                  "charged at its own time and rate")


def test_an_unchanged_schedule_has_no_line(binance):
    write_rates({t: 0.0003 for t in pd.date_range("2025-10-03 00:00", "2025-10-06 00:00", freq="8h", tz="UTC")})
    r = backtest(binance, hourly_bars("2025-10-03 00:00", "2025-10-06 00:00"), win(("2025-10-03 01:00", "2026-01-01", 1)))
    assert r.funding_schedule == ""


@pytest.fixture
def four_hour_study(tmp_path, monkeypatch):
    """A trend study on the venue's perpetual, 800 daily bars, its funding 8-hourly with a 4-hourly spell of 101 steps
    in the second test window, as SOL's in November 2022."""
    from sleeve_fund.data import synthetic_ohlcv
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.study import run_study
    from sleeve_fund.strategies.trend_filter import SPEC
    from sleeve_fund.venues import binance_contract, venue
    from test_binance import INFO

    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path / "funding")
    prices = synthetic_ohlcv(days=800, seed=4)
    first, last = prices.index[0] - pd.Timedelta(days=2), prices.index[-1] + pd.Timedelta(days=1)
    spell = prices.index[600].floor("8h")
    times = (list(pd.date_range(first.floor("8h"), spell, freq="8h"))
             + list(pd.date_range(spell + pd.Timedelta(hours=4), periods=101, freq="4h")))
    times += list(pd.date_range(times[-1] + pd.Timedelta(hours=8), last, freq="8h"))
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"rates": [[int(pd.Timestamp(t).timestamp() * 1000), 0.0001] for t in times]}))
    funding._cache.clear()
    r = run_study(SPEC, prices, b.instrument("BTC", "USDT"), dataset="syn", ledger=IdeaLedger(tmp_path / "l.jsonl"),
                  synthetic=True, holdout_days=0, train_days=300, test_days=200,
                  default_params={"fast": 20, "slow": 50})
    return r, spell, tmp_path


def test_a_study_across_a_4_hour_spell_is_judged_on_its_funding_and_the_sheet_says_why(four_hour_study):
    from sleeve_fund.research.ledger import IdeaLedger
    from sleeve_fund.research.tearsheet import render

    r, spell, tmp_path = four_hour_study
    verdict, words = r.funding_check
    assert verdict != "NOT JUDGED" and "schedule mismatch" not in words, (verdict, words)
    line = (f"BTC/USDT settled every 4 hours for 101 steps from {spell + pd.Timedelta(hours=4):%Y-%m-%d %H:%M} UTC; "
            "each charged at its own time and rate")
    assert r.full_period.funding_schedule == line
    assert line in render(r, IdeaLedger(tmp_path / "l.jsonl"))


def test_paper_start_still_refuses_a_shorter_latest_step(tmp_path, monkeypatch):
    """(b) is the next paper head's: until then the paper check is unchanged."""
    from types import SimpleNamespace

    from sleeve_fund.supervisor import check_funding_schedule

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    write_rates({t: 0.0001 for t in pd.date_range(utc("2025-10-03 00:00"), utc("2025-10-03 16:00"), freq="8h")}
                | {t: 0.0001 for t in pd.date_range(utc("2025-10-03 20:00"), utc("2025-10-04 00:00"), freq="4h")})
    s = SimpleNamespace(strategy="buy_and_hold", instrument="BTC/USDT", venue="BINANCE",
                        params={"market": "perp", "venue": "BINANCE"})
    with pytest.raises(ValueError, match="funding schedule mismatch"):
        check_funding_schedule(s)
