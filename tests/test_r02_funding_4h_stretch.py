"""R-02 (Advisor, 21 Oct): the backfilled SOL history holds a stretch of 4-hour settlements between 8-hour ones. The
engine charges funding at each STORED settlement time and rate, never on a fixed 00:00/08:00/16:00 schedule: a 4-hour
day charges 6, and once the venue goes back to 8 hours nothing is charged in between (no baseline fill)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sleeve_fund import funding, markets
from sleeve_fund.venues import binance_contract, venue
from test_funding_settlement import PERP, _held_throughout, _win, utc  # noqa: F401 - _win registers the test model


@pytest.fixture
def eight_four_eight(tmp_path, monkeypatch):
    """8-hourly to 3 Oct 16:00, 4-hourly from 20:00 to 4 Oct 20:00, 8-hourly again from 5 Oct 00:00."""
    from test_binance import INFO

    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    b = venue("binance")
    monkeypatch.setattr(b, "contract", lambda pair: binance_contract(pair, get_json=lambda url: INFO))
    times = (pd.date_range("2025-10-03 00:00", "2025-10-03 16:00", freq="8h", tz="UTC")
             .append(pd.date_range("2025-10-03 20:00", "2025-10-04 20:00", freq="4h", tz="UTC"))
             .append(pd.date_range("2025-10-05 00:00", "2025-10-06 00:00", freq="8h", tz="UTC")))
    rates = [[int(t.timestamp() * 1000), 0.0001 * (1 + i % 3)] for i, t in enumerate(times)]
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"rates": rates}))
    idx = pd.date_range("2025-10-03 00:00", "2025-10-06 00:00", freq="1h", tz="UTC")[1:]
    c = 60_000.0 + np.arange(len(idx))
    bars = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1e9}, index=idx)
    return b.instrument("BTC", "USDT"), bars, dict(zip(times, (r for _, r in rates)))


@pytest.mark.parametrize("side", [1, -1])
def test_a_4_hour_stretch_charges_6_a_day_at_the_stored_times_and_nothing_between_8_hour_ones(eight_four_eight, side):
    inst, bars, rate = eight_four_eight
    assert PERP["market"] == "perp"  # set-up: the venue's own perp
    rows = _held_throughout(inst, bars, side)
    charged = [pd.Timestamp(r["ts"]) for r in rows]
    day = lambda d: [t for t in charged if t.strftime("%Y-%m-%d") == d]  # noqa: E731
    assert [t.hour for t in day("2025-10-04")] == [0, 4, 8, 12, 16, 20]  # the 4-hour day: 6
    assert [t.hour for t in day("2025-10-05")] == [0, 8, 16]  # back to 8 hours: 3, none filled in between
    assert [t.hour for t in day("2025-10-03")] == [8, 16, 20]  # opened 01:00: 08:00, 16:00, then the first 4-hourly
    for r in rows:
        assert r["rate"] == pytest.approx(rate[pd.Timestamp(r["ts"])])


def test_the_settlement_times_follow_the_stored_records_through_the_stretch():
    t = lambda s: utc(s).to_pydatetime()  # noqa: E731
    rec = pd.Series(0.0001, index=pd.DatetimeIndex([utc(s) for s in (
        "2025-10-04 16:00", "2025-10-04 20:00", "2025-10-05 00:00", "2025-10-05 04:00", "2025-10-05 08:00",
        "2025-10-05 16:00", "2025-10-06 00:00")]))
    got = markets.settlement_times(t("2025-10-04 12:00"), t("2025-10-06 00:00"), (0, 8, 16), rec)
    assert got == [t(s) for s in ("2025-10-04 16:00", "2025-10-04 20:00", "2025-10-05 00:00", "2025-10-05 04:00",
                                  "2025-10-05 08:00", "2025-10-05 16:00", "2025-10-06 00:00")]


def _kept(tmp_path, monkeypatch, rates):
    monkeypatch.setattr(funding, "DEFAULT_ROOT", tmp_path)
    p = funding._path("BINANCE", "BTC/USDT")
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({"rates": rates}))


def test_the_spot_check_compares_each_sampled_rate_and_the_most_negative_with_the_venues_own_record(tmp_path,
                                                                                                   monkeypatch):
    from types import SimpleNamespace

    from sleeve_fund import history

    t0 = int(utc("2024-01-01").timestamp() * 1000)
    step = 8 * 3_600_000
    kept = [[t0 + i * step, r] for i, r in enumerate([0.0001, -0.00025, 0.0003, 0.00005])]
    _kept(tmp_path, monkeypatch, kept)
    venue_says = {t0 + i * step: s for i, s in enumerate(["0.00010000", "-0.00025000", "0.00030000", "0.00005000"])}

    def raw(pair, start_ms):  # the venue's records from start_ms, as it sends them (stamped a few ms late)
        return [{"fundingTime": t + 8, "fundingRate": s} for t, s in sorted(venue_says.items()) if t + 8 >= start_ms]

    profile = SimpleNamespace(name="BINANCE", label="Binance", funding_raw=raw)
    out = history.funding_sample(profile, "BTC/USDT", count=2)
    assert out["checked"] == 3 and out["matched"] == 3 and out["mismatches"] == []  # 2 drawn + the most negative
    assert out["most_negative"]["stored"] == -0.00025
    venue_says[t0 + step] = "0.00025000"  # the venue's sign is the other way: the most negative is always checked
    venue_says[t0] = "0.01000000"  # a percent kept as a fraction
    out = history.funding_sample(profile, "BTC/USDT", count=4)
    assert out["checked"] == 4 and out["matched"] == 2
    assert sorted(m["venue"] for m in out["mismatches"]) == ["0.00025000", "0.01000000"]
