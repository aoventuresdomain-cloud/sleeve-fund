"""The funding validity cap (Advisor, 6 Oct 2026, O17a-4): each instrument's published cap kept with when it took
effect, the venue's widest standing in where it is missing, and rejected rates quarantined raw, never dropped."""

from __future__ import annotations

import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.venues import binance_funding_caps

T = pd.Timestamp("2025-10-01", tz="UTC")


def test_binance_publishes_the_wider_of_cap_and_floor_per_symbol():
    rows = [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.02000000", "adjustedFundingRateFloor": "-0.02000000"},
            {"symbol": "XYZUSDT", "adjustedFundingRateCap": "0.00300000", "adjustedFundingRateFloor": "-0.00500000"},
            {"symbol": "OLDUSDT", "fundingIntervalHours": 4}]  # only its interval set: no cap published
    assert binance_funding_caps(get_json=lambda url: rows) == {"BTCUSDT": 0.02, "XYZUSDT": 0.005}
    with pytest.raises(ValueError):
        binance_funding_caps(get_json=lambda url: {"code": -1, "msg": "blocked"})


def test_a_cap_is_kept_only_when_it_changes_and_applies_point_in_time(tmp_path):
    assert funding.keep_cap("BINANCE", "BTC/USDT", 0.03, T, tmp_path)
    assert not funding.keep_cap("BINANCE", "BTC/USDT", 0.03, T + pd.Timedelta(days=1), tmp_path)  # unchanged
    assert funding.keep_cap("BINANCE", "BTC/USDT", 0.0075, T + pd.Timedelta(days=2), tmp_path)
    assert not funding.keep_cap("BINANCE", "BTC/USDT", float("nan"), T, tmp_path)
    assert [c for _, c in funding.caps("BINANCE", "BTC/USDT", tmp_path)] == [0.03, 0.0075]
    assert funding.cap_for("BINANCE", "BTC/USDT", T + pd.Timedelta(days=1), tmp_path) == (0.03, False)
    assert funding.cap_at("BINANCE", "BTC/USDT", T + pd.Timedelta(days=3), tmp_path) == 0.0075
    # Before any cap was kept: the venue's widest, else the sanity bound, and said to be missing.
    assert funding.cap_for("BINANCE", "BTC/USDT", T - pd.Timedelta(days=1), tmp_path) == (funding.CAP, True)
    funding.keep_widest("BINANCE", 0.04, tmp_path)
    assert funding.cap_for("BINANCE", "BTC/USDT", T - pd.Timedelta(days=1), tmp_path) == (0.04, True)


def test_a_quarantined_rate_keeps_its_first_raw_value(tmp_path):
    t = int(T.timestamp() * 1000)
    funding.quarantine("BINANCE", "BTC/USDT", [(t, float("inf")), (t + 1, None)], tmp_path)
    funding.quarantine("BINANCE", "BTC/USDT", [(t, 0.5)], tmp_path)
    q = funding.quarantined("BINANCE", "BTC/USDT", tmp_path)
    assert q.iloc[0] == float("inf") and pd.isna(q.iloc[1]) and len(q) == 2
