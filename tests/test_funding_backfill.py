"""FUNDING-BACKFILL (START CHECKLIST 5, HoE 14 Oct 02:25 UK): the settled funding a perpetual's store lacks before
its first kept rate, fetched from the venue's history. Only adds; a kept rate is never changed; the dry run counts
what the real run adds and writes nothing."""

import json

import pandas as pd
import pytest

from sleeve_fund import funding
from sleeve_fund.history import funding_backfill, main
from sleeve_fund.venues import venue as venue_profile

H8 = 8 * 3600 * 1000
T0 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)
KEPT_FROM = T0 + 30 * H8  # the store holds rates from the 31st settlement on, as the server did from 22 Apr


def _venue(n=60, rate=0.0001, odd=None):
    """The venue's history: n settlements every 8 h from T0, each at `rate` unless `odd` says otherwise."""
    odd = odd or {}
    return [(T0 + i * H8, odd.get(i, rate)) for i in range(n)]


def _loader(history):
    def load(pair, start):
        return [(t, r) for t, r in history if t >= start][:funding.PAGE]
    return load


def _keep(profile, root, rows):
    path = funding._path(profile.name, "BTC/USDT", root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"rates": [[t, r] for t, r in rows]}))
    return path


@pytest.fixture
def binance(monkeypatch):
    profile = venue_profile("binance")
    monkeypatch.setattr(funding, "PAGE", 7)  # several pages over the 30 missing settlements
    return profile


def test_the_stretch_before_the_first_kept_rate_is_added_and_no_kept_rate_changes(binance, tmp_path, monkeypatch):
    history = _venue()
    monkeypatch.setattr(binance, "funding_loader", _loader(history))
    # The kept rates differ from the venue's for the same settlements: they must stay exactly as kept.
    kept = [(t, 0.0003) for t, _ in history if t >= KEPT_FROM]
    path = _keep(binance, tmp_path, kept)
    before = path.read_bytes()

    dry = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path, dry_run=True)
    assert dry["would_add"] == 30 and dry["complete"] and dry["refused"] == 0
    assert path.read_bytes() == before  # the dry run writes nothing

    real = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path)
    assert real["added"] == dry["would_add"] and real["complete"]
    assert real["first_now"] == pd.Timestamp(T0, unit="ms", tz="UTC").isoformat()
    rows = json.loads(path.read_text())["rates"]
    assert [t for t, _ in rows] == [t for t, _ in history]  # in order, each settlement once
    assert {t: r for t, r in rows if t >= KEPT_FROM} == dict(kept)  # every kept rate as it was

    again = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path, dry_run=True)
    assert again["would_add"] == 0  # a second run has nothing left to add


def test_running_out_of_pages_is_not_complete_and_the_command_fails(binance, tmp_path, monkeypatch):
    monkeypatch.setattr(binance, "funding_loader", _loader(_venue()))
    _keep(binance, tmp_path, [(t, r) for t, r in _venue() if t >= KEPT_FROM])
    out = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path, pages=2, dry_run=True)
    assert not out["complete"] and out["would_add"] == 14
    assert main(["--root", str(tmp_path), "funding-backfill", "--venue", "binance", "--since", "2024-01-01",
                 "--pages", "2", "--dry-run", "BTC/USDT"]) == 2
    assert main(["--root", str(tmp_path), "funding-backfill", "--venue", "binance", "--since", "2024-01-01",
                 "--dry-run", "BTC/USDT"]) == 0


def test_a_rate_that_cant_be_believed_is_left_out_counted_and_stays_a_hole(binance, tmp_path, monkeypatch):
    monkeypatch.setattr(binance, "funding_loader", _loader(_venue(odd={5: float("nan"), 6: 0.5})))
    _keep(binance, tmp_path, [(t, r) for t, r in _venue() if t >= KEPT_FROM])
    out = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path)
    assert out["added"] == 28 and out["refused"] == 2
    held = set(funding.rates(binance.name, "BTC/USDT", tmp_path).index)
    assert len(held) == 30 + 28
    assert pd.Timestamp(T0 + 5 * H8, unit="ms", tz="UTC") not in held
    assert pd.Timestamp(T0 + 6 * H8, unit="ms", tz="UTC") not in held


def test_an_instrument_with_no_rates_kept_is_left_to_the_collector(binance, tmp_path, monkeypatch):
    monkeypatch.setattr(binance, "funding_loader", _loader(_venue()))
    out = funding_backfill(binance, "BTC/USDT", pd.Timestamp("2024-01-01", tz="UTC"), tmp_path)
    assert out["added"] == 0 and out["kept"] == 0 and out["complete"]
    assert not funding._path(binance.name, "BTC/USDT", tmp_path).exists()


def test_the_command_prints_one_line_per_instrument_and_a_spot_venue_is_refused(binance, tmp_path, monkeypatch,
                                                                                capfd):
    monkeypatch.setattr(binance, "funding_loader", _loader(_venue()))
    _keep(binance, tmp_path, [(t, r) for t, r in _venue() if t >= KEPT_FROM])
    assert main(["--root", str(tmp_path), "funding-backfill", "--venue", "binance", "--since", "2024-01-01",
                 "BTC/USDT"]) == 0
    line = capfd.readouterr().out.strip().splitlines()[-1]
    assert "'pair': 'BTC/USDT'" in line and "'added': 30" in line and "'complete': True" in line
    assert main(["--root", str(tmp_path), "funding-backfill", "--venue", "kraken", "--since", "2024-01-01"]) == 2


def test_a_settlement_a_page_repeats_is_added_once_and_the_dry_run_agrees(binance, tmp_path, monkeypatch):
    """CR F270-1: a venue page that gives one settlement twice writes it once, the first rate given."""
    history = _venue()
    repeated = history[:3] + [(history[2][0], 0.0002)] + history[3:]
    monkeypatch.setattr(binance, "funding_loader", _loader(repeated))
    _keep(binance, tmp_path, [(t, r) for t, r in history if t >= KEPT_FROM])
    since = pd.Timestamp("2024-01-01", tz="UTC")
    dry = funding_backfill(binance, "BTC/USDT", since, tmp_path, dry_run=True)
    real = funding_backfill(binance, "BTC/USDT", since, tmp_path)
    assert dry["would_add"] == real["added"] == 30
    rows = json.loads(funding._path(binance.name, "BTC/USDT", tmp_path).read_text())["rates"]
    assert [t for t, _ in rows] == [t for t, _ in history]
    assert dict(rows)[history[2][0]] == history[2][1]
