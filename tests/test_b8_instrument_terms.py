"""B8 instrument terms (scripts/b8_instrument_terms.py): the verdicts on stub venue data, no network."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("b8_instrument_terms", ROOT / "scripts" / "b8_instrument_terms.py")
b8 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(b8)


def _symbol(sym, tick="0.10", step="0.001", min_qty="0.001", min_notional="100"):
    return {"symbol": sym, "contractType": "PERPETUAL", "status": "TRADING", "pricePrecision": 2,
            "quantityPrecision": 3,
            "filters": [{"filterType": "PRICE_FILTER", "tickSize": tick},
                        {"filterType": "LOT_SIZE", "stepSize": step, "minQty": min_qty},
                        {"filterType": "MIN_NOTIONAL", "notional": min_notional}]}


def _venue(symbols, funding):
    def get(url):
        return {"symbols": symbols} if url.endswith("/exchangeInfo") else funding
    return get


def _sleeve(tmp_path, name, instrument, venue=None, market="perp"):
    v = f'venue = "{venue}"\n' if venue else ""
    (tmp_path / f"{name}.toml").write_text(f'[sleeve]\nname = "{name}"\ninstrument = "{instrument}"\n{v}'
                                           f'[params]\nmarket = "{market}"\n')


BTC_FUNDING = [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.003", "fundingIntervalHours": 8}]


def test_the_instruments_come_from_the_configs_each_perpetual_once():
    got = b8.configured(str(ROOT / "configs" / "sleeves" / "*.toml"))
    assert ("BINANCE", "BTC/USDT") in got
    assert len(got) == len(set(got))
    assert all(v.isupper() for v, _ in got)


def test_a_spot_strategy_is_not_listed(tmp_path):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    _sleeve(tmp_path, "b", "ETH/USDT", "binance", market="spot")
    assert b8.configured(str(tmp_path / "*.toml")) == [("BINANCE", "BTC/USDT")]


def test_terms_that_match_pass_with_only_the_hand_checks_left(tmp_path, capfd):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    assert b8.main([], get=_venue([_symbol("BTCUSDT")], BTC_FUNDING), pattern=str(tmp_path / "*.toml")) == 0
    out = capfd.readouterr().out
    assert "MUST" not in out.replace("B8: PASS (0 MUST row(s))", "")
    assert out.count("FOLLOW-UP") == 2  # maintenance margin and half spread: one constant each, by hand
    assert "B8: PASS" in out


@pytest.mark.parametrize("tick", ["0.5", "0.25", "5"])
def test_a_tick_that_is_not_a_power_of_ten_is_a_must(tmp_path, capfd, tick):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    assert b8.main([], get=_venue([_symbol("BTCUSDT", tick=tick)], BTC_FUNDING), pattern=str(tmp_path / "*.toml")) == 1
    assert any(line.split()[1] == "tick" and line.endswith("MUST") for line in capfd.readouterr().out.splitlines()
               if line.startswith("BTC/USDT"))


def test_a_different_funding_interval_is_a_must(tmp_path, capfd):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    funding = [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.003", "fundingIntervalHours": 4}]
    assert b8.main([], get=_venue([_symbol("BTCUSDT")], funding), pattern=str(tmp_path / "*.toml")) == 1
    assert "funding interval (h)                4              8  MUST" in capfd.readouterr().out


def test_a_different_funding_cap_is_a_follow_up_not_a_must(tmp_path, capfd):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    funding = [{"symbol": "BTCUSDT", "adjustedFundingRateCap": "0.02", "fundingIntervalHours": 8}]
    assert b8.main([], get=_venue([_symbol("BTCUSDT")], funding), pattern=str(tmp_path / "*.toml")) == 0
    assert "funding cap                      0.02          0.003  FOLLOW-UP" in capfd.readouterr().out


def test_a_minimum_below_the_venues_is_a_must_and_above_it_a_follow_up():
    assert b8._floor_row("min notional", b8.Decimal("100"), b8.Decimal("5"))[3] == "MUST"
    assert b8._floor_row("min notional", b8.Decimal("5"), b8.Decimal("100"))[3] == "FOLLOW-UP"
    assert b8._floor_row("min notional", b8.Decimal("5"), None)[3] == "MUST"


def test_a_simulated_perp_and_an_extra_pair_are_both_reported(tmp_path, capfd):
    _sleeve(tmp_path, "a", "BTC/USD")  # the default venue: the simulated perp
    get = _venue([_symbol("BTCUSDT"), _symbol("ETHUSDT", tick="0.01")], BTC_FUNDING)
    assert b8.main(["eth/usdt"], get=get, pattern=str(tmp_path / "*.toml")) == 0
    out = capfd.readouterr().out
    assert "BTC/USD        (simulated perp)" in out and "FOLLOW-UP" in out
    assert "ETH/USDT       tick" in out
