"""B8 instrument terms (scripts/b8_instrument_terms.py): the verdicts on stub venue data, no network."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

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


@pytest.fixture(autouse=True)
def contract(monkeypatch):
    """Paper's instrument for a pair: the venue's contract answered offline, as the stub venue publishes it (tick 0.1,
    lot 0.001, minimum 0.001 and 100). A cell sets `contract.limits` to a dict, or to an exception to raise."""
    from sleeve_fund import venues

    state = SimpleNamespace(limits={"price_precision": 1, "size_precision": 3, "min_quantity": 0.001,
                                    "min_notional": 100.0})

    def answer(pair):
        if isinstance(state.limits, Exception):
            raise state.limits
        return state.limits

    monkeypatch.setattr(venues.VENUES["BINANCE"], "contract", answer)
    return state


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
    assert not [line for line in out.splitlines() if line.endswith("MUST")]
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


def test_the_engine_side_is_the_instrument_paper_builds_so_a_contract_that_fails_shows_its_fallback(tmp_path, capfd,
                                                                                                   contract):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    contract.limits = OSError("venue unreachable")  # paper then builds the cautious fallback
    venue_terms = _venue([_symbol("BTCUSDT", min_qty="0.01")], BTC_FUNDING)
    assert b8.main([], get=venue_terms, pattern=str(tmp_path / "*.toml")) == 1
    got = {line[15:37].strip(): line for line in capfd.readouterr().out.splitlines()
           if line.startswith("BTC/USDT")}
    assert got["tick"].endswith("MUST")  # the fallback's 2-place tick, 0.01, is finer than the venue's 0.1
    assert got["min quantity"].endswith("MUST")  # no minimum given: paper's is one lot, 0.001, under the venue's 0.01
    assert got["min notional"].endswith("MUST")  # 5 under the venue's 100
    assert got["lot step"].endswith("PASS")  # the fallback caps the size at 3 places: 0.001, the venue's own


def test_a_finer_lot_than_the_venues_is_a_must(tmp_path, capfd, contract):
    _sleeve(tmp_path, "a", "BTC/USDT", "binance")
    contract.limits = {"price_precision": 1, "size_precision": 4, "min_quantity": 0.001, "min_notional": 100.0}
    assert b8.main([], get=_venue([_symbol("BTCUSDT")], BTC_FUNDING), pattern=str(tmp_path / "*.toml")) == 1
    assert any(line.startswith("BTC/USDT") and "lot step" in line and line.endswith("MUST")
               for line in capfd.readouterr().out.splitlines())


def test_nothing_compared_is_not_a_pass(tmp_path, capfd):
    _sleeve(tmp_path, "a", "BTC/USD")  # only the simulated perp: no venue with published terms
    assert b8.main([], get=_venue([], []), pattern=str(tmp_path / "*.toml")) == 1
    out = capfd.readouterr().out
    assert "B8: NOTHING CHECKED" in out and "B8: PASS" not in out
    assert b8.main([], get=_venue([], []), pattern=str(tmp_path / "none-*.toml")) == 1  # no configs at all
