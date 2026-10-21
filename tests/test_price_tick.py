"""B-36b (HoE, 20 Oct): the price tick per instrument is kept beside its VS-1 tiers (BTC/USDT 0.1), read through one
shared reader, markets.price_tick(venue, pair), which says None when no tick is kept. The start check FAILs, in words,
a perp Strategy whose venue's live contract tick differs from the kept one, or whose tick isn't kept or can't be
read: the same pattern as VS-1's maintenance tables. Recorded responses only."""
from __future__ import annotations

from decimal import Decimal

import pytest

from sleeve_fund.store import Store

pytestmark = [pytest.mark.no_maintenance_tables]
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


def test_the_kept_tick_is_read_through_one_reader_and_none_where_not_kept():
    from sleeve_fund.markets import price_tick

    assert price_tick("BINANCE", "BTC/USDT") == Decimal("0.1")
    assert price_tick("BINANCE", "BTCUSDT-PERP") == Decimal("0.1")  # the venue's own symbol too, as maintenance_for
    assert price_tick("BINANCE", "SUI/USDT") is None  # TEC-TIERS3-B: was ETH/USDT
    assert price_tick("KRAKEN", "BTC/USD") is None  # a spot venue keeps no contract tick


def test_the_venues_live_tick_comes_from_its_recorded_contract_listing():
    from test_b43_one_tick_source import _listing

    from sleeve_fund.venues import paper_venue_tick  # B-43: the paper venue's listing, not another venue's

    assert paper_venue_tick("BTC/USDT", get_json=lambda url: _listing("0.10")) == Decimal("0.1")
    assert paper_venue_tick("BTC/USDT", get_json=lambda url: _listing("0.50")) == Decimal("0.5")


def _store(pair="BTC/USDT", params=PERP):
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument=pair, bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, venue="BINANCE", params=params)
    return store


def _check(store, live):
    from sleeve_fund import start_check

    return start_check.check_price_ticks(store, live_tick=live)


def test_the_start_check_passes_a_perp_whose_live_tick_is_the_kept_one():
    sec = _check(_store(), lambda venue, pair: Decimal("0.10"))
    assert not sec.failed and any(k == "PASS" for k, _ in sec.lines), sec.lines


def test_the_start_check_fails_in_words_a_perp_whose_live_tick_differs():
    sec = _check(_store(), lambda venue, pair: Decimal("0.5"))
    assert sec.failed
    (line,) = [w for k, w in sec.lines if k == "FAIL"]
    assert "p1" in line and "BTC/USDT" in line and "0.5" in line and "0.1" in line, line


def test_the_start_check_fails_a_perp_whose_tick_is_not_kept_or_cant_be_read():
    sec = _check(_store("SUI/USDT"), lambda venue, pair: Decimal("0.01"))
    assert sec.failed and any("SUI/USDT" in w and "no price tick kept" in w for k, w in sec.lines if k == "FAIL")

    def unreachable(venue, pair):
        raise OSError("no route")

    sec = _check(_store(), unreachable)
    assert sec.failed and any("couldn't be read" in w for k, w in sec.lines if k == "FAIL"), sec.lines


def test_a_spot_strategy_is_not_checked():
    sec = _check(_store(params={"market": "spot"}), lambda venue, pair: pytest.fail("no contract read for spot"))
    assert not sec.failed


def test_the_start_check_runs_it():
    import inspect

    from sleeve_fund import start_check

    assert "check_price_ticks(store)" in inspect.getsource(start_check.run)  # the report includes the section
