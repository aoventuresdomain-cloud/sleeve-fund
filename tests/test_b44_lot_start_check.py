"""B-44 (HoE 23:19 UK, option ii; Advisor 23:21 UK): paper's instrument is the hub's (the data venue's definition), so
the paper venue's listed lot never reaches paper sizing. The start check FAILs, fail-closed as the price tick does, a
perp Strategy whose hub step or minimum differs from the paper venue's listing, or whose listed step the engine can't
round to, naming the symbol and both lots. Recorded responses only (tests/data/fee_live/instruments-info-BTCUSDT.json)."""
from __future__ import annotations

from decimal import Decimal as D

import pytest

from sleeve_fund.store import Store
from sleeve_fund.venues import paper_venue_lot
from test_b44_lot_size import listing

pytestmark = [pytest.mark.no_maintenance_tables]
PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


def _store(params=PERP):
    store = Store.in_memory()
    store.create_sleeve(name="p1", strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, venue="BINANCE", params=params)
    return store


def _check(store, hub=(D("0.001"), D("0.001")), step=None, least=None, paper=None):
    from sleeve_fund import start_check

    return start_check.check_lot_sizes(
        store, hub_lot=lambda venue, pair: hub,
        paper_lot=paper or (lambda venue, pair: paper_venue_lot(pair, get_json=listing(step, least))))


def _fails(sec):
    return [w for k, w in sec.lines if k == "FAIL"]


def test_the_hubs_lot_matching_the_listing_passes():
    sec = _check(_store())
    assert not sec.failed and any(k == "PASS" for k, _ in sec.lines), sec.lines


def test_a_listing_coarser_than_the_hubs_instrument_fails_naming_the_symbol_and_both_lots():
    sec = _check(_store(), step="0.01", least="0.01")
    (line,) = _fails(sec)
    assert sec.failed and "p1" in line and "BTC/USDT" in line and "0.01" in line and "0.001" in line, line


def test_a_differing_minimum_alone_fails():
    sec = _check(_store(), least="0.002")
    assert sec.failed and "0.002" in _fails(sec)[0]


def test_a_listed_step_the_engine_cant_round_to_fails():
    sec = _check(_store(), step="0.005", least="0.005")
    (line,) = _fails(sec)
    assert sec.failed and "0.005" in line and "power of ten" in line, line


def test_an_unreadable_listing_or_hub_contract_fails_never_an_assumed_lot():
    def down(venue, pair):
        raise OSError("no route")

    assert _check(_store(), paper=down).failed
    from sleeve_fund import start_check

    sec = start_check.check_lot_sizes(_store(), hub_lot=down,
                                      paper_lot=lambda venue, pair: paper_venue_lot(pair, get_json=listing()))
    assert sec.failed and "couldn't be read" in _fails(sec)[0]


def test_a_spot_strategy_is_not_checked():
    from sleeve_fund import start_check

    sec = start_check.check_lot_sizes(_store({"market": "spot"}), hub_lot=lambda v, p: pytest.fail("no read"),
                                      paper_lot=lambda v, p: pytest.fail("no read"))
    assert not sec.failed


def test_the_start_check_runs_it():
    import inspect

    from sleeve_fund import start_check

    assert "check_lot_sizes(store)" in inspect.getsource(start_check.run)


def test_the_hubs_lot_is_the_data_venues_contract(monkeypatch):
    from sleeve_fund import start_check
    from sleeve_fund.venues import venue

    monkeypatch.setattr(venue("BINANCE"), "contract",
                        lambda pair: {"price_precision": 1, "size_precision": 3, "min_quantity": 0.002})
    assert start_check._hub_lot("BINANCE", "BTC/USDT") == (D("0.001"), D("0.002"))


def test_cr2_on_start_calls_say_lot():
    """CR2 B-44 MUST 2: dropping on_start's _say_lot call goes red. The stand-in stops at on_start's first clock read
    (as B-43's on_start cell)."""
    from types import SimpleNamespace

    from sleeve_fund.strategies.base import LongFlatStrategy

    class Stop(Exception):
        pass

    said = []
    s = SimpleNamespace(instrument=object(), _backtest=False, recorder=None, _say_tick=lambda: None,
                        _say_lot=lambda: said.append(1))
    s.cache = SimpleNamespace(instrument=lambda iid: s.instrument)
    s._cfg = SimpleNamespace(instrument_id="BTCUSDT-PERP.BINANCE")
    s.runtime = SimpleNamespace(name="p1", pinned=False, store=SimpleNamespace(event=lambda *a, **k: None))
    s.clock = SimpleNamespace(timestamp_ns=lambda: (_ for _ in ()).throw(Stop()))
    with pytest.raises(Stop):
        LongFlatStrategy.on_start(s)
    assert said == [1]
