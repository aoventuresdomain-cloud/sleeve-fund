"""B-43 (HoE; CR FOLLOW-UPs on B-36b; Advisor 20 Oct): one price tick source. The kept tick (markets.price_tick, B-36b) is
what the engine's liquidation rounds to, as the dashboard's does; the start check holds it equal to the live tick. The
live tick is read from the paper venue's own instrument listing, the venue the tiers come from, never another venue's.
When the live read fails an instrument takes the kept tick, said once, never 0.01; a live tick that differs wins, said.
Recorded responses only (tests/data/fee_live)."""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from sleeve_fund import venues
from sleeve_fund.strategies.base import LongFlatStrategy

pytestmark = [pytest.mark.no_maintenance_tables, pytest.mark.price_tick_live]
DATA = Path(__file__).parent / "data" / "fee_live"
PAPER = venues.venue("BINANCE")


def _listing(tick="0.10", status="Trading"):
    a = json.loads((DATA / "instruments-info-BTCUSDT.json").read_text())
    a["result"]["list"][0]["priceFilter"]["tickSize"] = tick
    a["result"]["list"][0]["status"] = status
    return a


def test_the_live_tick_is_read_from_the_paper_venues_listing_the_tiers_come_from():
    asked = []

    def get(url):
        asked.append(url)
        return _listing()

    assert venues.paper_venue_tick("BTC/USDT", get_json=get) == Decimal("0.1")
    (url,) = asked
    tiers_host = PAPER.risk_limit_url.split("/v5/")[0]
    assert url.startswith(tiers_host) and "instruments-info" in url and "category=linear" in url and "BTCUSDT" in url
    assert PAPER.live_tick is venues.paper_venue_tick  # the start check's read
    with pytest.raises(ValueError, match="not trading"):
        venues.paper_venue_tick("BTC/USDT", get_json=lambda url: _listing(status="Settling"))
    with pytest.raises(ValueError):
        venues.paper_venue_tick("BTC/USDT", get_json=lambda url: {"retCode": 10001, "retMsg": "params error"})


def _engine(increment: float):
    iid = SimpleNamespace(venue="BINANCE", symbol="BTCUSDT-PERP")
    return SimpleNamespace(instrument=SimpleNamespace(price_increment=increment, id=iid))


def test_the_engines_liquidation_rounds_to_the_kept_tick_whatever_its_instrument_carries():
    assert LongFlatStrategy._tick(_engine(0.01)) == 0.1  # an instrument from another venue's definition: the kept one
    assert LongFlatStrategy._tick(_engine(0.1)) == 0.1
    other = SimpleNamespace(instrument=SimpleNamespace(price_increment=0.01,
                                                       id=SimpleNamespace(venue="BINANCE", symbol="ETHUSDT-PERP")))
    assert LongFlatStrategy._tick(other) == 0.01  # no tick kept: its own


def test_an_instrument_whose_live_read_fails_takes_the_kept_tick_and_says_so_never_0_01(monkeypatch, capfd):
    def down(pair):
        raise OSError("venue unreachable")

    monkeypatch.setattr(PAPER, "live_tick", down)
    monkeypatch.setattr(PAPER, "contract", None)  # the contract limits unreachable too: nothing else gives a tick
    inst = PAPER.instrument("BTC", "USDT")
    assert inst.price_precision == 1 and float(inst.price_increment) == 0.1
    assert "kept price tick 0.1" in capfd.readouterr().err


def test_a_live_tick_that_differs_from_the_kept_one_wins_and_says_so(monkeypatch, capfd):
    monkeypatch.setattr(PAPER, "live_tick", lambda pair: Decimal("0.01"))
    monkeypatch.setattr(PAPER, "contract", lambda pair: {"price_precision": 1, "size_precision": 3})
    inst = PAPER.instrument("BTC", "USDT")
    assert inst.price_precision == 2 and float(inst.price_increment) == 0.01
    assert "live price tick 0.01" in capfd.readouterr().err


def test_a_start_whose_instrument_carries_another_tick_journals_the_kept_one_once():
    events = []
    store = SimpleNamespace(event=lambda sleeve, level, kind, message: events.append((sleeve, level, kind, message)))
    s = _engine(0.01)
    s.runtime = SimpleNamespace(store=store, name="p1")
    LongFlatStrategy._say_tick(s)
    same = _engine(0.1)
    same.runtime = s.runtime
    LongFlatStrategy._say_tick(same)  # its instrument carries the kept tick: nothing to say
    assert [(e[0], e[1], e[2]) for e in events] == [("p1", "warning", "price_tick_kept")]
    assert "0.1" in events[0][3] and "0.01" in events[0][3]


def test_a_paper_start_says_the_kept_tick_once_from_on_start():
    """CR2 (FOLLOW-UP, HoE): on_start itself says it (the Advisor's "said once per start"), not only the helper. The
    stand-in stops on_start just after, at its first clock read."""
    class Stop(Exception):
        pass

    events = []
    s = _engine(0.01)
    inst = s.instrument
    s.cache = SimpleNamespace(instrument=lambda iid: inst)
    s._cfg = SimpleNamespace(instrument_id="BTCUSDT-PERP.BINANCE")
    s._backtest, s.recorder = False, None
    s.runtime = SimpleNamespace(name="p1", pinned=False, store=SimpleNamespace(
        event=lambda sleeve, level, kind, message, **k: events.append(kind)))
    s.clock = SimpleNamespace(timestamp_ns=lambda: (_ for _ in ()).throw(Stop()))
    s._say_tick = lambda: LongFlatStrategy._say_tick(s)
    with pytest.raises(Stop):
        LongFlatStrategy.on_start(s)
    assert events == ["price_tick_kept"]
