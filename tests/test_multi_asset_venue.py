"""F-B3C-VENUE (HoE 15 Oct 02:05 UK; DA; HoQA b3c in qa3_b3): a multi-asset Strategy trades one venue, named once, and
its assets share that venue's quote currency. create_legs refuses an asset naming a venue of its own, a mixed quote, or
an unknown venue, before anything is written, rather than the Strategy failing at its start. Both engines."""

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select, text

from sleeve_fund.store import Store, make_engine, metadata, sleeves_t

PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/venue.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    s = Store(engine=eng)
    at = datetime.now(UTC) - timedelta(minutes=5)
    s.settle_flow(s.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    yield s
    metadata.drop_all(eng)


def _make(store, assets, venue="BINANCE", name="m"):
    return store.create_legs(name=name, strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", assets=list(assets),
                             params=dict(PERP), starting_balance=0, universe_date="2026-10-15", venue=venue)


def _no_rows(store):
    with store.engine.connect() as c:
        assert c.execute(select(func.count()).select_from(sleeves_t)).scalar() == 0


@pytest.mark.parametrize("assets", [
    ("BTC/USDT", "ETH/USD@KRAKEN"),  # HoQA b3c: a second venue named on an asset
    ("BTC/USDT", "ETH/USDT@KRAKEN"),  # the same quote, still another venue
    ("BTC/USDT", "ETH/USDT.KRAKEN"),  # an instrument id's venue suffix
    ("BTC/USDT", "ETH/USDT@BINANCE"),  # even its own venue: named once, on the Strategy
])
def test_an_asset_naming_a_venue_of_its_own_is_refused_at_create(store, assets):
    with pytest.raises(ValueError, match="one venue"):
        _make(store, assets)
    _no_rows(store)


def test_assets_with_mixed_quote_currencies_are_refused_at_create(store):
    with pytest.raises(ValueError, match="one quote currency"):
        _make(store, ("BTC/USDT", "ETH/USD", "SOL/USDT"))
    _no_rows(store)


def test_an_unknown_venue_is_refused_at_create(store):
    with pytest.raises(ValueError, match="unknown venue"):
        _make(store, ("BTC/USDT", "ETH/USDT"), venue="NOWHERE")
    _no_rows(store)


def test_one_venue_and_one_quote_is_made(store):
    assert [s.name for s in _make(store, ("BTC/USDT", "ETH/USDT", "SOL/USDT"))] == ["m", "m~ETH", "m~SOL"]
