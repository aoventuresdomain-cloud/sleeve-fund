"""Drop B MUST (HoE 15 Oct 01:58 UK; the guard is PE1's 5ea970aa; DA's cells, both engines): a reset is refused for a
multi-asset Strategy, from its parent or any asset leg, and changes nothing; a one-row Strategy still resets and a
two-leg Strategy keeps its own refusal. split_run puts away one row's run: a parent reset alone would move the parent's
journal and leave its legs' under the live Strategy, so asset_book's total would mix two runs. A reset across every
leg is a later item."""

import os

import pytest
from sqlalchemy import func, insert, select, text

from sleeve_fund.store import Store, make_engine, metadata, resets_t, sleeves_t

REFUSED = "isn't built for a (multi-asset|two-leg) Strategy yet, so nothing was changed"


@pytest.fixture
def store(tmp_path):
    eng = make_engine(os.environ.get("TEST_DATABASE_URL") or f"sqlite:///{tmp_path}/ma.db")
    with eng.begin() as c:
        c.execute(text("DROP TABLE IF EXISTS alembic_version"))
    metadata.drop_all(eng)
    yield Store(engine=eng)
    metadata.drop_all(eng)


def _multi_asset(store, name="tx", assets=("ETH", "SOL")):
    """A parent and its asset legs "<name>~<ASSET>" as 0028 keys them; written as rows, so the cell holds on a head
    without create_legs too."""
    store.create_sleeve(name=name, strategy="buy_and_hold", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000.0)
    with store.engine.begin() as c:
        row = dict(c.execute(select(sleeves_t).where(sleeves_t.c.name == name)).one()._mapping)
        row.pop("id")
        for a in assets:
            c.execute(insert(sleeves_t).values(**{**row, "name": f"{name}~{a}", "parent": name,
                                                  "instrument": f"{a}/USDT"}))


def _nothing_changed(store):
    with store.engine.connect() as c:
        assert c.execute(select(func.count()).select_from(resets_t)).scalar() == 0
    assert all(store.pending_reset(n) is None for n in ("tx", "tx~ETH", "tx~SOL"))


@pytest.mark.parametrize("name", ["tx", "tx~ETH", "tx~SOL"])
def test_a_multi_asset_strategy_is_never_reset_from_its_parent_or_a_leg(store, name):
    _multi_asset(store)
    with pytest.raises(ValueError, match=REFUSED):
        store.request_reset(name, "Settings changed")
    _nothing_changed(store)


def test_a_one_row_strategy_still_resets(store):
    store.create_sleeve(name="solo", strategy="buy_and_hold", instrument="BTC/USDT",
                        bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=10_000.0)
    store.request_reset("solo", "Settings changed")
    assert store.pending_reset("solo") is not None


def test_a_two_leg_strategy_stays_a_pair_and_keeps_its_own_refusal(store):
    """Its "~2" row is no asset leg (Store._asset_legs leaves it out): the pair is never read as multi-asset, and its
    reset is refused in the two-leg words."""
    store.create_two_legs(name="carry", strategy="trend_filter", bar_spec="1-HOUR-LAST-INTERNAL",
                          starting_balance=10_000,
                          first={"instrument": "BTC/USD", "params": {}, "venue": "KRAKEN", "side": 1},
                          second={"instrument": "BTC/USD", "params": {"market": "perp"}, "venue": "KRAKEN", "side": -1})
    assert not store.multi_asset("carry") and not store.multi_asset("carry~2")
    assert store.legs("carry") == ["carry", "carry~2"]
    for name in ("carry", "carry~2"):
        with pytest.raises(ValueError, match="isn't built for a two-leg Strategy yet"):
            store.request_reset(name, "Settings changed")
