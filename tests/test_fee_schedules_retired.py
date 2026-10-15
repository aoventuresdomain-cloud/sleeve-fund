"""FEE-LIVE (DA, head 3): fees are read only from venue_terms_fetch (FEE-LIVE storage). The old fee_schedules path is
retired: nothing writes it or reads it, so no venue's fees can come from a silent fallback (HoE: a fallback is a
labelled published row, alerted and on the start check). The table and its rows stay, never dropped."""

from decimal import Decimal

import pytest
from sqlalchemy import text

from sleeve_fund import fees, venues
from sleeve_fund.store import Store


@pytest.fixture
def store(tmp_path):
    return Store(f"sqlite:///{tmp_path}/t.db")


def test_the_store_has_no_old_fee_writer_or_reader(store):
    assert not hasattr(Store, "record_fees") and not hasattr(Store, "latest_fees")
    with store.engine.connect() as c:  # the table stays, with whatever it holds
        assert c.execute(text("SELECT count(*) FROM fee_schedules")).scalar() == 0


def test_the_supervisor_writes_no_fee_schedule_for_a_live_account(store, monkeypatch):
    from sleeve_fund.instruments import FeeSchedule
    from sleeve_fund.supervisor import Supervisor

    store.create_account("kraken-main", "live", venue="kraken")
    monkeypatch.setattr(venues.KRAKEN, "fetch_fees", lambda key, secret: FeeSchedule(Decimal("0.002"), Decimal("0.004")))
    monkeypatch.setattr(Supervisor, "check_venue_terms", lambda self, environ=None, **kw: None)
    Supervisor(store).check_fees(environ={"KRAKEN_API_KEY__KRAKEN_MAIN": "k1", "KRAKEN_API_SECRET__KRAKEN_MAIN": "s1"})
    with store.engine.connect() as c:
        assert c.execute(text("SELECT count(*) FROM fee_schedules")).scalar() == 0


def test_a_venue_without_published_fee_bounds_is_refused_never_charged_silently(monkeypatch):
    import dataclasses

    monkeypatch.setitem(venues.VENUES, "KRAKEN", dataclasses.replace(venues.KRAKEN, fee_bounds=None))
    with pytest.raises(ValueError, match="no published fee bounds"):
        fees.resolve("KRAKEN", None)
    assert all(p.fee_bounds for name, p in venues.VENUES.items() if name != "KRAKEN")  # every real venue has them
