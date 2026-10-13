"""ORPHAN-GUARD's pure parts (paper.orphan_guard); HoQA's cells are test_orphan_guard_qa."""
from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.paper.orphan_guard import is_overdue, overdue_at, resting_entry_order

CLOSE, BAR = datetime(2026, 10, 8, 9, 0, tzinfo=UTC), timedelta(hours=1)


def test_og4_a_resting_entry_is_overdue_only_after_its_next_close_plus_two_minutes():
    due = overdue_at(CLOSE, BAR)
    assert due == CLOSE + BAR + timedelta(minutes=2)
    assert not is_overdue(CLOSE, BAR, due) and is_overdue(CLOSE, BAR, due + timedelta(microseconds=1))


def test_og1_a_bar_of_no_length_is_refused():
    with pytest.raises(ValueError):
        resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(0))


def test_og1_a_short_bars_expiry_is_clamped_past_the_venues_minimum_lead():
    """Advisor 13 Oct: expiry = max(the next close, now + the venue's minimum lead + 30 s)."""
    lead = timedelta(minutes=10)
    now = CLOSE + timedelta(seconds=2)
    _, one_minute, _ = resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(minutes=1),
                                           now=now, min_lead=lead)
    assert one_minute == now + lead + timedelta(seconds=30)
    _, four_hours, _ = resting_entry_order(gtd_supported=True, decision_close=CLOSE, bar=timedelta(hours=4),
                                           now=now, min_lead=lead)
    assert four_hours == CLOSE + timedelta(hours=4)


@pytest.mark.parametrize("venue,minutes,expect", [
    ("BINANCE", 60, "close"),  # an hour's bar (1-HOUR): good till its next close
    ("BINANCE", 1, "lead"),  # a minute's bar: clamped past the venue's 600 s minimum lead
    ("NOWHERE", 60, "gtc"),  # no venue profile: good-till-cancelled, with the backstop note
])
def test_og1_the_engines_resting_entry_takes_its_venues_terms(venue, minutes, expect):
    from types import SimpleNamespace

    from nautilus_trader.model import InstrumentId

    from sleeve_fund.strategies.base import LongFlatStrategy
    from sleeve_fund.venues import BINANCE_GTD_MIN_LEAD

    now = CLOSE + timedelta(seconds=3)
    fake = SimpleNamespace(
        _cfg=SimpleNamespace(instrument_id=InstrumentId.from_str(f"BTCUSDT-PERP.{venue}"),
                             bar_type=f"BTCUSDT-PERP.{venue}-" + ("1-HOUR" if minutes == 60 else f"{minutes}-MINUTE") + "-LAST-INTERNAL"),
        _deciding=(int(CLOSE.timestamp() * 1e9), 0), clock=SimpleNamespace(timestamp_ns=lambda: int(now.timestamp() * 1e9)))
    tif, expires, note = LongFlatStrategy._resting_entry_terms(fake)
    if expect == "gtc":
        assert tif.name == "GTC" and expires is None and "backstop" in note
    else:
        assert tif.name == "GTD" and note is None
        assert expires == (CLOSE + timedelta(minutes=minutes) if expect == "close" else
                           now + BINANCE_GTD_MIN_LEAD + timedelta(seconds=30))
