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
        _gtd_refused=None, _deciding=(int(CLOSE.timestamp() * 1e9), 0), clock=SimpleNamespace(timestamp_ns=lambda: int(now.timestamp() * 1e9)))
    tif, expires, note = LongFlatStrategy._resting_entry_terms(fake)
    if expect == "gtc":
        assert tif.name == "GTC" and expires is None and "backstop" in note
    else:
        assert tif.name == "GTD" and note is None
        assert expires == (CLOSE + timedelta(minutes=minutes) if expect == "close" else
                           now + BINANCE_GTD_MIN_LEAD + timedelta(seconds=30))


@pytest.mark.parametrize("gated", [True, False])
def test_og2_the_supervisor_marks_a_crashed_or_stopped_owner_down_only_where_the_gate_is_in_force(tmp_path, gated):
    """HoE 13 Oct (the pair): a crash seen by the supervisor, or its own stop, hands the strategy's resting entries to
    the supervisor's orphan pass (owner_down); a journal without the portfolio gate has no gated entries to hand."""
    from sleeve_fund import supervisor as sup
    from sleeve_fund.paper.orphan_guard import down_since
    from sleeve_fund.store import Store

    class Dead:
        pid, returncode = 4242, 1
        poll = lambda self: 1  # noqa: E731

    class Alive(Dead):
        returncode = None
        poll = lambda self: None  # noqa: E731
        send_signal = wait = kill = lambda self, *a, **k: 0  # noqa: E731

    store = Store(f"sqlite:///{tmp_path}/sv.db", portfolio_gate=gated)
    for name in ("crashes", "stops"):
        store.create_sleeve(name=name, strategy="trend_filter", instrument="BTC/USD",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=1000)
    sv = sup.Supervisor(store)
    sv.procs["crashes"] = sup.Proc(popen=Dead())
    sv.step()
    sv._stop("stops", sup.Proc(popen=Alive()), "stopped by PM")
    sv._stop("stops", sup.Proc(), "stopped by PM")  # nothing running: no second hand-over
    for name in ("crashes", "stops"):
        assert (down_since(store, name) is not None) is gated, name
    downs = [e for e in store.events("stops", limit=100) if e["kind"] == "owner_down"]
    assert len(downs) == (1 if gated else 0)


@pytest.mark.parametrize("reason, refused", [
    ("Order would immediately expire: GTD time too early", True), ("goodTillDate must be at least 600s ahead", True),
    ("Post only order will be rejected", False), ("insufficient margin", False)])
def test_og1_a_venue_refusing_the_expiry_is_told_apart_from_other_refusals(reason, refused):
    from types import SimpleNamespace

    from nautilus_trader.model import TimeInForce

    from sleeve_fund.strategies.base import LongFlatStrategy

    gtd, gtc = SimpleNamespace(time_in_force=TimeInForce.GTD), SimpleNamespace(time_in_force=TimeInForce.GTC)
    assert LongFlatStrategy._gtd_refusal(gtd, SimpleNamespace(reason=reason)) is refused
    assert LongFlatStrategy._gtd_refusal(gtc, SimpleNamespace(reason=reason)) is False  # only a good-till-date one
    assert LongFlatStrategy._gtd_refusal(None, SimpleNamespace(reason=reason)) is False


def test_og1_after_a_refused_expiry_the_entries_rest_good_till_cancelled_and_say_why():
    """Advisor 13 Oct: never refuse resting entries for the expiry; the next goes good-till-cancelled with the
    backstop note and the venue's reason."""
    from types import SimpleNamespace

    from nautilus_trader.model import InstrumentId

    from sleeve_fund.strategies.base import LongFlatStrategy

    fake = SimpleNamespace(_cfg=SimpleNamespace(instrument_id=InstrumentId.from_str("BTCUSDT-PERP.BINANCE"),
                                                bar_type="BTCUSDT-PERP.BINANCE-1-MINUTE-LAST-INTERNAL"),
                           _gtd_refused="GTD time too early", _deciding=(int(CLOSE.timestamp() * 1e9), 0),
                           clock=SimpleNamespace(timestamp_ns=lambda: int(CLOSE.timestamp() * 1e9)))
    tif, expires, note = LongFlatStrategy._resting_entry_terms(fake)
    assert tif.name == "GTC" and expires is None and "backstop" in note and "GTD time too early" in note
