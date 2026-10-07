"""Funding on a quiet market (DA 7 Oct, on #146): paper holds a settlement back while no trade has reached the strategy
for UNSEEN_GAP_NS, since unseen minutes may show the venue's stop closed the position first (QA P1-L19). On a market
that trades every 30 s that is most of the time, and in step with the tick it is every time: the settlement was never
charged. The hold now ends by a hard deadline: FUNDING_DEFER_MAX after the settlement, once a trade from after it has
arrived. The run goes on to 08:30, past the 08:15 deadline."""

import pandas as pd
import pytest

import test_hub_146_qa as qa
from test_hub_146_qa import _probe, flat_prices, paper  # noqa: F401 - _probe registers the probe strategy

SETTLED = pd.Timestamp("2025-10-03 08:00", tz="UTC")


def _quiet(monkeypatch, every: int, side: int, offset: int = 0):
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now, *_: 0.0001)  # #155 passes the wait too
    p = flat_prices(40)
    quiet = frozenset(s for s in range(len(p)) if (s - offset) % every)  # one trade every `every` seconds, nothing else
    return paper(p, side=side, perp=True, profile="aggressive", leave=35, gone=quiet)


@pytest.mark.parametrize("offset", [0, 10, 20, 25])
@pytest.mark.parametrize("every", [30, 45, 60])
@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_quiet_market_is_charged_its_settlement_by_the_deadline(side, every, offset, monkeypatch):
    from sleeve_fund.strategies.base import LongFlatStrategy

    assert LongFlatStrategy.FUNDING_DEFER_MAX == pd.Timedelta(minutes=15)
    run = _quiet(monkeypatch, every, side, offset)
    assert [r for r in run.sequence() if r[0] == "entry"], ("set-up: the position was held", run.sequence())
    rows = [r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED]
    assert len(rows) == 1, run.store.funding("q146")
    assert [e for e in run.events if e["kind"] == "funding"]
    assert rows[0]["amount"] == pytest.approx(-rows[0]["qty"] * rows[0]["price"] * 0.0001)


def test_a_market_trading_every_second_is_charged_as_before(monkeypatch):
    run = _quiet(monkeypatch, 1, 1)
    assert len([r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED]) == 1


def test_without_a_trade_after_the_settlement_it_stays_held_however_long(monkeypatch):
    """The feed away from 07:58 to the end: the minutes may yet show the stop went first, so nothing is charged."""
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now, *_: 0.0001)  # #155 passes the wait too
    p = flat_prices(40)
    run = paper(p, side=1, perp=True, profile="aggressive", leave=35, gone=frozenset(range(8 * 60, len(p))))
    assert [r for r in run.sequence() if r[0] == "entry"], run.sequence()
    assert [r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED] == []


# ============================================ a refill later than the deadline (Advisor 7 Oct 00:40, QA pins 3 and 4)


def _late_refill(monkeypatch, *, side, stop_in_outage: bool):
    """Hub away 07:56-08:02 with the trades back at 08:02, but the REST refill of 07:57-08:02 only lands at 08:20,
    after the 08:15 deadline charged 08:00. With stop_in_outage the refill shows the venue's stop closed the position
    in the minute to 07:58, before the settlement."""
    from test_hub_146_qa import M, adverse, shape

    from sleeve_fund.strategies.base import LongFlatStrategy

    start = int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value)
    monkeypatch.setattr(qa, "START", start)
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now, *_: 0.0001)  # #155 passes the wait too
    p = flat_prices(40)
    if stop_in_outage:
        p = shape(p, 7.5, 7 + 50 / 60, adverse(side, 0.02))  # 07:57:30-07:57:50 through the 1 % stop
    return paper(p, side=side, perp=True, profile="aggressive", leave=35, gone=frozenset(range(6 * 60, 12 * 60)),
                 away=(start + 6 * M, start + 12 * M), back_at=start + 30 * M, refill_after_live=True)


def _at_settlement(run):
    return [r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED]


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_refill_after_the_deadline_showing_a_stop_before_it_reverses_the_charge(side, monkeypatch):
    """The 08:00 row stays as booked; a separate correcting row of the opposite amount nets it to zero, and the
    journal says funding_charged_while_flat with the amount."""
    from test_hub_146_qa import first_exit

    run = _late_refill(monkeypatch, side=side, stop_in_outage=True)
    assert first_exit(run.sequence())[0] == "stop_loss", run.sequence()
    rows = _at_settlement(run)
    assert len(rows) == 2, rows
    charged, correction = sorted(rows, key=lambda r: r["id"])
    assert charged["amount"] == pytest.approx(-charged["qty"] * charged["price"] * 0.0001)
    assert correction["amount"] == pytest.approx(-charged["amount"])
    assert (correction["qty"], correction["price"], correction["rate"]) == (
        charged["qty"], charged["price"], charged["rate"])
    flat = run.kinds("funding_charged_while_flat")
    assert len(flat) == 1 and flat[0]["level"] == "warning", flat
    assert f"{abs(charged['amount']):,.2f}" in flat[0]["message"], flat[0]["message"]


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_refill_after_the_deadline_that_leaves_the_position_open_never_charges_twice(side, monkeypatch):
    run = _late_refill(monkeypatch, side=side, stop_in_outage=False)
    rows = _at_settlement(run)
    assert len(rows) == 1, rows
    assert rows[0]["amount"] == pytest.approx(-rows[0]["qty"] * rows[0]["price"] * 0.0001)
    assert run.kinds("funding_charged_while_flat") == []
