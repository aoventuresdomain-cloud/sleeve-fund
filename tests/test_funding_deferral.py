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
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now: 0.0001)
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
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", lambda self, terms, ts, now: 0.0001)
    p = flat_prices(40)
    run = paper(p, side=1, perp=True, profile="aggressive", leave=35, gone=frozenset(range(8 * 60, len(p))))
    assert [r for r in run.sequence() if r[0] == "entry"], run.sequence()
    assert [r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED] == []
