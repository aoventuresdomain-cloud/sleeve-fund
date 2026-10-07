"""Funding on a quiet market (DA 7 Oct, on #146): paper holds a settlement back while no trade has reached the strategy
for UNSEEN_GAP_NS, since unseen minutes may show the venue's stop closed the position first (QA P1-L19). On a market
that trades every 30 s that is most of the time, and in step with the tick it is every time: the settlement was never
charged. The hold now ends by a hard deadline: FUNDING_DEFER_MAX after the settlement, once a trade from after it has
arrived. The run goes on to 08:30, past the 08:15 deadline."""

import pandas as pd
import pytest

import test_hub_146_qa as qa
from test_hub_146_full_round import _funding_rate_stub  # either _funding_rate signature (#163 adds held=)
from test_hub_146_qa import _probe, flat_prices, paper  # noqa: F401 - _probe registers the probe strategy

SETTLED = pd.Timestamp("2025-10-03 08:00", tz="UTC")


def _quiet(monkeypatch, every: int, side: int, offset: int = 0):
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, 0.0001))
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
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, 0.0001))
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
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, 0.0001))
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
    # The replayed close is journaled on the exit's order, for a restart before the next booking (CR #179)
    closes = [o["signal"].get("replayed_close") for o in run.orders if (o.get("signal") or {}).get("replayed_close")]
    assert closes and closes[0] < SETTLED.isoformat(), [o["signal"] for o in run.orders]


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_reversed_deadline_charge_is_its_own_reversal_row_and_is_never_trued_up(side, monkeypatch):
    """The correcting row is kind "reversal", so a true-up never reads it as a settlement to correct, and reversing
    a charge raises no funding outage (CR-3, M-1 on #163)."""
    run = _late_refill(monkeypatch, side=side, stop_in_outage=True)
    charged, correction = sorted(_at_settlement(run), key=lambda r: r["id"])
    assert (charged["kind"], correction["kind"]) == ("settled", "reversal"), (charged, correction)
    assert [r for r in run.store.funding("q146") if r["kind"] == "true_up"] == []
    assert run.store.events_of(("funding_missing", "funding_stale"), limit=1000) == []  # global alerts


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_refill_after_the_deadline_that_leaves_the_position_open_never_charges_twice(side, monkeypatch):
    run = _late_refill(monkeypatch, side=side, stop_in_outage=False)
    rows = _at_settlement(run)
    assert len(rows) == 1, rows
    assert rows[0]["amount"] == pytest.approx(-rows[0]["qty"] * rows[0]["price"] * 0.0001)
    assert run.kinds("funding_charged_while_flat") == []


# ================================================= CR #179: a later settlement keeps its own deadline; a restart


def _booked_at(monkeypatch) -> list:
    """(strategy clock, settlement time) at each funding booking."""
    from sleeve_fund.store import Store
    from sleeve_fund.strategies.base import LongFlatStrategy

    seen, live = [], {}
    on_start, record = LongFlatStrategy.on_start, Store.record_funding

    def started(self):
        live["s"] = self
        return on_start(self)

    def booking(self, *a, **kw):
        seen.append((pd.Timestamp(live["s"].clock.utc_now()), pd.Timestamp(kw["ts"])))
        return record(self, *a, **kw)

    monkeypatch.setattr(LongFlatStrategy, "on_start", started)
    monkeypatch.setattr(Store, "record_funding", booking)
    return seen


def test_a_settlement_younger_than_the_deadline_is_not_booked_with_an_older_one(monkeypatch):
    """Hourly settlements; trades at :10 and :40 of each minute, none 07:59-09:00. At 09:00:30 the 08:00 settlement is
    past its deadline and booked; 09:00 is 30 s old and waits for its own, 09:15."""
    import dataclasses

    from sleeve_fund import markets
    from sleeve_fund.strategies.base import LongFlatStrategy

    monkeypatch.setattr(markets, "LOW_FEE_PERP", dataclasses.replace(markets.LOW_FEE_PERP, funding_hours=tuple(range(24))))
    monkeypatch.setattr(qa, "START", int(pd.Timestamp("2025-10-03 07:50", tz="UTC").value))
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, 0.0001))
    booked = _booked_at(monkeypatch)
    n = 90
    p = flat_prices(n)
    gone = frozenset(s for s in range(n * 60) if s % 60 not in (10, 40)) | frozenset(range(9 * 60, 70 * 60))
    run = paper(p, side=1, perp=True, profile="aggressive", leave=85, gone=gone)
    assert [r for r in run.sequence() if r[0] == "entry"], run.sequence()
    nine = SETTLED + pd.Timedelta(hours=1)
    assert sorted(t for _, t in booked) == [SETTLED, nine], booked
    at = dict((t, c) for c, t in booked)
    assert at[SETTLED] <= nine + pd.Timedelta(minutes=2), booked
    assert nine + pd.Timedelta(minutes=15) <= at[nine] <= nine + pd.Timedelta(minutes=16), booked
    said = [e["message"] for e in run.kinds("funding_deadline_booked")]
    assert any(m.startswith("The 09:00 settlement was booked 15 minutes after it") for m in said), said


@pytest.mark.parametrize("replayed", [True, False], ids=["replayed-close", "closed-after"])
def test_after_a_restart_a_settlement_after_the_replayed_close_is_not_charged(replayed, monkeypatch):
    """The previous process held a long from 07:55; the replay of an outage found the venue's stop closed it at 07:58,
    and our exit was journaled at 08:05, before 08:00 was booked; then it stopped. The new process (from 08:06, trades
    from 08:08) reads the replayed close from the exit's journaled order, so 08:00 is not charged. The control: an exit at 08:05 with no replay
    behind it charges the long held at 08:00, once."""
    from datetime import datetime, timezone

    from sleeve_fund.store import Store
    from sleeve_fund.strategies.base import LongFlatStrategy

    start = int(pd.Timestamp("2025-10-03 08:06", tz="UTC").value)  # the new process
    monkeypatch.setattr(qa, "START", start)
    monkeypatch.setattr(LongFlatStrategy, "_funding_rate", _funding_rate_stub(LongFlatStrategy, 0.0001))
    p = flat_prices(20)
    at = lambda hhmm: datetime.fromisoformat(f"2025-10-03T{hhmm}:00+00:00")  # noqa: E731
    create = Store.create_sleeve

    def create_and_seed(self, *a, **kw):
        out = create(self, *a, **kw)
        px = float(p[0])
        self.record_order("q146", order_id="O-in", side="BUY", qty=0.05, intent="entry", reason="held", ts=at("07:55"),
                          signal={"stop_frac": 0.01})
        self.record_fill("q146", side="BUY", qty=0.05, price=px, fee=0.0, order_id="O-in", trade_id="T-in",
                         ts=at("07:55"))
        self.record_funding("q146", qty=0.05, price=px, rate=0.0001, amount=round(-0.05 * px * 0.0001, 8),
                            ts=datetime(2025, 10, 3, 0, 0, tzinfo=timezone.utc))  # the last settlement booked
        sig = {"price_source": "replay_model", "replayed_close": at("07:58").isoformat()} if replayed else {}
        self.record_order("q146", order_id="O-out", side="SELL", qty=0.05, intent="stop_loss", reason="stop",
                          ts=at("08:05"), signal=sig)
        self.record_fill("q146", side="SELL", qty=0.05, price=px * 0.98, fee=0.0, order_id="O-out", trade_id="T-out",
                         ts=at("08:05"))
        return out

    monkeypatch.setattr(Store, "create_sleeve", create_and_seed)
    run = paper(p, side=1, enter=10**6, perp=True, profile="aggressive", leave=10**6,
                gone=frozenset(range(0, 2 * 60)), heartbeat=start - 30 * qa.S)
    rows = [r for r in run.store.funding("q146") if pd.Timestamp(r["ts"]) == SETTLED]
    if replayed:
        assert rows == [], rows
    else:
        assert len(rows) == 1 and rows[0]["qty"] == pytest.approx(0.05), rows
