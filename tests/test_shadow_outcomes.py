"""Shadow outcomes (multi-asset spec §5; Advisor MA-SPEC, settled 15 Oct ~00:10 UK). A trim's outcome is the real
trade's per-unit P&L times what the trim held back (asked - approved), no replay. A refusal's is a replay of its own
planned stop and target on the leg's bars from its decision, at the asked size, net of fees both ways: marked
hypothetical, and it never touches the leg's book. Written once (Store.record_shadow_outcome)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal as D

import pandas as pd
import pytest
from sqlalchemy import insert, select, update

from sleeve_fund import shadow
from sleeve_fund.store import Store, gate_decisions_t, sleeves_t

T0 = datetime(2026, 10, 15, 9, tzinfo=UTC)


def _decision(store, sleeve, outcome, asked, approved, *, side=1, stop=None, target=None, price=100, order=None,
              at=T0):
    with store.engine.begin() as c:
        sid = c.execute(select(sleeves_t.c.id).where(sleeves_t.c.name == sleeve)).scalar()
        seq = (c.execute(select(gate_decisions_t.c.seq).order_by(gate_decisions_t.c.seq.desc())).scalar() or 0) + 1
        return c.execute(insert(gate_decisions_t).values(
            seq=seq, sleeve_id=sid, bar_ts=at, intent_id=f"i{seq}", intent="entry", stage="submit", underlying="ETH",
            outcome=outcome, test_hit="asset_stop_loss", compliance_version=1, profile_version=1,
            requested_qty=D(asked), approved_qty=D(approved), price=D(price), decided_at=at, side=side,
            planned_stop=None if stop is None else D(stop), planned_target=None if target is None else D(target),
            order_id=order)).inserted_primary_key[0]


@pytest.fixture
def store(tmp_path):
    s = Store(f"sqlite:///{tmp_path}/sh.db")
    s.create_sleeve(name="eth", strategy="ping_pong", instrument="ETH/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                    starting_balance=10_000)
    return s


def test_a_trim_is_the_real_trades_per_unit_pnl_times_what_was_held_back(store):
    store.record_order("eth", order_id="e1", side="BUY", qty=D(2), intent="entry", reason="cell", ts=T0)
    store.record_fill("eth", side="BUY", qty=2, price=100.0, fee=0.2, order_id="e1", trade_id="t1",
                      ts=T0 + timedelta(seconds=1))
    did = _decision(store, "eth", "trimmed", "5", "2", order="e1")
    assert shadow.fill_outcomes(store, bars={}) == {"trade_scaled": 0, "replay": 0, "waiting": 1}  # still open
    store.record_fill("eth", side="SELL", qty=2, price=110.0, fee=0.22, order_id="x1", trade_id="t2",
                      ts=T0 + timedelta(hours=1))
    assert shadow.fill_outcomes(store, bars={}) == {"trade_scaled": 1, "replay": 0, "waiting": 0}
    [row] = store.shadow_outcomes("eth")
    per_unit = (D(20) - D("0.42")) / 2  # the trade's P&L after fees, per unit bought
    assert (row["decision_id"], row["kind"], row["method"], row["hypothetical"]) == (did, "trim", "trade_scaled",
                                                                                     False)
    assert row["pnl"] == per_unit * 3
    assert shadow.fill_outcomes(store, bars={}) == {"trade_scaled": 0, "replay": 0, "waiting": 0}  # written once


def _bars(*hl):
    idx = pd.date_range(T0 + timedelta(minutes=1), periods=len(hl), freq="min", tz="UTC")
    return pd.DataFrame({"high": [h for h, _ in hl], "low": [lo for _, lo in hl]}, index=idx)


@pytest.mark.parametrize("side, stop, target, path, exit_px", [
    (1, 98, 104, [(101, 99), (104.5, 100)], 104),  # a long reaches its target
    (1, 98, 104, [(101, 97.5), (110, 100)], 98),  # a long's stop first
    (-1, 102, 96, [(101, 95)], 96),  # a short reaches its target
    (1, 98, 104, [(104.5, 97)], 98),  # both in one bar: the stop, the worse of the two
])
def test_a_refusal_replays_its_own_stop_and_target_at_the_asked_size_net_of_fees(store, side, stop, target, path,
                                                                                  exit_px):
    did = _decision(store, "eth", "rejected", "3", "0", side=side, stop=stop, target=target)
    assert shadow.fill_outcomes(store, bars={"ETH/USDT": _bars(*path)}, fee_rate=D("0.0005")) == {
        "trade_scaled": 0, "replay": 1, "waiting": 0}
    [row] = store.shadow_outcomes("eth")
    want = side * (D(exit_px) - D(100)) * 3 - D("0.0005") * (D(100) + D(exit_px)) * 3
    assert (row["decision_id"], row["kind"], row["method"], row["hypothetical"], row["pnl"]) == (
        did, "refusal", "replay", True, want)
    assert store.fills("eth") == []  # never touches the leg's book


def test_a_refusal_with_no_stop_and_target_hit_yet_or_no_plan_waits(store):
    _decision(store, "eth", "rejected", "3", "0", stop=98, target=104)
    _decision(store, "eth", "rejected", "3", "0")  # no plan to replay: stays without an outcome
    assert shadow.fill_outcomes(store, bars={"ETH/USDT": _bars((101, 99))}) == {"trade_scaled": 0, "replay": 0,
                                                                                 "waiting": 2}
