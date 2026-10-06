"""GAP-LIQ (Independent Quant Advisor, relayed by the HoE 6 Oct ~20:06): a stop whose fill, after slippage, is at or past
the liquidation price books as a liquidation (the D3 loss, the halt with X and Y, an incident, a reset after liquidation
required). A stop that fills short of the liquidation price is still a stop-loss. One test decides both, in a backtest
and in paper: strategies.base.through_liquidation. QA's guards-on paper cases are in test_gap_liq_qa.py."""
import re

import pytest

from sleeve_fund.research.runner import run_backtest
from sleeve_fund.store import Store
from sleeve_fund.strategies.base import through_liquidation
from test_long_short import PERP, _gapped
from test_sleeve_runtime import store  # noqa: F401  (Postgres when TEST_DATABASE_URL is set)

STOPPED = {**PERP, "stop_loss": 0.02}
# Shorted at 101.5 on aggressive (3x, full margin): the 10% stop rests at about 111.7 and the liquidation price is
# about 134. A gap to 160 goes through both; one to 118 only through the stop.
THROUGH_BOTH = [100.0, 100.5, 100.8, 101.5, 101.5, 160.0, 160.0, 160.0]
STOP_ONLY = [100.0, 100.5, 100.8, 101.5, 101.5, 103.8, 103.8, 103.8]


def _closing(res):
    """The orders that closed the last position opened: those filled after its entry."""
    filled = list(res.fills.sort_values("ts_last", kind="stable").index)
    last = max(i for i, o in enumerate(filled) if res.decisions[o]["intent"] == "entry")
    return [(o, res.decisions[o]) for o in filled[last + 1:]]


def test_through_liquidation_is_at_or_past_it_on_the_losing_side():
    assert through_liquidation(1, 90.0, 90.0) and through_liquidation(1, 80.0, 90.0)
    assert not through_liquidation(1, 90.1, 90.0)
    assert through_liquidation(-1, 110.0, 110.0) and through_liquidation(-1, 150.0, 110.0)
    assert not through_liquidation(-1, 109.9, 110.0)
    assert not through_liquidation(1, 50.0, None) and not through_liquidation(-1, 0.0, 10.0)


def test_a_backtest_stop_gapped_through_liquidation_books_a_liquidation_at_the_d3_loss(prices, instrument):
    """Done when 1-3: the resting stop fills at the gap's open, past the liquidation price: journaled with intent
    liquidation, the liquidation event and an incident, the halt in the Advisor's words with X, and the loss the
    position's margin plus its fees (the insurance fund takes the rest), never more."""
    res = run_backtest("ping_pong", _gapped(prices, THROUGH_BOTH), instrument, STOPPED, half_spread=0,
                       risk_profile="aggressive")
    assert not res.handler_errors, res.handler_errors
    (oid, d), = _closing(res)
    assert d["intent"] == "liquidation" and d["reason"].startswith("Liquidated: the stop filled at 160"), d
    assert res.journal.orders_[oid]["intent"] == "liquidation" and res.journal.orders_[oid]["order_type"] == "STOP"
    kinds = [e["kind"] for e in res.journal.events_]
    assert "liquidation" in kinds and "incident" in kinds
    (halt,) = [e for e in res.risk_events if e["kind"] == "risk_halt"]
    assert re.match(r"Position margin lost \(liquidated\): [\d,]+\.\d\d, \d+(\.\d)?% of strategy equity", halt["message"])
    filled = list(res.fills.sort_values("ts_last", kind="stable").index)
    opened = filled[max(i for i, o in enumerate(filled) if res.decisions[o]["intent"] == "entry")]
    fills = {f["order_id"]: f for f in res.journal.fills_}
    entry, close = fills[opened], fills[oid]
    margin = entry["qty"] * entry["price"] / 3  # aggressive: 3x
    assert close["qty"] == pytest.approx(entry["qty"]) and close["side"] == "BUY"
    # The D3 loss: the short's price loss less what the insurance fund took is its isolated margin, to the cent; the
    # fees are charged as filled, so the trade loses the margin and both fees, never more.
    (covered,) = res.insurance
    assert (close["price"] - entry["price"]) * entry["qty"] - covered["amount"] == pytest.approx(margin, abs=0.01)
    assert res.exposure.iloc[-1] == pytest.approx(0, abs=1e-9)  # halted: nothing reopened


def test_a_backtest_stop_filled_short_of_liquidation_is_still_a_stop_loss(prices, instrument, full_margin):
    """Done when 4: a gap through the stop that stops short of the liquidation price is an ordinary stop-loss."""
    res = run_backtest("ping_pong", _gapped(prices, STOP_ONLY), instrument, STOPPED, half_spread=0,
                       risk_profile="aggressive")
    (oid, d), *_ = _closing(res)
    assert d["intent"] == "stop_loss" and res.journal.orders_[oid]["intent"] == "stop_loss", d
    assert not {"liquidation", "incident"} & {e["kind"] for e in res.journal.events_}
    assert not res.insurance


@pytest.mark.parametrize("gap,booked", [(0.6, "liquidation"), (0.15, "stop_loss")])
def test_a_paper_stop_is_judged_by_the_same_test(tmp_path, monkeypatch, gap, booked):
    """Done when 3-4 in paper (a recorded session replayed through the paper runtime, every guard on): the stop's
    market order books as a liquidation when it fills past the liquidation price, else as a stop-loss."""
    import dataclasses

    from sleeve_fund import risk
    from sleeve_fund.research.replay import replay
    from test_degraded_155_qa import _record_session, _session_meta

    for nm, p in list(risk.PROFILES.items()):  # a 10% margin cap, as QA's guards-on cases
        monkeypatch.setitem(risk.PROFILES, nm, dataclasses.replace(p, max_position_pct=0.1))
    params = {"rise": 0.01, "dip": 0.005, "stop_loss": 0.10, **PERP}
    store = Store(f"sqlite:///{tmp_path}/t.db")
    path = tmp_path / "gap.jsonl.gz"
    _record_session(path, _session_meta(10_000.0, params), [(5, 0.0), (20, 0.015), (0, gap), (5, 0.0)])
    orders = replay(path, store=store)
    entry = max(i for i, o in enumerate(orders) if o["intent"] == "entry")
    assert [o["intent"] for o in orders[entry + 1:]] == [booked], orders[entry:]
    kinds = {e["kind"] for e in store.events("ping-pong-test", limit=1000)}
    assert ({"liquidation", "incident", "order_rebooked"} <= kinds) == (booked == "liquidation"), kinds


@pytest.mark.parametrize("stop", [0.02, None], ids=["its-stop", "stopless"])
def test_a_restart_past_the_liquidation_price_books_a_liquidation_not_a_drawdown_halt(tmp_path, stop):
    """Done when 6: the process was down while the price went through a short's liquidation price; it restarts holding
    the short at 97,440. The venue would have liquidated it, so the restart books a liquidation (an incident, the
    halt with X and Y), not a drawdown flatten at today's price: on the stop's fill (GAP-LIQ) or, stopless, on the
    engine's own liquidation check. Both judge it on the journal's entry, not the price the simulated venue was given
    the position back at."""
    from datetime import datetime, timezone

    import test_replay
    from test_ral_xfails import GAP_PX, NAME, _meta, _record_at, _replay_into

    from sleeve_fund.store import replay_book

    params = {"rise": 0.01, "dip": 0.005, **PERP, **({"stop_loss": stop} if stop else {})}
    store = Store(f"sqlite:///{tmp_path}/t.db")
    opened = datetime.fromtimestamp(test_replay.START / 1e9 - 3600, tz=timezone.utc)
    store.create_sleeve(name=NAME, strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=10_000, params=params)
    store.record_equity(NAME, equity=10_000, cash=10_000, qty=0, price=60_600, benchmark=10_000, ts=opened)
    fill = {"side": "SELL", "qty": 0.11, "price": 60_600.0, "fee": 3.3, "order_id": "carried", "trade_id": "carried",
            "ts": opened}
    store.record_fill(NAME, **fill)
    book = replay_book([fill], 10_000.0)
    path = tmp_path / "outage.jsonl.gz"
    _record_at(path, _meta(book["cash"] + book["qty"] * book["entry_px"], params), [(5, 0.0)], GAP_PX, 0)
    _replay_into(store, path)
    s = store.sleeve(NAME)
    kinds = [e["kind"] for e in store.events(NAME, limit=200)]
    assert s.status == "halted" and s.status_reason.startswith("Position margin lost (liquidated): "), s.status_reason
    assert "liquidation" in kinds and "incident" in kinds and "drawdown" not in s.status_reason
    closing = [o for o in store.orders(NAME, limit=50)  # paper's watched stop is the journal's view, not an order sent
               if o["side"] == "BUY" and not (o.get("signal") or {}).get("watched")]
    assert closing and {o["intent"] for o in closing} == {"liquidation"}, closing


@pytest.mark.parametrize("kind", ["store", "backtest_journal"])
def test_only_a_filled_stop_loss_is_rebooked_as_a_liquidation_and_the_journal_says_so(store, kind):
    """Data Architect's conditions: the one re-booking allowed is a stop-loss that has filled becoming a liquidation;
    it writes an order_rebooked event naming the order, both intents and the prices; fills stay as they are."""
    from sleeve_fund.paper.journal import MemoryJournal

    j = store if kind == "store" else MemoryJournal()
    j.create_sleeve(name="s", strategy="ping_pong", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                    starting_balance=1_000)
    for oid, intent in (("stop", "stop_loss"), ("unfilled", "stop_loss"), ("exit", "exit")):
        j.record_order("s", order_id=oid, side="BUY", qty=1.0, intent=intent, reason=intent, signal={"a": 1})
    for oid in ("stop", "exit"):
        j.record_fill("s", side="BUY", qty=1.0, price=2.0, fee=0.01, order_id=oid, trade_id=f"t-{oid}")
        j.update_order(oid, fill_qty=1.0, fill_px=2.0, fee=0.01)
    why = "Liquidated: the stop filled at 2, at or past the liquidation price 1.9, so the venue took the position first"
    j.rebook_liquidation("stop", why, {"a": 1, "price": 2.0})
    orders = {o["order_id"]: o for o in j.orders("s")}
    assert (orders["stop"]["intent"], orders["stop"]["reason"], orders["stop"]["signal"]) == (
        "liquidation", why, {"a": 1, "price": 2.0})
    (ev,) = [e for e in j.events("s", limit=50) if e["kind"] == "order_rebooked"]
    assert ev["level"] == "info" and ev["message"] == f"Order stop re-booked from stop_loss to liquidation: {why}"
    for oid in ("unfilled", "exit", "stop", "missing"):  # unfilled, not a stop, already re-booked, no such order
        with pytest.raises(ValueError):
            j.rebook_liquidation(oid, why, {})
    assert [f["price"] for f in j.fills("s", limit=10)] == [2.0, 2.0]  # fills untouched
