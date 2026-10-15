"""A8 fills vs model (HoE Done-when; Advisor's ruling, 15 Oct 09:47 UK): per Strategy, decision fills and trigger fills
each notional-weighted against what the backtest assumes for that order type (a decision fill: the half spread; a
trigger fill: the taker's slippage, the half spread at least its floor), the total execution cost in currency over every
fill (fees and slippage), and under 20 fills a line says how many and that they are too few to judge. QT3's master
(quant-review/v2-p2/drop-4) holds the per-fill and aggregate pins; these cells pin the split and its words."""

import re
from datetime import timedelta

import pytest
from test_halt_ref import T0, _make, store  # noqa: F401

AUTH = ("pm", "pw")
S = timedelta(seconds=1)


def _ns(t) -> int:
    return int(t.timestamp() * 1e9)


def _fill(db, name, oid, side, px, fee, at, *, qty=1.0, intent="entry", signal=None):
    timing = {"bar_close": _ns(at), "decided": _ns(at)} if signal and "price" in signal else None
    db.record_order(name, order_id=oid, side=side, qty=qty, intent=intent, reason="r", signal=signal or {}, ts=at,
                    timing=timing)
    db.record_fill(name, side=side, qty=qty, price=px, fee=fee, order_id=oid, trade_id=f"t-{oid}", ts=at + 2 * S)
    db.record_timing(name, oid, fill=_ns(at + 2 * S))
    db.update_order(oid, status="filled")


@pytest.fixture
def a(store):
    """"a": two decision fills (+5 bp on 1,000.50, -5 bp on 3,001.50) and one trigger fill (+10 bp on 989.01)."""
    _make(store, "a", 10_000)
    _fill(store, "a", "d1", "BUY", 1_000.5, 0.5, T0, signal={"price": 1_000.0})
    _fill(store, "a", "d2", "SELL", 1_000.5, 1.5, T0 + 60 * S, qty=3.0, intent="exit", signal={"price": 1_000.0})
    _fill(store, "a", "t1", "SELL", 989.01, 0.5, T0 + 120 * S, intent="stop_loss", signal={"stop_px": 990.0})
    return store


def test_a8_decision_and_trigger_fills_are_split_each_against_what_the_model_assumes(a):
    from sleeve_fund import markets
    from sleeve_fund.dashboard import fills_model
    from sleeve_fund.instruments import taker_slippage
    from sleeve_fund.venues import venue as venue_profile

    s = fills_model.view(a, T0 + 600 * S)["by_strategy"]["a"]
    venue = a.orders("a")[0]["venue"]
    half = markets.half_spread_for(a.sleeve("a").params, venue_profile(venue).assumed_half_spread, venue)
    dec, trig = s["decision"], s["trigger"]
    assert dec["fills"] == 2 and dec["slip_bp"] == pytest.approx((5 * 1_000.5 - 5 * 3_001.5) / 4_002.0, abs=1e-6)
    assert trig["fills"] == 1 and trig["slip_bp"] == pytest.approx(10.0, abs=1e-6)
    assert dec["assumed_bp"] == pytest.approx(half * 1e4) and trig["assumed_bp"] == pytest.approx(
        float(taker_slippage(half)) * 1e4)
    assert dec["gap_bp"] == pytest.approx(dec["slip_bp"] - dec["assumed_bp"])


def test_a8_execution_cost_is_every_fee_and_the_slippage_in_currency(a):
    from sleeve_fund.dashboard import fills_model

    s = fills_model.view(a, T0 + 600 * S)["by_strategy"]["a"]
    slip = (1_000.5 - 1_000.0) * 1 - (1_000.5 - 1_000.0) * 3 + (990.0 - 989.01) * 1  # adverse is a cost: -0.01
    assert s["fees"] == pytest.approx(2.5) and s["slip_cost"] == pytest.approx(slip, abs=1e-9)
    assert s["cost"] == pytest.approx(2.5 + slip, abs=1e-9)


def _words(db, monkeypatch):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as dash

    monkeypatch.setenv("DASHBOARD_PASSWORD", "pw")
    html = TestClient(dash.create_app(db)).get("/fills", auth=AUTH).text
    return " ".join(re.sub(r"<[^>]+>", " ", html).split()).replace("−", "-")


def test_a8_the_page_gives_each_line_with_its_count_too_few_under_20(a, monkeypatch):
    words = _words(a, monkeypatch)
    assert "Decision fills: -2.5 bp (model assumes" in words and "n = 2, too few to judge" in words
    assert "Trigger fills: +10.0 bp (model assumes" in words and "n = 1, too few to judge" in words
    assert "Execution cost: 2.49" in words


def test_a8_twenty_fills_are_enough_to_judge(store, monkeypatch):
    _make(store, "a", 10_000)
    for i in range(20):
        _fill(store, "a", f"d{i}", "BUY" if i % 2 == 0 else "SELL", 1_000.0, 0.1, T0 + i * 60 * S,
              signal={"price": 1_000.0})
    words = _words(store, monkeypatch)
    assert "Decision fills: 0.0 bp (model assumes" in words and "n = 20" in words
    assert "too few to judge" not in words.split("Trigger fills")[0]
    assert "Trigger fills: none yet" in words
