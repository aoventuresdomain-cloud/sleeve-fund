# Verbatim copy of quant-review/v2-p1/gap-liq-cap-189-scripts/test_qa_glc189_probes.py (m, dec, TOL), which
# tests/test_qa_ir_glc_bars.py (integration-1709cd9 re-pin ir4) imports. Not collected. Keep it unchanged.
"""QA probes for PR #189 GAP-LIQ-CAP (Head of QA round, 7 Oct). Decimal checks of loss == X, beyond the master.

Uses the GAP-LIQ-CAP master's harness (tests/test_gap_liq_cap_xfails.py: probe strategy, 1-minute backtest, recorded
paper session, #146 outage replay). Each cell asserts to 1e-6 (sub-cent), in Decimal, from the journal:
  loss (10,000 - journal cash + funding) == X = posted margin + entry fee + liquidation fee
  fill at the bankruptcy price entry x (1 - side / leverage)
  liquidation fee == qty x trigger (entry signal's liquidation_px) x the run's taker rate
  trip P&L == -X, no insurance row anywhere, equity marks / curve never below 10,000 - X and ending there
  the diagnostic names the excess (past bankruptcy) or the forfeited margin (short of it)
"""
from __future__ import annotations

import dataclasses
import functools
import re
from decimal import Decimal as D

import pandas as pd
import pytest

import test_gap_liq_cap_xfails as m

TOL = D("0.000001")
MM = 0.005  # LOW_FEE_PERP maintenance margin (KRAKEN simulated perp)

# (side, profile, leverage)
GRID = {
    "1x-short": (-1, "conservative", 1.0),
    "2x-short": (-1, "balanced", 2.0),
    "3x-short": (-1, "aggressive", 3.0),
    "2x-long": (1, "balanced", 2.0),
    "3x-long": (1, "aggressive", 3.0),
}


def dec(v) -> D:
    return D(repr(float(v)))


def _liq_rel(side: int, lev: float) -> float:
    """Liquidation price / entry (markets.liquidation_price with cash = margin - qty x entry)."""
    from sleeve_fund import markets

    return markets.liquidation_price(1 / lev - side, side, MM)


def gap_factor(case: str, where: str) -> float:
    side, _, lev = GRID[case]
    bk = 1 - side / lev
    if where == "past_bankruptcy":
        return 1 - side * (1 / lev + 0.20)
    return (_liq_rel(side, lev) + bk) / 2  # between the liquidation price and bankruptcy


def run(mode: str, case: str, where: str) -> dict:
    for k, v in GRID.items():
        m.CASES[k] = v
    return (m._backtest if mode == "backtest" else m._paper)(case, round(gap_factor(case, where), 6))


def check_exact_x(d: dict, where: str) -> dict:
    side, lev, q = d["side"], d["lev"], dec(d["q"])
    opened, closed = d["opened"], d["closed"]
    entry_px = sum(dec(f["qty"]) * dec(f["price"]) for f in opened) / q
    margin = q * entry_px / dec(lev)
    entry_fee = sum(dec(f["fee"]) for f in opened)
    liq_fee = sum(dec(f["fee"]) for f in closed)
    x = margin + entry_fee + liq_fee
    bk = entry_px * (1 - D(side) / dec(lev))
    close_px = sum(dec(f["qty"]) * dec(f["price"]) for f in closed) / q
    rate = entry_fee / (q * entry_px)
    want_liq_fee = q * dec(d["liq"]) * rate
    loss = D(10_000) - dec(d["cash_after"]) + dec(d["funding"])
    trip = d["trips"][-1]
    floor = D(10_000) - x
    series = [dec(v) for v in d["marks"]] + ([dec(v) for v in d["curve"].values] if d["curve"] is not None else [])
    last = dec(d["curve"].iloc[-1]) if d["curve"] is not None else dec(d["marks"][-1])
    excess = D(side) * (bk - dec(d["gap_px"])) * q  # > 0 past bankruptcy
    out = {
        "intents": d["closing_intents"], "x": x, "loss": loss, "loss_minus_x": loss - x,
        "close_px_minus_bk": close_px - bk, "liq_fee": liq_fee, "liq_fee_minus_want": liq_fee - want_liq_fee,
        "trip_pnl_plus_x": dec(trip["pnl"]) + x, "trip_insurance": trip.get("insurance", 0.0),
        "insurance_rows": d["insurance"], "lowest_minus_floor": min(series) - floor, "last_minus_floor": last - floor,
        "excess": excess, "events": [(k, msg) for k, msg in d["events"]
                                     if k in ("insurance_fund", "liquidation_forfeit")],
    }
    return out


CELLS = [(mode, case, where) for mode in ("backtest", "paper") for case in GRID
         for where in ("short_of_bankruptcy", "past_bankruptcy")]


@pytest.mark.parametrize("mode,case,where", CELLS, ids=["-".join(c) for c in CELLS])
def test_p2_loss_is_exactly_x_in_decimal(mode, case, where):
    from sleeve_fund import risk

    with m._patched():
        assert risk.profile(GRID[case][1]).max_leverage == GRID[case][2]
    d = run(mode, case, where)
    assert not d.get("handler_errors"), d.get("handler_errors")
    assert d["closed"] and d["opened"], ("set-up: opened and closed", d["closing_intents"])
    r = check_exact_x(d, where)
    print(f"\n{mode} {case} {where}: X={r['x']:.8f} loss={r['loss']:.8f} diff={r['loss_minus_x']:.2E} "
          f"px-bk={r['close_px_minus_bk']:.2E} liqfee={r['liq_fee']:.8f} (diff {r['liq_fee_minus_want']:.2E}) "
          f"trip+X={r['trip_pnl_plus_x']:.2E} low-floor={r['lowest_minus_floor']:.2E} "
          f"last-floor={r['last_minus_floor']:.2E} excess={r['excess']:.4f} events={r['events']}")
    assert set(r["intents"]) == {"liquidation"}, r["intents"]
    assert abs(r["loss_minus_x"]) <= TOL, r
    assert abs(r["close_px_minus_bk"]) <= D("0.00000001") * dec(d["entry_px"]), r
    assert abs(r["liq_fee_minus_want"]) <= TOL, r
    assert r["trip_insurance"] == 0.0 and not r["insurance_rows"], r
    assert abs(r["trip_pnl_plus_x"]) <= D("0.005") and r["lowest_minus_floor"] >= D("-0.005"), r  # to the cent
    if where == "past_bankruptcy":
        named = [msg for k, msg in r["events"] if k == "insurance_fund" and m.INSURANCE_WORDS in msg.lower()
                 and f"{r['excess']:,.2f}" in msg]
        assert len(named) == 1 and len(r["events"]) == 1, (f"{r['excess']:,.2f}", r["events"])
    else:
        forfeits = [msg for k, msg in r["events"] if k == "liquidation_forfeit"]
        assert len(forfeits) == 1 and len(r["events"]) == 1 and f"{-r['excess']:,.2f}" in forfeits[0], (
            f"{-r['excess']:,.2f}", r["events"])


# ---------------------------------------------------------------------------------- outage replay, wider grid

OUTAGE = [(-1, "aggressive", 3.0, 0.40, 0.10), (1, "balanced", 2.0, 0.60, 0.30), (1, "aggressive", 3.0, 0.45, 0.30),
          (-1, "balanced", 2.0, 0.60, 0.10), (-1, "conservative", 1.0, 1.20, 0.10),
          # short of bankruptcy: past liq (~0.4975 for the 2x long: liq at -49.75%) but not bankruptcy (-50%)
          (1, "balanced", 2.0, 0.4988, 0.30), (-1, "aggressive", 3.0, 0.3317, 0.10)]


@pytest.mark.parametrize("side,profile,lev,gap,back", OUTAGE,
                         ids=[f"{lev:g}x-{'long' if s > 0 else 'short'}-{gap}" for s, _, lev, gap, _ in OUTAGE])
def test_p2_outage_replay_loss_is_exactly_x_in_decimal(side, profile, lev, gap, back):
    from sleeve_fund.strategies import REGISTRY

    h = m._hub_harness()
    saved = REGISTRY.get("probe")
    REGISTRY["probe"] = h._probe_classes()
    try:
        p = h.flat_prices(30)
        p = h.shape(h.shape(p, 7.0, 8.0, h.adverse(side, gap)), 8.0, 30, h.adverse(side, back))
        r = h.restart(p, 6, 12, side=side, perp=True, profile=profile, stop=0.01, qty=0.25)
    finally:
        if saved is None:
            REGISTRY.pop("probe", None)
        else:
            REGISTRY["probe"] = saved
    held = [f for f in r.fills if f["order_id"] == "O-held"]
    closed = [f for f in r.fills if f["order_id"] != "O-held"]
    q = sum(dec(f["qty"]) for f in held)
    entry_px = sum(dec(f["qty"]) * dec(f["price"]) for f in held) / q
    x = q * entry_px / dec(lev) + sum(dec(f["fee"]) for f in held) + sum(dec(f["fee"]) for f in closed)
    bk = entry_px * (1 - D(side) / dec(lev))
    close_px = sum(dec(f["qty"]) * dec(f["price"]) for f in closed) / q
    funding = dec(r.store.funding_total("q146"))
    loss = D(10_000) - dec(r.store.journal_book("q146", 10_000)["cash"]) + funding
    intents = sorted({o["intent"] for o in r.orders if o["order_id"] != "O-held"})
    gap_open = dec(h.price_at(p, 7.0))
    marks = [dec(mk["equity"]) for mk in r.store.equity_series("q146", limit=100_000)]
    ins = list(r.store.insurance("q146"))
    print(f"\noutage {lev:g}x side {side} gap {gap}: X={x:.8f} loss={loss:.8f} diff={loss - x:.2E} "
          f"px-bk={close_px - bk:.2E} gap_open={gap_open} bk={bk:.4f} funding={funding} intents={intents} "
          f"insurance_rows={ins} low-floor={(min(marks) - (D(10_000) - x)) if marks else None} "
          f"events={[e['message'][:120] for e in r.events if e.get('kind') in ('insurance_fund', 'liquidation_forfeit')]}")
    assert intents == ["liquidation"], intents
    assert abs(loss - x) <= TOL and abs(close_px - bk) <= D("0.00000001") * entry_px, (loss - x, close_px - bk)
    assert not ins
    assert marks and min(marks) >= D(10_000) - x - TOL


@pytest.mark.parametrize("mode,case,where", CELLS, ids=["-".join(c) for c in CELLS])
def test_p2b_trip_and_marks_match_the_journal_below_the_cent(mode, case, where):
    """Sub-cent: the trip row (backtest: fills report) and the equity marks / curve after the liquidation equal the
    journal's -X / 10,000 - X to 1e-6. Fails where they differ by up to half a cent (P1-GLC-2)."""
    d = run(mode, case, where)
    r = check_exact_x(d, where)
    got = {"trip+X": r["trip_pnl_plus_x"], "lowest-floor": r["lowest_minus_floor"], "last-floor": r["last_minus_floor"]}
    assert all(abs(v) <= TOL for v in got.values()), got
