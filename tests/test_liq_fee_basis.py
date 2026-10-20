"""B15-LIQFEE-BASIS (Advisor, 20 Oct; asked of B-15 before HoQA signs): in both gap-liquidation scenarios of the QA
master (a stop gapped through liquidation, short and long), the total booked loss is the margin plus the entry fee
plus the liquidation fee, to the cent; the liquidation fee is the position x the liquidation price x the taker rate;
and the entry fee booked is the very fee term inside margin.isolated_liquidation(fees=) that put the liquidation price
where it was. Expected values come from the fills and markets.isolated_liquidation, never from the engine's own
figures."""
from __future__ import annotations

import os
import sys
from decimal import Decimal

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_degraded_155_qa import _session_meta  # noqa: E402
from test_gap_liq_qa import PERP, _gap_through_stop, risk  # noqa: E402

START = 10_000.0
TICK = 0.1  # the recorded session's BTC/USD, price_precision 1


def cents(x) -> Decimal:
    return Decimal(str(x)).quantize(Decimal("0.01"))


@pytest.mark.parametrize("side,profile", [("short", "balanced"), ("long", "aggressive")])
def test_a_gap_liquidation_loses_margin_entry_fee_and_liquidation_fee_on_the_fee_basis_it_was_priced_on(
        tmp_path, side, profile):
    from sleeve_fund import markets
    from sleeve_fund.store import replay_book

    out = _gap_through_stop(tmp_path, side=side, profile=profile)
    assert [o["intent"] for o in out["closing"]] == ["liquidation"]  # set-up: the gap booked one liquidation
    (opened,), closed = out["opened"], out["closed"]
    lev = risk.profile(profile).max_leverage
    qty, entry, entry_fee = float(opened["qty"]), float(opened["price"]), float(opened["fee"])
    signed = qty if side == "long" else -qty
    trigger = float(out["closing"][0]["signal"]["liquidation_px"])
    perp = markets.terms(PERP, "KRAKEN")
    taker = float(_session_meta(START, PERP)["sleeve"]["taker_fee"])  # the recorded session's own rate (0.05%)

    # The price the liquidation fired at is margin.isolated_liquidation with the entry fee booked as its fees= term.
    book = replay_book([opened], START)
    priced = markets.isolated_liquidation(float(book["cash"]), signed, entry, lev, perp.maintenance_margin,
                                          fees=entry_fee, tick=TICK)
    assert trigger == pytest.approx(priced, abs=1e-6)
    assert float(book["entry_fees"]) == pytest.approx(entry_fee, abs=1e-9)  # the journal's fees= term is the booked fee
    # The liquidation fee: the position x that price x the taker rate.
    liq_fee = sum(float(f["fee"]) for f in closed)
    assert cents(liq_fee) == cents(qty * trigger * taker)
    # The total booked loss, from the journal's cash: margin + entry fee + liquidation fee, to the cent.
    lost = START - float(replay_book(out["opened"] + closed, START)["cash"])
    assert cents(lost) == cents(qty * entry / lev + entry_fee + liq_fee)
