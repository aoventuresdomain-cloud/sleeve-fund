"""F-CLOSE-OVERFILL (HoE, Drop A): a close (a risk halt's, a fund halt's, a PM close) is sized from the position the
venue holds, rounded to the lot, never from a sum of floats, and on a perp is reduce-only, so it can never open the
other side. HoQA's f_c5 overfill was the pair's equal-notional second leg entering 0.01500002 (Pair._sized at two
prices a tick apart) and closing exactly that: these cells pin the close equal to the position whatever its digits."""

from __future__ import annotations

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from sleeve_fund.strategies.base import OPENING_INTENTS, LongFlatStrategy

FLOAT_SUM = 0.005 + 0.005 + 0.005  # 0.015 built in binary floating point


def _seller(position: D, lot: D, margin: bool):
    sent = []
    s = SimpleNamespace(_pending_exit=None, pair=None, _margin=margin, _drop_kept=lambda earn: None,
                        _working=lambda: [], _lot=lambda: lot, _min_qty=lambda: lot,
                        _position_qty=lambda free=False: position, _pos_side=lambda: 1, _cancel_on_accept=set(),
                        _liq_signal=lambda: {},
                        _submit=lambda side, qty, intent, reason, values: sent.append((intent, qty)))
    return s, sent


@pytest.mark.parametrize("intent", ["risk_halt", "risk_pause", "pm_flatten", "exit"])
@pytest.mark.parametrize("position, lot, margin", [
    (D(repr(FLOAT_SUM)), D("0.001"), True),     # a perp holding 0.015 from three fills of 0.005
    (D(FLOAT_SUM), D("0.001"), True),           # the same, as the float's exact binary value
    (D("0.01500002"), D("0.00000001"), False),  # f_c5's spot leg: an equal-notional entry a tick apart
    (D("0.01500002"), D("0.00000001"), True),
])
def test_a_close_is_exactly_the_position_held_rounded_to_the_lot(intent, position, lot, margin):
    s, sent = _seller(position, lot, margin)
    LongFlatStrategy._sell_all(s, intent, "close")
    ((what, qty),) = sent
    assert what == intent and qty == position.quantize(lot) and qty - position.quantize(lot) == 0
    assert abs(qty - position) < lot  # never more than the venue holds by a lot: nothing left on the other side


@pytest.mark.parametrize("intent", ["risk_halt", "risk_pause", "pm_flatten", "exit", "stop_loss", "liquidation"])
def test_every_close_on_a_perp_is_reduce_only(intent):
    """_submit: reduce = margin and intent not in OPENING_INTENTS; a close can't open the other side at the venue."""
    assert intent not in OPENING_INTENTS


def test_the_pairs_second_leg_matches_the_first_in_notional_and_equals_it_at_one_price():
    from sleeve_fund.strategies.pair import Pair

    follow = SimpleNamespace(_price=lambda: 60_000.0, _lot=lambda: D("0.00000001"))
    pair = SimpleNamespace(ratio=D(1), follow=follow, lead=object())
    assert Pair._sized(pair, follow, D("0.015"), pair.lead, 60_000.0) == D("0.015")
    assert Pair._sized(pair, follow, D("0.015"), pair.lead, 60_000.1) == D("0.01500002")  # f_c5's entry
