"""F283-P1 (Advisor 14 Oct 15:00 UK), the single-leg part: GAP-LIQ on every exit path. A closing fill at or beyond the
position's liquidation price is a LIQUIDATION, whatever closed it (a stop, a halt's flatten, a PM close): loss exactly X
at bankruptcy, the insurance diagnostic, the incident and a reset after liquidation. A liquidation waiting on its
working orders' cancels is never overwritten by a later close on the same mark."""

from decimal import Decimal as D
from types import SimpleNamespace

import pytest

from sleeve_fund.strategies.base import LongFlatStrategy


def _closing(intent, liq=30_226.13):
    """A long perp position's close as the engine decided it: its decision row, with the liquidation price it was
    judged against (the engine's _sell_all puts it there), and stand-ins for the liquidation's own journaling."""
    booked = []
    s = SimpleNamespace(decisions={"c1": {"intent": intent, "reason": "a close", "signal": {"liquidation_px": liq}}},
                        _entry_side=1, _entry_px=60_000.0, _rebook=None, _exit_lock=None, _flip=None,
                        runtime=SimpleNamespace(), _liquidation_events=lambda *a: booked.append(("events", a)),
                        _on_liquidation=lambda *a: booked.append(("on", a)))
    return s, booked


@pytest.mark.parametrize("intent", ["risk_halt", "risk_pause", "pm_flatten", "exit", "stop_loss"])
def test_any_close_filled_past_the_liquidation_price_is_booked_as_a_liquidation(intent):
    s, booked = _closing(intent)
    assert LongFlatStrategy._gap_liquidation(s, "c1", "c1", 29_000.0)
    d = s.decisions["c1"]
    assert d["intent"] == "liquidation" and d["signal"]["market_px"] == 29_000.0 and booked
    assert d["signal"].get("relabelled_from", "stop_loss") == intent and s._rebook == "c1"


@pytest.mark.parametrize("intent, px", [("risk_halt", 31_000.0), ("entry", 29_000.0), ("liquidation", 29_000.0)])
def test_a_close_inside_the_liquidation_price_or_an_entry_is_left_as_it_is(intent, px):
    s, booked = _closing(intent)
    assert not LongFlatStrategy._gap_liquidation(s, "c1", "c1", px)
    assert s.decisions["c1"]["intent"] == intent and not booked


def _seller(pending=None, working=()):
    sent, cancelled = [], []
    s = SimpleNamespace(_pending_exit=pending, pair=None, _margin=True, _drop_kept=lambda earn: None,
                        _working=lambda: list(working), _lot=lambda: D("0.001"), _min_qty=lambda: D("0.001"),
                        _position_qty=lambda free: D("0.05"), _pos_side=lambda: 1, _cancel_on_accept=set(),
                        cancel_order=cancelled.append, _liq_signal=lambda: {"liquidation_px": 30_226.13},
                        _submit=lambda side, qty, intent, reason, values: sent.append((intent, values)))
    return s, sent


def test_every_close_of_a_margin_position_carries_its_liquidation_price():
    s, sent = _seller()
    LongFlatStrategy._sell_all(s, "risk_halt", "Risk halt", {"equity": 1.0})
    assert sent == [("risk_halt", {"equity": 1.0, "liquidation_px": 30_226.13})]


@pytest.mark.parametrize("intent", ["risk_halt", "risk_pause", "pm_flatten", "exit"])
def test_a_later_close_on_the_same_mark_never_overwrites_a_pending_liquidation(intent):
    pending = ("liquidation", "Liquidated at 30,226", {"liquidation_px": 30_226.13})
    s, sent = _seller(pending=pending)
    LongFlatStrategy._sell_all(s, intent, "Risk halt")
    assert s._pending_exit == pending and not sent


def test_a_liquidation_still_replaces_a_pending_plain_close():
    s, sent = _seller(pending=("risk_halt", "Risk halt", None))
    LongFlatStrategy._sell_all(s, "liquidation", "Liquidated")
    assert sent and sent[0][0] == "liquidation"
