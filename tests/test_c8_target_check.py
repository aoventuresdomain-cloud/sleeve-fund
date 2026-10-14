"""C8 (Drop B): an entry's take-profit that a hit would not pay the round trip's costs on is dropped, or kept when the
Strategy's settings say so (keep_sub_cost_target); either way that entry's order says what was found, in the PM's
words and as a code, with the target and its break-even as shares of the entry price."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from sleeve_fund.strategies.base import LongFlatStrategy, gain_at_target

LEG = 0.0015  # one leg's fee and half spread: the round trip needs ~0.30%


def _leg(tp, keep=False):
    notes = []
    s = SimpleNamespace(_tp_frac=tp, _tp_signal={}, _noted=set(), _cfg=SimpleNamespace(keep_sub_cost_target=keep))
    s._note = lambda kind, msg, level="warning": notes.append((kind, msg, level))
    return s, notes


@pytest.mark.parametrize("side", [1, -1])
def test_a_target_below_the_round_trip_is_dropped_and_the_entry_says_so(side):
    s, notes = _leg(0.002)
    LongFlatStrategy._check_target(s, LEG, side)
    even = s._tp_signal["tp_break_even"]
    assert s._tp_frac is None and gain_at_target(even, LEG, side) == pytest.approx(0, abs=1e-6)  # to 6 places
    assert s._tp_signal["tp_drop_code"] == "below_round_trip_cost" and s._tp_signal["tp_target"] == 0.002
    assert s._tp_signal["tp_drop_step"] == "entry_target_check" and "tp_kept" not in s._tp_signal
    assert s._tp_signal["tp_dropped"] == f"Take-profit 0.20% dropped: below the round trip's costs (break-even {even:.2%})"
    assert [n[0] for n in notes] == ["target_below_costs"]


def test_its_settings_keep_a_target_below_the_round_trip_and_the_entry_says_so():
    s, notes = _leg(0.002, keep=True)
    LongFlatStrategy._check_target(s, LEG)
    assert s._tp_frac == 0.002 and s._tp_signal["tp_drop_code"] == "kept_by_config"
    assert "kept by its settings" in s._tp_signal["tp_kept"] and "tp_dropped" not in s._tp_signal
    assert [(n[0], n[2]) for n in notes] == [("target_kept_below_costs", "info")]


@pytest.mark.parametrize("tp", [0.01, None])
def test_a_target_that_pays_its_costs_or_none_leaves_the_entry_as_it_was(tp):
    s, notes = _leg(tp)
    s._noted = {"target_below_costs", "target_kept_below_costs"}
    s._tp_signal = {"stale": 1}
    LongFlatStrategy._check_target(s, LEG)
    assert s._tp_frac == tp and s._tp_signal == {} and not notes and not s._noted


def test_the_entry_order_carries_what_the_check_found_once():
    sent = []
    s = SimpleNamespace(pair=None, two_leg=False, _portfolio=None, _paper_gated=lambda: None,
                        _tp_signal={"tp_drop_code": "below_round_trip_cost"},
                        _submit=lambda side, qty, what, reason, signal: sent.append((what, signal)))
    LongFlatStrategy._send_entry(s, 1, Decimal("0.01"), "why", {"close": 100.0})
    LongFlatStrategy._send_entry(s, 1, Decimal("0.01"), "why", {"close": 100.0})
    assert sent == [("entry", {"close": 100.0, "tp_drop_code": "below_round_trip_cost"}), ("entry", {"close": 100.0})]


def test_keep_sub_cost_target_is_a_strategy_setting_off_by_default():
    from sleeve_fund.strategies.base import LongFlatConfig

    assert "keep_sub_cost_target" in LongFlatConfig.__init__.__code__.co_varnames
    assert LongFlatConfig.__init__.__kwdefaults__["keep_sub_cost_target"] is False
