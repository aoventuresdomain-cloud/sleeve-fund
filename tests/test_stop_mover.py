"""Advisor STOP-EXEC (15 Oct): a resting stop's level moves by PLACE-THEN-CANCEL through the exchange client, never
cancel-then-place and never an amend of the trigger (the client drops it). The planner (paper.stop_mover) only says
what to send; these cells drive it with the venue's answers. Conditions, all MUST: the new stop is confirmed accepted
before the old one's cancel goes; a refused place keeps the old stop and alerts; a failed or unknown cancel is looked
up, retried and alerted; never more than 2 resting per position, a third move waits for the earlier cancel; a fill on
either stop leaves the other resized to what is left; a restart cancels every resting stop the journal doesn't own."""

from __future__ import annotations

import pytest

from sleeve_fund.paper.exec_client import TriggerAmendRefused, guard_modify
from sleeve_fund.paper.stop_mover import MAX_RESTING, StopMover


def kinds(actions):
    return [a[0] for a in actions]


def placed(actions):
    return [a for a in actions if a[0] == "place"]


@pytest.fixture
def mover():
    m = StopMover(side="SELL", qty="0.010")
    [place] = m.move("60000.0")
    assert place[0] == "place" and place[2:] == ("60000.0", "0.010")
    m.on_accepted(place[1])
    return m, place[1]


def test_a_a_level_move_places_the_new_stop_first_and_cancels_the_old_only_once_it_is_accepted(mover):
    m, old = mover
    acts = m.move("61000.0")
    assert kinds(acts) == ["place"]  # nothing cancelled yet: the old stop keeps protecting
    new = acts[0][1]
    assert new != old and m.resting() == {old, new}
    assert m.on_accepted(new) == [("cancel", old)]
    assert m.current == new
    assert m.on_cancelled(old) == [] and m.resting() == {new}


def test_b_a_refused_place_keeps_the_old_stop_and_alerts(mover):
    m, old = mover
    new = m.move("61000.0")[0][1]
    acts = m.on_rejected(new, "venue said no")
    assert kinds(acts) == ["alert"] and "61000.0" in acts[0][2] and "venue said no" in acts[0][2]
    assert m.current == old and m.resting() == {old}  # never cancelled
    assert "cancel" not in kinds(acts)


@pytest.mark.parametrize("answer", ["failed", "unknown"])
def test_c_a_cancel_that_fails_or_goes_unanswered_is_looked_up_retried_and_alerted_and_both_stops_count(mover, answer):
    m, old = mover
    new = m.move("61000.0")[0][1]
    m.on_accepted(new)
    acts = m.on_cancel_failed(old, "timeout") if answer == "failed" else m.on_cancel_unknown(old)
    assert ("lookup", old) in acts and any(a[0] == "alert" for a in acts)
    assert m.resting() == {old, new}  # two resting, never assumed gone
    assert m.on_lookup(old, resting=True) == [("cancel", old)]  # still there: cancel again
    assert m.on_lookup(old, resting=False) == [] and m.resting() == {new}


def test_d_never_more_than_two_resting_a_third_move_waits_for_the_earlier_cancel(mover):
    m, old = mover
    second = m.move("61000.0")[0][1]
    m.on_accepted(second)  # cancel of old sent, not confirmed
    assert m.move("62000.0") == []  # a third would make three resting: it waits
    assert m.move("62500.0") == []  # the latest level wins while it waits
    assert len(m.resting()) <= MAX_RESTING == 2
    acts = m.on_cancelled(old)
    assert [(a[0], a[2]) for a in placed(acts)] == [("place", "62500.0")]
    assert len(m.resting()) == 2


def test_e_a_move_while_the_new_stop_is_still_unconfirmed_waits_too(mover):
    m, _ = mover
    m.move("61000.0")
    assert m.move("62000.0") == []


def test_f_a_part_fill_on_one_stop_resizes_the_other_to_what_is_left(mover):
    m, old = mover
    new = m.move("61000.0")[0][1]
    m.on_accepted(new)
    acts = m.on_filled(new, "0.004")  # the new stop part-filled; the old one's cancel still in flight
    assert ("amend", old, "0.006") in acts  # whichever rests on reduces only the remainder
    assert m.qty == "0.006"


def test_g_a_full_fill_leaves_nothing_to_protect_and_cancels_the_other(mover):
    m, old = mover
    new = m.move("61000.0")[0][1]
    m.on_accepted(new)
    m.on_cancel_unknown(old)
    acts = m.on_filled(new, "0.010")
    assert ("cancel", old) in acts and m.qty == "0"
    assert m.move("62000.0") == []  # flat: no stop to move


def test_h_a_quantity_change_alone_is_an_amend_of_the_one_resting_stop(mover):
    m, old = mover
    assert m.resize("0.020") == [("amend", old, "0.020")]
    assert m.resting() == {old}


def test_i_a_restart_cancels_every_resting_stop_the_journal_does_not_own():
    venue = ["s-a", "s-b", "s-c"]
    acts = StopMover.reconcile(venue, journal_current="s-b")
    assert sorted(acts) == [("cancel", "s-a"), ("cancel", "s-c")]
    assert StopMover.reconcile(venue, journal_current=None) == [("cancel", v) for v in venue]


def test_j_order_ids_fit_the_venue():
    m = StopMover(side="SELL", qty="0.010")
    ids = {m.move(str(60000 + i))[0][1] for i in range(1)}
    assert all(len(i) <= 36 for i in ids)


def test_k_a_trigger_price_amend_through_the_client_is_refused_never_sent_without_it():
    """Advisor line 14: the client drops a new trigger price from an amend without a word, so asking for one raises."""
    with pytest.raises(TriggerAmendRefused):
        guard_modify(on=True, trigger_price="61000.0")
    guard_modify(on=True, trigger_price=None)  # a quantity-only amend passes
    guard_modify(on=False, trigger_price="61000.0")  # the simulated client moves triggers itself


# Advisor STOP-RETRY: a refused place is retried by its reject class ------------------------------------------------
def _refused(mover, level="61000.0"):
    m, old = mover
    new = m.move(level)[0][1]
    return m, old, new


def test_l_a_transient_refusal_is_retried_once_after_the_back_off_then_alerted_never_looped(mover):
    m, old, new = _refused(mover)
    acts = m.on_rejected(new, "rate limit", kind="transient")
    [retry] = [a for a in acts if a[0] == "retry"]
    assert retry[2:] == ("61000.0", "0.010") and retry[1] != new  # a fresh order id, placed after the back-off
    assert m.resting() == {old, retry[1]}
    acts = m.on_rejected(retry[1], "timeout", kind="transient")
    assert kinds(acts) == ["alert"] and m.resting() == {old}  # once only: left to the next stop update
    assert m.move("61000.0")[0][0] == "place"  # the next update tries again, fresh


def test_m_a_refusal_because_the_price_is_already_past_the_new_level_closes_at_market_at_once(mover):
    """A tightening move in a fast fall: the venue refuses the new stop because it would trigger at once. The
    Strategy's stop has been hit, so its quantity closes now; only the old looser stop resting would be wrong."""
    m, old, new = _refused(mover)
    acts = m.on_rejected(new, "trigger price would trigger immediately", kind="crossed")
    assert ("close", "0.010", "stop level already crossed") in acts
    assert not [a for a in acts if a[0] in ("retry", "place")]
    assert m.resting() == {old}  # reduce-only: harmless while the close goes out
    assert m.on_filled(None, "0.010") == [("cancel", old)]  # the close filled: nothing left to protect


@pytest.mark.parametrize("current", [True, False])
def test_n_any_other_refusal_is_not_retried_and_is_critical_when_no_stop_is_left(current):
    m = StopMover(side="SELL", qty="0.010")
    first = m.move("60000.0")[0][1]
    if current:
        m.on_accepted(first)
        first = m.move("61000.0")[0][1]
    acts = m.on_rejected(first, "reduce-only conflict", kind="other")
    [alert] = [a for a in acts if a[0] == "alert"]
    assert alert[1] == ("warning" if current else "critical")
    assert not [a for a in acts if a[0] in ("retry", "place", "close")]


@pytest.mark.parametrize("code, msg, kind", [
    (10006, "Too many visits!", "transient"), (10016, "Server error", "transient"), (None, "timeout", "transient"),
    (110092, "expect Rising, but trigger_price[59000] <= current[59500]", "crossed"),
    (110093, "expect Falling, but trigger_price[61000] >= current[60500]", "crossed"),
    (110017, "reduce-only rule not satisfied", "other"), (10001, "params error", "other"),
    (999999, "a code nobody listed", "other")])  # never guessed into transient (Advisor STOP-RETRY)
def test_o_the_venue_s_reject_codes_map_to_their_class(code, msg, kind):
    from sleeve_fund.paper.exec_client import reject_class

    assert reject_class(code, msg) == kind
