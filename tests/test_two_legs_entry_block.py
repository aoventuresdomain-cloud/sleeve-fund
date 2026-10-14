"""HoQA F283-L1 (QT2 L1, L2; Advisor 05:10 atomic entry, agreed 15:00 UK): a two-leg Strategy's entry asks BOTH legs
whether they may open (can_open, and on paper CHOKE: runtime.entry_blocked) before the pair's check sends anything.
A blocked leg (liquidated, halted, paused, stale data, the portfolio's block) means neither leg is sent, in that leg's
own words: never one leg sent, filled and unwound."""

from decimal import Decimal as D
from types import SimpleNamespace

import pytest
from nautilus_trader.model import OrderSide

from sleeve_fund.strategies.pair import Pair

PX = 60_000.0


def entry(lead_rt, follow_rt):
    """Pair.entry on the lead's signal, with these runtimes on its legs: the notes it left, and whether it went on to
    the pair's check (its _gate_both, which sends)."""
    notes, sent = [], []
    p = Pair.__new__(Pair)
    p.lead = SimpleNamespace(runtime=lead_rt, _portfolio=None, _note=lambda k, m: notes.append((k, m)),
                             _price=lambda: PX)
    p.follow = SimpleNamespace(runtime=follow_rt, _portfolio=None, _price=lambda: PX)
    p.lead.pair = p.follow.pair = p
    p.sides = {id(p.lead): 1, id(p.follow): -1}
    p._state, p.recovered, p._store, p.on_state, p.ratio = "flat", True, None, None, D(1)
    p._flat = lambda leg: True
    p._sized = lambda *a: D(1)
    p._gate_both = lambda *a: sent.append(a) or False
    Pair.entry(p, p.lead, OrderSide.BUY, D(1), "the signal", {"close": PX})
    return notes, bool(sent)


def stub(name, block=(False, None), status="running", backtest=False):
    """A leg's runtime saying what CHOKE says and its own status (can_open only while running)."""
    return SimpleNamespace(name=name, backtest=backtest, status=status, can_open=lambda: status == "running",
                           entry_blocked=lambda: block)


@pytest.mark.parametrize("leg, block, status, words", [
    ("lead", (True, "Stale data: no trade or quote for 90 s"), "running", "the first leg can't open: Stale data"),
    ("lead", (True, "Liquidated: position margin lost"), "halted", "the first leg can't open: Liquidated"),
    ("lead", (False, None), "halted", "the first leg can't open: a leg is halted"),
    ("follow", (True, "The portfolio is halted"), "running", "the second leg can't open: The portfolio is halted"),
    ("follow", (True, "Stale data: no trade or quote for 90 s"), "running", "the second leg can't open: Stale data"),
    ("follow", (False, None), "paused", "the second leg can't open: a leg is paused"),
])
def test_a_pair_entry_with_either_leg_blocked_sends_nothing_and_says_why(leg, block, status, words):
    rts = {"lead": stub("carry"), "follow": stub("carry~2")}
    rts[leg] = stub(rts[leg].name, block, status)
    notes, sent = entry(rts["lead"], rts["follow"])
    assert not sent and notes and notes[-1][0] == "pair_entry_refused" and words in notes[-1][1], notes


def test_a_pair_entry_with_both_legs_clear_goes_on_to_the_pairs_check():
    notes, sent = entry(stub("carry"), stub("carry~2"))
    assert sent and not notes


@pytest.mark.parametrize("leg", ["lead", "follow"])
def test_in_a_backtest_a_leg_that_cant_open_blocks_the_pair_too(leg):
    """A backtest has no CHOKE journal: can_open decides (as the leg-2 retry's rule)."""
    rts = {"lead": stub("carry", backtest=True), "follow": stub("carry~2", backtest=True)}
    rts[leg] = stub(rts[leg].name, (True, "never asked in a backtest"), "halted", backtest=True)
    notes, sent = entry(rts["lead"], rts["follow"])
    assert not sent and "a leg is halted" in notes[-1][1], notes
