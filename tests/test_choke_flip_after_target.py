"""CHOKE and the turn after a target (HoE, 7 Oct): a perp whose target fills while the signal has turned opens the
other side on that close (_flip_after_target, QA P1-L10). That opening is an entry like any other, so while nothing
may open (halted, paused, exits only, stale data, Stopped) it is refused with its decision row and no opposite-side
order is sent. The block is applied as the target's fill arrives, the moment the turn would be taken."""
from __future__ import annotations

import pytest
from test_hub_146_full_round import wick
from test_hub_146_qa import _probe, _probe_classes, paper  # noqa: F401 - _probe registers the probe strategy

from sleeve_fund.paper.runtime import EXITS_ONLY, REFUSED

BLOCKS = ("none", "halted", "paused", "exits_only", "stale_data", "stopped")


def _block(strategy, how: str) -> None:
    rt = strategy.runtime
    if how == "halted":
        rt._set("halted", "drawdown 25.0% hit the 20.0% limit")
    elif how == "paused":
        rt._set("paused", "paused by PM: QA")
    elif how == "exits_only":
        rt._set("paused", f"{EXITS_ONLY}: QA")
    elif how == "stale_data":
        rt.holds["stale_data"] = "last price 400 s old. It clears when data resumes"
    elif how == "stopped":
        rt.store.set_desired_state(rt.name, "stopped")


@pytest.mark.parametrize("how", BLOCKS)
@pytest.mark.parametrize("side", [1, -1], ids=["long-to-short", "short-to-long"])
def test_the_turn_after_a_target_is_refused_while_nothing_may_open(monkeypatch, how, side):
    from sleeve_fund.strategies import REGISTRY

    probe, config = _probe_classes()

    class Blocking(probe):
        blocked = False

        def on_order_filled(self, event) -> None:
            if how != "none" and self.decisions.get(str(event.client_order_id), {}).get("intent") == "take_profit":
                self.blocked = True
                _block(self, how)
            super().on_order_filled(event)

        def _market_seen(self) -> None:
            super()._market_seen()
            if self.blocked and how == "stale_data":  # the replay's prices keep coming; the hold stands for stale data
                _block(self, how)

    monkeypatch.setitem(REGISTRY, "probe", (Blocking, config))
    run = paper(wick(side, 10.4), side=side, perp=True, profile="aggressive", tp=0.02, leave=11,
                extra={"after": -side})
    assert [o["intent"] for o in run.orders].count("take_profit") == 1, [o["intent"] for o in run.orders]
    entries = [o for o in run.orders if o["intent"] == "entry"]
    if how == "none":  # the control: unblocked, the turn opens the other side
        assert len(entries) == 2 and entries[1]["side"] == ("SELL" if side > 0 else "BUY"), entries
        return
    assert len(entries) == 1, f"the turn after the target opened the other side while {how}: {entries}"
    refused = run.store.decisions("q146", action=REFUSED, limit=100)
    assert refused, f"no decision row for the refused turn while {how}"
