"""Whether a strategy is liquidated: its position margin was lost and the PM hasn't yet used Reset after
liquidation. While it is, nothing restarts it: the dashboard refuses Start, Resume and Reset, and the
supervisor won't carry out a reset asked for before the liquidation landed (Advisor 6 Oct 17:57 and 20:41)."""
from __future__ import annotations

from sleeve_fund.store import Store

# How the engine's halt message starts after a liquidation (runtime.WIPED_OUT).
HALT = "Position margin lost (liquidated)"
# Why Start, Resume and Reset are refused then, in the page's words (QA P1-U25, U27, U31, U33).
REFUSAL = ("its position margin was lost (liquidated), so it can't start, resume or be reset: it trades again only "
           "after you use Reset after liquidation, which asks for an incident note")


def liquidated_since_reset(store: Store, sleeve: str) -> bool:
    """Whether the strategy's position margin was lost with no reset after liquidation since: the engine's own rule
    (runtime.liquidation_head), so the supervisor, the dashboard and the gate can't disagree (CR 7)."""
    from sleeve_fund.paper.runtime import liquidation_head  # the engine imports the store, as this module does

    return liquidation_head(store, sleeve) is not None
