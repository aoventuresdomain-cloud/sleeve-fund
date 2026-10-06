"""Whether a strategy is liquidated: its position margin was lost and the PM hasn't yet used Reset after
liquidation. While it is, nothing restarts it: the dashboard refuses Start, Resume and Reset, and the
supervisor won't carry out a reset asked for before the liquidation landed (Advisor 6 Oct 17:57 and 20:41)."""
from __future__ import annotations

from sleeve_fund.store import LIQUIDATION_RESET, Store

# How the engine's halt message starts after a liquidation (#155's runtime.WIPED_OUT; import it once that lands).
HALT = "Position margin lost (liquidated)"
# Why Start, Resume and Reset are refused then, in the page's words (QA P1-U25, U27, U31, U33).
REFUSAL = ("its position margin was lost (liquidated), so it can't start, resume or be reset: it trades again only "
           "after you use Reset after liquidation, which asks for an incident note")


def liquidated_since_reset(store: Store, sleeve: str, halt_words: str = HALT) -> bool:
    """Whether the strategy's position margin was lost (a liquidation event or order, or a halt whose message
    starts with halt_words) with no reset after liquidation since. Read from the journal, not the latest halt:
    a Stop/Start that halts it again on drawdown must not make a Resume restart it (QA P1-U22). The halt words
    match in any case and spacing (P1-U28a)."""
    reset = store.last_event(sleeve, (LIQUIDATION_RESET,))
    since_id, since_ts = (reset["id"], reset["ts"]) if reset else (0, None)
    words = _fold(halt_words)
    if any(e["kind"] == "liquidation" or _fold(e["message"]).startswith(words)
           for e in store.sleeve_events_since(sleeve, ("liquidation", "risk_halt"), after_id=since_id)):
        return True
    order = store.last_order(sleeve, ("liquidation",))
    if order is None:
        return False
    if since_ts is None or order["ts"] > since_ts:
        return True
    if order["ts"] < since_ts:
        return False
    # A liquidation order in the same instant as the reset: orders and events share no id, so the liquidation
    # wins the tie (P1-U28b) unless the reset answered a liquidation event of that same instant.
    answered = store.last_event(sleeve, ("liquidation",))
    return not (answered is not None and answered["id"] < since_id and answered["ts"] == order["ts"])


def _fold(text: str) -> str:
    """Text for a loose match: one plain space between words, in any case (a no-break space counts as one)."""
    return " ".join((text or "").split()).casefold()
