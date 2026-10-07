"""RAL-ANCHOR (Advisor 15:22 UK): a reset after liquidation starts a fresh run. The strategy's cycle no longer picks
up from the last fill (the liquidation, booked at the bankruptcy price since GAP-LIQ-CAP), so its first entry after
the reset follows its fresh-start rule. Uses QA's RAL set-up (test_ral_xfails)."""

from sleeve_fund.paper.runtime import SleeveRuntime
from tests.test_ral_xfails import NAME, _liquidate, _noted, _ral, _restart, store  # noqa: F401


def test_after_a_ral_the_first_entry_follows_the_fresh_start_rule(store, tmp_path):
    """ping_pong buys on its first bar after a fresh start. Restarting at the gap's price, 7% above the booked
    liquidation price, used to read as a long leg past its rise from that fill, and so opened a short."""
    f = _liquidate(tmp_path, store)
    _ral(store, incident=_noted(store, f.liq))
    new = _restart(tmp_path, store, [(5, 0.0)], hours=25)
    assert [(o["side"], o["intent"]) for o in new][:1] == [("BUY", "entry")], new


def test_the_last_fill_anchors_a_restart_until_a_ral_comes_after_it(store, tmp_path):
    f = _liquidate(tmp_path, store)
    last = store.fills(NAME, limit=1)[0]
    assert SleeveRuntime(store, NAME).last_fill_this_run()["id"] == last["id"]  # a plain restart keeps its cycle
    _ral(store, incident=_noted(store, f.liq))
    _restart(tmp_path, store, [(1, 0.0)], hours=25, tag="applied")  # the process applies the reset
    assert store.last_event(NAME, ("liquidation_reset",)) is not None
    assert SleeveRuntime(store, NAME).last_fill_this_run() is None
