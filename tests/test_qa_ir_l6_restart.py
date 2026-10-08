"""Integration round on main 1709cd9: hub-146 l6/na1 outage-target cell, restart path, re-pinned.

Ported from quant-review/v2-p1/integration-1709cd9-scripts/test_ir_l6_restart.py (round integration-1709cd9.md). The
source loaded hub-146-scripts/test_hub_146_qa.py from QA's folder; it is carried verbatim as tests/qa_ir_hub146_lib.py
(the repo's tests/test_hub_146_qa.py now seeds the held entry at the touch, which is not the set-up this cell checks).

D13 (Advisor 6 Oct 20:55/21:03): the backtest's entry fills at mid +/- half spread and its levels derive from the
fill. The restart harness seeds the held entry at MID (test_hub_146_qa.restart writes O-held at the bare price), so
its target sits exactly one half spread from the backtest's. The engine still places the target from its booked
entry (first assert, unchanged). The second assert allows exactly that half spread, no more.
"""
import pytest

import tests.qa_ir_hub146_lib as h
from tests.qa_ir_hub146_lib import _guard_marks, _probe  # noqa: F401 - autouse fixtures of the source file

HALF = h.SPREAD / 2 / h.BASE


@pytest.mark.parametrize("label, perp, profile, side", h.NA_SETUPS, ids=h.NA_IDS)
def test_ir3_restart_target_fills_at_its_level_and_one_seeded_half_spread_from_the_backtest(label, perp, profile,
                                                                                            side):
    """Source: quant-review/v2-p1/integration-1709cd9-scripts/test_ir_l6_restart.py::
    test_ir3_restart_target_fills_at_its_level_and_one_seeded_half_spread_from_the_backtest. Finding: ir3 (re-pin of
    hub-146 l6/na1 target, restart path; integration-1709cd9.md, D13, 20:55)."""
    p = h.shape(h.flat_prices(30), 7.5, 7 + 50 / 60, h.favourable(side, 0.03))
    run = h._outage("restart", p, side=side, perp=perp, profile=profile, tp=0.02)
    ex = h.first_exit(run.sequence())
    level = h._entry(run) * (1 + side * 0.02)
    assert ex[0] == "take_profit" and abs(ex[3] / h.tp_model(level, side) - 1) < 1e-4, (ex, h.tp_model(level, side))
    bt = h._bt_exit(p, side=side, perp=perp, profile=profile, tp=0.02, leave=25)
    # paper target / backtest target = (seeded mid entry) / (backtest entry at mid + side x half spread)
    assert abs(ex[3] / bt[3] * (1 + side * HALF) - 1) < 1e-5, (ex, bt)
