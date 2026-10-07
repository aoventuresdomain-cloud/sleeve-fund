"""Integration round on main 1709cd9: hub-path parity re-pinned to tonight's rulings.

Ported from quant-review/v2-p1/integration-1709cd9-scripts/test_ir_parity.py (round integration-1709cd9.md). The
source loaded hub-client-scripts/test_hub_path_parity_qa.py from QA's folder; it is carried verbatim as
tests/qa_ir_hub_path_parity_lib.py. Only ir1 is ported: ir2 (the c5 stop band) passes on every older head tried
(1918f8b, 018bb23, a83de76, 1582c3a, 9ae1f8a, 00f4b5a), so it pins no regression.

Replaces two asserts in hub-client-scripts/test_hub_path_parity_qa.py that encode superseded rules:
- c2_c5's fee line subtracted the backtest's half spread from its commission. Advisor 6 Oct 20:55: backtest taker
  fills pay the spread IN THE PRICE and it is removed from the cost line, so the fees now match the hub's directly.
- c5's stop band (-2.5 bp) assumed the backtest stop pays half a spread. Advisor 18:36/21:03 (D13): backtest stops pay
  max(half spread, 0.05 %); paper books the real fill. With a 1 bp half spread the backtest is up to ~4 bp worse.
"""
import numpy as np
import pytest

import tests.qa_ir_hub_path_parity_lib as hp
from tests.qa_ir_hub_path_parity_lib import _probe  # noqa: F401 - autouse: registers the probe strategy


@pytest.mark.parametrize("minutes", [1, 15])
def test_ir1_backtest_fees_equal_the_hub_paths_with_the_spread_in_the_price(minutes):
    """Source: quant-review/v2-p1/integration-1709cd9-scripts/test_ir_parity.py::
    test_ir1_backtest_fees_equal_the_hub_paths_with_the_spread_in_the_price. Finding: ir1 (re-pin of hub-path parity
    c2_c5 [1], [15]; integration-1709cd9.md, Advisor 6 Oct 20:55)."""
    prices = hp.WAVE(np.arange(240 * 60))
    params = {"period": 3 if minutes == 15 else 7}
    o, f, dec, _ = hp.hub_paper(prices, params, minutes=minutes)
    hub = hp.per_order(o, f)
    bo, bf = hp.backtest(prices, params, minutes=minutes)
    bt = hp.per_order(bo, bf)
    assert len(hub) >= 4 and dec.late == 0
    hp._assert_same(hub, bt)  # same minute, size and all-in price to 0.3 bp
    assert sum(r[5] for r in bt) == pytest.approx(sum(r[5] for r in hub), abs=0.01 * len(hub) + 0.01)
