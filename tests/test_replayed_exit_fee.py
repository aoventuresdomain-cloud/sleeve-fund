"""QA P1-D25: an exit the outage replay books (Advisor NA-1) is journaled at the modelled price with the fee rescaled to
it; the engine's cash carries that fee too, not the one charged at the market-on-return price, so the incident's
"equity left" and the equity series agree with the journal to the cent, with no restart in between."""
import re

import pytest

import test_hub_146_qa as qa
from test_hub_146_qa import _probe  # noqa: F401 - registers the probe strategy


@pytest.mark.parametrize("path", [pytest.param("reconnect", marks=pytest.mark.no_open_risk_limit), "restart"])
@pytest.mark.parametrize("setup", [0, 2], ids=["perp-2x-long", "perp-3x-short"])
def test_a_replayed_liquidation_leaves_the_equity_the_journal_does(path, setup):
    _, perp, profile, side = qa.LIQ_SETUPS[setup]
    p = qa.shape(qa.flat_prices(30), 7.0, 8.0, qa.adverse(side, qa.LIQ_DEPTH[profile]))
    run = qa._outage(path, p, side=side, perp=perp, profile=profile, stop=0.10, qty=0.25)
    assert qa.first_exit(run.sequence())[0] == "liquidation"  # set-up: the replay found it
    left = run.store.journal_book("q146", 10_000)["cash"]
    (inc,) = run.kinds("incident")
    m = re.search(r"; ([\d,]+\.\d\d) of equity left", inc["message"])
    assert m and float(m.group(1).replace(",", "")) == pytest.approx(max(left, 0.0), abs=0.011), (inc["message"], left)
    flat = [e for e in run.store.equity_series("q146") if e["qty"] == 0.0]
    assert flat and flat[-1]["equity"] == pytest.approx(max(left, 0.0), abs=0.011), (flat[-1], left)


@pytest.mark.parametrize("side", [1, -1], ids=["long", "short"])
def test_a_replayed_stop_leaves_the_equity_the_journal_does(side):
    run = qa._outage("reconnect", qa._outage_prices("stop", side, 30), side=side, perp=True, profile="aggressive")
    assert qa.first_exit(run.sequence())[0] == "stop_loss"  # set-up: the replay found it
    cash = run.store.journal_book("q146", 10_000)["cash"]
    marks = run.store.equity_series("q146")
    last = max(marks, key=lambda e: e["ts"])
    assert last["qty"] == 0.0 and last["equity"] == pytest.approx(cash, abs=0.011), (last, cash)
