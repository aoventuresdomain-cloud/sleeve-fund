"""P2-1b band variant, the Advisor's pins of 7 Oct 02:08 on top of QA's tests (test_p2_1b.py): the position's one stop
only ever tightens through its adds, and an add the margin cap holds back is said once and never sent as an order."""

import pytest

from sleeve_fund.data import synthetic_ohlcv
from sleeve_fund.paper.runtime import SleeveRuntime
from sleeve_fund.research.runner import run_backtest
from sleeve_fund.venues import venue
from test_backtest import _path
from test_p2_1b import CLOSES, DONCHIAN, MODELS


def _run(strategy, params):
    px = _path(synthetic_ohlcv(days=900, seed=11), CLOSES)
    res = run_backtest(strategy, px, venue("KRAKEN").instrument("BTC", "USD"), {**params, "resize_band": 0.25},
                       risk_profile="balanced", half_spread=0)
    assert not res.handler_errors, res.handler_errors
    return res


def _band_adds(res):
    return [d for d in res.decisions.values() if d["intent"] == "entry" and "band add" in d["signal"]["sized_by"]]


@pytest.mark.parametrize("strategy, params", MODELS)
def test_the_positions_stop_never_loosens_through_its_adds(strategy, params):
    res = _run(strategy, params)
    triggers, adds = [], 0
    for d in res.decisions.values():
        if d["intent"] == "entry" and "band add" not in d["signal"]["sized_by"]:
            triggers = []  # a new position from flat
        elif d["intent"] == "entry":
            adds += 1
            assert d["signal"]["position_stop"] >= d["signal"]["close"] * (1 - d["signal"]["stop_frac"]) - 1e-9
        elif d["intent"] == "stop_loss" and "trigger" in d["signal"]:
            triggers.append(d["signal"]["trigger"])
            assert triggers == sorted(triggers), triggers  # a long's stop only ever moves up
    if strategy == "donchian":
        assert adds  # the path makes the Donchian ensemble add, so the check above has something to check


def test_an_add_the_margin_cap_holds_back_is_said_once_per_position_and_never_sent(monkeypatch):
    real = SleeveRuntime.position_budget
    monkeypatch.setattr(SleeveRuntime, "position_budget", lambda self, e, posted=0.0: 0.0 if posted else real(self, e))
    res = _run("donchian", DONCHIAN)
    assert not _band_adds(res)
    entries = [d for d in res.decisions.values() if d["intent"] == "entry"]
    capped = [e for e in res.journal.events(None, limit=100_000) if e["kind"] == "add_capped"]
    # Unbound, this path adds on several bars of the second position; held back, each position says so once.
    assert 1 <= len(capped) <= len(entries), (len(capped), len(entries))
    assert len({e["ts"] for e in capped}) == len(capped)
