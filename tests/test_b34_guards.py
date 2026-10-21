"""CR on B-34 d528c7a (MUST): three money guards the end-to-end cells left unpinned, one cell each.

(a) _fund_held: a gate-on Strategy before the bridge sizes off its own capital, not the fund's.
(b) _fund_halt_room: the looser-stop room is measured from the fund halt's reference (the high-water mark), not
    from today's equity.
(c) _rebalance: a stale fund mark skips the rebalance (and says so); it never falls back to the Strategy's own P&L."""

from __future__ import annotations

from types import SimpleNamespace as NS

import pytest
from test_cut_a_fund_equity import T0, _store
from test_exposure_gate_xfails import _offline  # noqa: F401

from sleeve_fund.strategies.base import LongFlatStrategy


@pytest.mark.parametrize("bridged", [False, True])
def test_a_gate_on_strategy_is_fund_held_only_once_the_bridge_lands(bridged):
    """Gate on (a paper gate set) in both; only the runtime's fund_held (bridge_start) puts it inside the fund, so
    before the bridge an entry's weight is of its own equity (_buy_all: `base if _fund_held else own`)."""
    s = NS(runtime=NS(backtest=False, fund_held=bridged), _paper_gated=lambda: object())
    assert LongFlatStrategy._fund_held(s) is bridged


def test_b_the_looser_stop_room_is_measured_from_the_funds_high_water_mark(tmp_path):
    """Fund marked at 90,000 below a 100,000 peak; T5 halt 15% (seeded): the halt is at 85,000, so 5,000 of room,
    not the 13,500 that 15% below today's 90,000 would give."""
    from sleeve_fund.dashboard.app import _fund_halt_room
    from sleeve_fund.gate_ledger import DbLedger
    from sleeve_fund.money import money
    from sleeve_fund.portfolio.gate import PortfolioState
    from sleeve_fund.risk import trading_day

    store = _store(tmp_path, True)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="B-34: the fund's capital", rebase=True,
                                      now=T0), T0)
    DbLedger(store, lambda: ()).set_state(PortfolioState(equity=money(90_000), mark_ts=T0, hwm=money(100_000),
                                                         day=trading_day(T0), day_start_equity=money(90_000)))
    room, words = _fund_halt_room(store)
    assert room == pytest.approx(5_000.0) and "15%" in words, (room, words)


def test_c_a_stale_fund_mark_skips_the_rebalance_and_says_so_never_its_own_pnl():
    """_weight_equity answers None for a stale mark: the rebalance sends nothing and notes why, whatever its own
    equity (here -500, its P&L from 0)."""
    notes, sent = [], []
    s = NS(_mark=lambda: (-500.0, 0.0, 0.1, None), _weight_equity=lambda own: None, _noted=set(),
           _note=lambda kind, msg, level="warning": notes.append((kind, msg)),
           submit_order=lambda *a, **k: sent.append(a))
    assert LongFlatStrategy._rebalance(s, NS(), 0.3, "r", {}) is False
    assert notes and notes[0][0] == "rebalance_skipped_fund_equity" and "fund equity stale" in notes[0][1]
    assert sent == []
