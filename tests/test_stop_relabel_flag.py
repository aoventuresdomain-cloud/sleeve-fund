"""FU-STOP-RELABEL-FLAG (Front-end Engineer's ask, HoE OK 7 Oct 18:54 UK): a stop that fills at or past the liquidation
price is re-labelled a liquidation (GAP-LIQ, Advisor 6 Oct), and its decision's signal says so, so the screens never
parse the reason. Additive only: the intent, the reason and the booking are unchanged (test_gap_liq.py pins those)."""

from sleeve_fund.research.runner import run_backtest
from test_gap_liq import STOPPED, THROUGH_BOTH, _closing
from test_long_short import PERP, _gapped


def test_a_stop_relabelled_as_a_liquidation_carries_the_flag_its_level_and_its_raw_fill(prices, instrument):
    res = run_backtest("ping_pong", _gapped(prices, THROUGH_BOTH), instrument, STOPPED, half_spread=0,
                       risk_profile="aggressive")
    (_, d), = _closing(res)
    sig = d["signal"]
    assert d["intent"] == "liquidation" and sig["stop_relabelled"] is True
    assert sig["stop_px"] == sig["trigger"] and 101.5 < sig["stop_px"] < 160  # the short's stop level, below the gap
    assert sig["stop_fill_px"] == sig["market_px"] == 160.0  # the raw fill, before the liquidation's booking


def test_a_liquidation_that_was_never_a_stop_carries_no_flag(prices, instrument):
    res = run_backtest("ping_pong", _gapped(prices, THROUGH_BOTH), instrument, PERP, half_spread=0,
                       risk_profile="aggressive")
    liquidations = [d for d in res.decisions.values() if d["intent"] == "liquidation"]
    assert liquidations
    for d in liquidations:
        assert not {"stop_relabelled", "stop_px", "stop_fill_px"} & set(d.get("signal") or {})
