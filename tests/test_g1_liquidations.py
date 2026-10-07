"""GAP-LIQ-CAP, the G1 Liquidations check (Advisor 7 Oct 00:19 (5); DA and HoE on g1_acks): a gap past a correctly
placed stop stops a pass until the PM acknowledges it; with no stop, or one beyond half the distance to liquidation,
it is a FAIL whatever the acknowledgements hold. An acknowledgement only records that the PM has seen it."""

from types import SimpleNamespace

import pandas as pd
import pytest

from sleeve_fund.research.tearsheet import LIQUIDATION_CHECK, NEEDS_ACK, _liquidation_check, g1_verdict

TS = pd.Timestamp("2024-02-20", tz="UTC")


def _liq(why: str, ts=TS) -> dict:
    ok = why == "gapped past the stop"
    return {"window": "out-of-sample test window 1", "ts": ts, "x": 1003.9, "stop_px": 220.0 if why != "missing"
            else None, "stop_ok": ok, "stop_why": why, "needs_ack": ok}


def _check(liqs, ack=None):
    name, verdict, detail = _liquidation_check(SimpleNamespace(liquidations=liqs), ack)
    return verdict, g1_verdict([(name, verdict, detail)])[1], detail


def test_no_liquidation_passes():
    assert _check([])[:2] == ("PASS", [])


@pytest.mark.parametrize("ack", [None, "", {}])
def test_a_gap_past_a_correctly_placed_stop_stops_a_pass_until_acknowledged(ack):
    verdict, failed, detail = _check([_liq("gapped past the stop")], ack)
    assert verdict == NEEDS_ACK and failed == [LIQUIDATION_CHECK] and "needs the PM's acknowledgement" in detail


@pytest.mark.parametrize("ack", ["seen", {TS: {"note": "seen", "actor": "pm"}}])
def test_the_pms_acknowledgement_clears_a_gap_past_a_correctly_placed_stop(ack):
    verdict, failed, detail = _check([_liq("gapped past the stop")], ack)
    assert verdict == "PASS" and failed == [] and "acknowledged by the PM: seen" in detail


def test_an_acknowledgement_for_another_liquidation_clears_nothing():
    verdict, failed, _ = _check([_liq("gapped past the stop")], {TS + pd.Timedelta(days=1): {"note": "seen"}})
    assert verdict == NEEDS_ACK and failed == [LIQUIDATION_CHECK]


@pytest.mark.parametrize("why", ["missing", "beyond half the distance to liquidation"])
@pytest.mark.parametrize("ack", [None, "seen", {TS: {"note": "seen", "actor": "pm"}}])
def test_a_missing_or_too_wide_stop_is_a_fail_whatever_the_acknowledgements_hold(why, ack):
    verdict, failed, detail = _check([_liq(why)], ack)
    assert verdict == "FAIL" and failed == [LIQUIDATION_CHECK] and "can't clear" in detail
    assert "acknowledged" not in detail


def test_one_fail_among_acknowledged_gaps_still_fails():
    later = TS + pd.Timedelta(days=3)
    verdict, failed, _ = _check([_liq("gapped past the stop"), _liq("missing", later)],
                                {TS: {"note": "seen"}, later: {"note": "seen"}})
    assert verdict == "FAIL" and failed == [LIQUIDATION_CHECK]


def test_what_a_liquidation_lost_leaves_out_a_part_of_the_position_closed_before_it():
    """CR minor on #189 (HoE: must fix): x is the loss on the quantity the liquidation closed, at the average entry
    with its share of the entry fee, plus its own fee; a trim taken at a profit before it is not in it."""
    from sleeve_fund.research.study import _liquidations

    t = lambda h: TS + pd.Timedelta(hours=h)  # noqa: E731
    orders = [{"order_id": "E", "intent": "entry", "side": "BUY", "ts": t(0), "signal": {"liquidation_px": 66.6}},
              {"order_id": "S", "intent": "stop_loss", "side": "SELL", "ts": t(0), "signal": {"trigger": 90.0}},
              {"order_id": "T", "intent": "exit", "side": "SELL", "ts": t(1), "signal": {}},
              {"order_id": "L", "intent": "liquidation", "side": "SELL", "ts": t(2), "signal": {}}]
    fills = [{"id": 1, "order_id": "E", "side": "BUY", "qty": 1.0, "price": 100.0, "fee": 0.1, "ts": t(0)},
             {"id": 2, "order_id": "T", "side": "SELL", "qty": 0.5, "price": 110.0, "fee": 0.055, "ts": t(1)},
             {"id": 3, "order_id": "L", "side": "SELL", "qty": 0.5, "price": 67.0, "fee": 0.05, "ts": t(2)}]
    journal = SimpleNamespace(sleeve_row=SimpleNamespace(name="s"), orders=lambda name, limit: orders,
                              fills=lambda name, limit: fills)
    (liq,) = _liquidations(SimpleNamespace(journal=journal), t(0), t(3), "holdout")
    # 0.5 x (100 - 67) lost on the half liquidated, its half of the 0.1 entry fee, and the 0.05 liquidation fee;
    # the whole position's cash flows would read 11.71, the trim's 5.00 profit netted in
    assert liq["x"] == pytest.approx(16.6) and liq["stop_px"] == 90.0
