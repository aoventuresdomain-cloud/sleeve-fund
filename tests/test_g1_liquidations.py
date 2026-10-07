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
