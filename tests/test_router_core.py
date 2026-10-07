"""P2-4 router pure core (sleeve_fund.router). Cells follow pe1-plans/phase2-pe1-plans.md section 5; V3 to V6 (the
journal column, the five submit sites, the trade_from order and the migration) land with the wiring."""

import pytest

from sleeve_fund import router

PROFILES = {"ALPHA": object(), "BETA": object()}


def test_v1_route_for_a_pinned_strategy():
    r = router.route("trend-a", "alpha", PROFILES, tag="TREND_A_0123456789ABCDEFGH")
    assert r == router.Route("trend-a", "ALPHA", "ALPHA-PAPER-TREND_A_0123456789AB")


def test_a_new_profile_needs_no_router_change():
    assert router.route("s", "GAMMA", {**PROFILES, "GAMMA": object()}, tag="S").venue == "GAMMA"


@pytest.mark.parametrize("kind", ["live", "LIVE", ""])
def test_paper_accounts_only(kind):
    with pytest.raises(router.RouteError, match="paper accounts only"):
        router.route("s", "ALPHA", PROFILES, tag="S", account_kind=kind)


def test_unknown_or_missing_venue_is_refused():
    with pytest.raises(router.RouteError):
        router.route("s", "GAMMA", PROFILES, tag="S")
    with pytest.raises(router.RouteError):
        router.route("s", "", PROFILES, tag="S")


def test_venue_of():
    assert router.venue_of("BTC/USD.ALPHA") == "ALPHA"
    assert router.venue_of("BTCUSDT-PERP.beta") == "BETA"
    for bad in ("BTCUSD", ".ALPHA", "BTC/USD."):
        with pytest.raises(ValueError):
            router.venue_of(bad)


def test_v2_an_order_on_its_pinned_venue_passes():
    assert router.check("s", "BTC/USD.ALPHA", "alpha") is None


def test_v2_a_mismatched_order_is_refused_with_a_reason():
    with pytest.raises(router.VenueMismatch) as e:
        router.check("s", "BTC/USD.BETA", "ALPHA")
    assert e.value.reason.startswith("Order refused") and "BETA" not in e.value.reason
