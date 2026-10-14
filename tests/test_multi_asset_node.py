"""Multi-asset Strategies, the paper node (multi-asset spec §1: one process per Strategy, one hub connection, the
parent's first asset on its venue's account and each asset leg on a clone of it; legs are independent ideas, so no Pair
and no follower: every leg runs its model's own strategy)."""

from __future__ import annotations

import pytest

from sleeve_fund.paper import node as node_mod
from sleeve_fund.paper.config import SleeveConfig
from sleeve_fund.paper.hub_client import HubStatus, HubStatusView
from test_two_legs_node import HUB, built  # noqa: F401 (the fixture)


def _legs(**kw):
    out = []
    for name, instrument in (("trend-x", "BTC/USD"), ("trend-x~ETH", "ETH/USD"), ("trend-x~SOL", "SOL/USD")):
        out.append(SleeveConfig(name=name, strategy="trend_filter", instrument=instrument, venue=kw.get("venue", "kraken"),
                                bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=0.0, params={}))
    return out


def test_each_asset_leg_runs_its_own_model_on_a_clone_with_an_account_of_its_own(built):
    node = node_mod.build_legs_node(_legs(), log_level="ERROR", asset_fetch=dict, hub=HUB)
    strategies = built["strategies"]
    assert [type(s).__name__ for s in strategies] == ["TrendFilter"] * 3  # never a LegFollower
    assert all(getattr(s, "pair", None) is None for s in strategies)  # never a pair
    assert [str(s._cfg.instrument_id) for s in strategies] == ["BTC/USD.KRAKEN", "ETH/USD.KRAKEN_P2",
                                                               "SOL/USD.KRAKEN_P3"]
    assert [(str(a[0]), a[2]) for a in built["accounts"]] == [
        ("KRAKEN", "KRAKEN-PAPER-TREND-X"), ("KRAKEN_P2", "KRAKEN_P2-PAPER-TREND-X-ETH"),
        ("KRAKEN_P3", "KRAKEN_P3-PAPER-TREND-X-SOL")]
    node.dispose()


def test_one_hub_connection_feeds_every_leg_each_by_its_clone(built):
    node_mod.build_legs_node(_legs(), log_level="ERROR", asset_fetch=dict, hub=HUB)
    (cfg,) = built["data"]
    assert cfg.instrument_ids == ("BTC/USD.KRAKEN", "ETH/USD.KRAKEN", "SOL/USD.KRAKEN")
    assert cfg.aliases == (("ETH/USD.KRAKEN", "ETH/USD.KRAKEN_P2"), ("SOL/USD.KRAKEN", "SOL/USD.KRAKEN_P3"))
    lead, *rest = built["strategies"]
    assert isinstance(lead.hub_status, HubStatus)
    assert all(isinstance(s.hub_status, HubStatusView) and s.hub_status.base is lead.hub_status for s in rest)


def test_a_pair_leg_or_a_second_venue_is_refused(built):
    legs = _legs()
    with pytest.raises(ValueError, match="one venue"):
        other = SleeveConfig(name="trend-x~ETH", strategy="trend_filter", instrument="ETH/USDT", venue="binance",
                             bar_spec="15-MINUTE-LAST-INTERNAL", starting_balance=0.0, params={"market": "perp"})
        node_mod.build_legs_node([legs[0], other], log_level="ERROR", asset_fetch=dict, hub=HUB)
    with pytest.raises(ValueError, match="at least two"):
        node_mod.build_legs_node(legs[:1], log_level="ERROR", asset_fetch=dict, hub=HUB)
