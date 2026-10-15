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


def test_one_assets_stale_data_holds_only_that_legs_entries(built, tmp_path):
    """Done-when 4: each leg runs on its own runtime, so a hold is that leg's alone. Stale data is measured on each
    leg's own instrument (test_stale_data: another instrument's prints never keep it fresh) and a clone is stale exactly
    when its source is (test_two_legs_hub_alias); here, the stale hold one leg's watchdog sets blocks that leg's entries
    and no other leg's."""
    from datetime import UTC, datetime, timedelta

    from sleeve_fund.paper.runtime import SleeveRuntime, block_codes
    from sleeve_fund.store import Store

    store = Store(f"sqlite:///{tmp_path}/ma.db")
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    store.create_legs(name="trend-x", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL",
                      assets=["BTC/USD", "ETH/USD", "SOL/USD"], params={}, starting_balance=0,
                      universe_date="2026-10-15", venue="kraken")
    runtimes = [SleeveRuntime(store, n) for n in store.legs("trend-x")]
    node = node_mod.build_legs_node(_legs(), log_level="ERROR", asset_fetch=dict, hub=HUB, runtimes=runtimes)
    assert [s.runtime for s in built["strategies"]] == runtimes
    runtimes[1].holds["stale_data"] = "last price 400 s old. It clears when data resumes"
    assert "stale_data" in block_codes(runtimes[1].entry_blocked()[1])
    assert all("stale_data" not in block_codes(r.entry_blocked()[1]) for r in (runtimes[0], runtimes[2]))
    node.dispose()


def _stored_legs():
    """MA-PARAMS-1 (CR): the rows a real create_legs writes, whose params carry the Strategy's asset list and the
    date it was chosen beside the model's own settings (the cells above build from params={}, which hid it)."""
    from test_multi_asset_legs import _fund

    from sleeve_fund.store import Store

    store = _fund(Store.in_memory())
    return store.create_legs(name="trend-x", strategy="trend_filter", bar_spec="15-MINUTE-LAST-INTERNAL",
                             assets=["BTC/USD", "ETH/USD", "SOL/USD"], params={}, starting_balance=0,
                             universe_date="2026-10-15", venue="kraken")


def test_a_multi_asset_strategy_starts_from_its_stored_rows(built):
    """MA-PARAMS-1: the asset list and universe_date are the Strategy's, never its model's: each row's config leaves
    them out, so its model takes its settings and the node starts every leg."""
    from sleeve_fund.paper.config import from_store

    rows = _stored_legs()
    assert rows[0].params["assets"] == ["BTC/USD", "ETH/USD", "SOL/USD"]  # still kept on the row
    legs = [from_store(r) for r in rows]
    assert all("assets" not in c.params and "universe_date" not in c.params for c in legs)
    node = node_mod.build_legs_node(legs, log_level="ERROR", asset_fetch=dict, hub=HUB)
    assert [type(s).__name__ for s in built["strategies"]] == ["TrendFilter"] * 3
    node.dispose()


def test_a_multi_asset_rows_model_builds_and_counts_its_warm_up_from_its_stored_params():
    """MA-PARAMS-1: the chart's indicator lines and warm-up (strategies.series) read the row's params as stored."""
    from sleeve_fund.instruments import spot_pair
    from sleeve_fund.strategies import series
    from sleeve_fund.venues import venue as venue_profile

    row, profile = _stored_legs()[1], venue_profile("kraken")
    inst = spot_pair("ETH", "USD", fees=profile.fees, venue=profile.venue, price_precision=8)
    assert series.warmup(row.strategy, row.params, 15) > 0
    assert series.build(row.strategy, inst, row.params, 15) is not None
