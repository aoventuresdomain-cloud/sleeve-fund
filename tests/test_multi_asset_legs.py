"""Multi-asset Strategies, the store (pe1-plans/multi-asset-spec.md §1-3 and §8 cells 1, 5, 10; HoE: one leg row per
asset, lifecycle fans out to every leg in one transaction; QT3 b3c: an asset-list edit is refused). A Strategy lists its
assets; the parent row trades the first, and each other asset is a leg row "<strategy>~<UNDERLYING>" under it. Legs are
independent ideas, never a hedged pair: no pair machinery applies. Made inside the fund only, every leg with 0."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from sleeve_fund.store import Store

PERP = {"market": "perp", "allow_short": True, "stop_loss": 0.02}
ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _fund(store):
    at = datetime.now(UTC) - timedelta(minutes=5)
    store.settle_flow(store.fund_flow("deposit", 100_000, actor="pm", reason="test", rebase=True, now=at), at)
    return store


def _make(store, name="trend-x", assets=ASSETS, **kw):
    return store.create_legs(name=name, strategy="ping_pong", bar_spec="1-MINUTE-LAST-INTERNAL", assets=assets,
                             params=PERP, starting_balance=0, universe_date="2026-10-15", **kw)


def test_a_three_asset_strategy_is_one_parent_and_two_asset_legs():
    store = _fund(Store.in_memory())
    rows = _make(store)
    assert [(r.name, r.instrument, r.parent) for r in rows] == [
        ("trend-x", "BTC/USDT", None), ("trend-x~ETH", "ETH/USDT", "trend-x"), ("trend-x~SOL", "SOL/USDT", "trend-x")]
    assert all(r.starting_balance == 0 for r in rows)
    assert store.sleeve("trend-x").params["assets"] == ASSETS
    assert store.sleeve("trend-x").params["universe_date"] == "2026-10-15"
    assert store.legs("trend-x~SOL") == store.legs("trend-x") == ["trend-x", "trend-x~ETH", "trend-x~SOL"]
    assert [s.name for s in store.sleeves(include_legs=False)] == ["trend-x"]  # a leg is never listed as a Strategy
    assert store.multi_asset("trend-x~ETH") and not store.two_leg("trend-x")  # never a pair


@pytest.mark.parametrize("assets, why", [
    (["BTC/USDT"], "at least two"), (["BTC/USDT", "BTC/USDT"], "twice"), (["BTC/USDT", "BTC/USD"], "twice"),
    (["BTC/USDT", "ETH-X/USDT"], "letters or digits")])  # 0028's leg-name check
def test_an_asset_list_with_one_asset_or_a_repeat_is_refused(assets, why):
    with pytest.raises(ValueError, match=why):
        _make(_fund(Store.in_memory()), assets=assets)


def test_a_leg_under_a_leg_and_a_tilde_in_the_name_are_refused():
    store = _fund(Store.in_memory())
    _make(store)
    with pytest.raises(ValueError, match="~"):
        _make(store, name="trend-x~ETH")
    with pytest.raises(ValueError, match="~"):
        _make(store, name="a~b")


def test_outside_the_fund_a_multi_asset_strategy_is_refused():
    with pytest.raises(ValueError, match="inside the fund"):
        _make(Store.in_memory())


def test_an_asset_list_edit_is_refused_and_the_strategy_keeps_its_legs():
    store = _fund(Store.in_memory())
    _make(store)
    with pytest.raises(ValueError, match="assets"):
        store.change_settings("trend-x", risk_profile="balanced", warmup_bars=0,
                              params={**store.sleeve("trend-x").params, "assets": ["BTC/USDT", "ETH/USDT"]})
    assert store.legs("trend-x") == ["trend-x", "trend-x~ETH", "trend-x~SOL"]


def test_a_settings_change_applies_to_every_leg():
    store = _fund(Store.in_memory())
    _make(store)
    store.change_settings("trend-x", risk_profile="aggressive", warmup_bars=0,
                          params={**store.sleeve("trend-x").params, "stop_loss": 0.03})
    for name in store.legs("trend-x"):
        s = store.sleeve(name)
        assert (s.risk_profile, s.params["stop_loss"]) == ("aggressive", 0.03), name
    with pytest.raises(ValueError, match="leg"):  # one settings block: a leg is never edited on its own
        store.change_settings("trend-x~ETH", risk_profile="balanced", warmup_bars=0, params=PERP)


def test_start_and_stop_act_on_every_leg():
    store = _fund(Store.in_memory())
    _make(store)
    store.set_desired_state("trend-x", "stopped")
    assert {store.sleeve(n).desired_state for n in store.legs("trend-x")} == {"stopped"}


@pytest.mark.parametrize("action, to", [("wind_down", "winding_down"), ("close_now", "retired")])
def test_a_lifecycle_move_acts_on_every_leg_in_one_go(action, to, tmp_path):
    store = _fund(Store(f"sqlite:///{tmp_path}/ma.db"))  # a put-away's writes need their own connection (F286-ENG)
    _make(store)
    store.set_lifecycle("trend-x", action, "pm", "PM's call")
    assert {store.sleeve(n).lifecycle for n in store.legs("trend-x")} == {to}
    with pytest.raises(ValueError, match="leg"):
        store.set_lifecycle("trend-x~ETH", "wind_down", "pm", "on one leg")


def test_a_retire_after_wind_down_waits_for_every_leg_then_puts_each_away(tmp_path):
    store = _fund(Store(f"sqlite:///{tmp_path}/ma.db"))
    _make(store)
    store.set_lifecycle("trend-x", "wind_down", "pm", "PM's call")
    assert store.complete_retire("trend-x~ETH")  # one leg's process finding every leg flat retires the Strategy
    assert {store.sleeve(n).lifecycle for n in store.legs("trend-x")} == {"retired"}
    assert not store.complete_retire("trend-x~SOL")  # once, never twice


def test_the_asset_list_is_part_of_the_idea_hash_and_its_order_is_not():
    from sleeve_fund.research.candidates import idea_hash

    d = {"blocks": {}, "long": {"all": []}}
    two, other = {**d, "assets": ["BTC/USDT", "ETH/USDT"]}, {**d, "assets": ["BTC/USDT", "SOL/USDT"]}
    assert idea_hash(two) != idea_hash(d) and idea_hash(two) != idea_hash(other)
    assert idea_hash(two) == idea_hash({**d, "assets": ["ETH/USDT", "BTC/USDT"]})
    assert idea_hash(d) == idea_hash({**d, "assets": []})  # no list: the hash every single-asset idea has today


def test_a_put_away_with_a_leg_held_waits_for_that_leg_then_puts_every_leg_away(tmp_path):
    """Cell 10 (HoE, spec §10): close now with one leg still holding winds every leg down, due now, so nothing opens;
    nothing is put away until that leg is flat. Its exit's fill makes it flat and the next check puts every leg away.
    With every leg flat, close now puts them all away at once (the cell above)."""
    from decimal import Decimal

    store = _fund(Store(f"sqlite:///{tmp_path}/ma.db"))
    _make(store)
    store.record_fill("trend-x~ETH", side="BUY", qty=Decimal("0.5"), price=Decimal(2400), fee=Decimal("0.6"),
                      order_id="o1", trade_id="t1")
    store.set_lifecycle("trend-x", "close_now", "pm", "PM's call")
    assert {store.sleeve(n).lifecycle for n in store.legs("trend-x")} == {"winding_down"}
    assert all(store.sleeve(n).wind_down_deadline is not None for n in store.legs("trend-x"))  # due now: no entry
    assert not store.complete_retire("trend-x~SOL")  # a flat leg's process can't put the Strategy away alone
    assert "trend-x" not in store.archived()
    store.record_fill("trend-x~ETH", side="SELL", qty=Decimal("0.5"), price=Decimal(2410), fee=Decimal("0.6"),
                      order_id="o2", trade_id="t2")
    assert store.complete_retire("trend-x~ETH")
    assert {store.sleeve(n).lifecycle for n in store.legs("trend-x")} == {"retired"}
    assert {store.sleeve(n).desired_state for n in store.legs("trend-x")} == {"stopped"}
    assert set(store.legs("trend-x")) <= set(store.archived())
