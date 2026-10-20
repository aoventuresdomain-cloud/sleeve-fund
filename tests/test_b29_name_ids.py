"""B-29: a Strategy whose name's node tag, cut to 20 characters, ends in "-" crashed its node at start.

The form's own suggested names did it (candle-step-btcusdt-1m, trend-filter-btcusd-1h): the trader id became
"PAPER-CANDLE-STEP-BTCUSDT-", which the engine refuses, so the node exited 1 and restarted in a loop. Every id
built from the cut tag must be one the engine takes, and a name that works today must keep the ids it has."""
import itertools
import pathlib
import re

import pytest
from nautilus_trader.model import AccountId, StrategyId, TraderId

from sleeve_fund import router
from sleeve_fund.paper import node
from sleeve_fund.strategies import REGISTRY
from sleeve_fund.venues import VENUES

PAIRS = ("BTC/USDT", "ETH/USDT", "SOL/USDT", "BTC/USD", "ETH/USD", "BTC/USDC")
CANDLES = ("1m", "3m", "5m", "15m", "30m", "1h", "4h", "1d")


def suggested(model: str, pair: str, candle: str) -> str:
    """The New Strategy form's suggested name, as console.js builds it (model, instrument, candle; 41 characters)."""
    name = "-".join(x for x in (model.replace("_", "-"), pair.replace("/", "").lower(), candle) if x)
    return re.sub(r"[^a-z0-9-]", "", name)[:41]


SUGGESTED = sorted({suggested(m, p, c) for m, p, c in itertools.product(REGISTRY, PAIRS, CANDLES)})
EDGES = ["candle-step-btcusdt-1m", "trend-filter-btcusd-1h", "x" * 19 + "-y", "a" * 41, "timed-test", "q"]


def _old_cut(tag: str) -> str:
    return tag[:20]


def test_the_suggested_names_include_the_two_that_crashed():
    assert {"candle-step-btcusdt-1m", "trend-filter-btcusd-1h"} <= set(SUGGESTED)
    assert any(_old_cut(node._tag(n)).endswith("-") for n in SUGGESTED)


@pytest.mark.parametrize("name", SUGGESTED + EDGES)
def test_every_id_the_paper_node_builds_from_a_name_is_one_the_engine_takes(name):
    tag = node._tag(name)
    TraderId.from_str(node.trader_id(tag))
    StrategyId.from_str(str(node.strategy_id("CandleStep", name)))
    venue = next(iter(VENUES)).upper()
    AccountId.from_str(router.route(name, venue, VENUES, tag=tag).account_id)
    AccountId.from_str(node.account_id(f"{venue}_P1", name))


@pytest.mark.parametrize("name", SUGGESTED + EDGES)
def test_a_name_that_works_today_keeps_its_ids(name):
    tag = node._tag(name)
    if _old_cut(tag).endswith("-"):
        pytest.skip("this name crashed before; it had no ids to keep")
    venue = next(iter(VENUES)).upper()
    assert node.trader_id(tag) == f"PAPER-{tag[:20]}"
    assert str(node.strategy_id("CandleStep", name)).startswith(f"CandleStep-{tag[:20]}-")
    assert router.route(name, venue, VENUES, tag=tag).account_id == f"{venue}-PAPER-{tag[:20]}"
    assert node.account_id(f"{venue}_P1", name) == f"{venue}_P1-PAPER-{tag[:20]}"


def test_no_id_in_the_paper_node_or_router_cuts_a_tag_by_hand():
    """Pins the call sites: every cut goes through router.cut_tag, so no builder can drift back to a raw [:20]."""
    for path in ("sleeve_fund/paper/node.py", "sleeve_fund/router.py"):
        src = (pathlib.Path(__file__).resolve().parents[1] / path).read_text()
        assert "[:20]" not in src.replace("tag[:20].rstrip", ""), path
    src = (pathlib.Path(__file__).resolve().parents[1] / "sleeve_fund/paper/node.py").read_text()
    assert src.count("TraderId.from_str(trader_id(tag))") == 3


@pytest.mark.parametrize("name", ["candle-step-btcusdt-1m", "trend-filter-btcusd-1h"])
def test_the_real_paper_node_builds_for_the_names_that_crashed(name, monkeypatch):
    """The real build_node, as `paper node main` calls it: on main 274ecd88 this raised ValueError at the trader id."""
    import os

    from sleeve_fund.paper.config import SleeveConfig

    for k in list(os.environ):
        if k.upper().startswith("KRAKEN_"):
            monkeypatch.delenv(k)
    cfg = SleeveConfig(name=name, strategy="trend_filter", instrument="BTC/USD", bar_spec="1-MINUTE-LAST-INTERNAL",
                       starting_balance=1000.0)
    built = node.build_node(cfg, log_level="ERROR", asset_fetch=dict)
    try:
        assert str(built.trader_id) == node.trader_id(node._tag(name))
        assert not str(built.trader_id).endswith("-")
    finally:
        built.dispose()
