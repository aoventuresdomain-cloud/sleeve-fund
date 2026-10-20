"""B-29: a Strategy whose name's node tag, cut to 20 characters, ends in "-" crashed its node at start.

The form's own suggested names did it (candle-step-btcusdt-1m, trend-filter-btcusd-1h): the trader id became
"PAPER-CANDLE-STEP-BTCUSDT-", which the engine refuses, so the node exited 1 and restarted in a loop. Every id
built from the cut tag must be one the engine takes, and a name that works today must keep the ids it has.

HoE QA ruling (20 Oct): ids must also be distinct for distinct names. A bare 20-character cut gave
candle-step-btcusdt-1m and candle-step-btcusdt-5m one trader and one account (PAPER-CANDLE-STEP-BTCUSDT), so their
orders and fills mixed. A tag of 20 characters or fewer keeps its ids exactly; a longer one is its cut plus a hash of
the whole tag; and a name whose ids would equal a live Strategy's is refused at create."""
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
def test_a_short_name_keeps_the_ids_it_has_today(name):
    tag = node._tag(name)
    if len(tag) > 20:
        pytest.skip("a longer tag gets a hash (HoE ruling 2): see the next cells")
    venue = next(iter(VENUES)).upper()
    assert node.trader_id(tag) == f"PAPER-{tag}"
    assert str(node.strategy_id("CandleStep", name)).startswith(f"CandleStep-{tag}-")
    assert router.route(name, venue, VENUES, tag=tag).account_id == f"{venue}-PAPER-{tag}"
    assert node.account_id(f"{venue}_P1", name) == f"{venue}_P1-PAPER-{tag}"


def test_the_pms_short_names_keep_their_ids_exactly():
    assert node.trader_id(node._tag("cs-default")) == "PAPER-CS-DEFAULT"
    assert node.trader_id(node._tag("timed-test")) == "PAPER-TIMED-TEST"
    assert router.cut_tag("A" * 20) == "A" * 20


@pytest.mark.parametrize("name", [n for n in SUGGESTED + EDGES if len(node._tag(n)) > 20])
def test_a_long_name_gets_its_cut_plus_a_hash_of_the_whole_tag(name):
    tag = node._tag(name)
    cut = router.cut_tag(tag)
    head, _, digest = cut.rpartition("-")
    assert len(cut) <= 20 and re.fullmatch(r"[0-9A-F]{6}", digest) and tag.startswith(head) and not head.endswith("-")


def test_two_names_sharing_their_first_20_tag_characters_get_different_ids():
    """RED on 274ecd88 and on the first B-29 cut: both were trader PAPER-CANDLE-STEP-BTCUSDT and one account."""
    a, b = "candle-step-btcusdt-1m", "candle-step-btcusdt-5m"
    venue = next(iter(VENUES)).upper()
    assert node.trader_id(node._tag(a)) != node.trader_id(node._tag(b))
    assert (router.route(a, venue, VENUES, tag=node._tag(a)).account_id
            != router.route(b, venue, VENUES, tag=node._tag(b)).account_id)
    assert node.account_id(f"{venue}_P1", a) != node.account_id(f"{venue}_P1", b)


def test_every_suggested_and_edge_name_gets_its_own_trader_id():
    names = sorted(set(SUGGESTED + EDGES))
    ids = {node.trader_id(node._tag(n)) for n in names}
    assert len(ids) == len(names)


def test_no_id_in_the_paper_node_or_router_cuts_a_tag_by_hand():
    """Pins the call sites: every cut goes through router.cut_tag, so no builder can drift back to a raw [:20]."""
    for path in ("sleeve_fund/paper/node.py", "sleeve_fund/router.py"):
        src = (pathlib.Path(__file__).resolve().parents[1] / path).read_text()
        assert "[:20]" not in src, path
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


def test_a_name_with_no_letter_or_digit_is_refused_not_built_into_an_empty_id():
    with pytest.raises(ValueError, match="letter or digit"):
        router.cut_tag("")
    with pytest.raises(ValueError, match="letter or digit"):
        node.trader_id(node._tag("---"))
    from sleeve_fund.dashboard.app import NAME_RE
    assert not NAME_RE.match("---") and not NAME_RE.match("-a")  # the form already refuses such a name


def test_a_strategy_put_away_after_an_error_no_longer_counts_as_halted(tmp_path, monkeypatch):
    """The 15 Oct crash left the archived candle-step Strategy in "error", and the top bar kept saying "1 halted"."""
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod
    from sleeve_fund.store import Store

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    store = Store(f"sqlite:///{tmp_path}/t.db")
    for name in ("candle-step-btcusdt-1m", "timed-test"):
        store.create_sleeve(name=name, strategy="candle_step", instrument="BTC/USDT",
                            bar_spec="1-MINUTE-LAST-INTERNAL", starting_balance=0, desired_state="stopped")
    store.set_desired_state("candle-step-btcusdt-1m", "running")  # TEC-ERR-STOPPED (HoE): it errored while running
    store.set_status("candle-step-btcusdt-1m", "error", "node exited 1")
    client = TestClient(app_mod.create_app(store))

    def bar():
        page = client.get("/", auth=("pm", "test-pw")).text
        return page.split('data-live="health"', 1)[1].split("</span></span>", 1)[0]

    assert "1 halted" in bar()  # red-first: it waits for the PM while it is still a live Strategy
    store.set_desired_state("candle-step-btcusdt-1m", "stopped")  # archiving needs a stop
    store.archive("candle-step-btcusdt-1m")
    assert "halted" not in bar()


@pytest.fixture(params=["sqlite", "postgres"])
def journal(request, tmp_path):
    import os

    from sleeve_fund.store import Store, make_engine, metadata

    if request.param == "sqlite":
        return Store(f"sqlite:///{tmp_path}/t.db")
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("CI runs these against Postgres too (TEST_DATABASE_URL)")
    engine = make_engine(url)
    metadata.drop_all(engine)
    return Store(engine=engine)


def _make(store, name):
    store.create_sleeve(name=name, strategy="candle_step", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                        starting_balance=0, desired_state="stopped")


def test_a_name_whose_ids_equal_a_live_strategys_is_refused_at_create(journal):
    """HoE ruling 3: Timed Test and timed-test share one tag, so one trader and one account."""
    store = journal
    _make(store, "timed-test")
    with pytest.raises(ValueError, match="share ids with Strategy timed-test"):
        _make(store, "Timed Test")
    assert [s.name for s in store.sleeves()] == ["timed-test"]
    _make(store, "candle-step-btcusdt-1m")
    _make(store, "candle-step-btcusdt-5m")  # distinct ids now: both are made


def test_a_put_away_strategy_doesnt_block_a_name_with_its_ids(journal):
    store = journal
    _make(store, "timed-test")
    store.archive("timed-test")
    _make(store, "Timed_Test")
    assert "Timed_Test" in {s.name for s in store.sleeves()}


def test_bringing_back_a_strategy_whose_ids_a_live_one_now_has_is_refused_in_words(journal):
    """CR MUST (B-29 v2): put timed-test away, make Timed Test, bring timed-test back: both would be live on one
    trader and one account. Bring-back refuses as create does, and moves nothing."""
    store = journal
    _make(store, "timed-test")
    store.archive("timed-test")
    _make(store, "Timed Test")
    with pytest.raises(ValueError, match="timed-test can't be brought back: its orders would share ids with "
                                         "Strategy Timed Test"):
        store.unarchive("timed-test")
    assert "timed-test" in store.archived()  # still put away
    store.archive("Timed Test")
    store.unarchive("timed-test")  # once the other is put away, it comes back


def test_the_dashboards_bring_back_shows_the_refusal(journal, monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from sleeve_fund.dashboard import app as app_mod

    monkeypatch.setenv("DASHBOARD_PASSWORD", "test-pw")
    monkeypatch.setenv("TEARSHEET_DIR", str(tmp_path))
    store = journal
    _make(store, "timed-test")
    store.archive("timed-test")
    _make(store, "Timed Test")
    c = TestClient(app_mod.create_app(store))
    r = c.post("/sleeves/timed-test/archive", data={"action": "restore", "reason": "back"}, auth=("pm", "test-pw"),
               headers={"origin": "http://testserver"}, follow_redirects=False)
    assert r.status_code == 400 and "can't be brought back" in r.text and "Timed Test" in r.text
