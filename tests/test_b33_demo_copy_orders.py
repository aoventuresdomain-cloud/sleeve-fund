"""B-33: the demo copy sends one demo order per paper ORDER, and every exit goes reduce-only.

On 15 Oct a paper order filled in pieces became several demo orders, and an exit larger than the demo position
opened the other side (the demo showed "Open Short"). Fake demo endpoint only: nothing here reaches a real account."""
import json
import os
import uuid

import pytest
from sqlalchemy import create_engine, text

from sleeve_fund import mirror
from sleeve_fund.store import CLOSING_INTENTS, Store


class FakeDemo(mirror.BybitDemo):
    """The demo account's answers, without the network. A reduce-only order never moves the position past flat."""

    def __init__(self, held=0.0):
        super().__init__(mirror.Settings("k", "s"), http=self.http)
        self.orders, self.held = [], held

    def http(self, method, url, headers, body):
        if "/v5/position/list" in url:
            side = "Buy" if self.held > 0 else "Sell" if self.held < 0 else ""
            return {"retCode": 0, "result": {"list": [{"symbol": "BTCUSDT", "side": side, "size": str(abs(self.held))}]}}
        if "/v5/order/realtime" in url:
            return {"retCode": 0, "result": {"list": [{"avgPrice": "60000"}]}}
        o = json.loads(body)
        self.orders.append(o)
        move = (1 if o["side"] == "Buy" else -1) * float(o["qty"])
        if o.get("reduceOnly") and (self.held == 0 or (self.held > 0) == (move > 0)):
            return {"retCode": 110017, "retMsg": "reduce-only order would increase the position"}
        if o.get("reduceOnly"):
            move = max(-abs(self.held), min(abs(self.held), move)) if self.held else 0.0
        self.held = round(self.held + move, 8)
        return {"retCode": 0, "result": {"orderId": str(len(self.orders))}}


PG = os.environ.get("TEST_DATABASE_URL")
_SCHEMAS: list[str] = []
_ENGINES: list = []


@pytest.fixture(scope="module", autouse=True)
def _drop_schemas():
    yield
    for eng in _ENGINES:
        eng.dispose()
    if PG and _SCHEMAS:
        admin = create_engine(PG)
        for name in _SCHEMAS:
            with admin.begin() as c:
                c.execute(text(f"DROP SCHEMA IF EXISTS {name} CASCADE"))
        admin.dispose()


def _journal():
    """SQLite in memory, or a fresh Postgres schema when TEST_DATABASE_URL is set (CI): both engines."""
    if not PG:
        return Store.in_memory()
    name = f"b33_{uuid.uuid4().hex[:12]}"
    admin = create_engine(PG)
    with admin.begin() as c:
        c.execute(text(f"CREATE SCHEMA {name}"))
    admin.dispose()
    _SCHEMAS.append(name)
    eng = create_engine(PG, connect_args={"options": f"-csearch_path={name}"})
    _ENGINES.append(eng)
    return Store(engine=eng)


def _store(*names):
    store = _journal()
    for name in names or ("bn-ls",):
        store.create_sleeve(name=name, strategy="ping_pong", instrument="BTC/USDT", bar_spec="1-MINUTE-LAST-INTERNAL",
                            starting_balance=10_000, params={"market": "perp", "allow_short": True, "demo_mirror": True},
                            venue="binance")
    return store


def _order(store, name, oid, side, qty, intent="entry", reason=""):
    store.record_order(name, order_id=oid, side=side, qty=qty, intent=intent, reason=reason)


def _piece(store, name, oid, side, qty, n, status):
    store.record_fill(name, side=side, qty=qty, price=60_000.0, fee=1.0, order_id=oid, trade_id=f"{oid}-t{n}")
    store.update_order(oid, status=status, fill_qty=qty, fill_px=60_000.0)


def _start(store, demo):
    targets = {"BYBIT": demo}
    mirror.mirror_once(store, targets)  # first sight: the watermark
    return targets


def test_a_paper_order_filled_in_three_pieces_is_one_demo_order_for_the_sum():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.3)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "partially_filled")
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 3, "filled")
    mirror.mirror_once(store, targets)
    assert [(o["side"], o["qty"]) for o in demo.orders] == [("Buy", "0.3")]
    assert demo.held == pytest.approx(0.3)
    assert mirror.mirror_once(store, targets) == 0  # nothing sent twice


def test_an_order_still_filling_is_not_copied_until_it_is_done():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.2)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    assert mirror.mirror_once(store, targets) == 0 and demo.orders == []
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "filled")
    assert mirror.mirror_once(store, targets) == 1
    assert [o["qty"] for o in demo.orders] == ["0.2"]


def test_catch_up_between_two_pieces_of_one_order_sends_nothing():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.2)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "partially_filled")
    mirror.mirror_once(store, targets)
    seen = mirror.catch_up(store, targets, {})
    seen = mirror.catch_up(store, targets, seen)  # the same gap twice: still being copied, not a missed copy
    assert demo.orders == []
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 2, "filled")
    mirror.mirror_once(store, targets)
    mirror.catch_up(store, targets, mirror.catch_up(store, targets, {}))
    assert [o["qty"] for o in demo.orders] == ["0.2"] and demo.held == pytest.approx(0.2)


@pytest.mark.parametrize("intent", sorted({*CLOSING_INTENTS, "liquidation"}))
def test_every_closing_order_is_sent_reduce_only(intent):
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent=intent)
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    entry, close = demo.orders
    assert not entry.get("reduceOnly") and close.get("reduceOnly") is True
    assert demo.held == 0.0


def test_a_reset_close_is_reduce_only():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="pm_flatten", reason="Reset strategy")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    assert demo.orders[-1].get("reduceOnly") is True and "-reset-" in demo.orders[-1]["orderLinkId"]


def test_an_exit_larger_than_the_demo_position_never_opens_the_other_side():
    """The 15 Oct case: the demo held less than the paper book, and the full exit opened a short."""
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.held = 0.04  # part of the copy never landed
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="exit")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    assert demo.held >= 0.0  # flat at worst, never short
    # capped by the mirror itself, not left to the venue: only what the account holds is sent
    assert demo.orders[-1]["qty"] == "0.04" and demo.orders[-1]["reduceOnly"] is True
    assert store.mirror_positions() == {"BTCUSDT": pytest.approx(0.06)}  # 0.1 put on, 0.04 taken off


def test_a_close_when_the_demo_account_holds_nothing_on_that_side_sends_nothing():
    store, demo = _store(), FakeDemo()
    targets = _start(store, demo)
    _order(store, "bn-ls", "o1", "BUY", 0.1)
    _piece(store, "bn-ls", "o1", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    demo.held = 0.0  # closed by hand on the demo account
    _order(store, "bn-ls", "o2", "SELL", 0.1, intent="stop_loss")
    _piece(store, "bn-ls", "o2", "SELL", 0.1, 1, "filled")
    assert mirror.mirror_once(store, targets) == 0
    assert len(demo.orders) == 1 and demo.held == 0.0
    assert store.mirror_rows("bn-ls")[0]["status"] == "skipped"


def test_two_long_names_never_share_a_demo_link_id():
    a, b = "candle-step-btcusdt-1m-" + "z" * 17 + "a", "candle-step-btcusdt-1m-" + "z" * 17 + "b"
    assert len(a) == len(b) == 41
    store, demo = _store(a, b), FakeDemo()
    targets = _start(store, demo)
    for n, name in enumerate((a, b)):
        _order(store, name, f"o{n}", "BUY", 0.1)
        _piece(store, name, f"o{n}", "BUY", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    links = [o["orderLinkId"] for o in demo.orders]
    assert len(links) == 2 and len(set(links)) == 2 and all(len(x) <= 36 for x in links)


def test_two_long_names_never_share_a_reset_link_id():
    a, b = "x" * 20 + "-one", "x" * 20 + "-two"
    store, demo = _store(a, b), FakeDemo()
    targets = _start(store, demo)
    for n, name in enumerate((a, b)):
        _order(store, name, f"e{n}", "BUY", 0.1)
        _piece(store, name, f"e{n}", "BUY", 0.1, 1, "filled")
        _order(store, name, f"r{n}", "SELL", 0.1, intent="pm_flatten", reason="Reset strategy")
        _piece(store, name, f"r{n}", "SELL", 0.1, 1, "filled")
    mirror.mirror_once(store, targets)
    resets = [o["orderLinkId"] for o in demo.orders if "-reset-" in o["orderLinkId"]]
    assert len(resets) == 2 and len(set(resets)) == 2
